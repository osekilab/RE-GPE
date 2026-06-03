import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_bow_token_mask(tokenizer: PreTrainedTokenizer, device: torch.device) -> torch.Tensor:
    """
    Create a boolean mask for Beginning-of-Word (BOW) tokens in the vocabulary.

    Supports multiple tokenizer types:
    - GPT-2 family: BOW tokens start with 'Ġ' (U+0120, G with combining breve)
    - LLaMA/SentencePiece: BOW tokens start with '▁' (U+2581, lower one eighth block)

    Args:
        tokenizer: The tokenizer with a vocab
        device: Device to place the mask tensor

    Returns:
        Boolean tensor of shape (vocab_size,) where True indicates BOW tokens
    """
    vocab_size = len(tokenizer)
    bow_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)

    # Auto-detect tokenizer type by checking a sample token
    # Common space-marked token: look for token containing space representation
    bow_marker = None
    for token_id in range(min(1000, vocab_size)):  # Check first 1000 tokens
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(token_str, str):
            if 'Ġ' in token_str:
                bow_marker = 'Ġ'  # GPT-2 style (BPE)
                break
            elif '▁' in token_str:
                bow_marker = '▁'  # LLaMA/SentencePiece style
                break

    # Fallback: assume GPT-2 style if no marker detected
    if bow_marker is None:
        bow_marker = 'Ġ'

    # Check each token in vocabulary
    for token_id in range(vocab_size):
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        # BOW tokens start with the detected marker
        if isinstance(token_str, str) and token_str.startswith(bow_marker):
            bow_mask[token_id] = True

    return bow_mask


def compute_surprisal(
    inputs: dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    word_ids: torch.Tensor,  # (batch_size, seq_len)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute surprisal values given the inputs and model.

    Args:
        inputs (dict): Tokenized inputs with input_ids and attention_mask
        model (PreTrainedModel): The language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        word_ids (torch.Tensor): Word ids for the input, only used as mask for loss

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - surprisal: (batch_size, seq_len) - token-level surprisal
            - loss: scalar - cross-entropy loss
            - log_probs: (batch_size, seq_len+1, vocab_size) - log probabilities

    """

    # Manually add bos token to input since add_special tokens has no effect for this tokenizer
    bos_tokens_tensor = torch.tensor(
        [[tokenizer.bos_token_id]] * inputs["input_ids"].size(dim=0)
    ).to(model.device)  # (batch_size, 1)

    input_ids = torch.cat(
        (
            bos_tokens_tensor,
            inputs["input_ids"],  # (batch_size, seq_len)
        ),
        dim=1,
    )  # (batch_size, seq_len+1)
    # Mask bos token
    attention_mask = torch.cat(
        [
            torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(model.device),  # (batch_size, 1)
            inputs["attention_mask"],  # (batch_size, seq_len)
        ],
        dim=1,
    )  # (batch_size, seq_len+1)

    # Mask labels with -100 where input ids are -1 for correct ce loss and perplexity calculation
    mask = word_ids != -1  # (batch_size, seq_len)
    mask = torch.cat(
        [
            torch.ones(bos_tokens_tensor.size(), dtype=torch.bool).to(model.device),  # (batch_size, 1)
            mask,  # (batch_size, seq_len)
        ],
        dim=1,
    )  # (batch_size, seq_len+1)
    labels = torch.where(mask, input_ids, -100)  # (batch_size, seq_len+1)

    # Compute logits and loss
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # outputs.logits: (batch_size, seq_len+1, vocab_size)

    # Shift tokens
    # Use natural log for surprisal (consistent with original implementation and existing trained models)
    log_probs = torch.log_softmax(outputs.logits, dim=-1)  # (batch_size, seq_len+1, vocab_size)
    log_probs_shifted = log_probs[:, :-1, :]  # (batch_size, seq_len, vocab_size)
    input_ids_shifted = input_ids[:, 1:]  # (batch_size, seq_len)

    # Gather log probabilities for the correct tokens
    surprisal = torch.mul(
        -1,
        torch.gather(log_probs_shifted, 2, input_ids_shifted[:, :, None]).squeeze(-1),
    )  # (batch_size, seq_len)

    return surprisal, outputs.loss, log_probs


def compute_surprisal_with_wt_decoding(
    inputs: dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    word_ids: torch.Tensor,  # (batch_size, seq_len)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute surprisal with Whitespace-Trailing (WT) decoding correction.

    This implements the correction from Oh & Schuler (2024) and Pimentel & Meister (2024)
    to properly compute word probabilities from subword-tokenized language models with
    leading whitespace tokens (e.g., GPT-2 family).

    The WT decoding formula (Oh & Schuler 2024, Eq. 2):
        P(w'_{t+1} | w'_{1..t}) = P(w_{t+1} | w_{1..t}) × [P(next∈V_B | after_word) / P(next∈V_B | before_word)]

    In surprisal form:
        surprisal_wt = surprisal_original - log(P(next∈V_B | after_word)) + log(P(next∈V_B | before_word))

    Args:
        inputs (dict): Tokenized inputs with input_ids and attention_mask
        model (PreTrainedModel): The language model
        tokenizer (PreTrainedTokenizer): The tokenizer
        word_ids (torch.Tensor): Word ids for the input (batch_size, seq_len)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - surprisal_wt: (batch_size, seq_len) - WT-corrected token-level surprisal
            - loss: scalar - cross-entropy loss
            - log_probs: (batch_size, seq_len+1, vocab_size) - log probabilities
    """
    # First, compute standard surprisal and get log probabilities
    surprisal_original, loss, log_probs = compute_surprisal(inputs, model, tokenizer, word_ids)

    # Get BOW token mask (which tokens start with 'Ġ')
    bow_mask = get_bow_token_mask(tokenizer, model.device)

    # Compute probabilities from log probabilities
    # log_probs: (batch_size, seq_len+1, vocab_size) in natural log space
    probs = torch.exp(log_probs)

    # Compute P(next token ∈ V_B) at each position
    # bow_mask: (vocab_size,) -> (1, 1, vocab_size) for broadcasting
    bow_mask_expanded = bow_mask.unsqueeze(0).unsqueeze(0)

    # Sum probabilities over BOW tokens: (batch_size, seq_len+1)
    prob_next_is_bow = (probs * bow_mask_expanded).sum(dim=-1)

    # Get input_ids to check which tokens are actually BOW
    input_ids = inputs['input_ids']  # (batch_size, seq_len)
    seq_len = input_ids.shape[1]

    # Create mask for actual BOW tokens in the input
    # Check if each token in input_ids is a BOW token
    is_bow_token = bow_mask[input_ids]  # (batch_size, seq_len)

    # Initialize WT-corrected surprisal as copy of original
    surprisal_wt = surprisal_original.clone()

    # Apply correction ONLY at positions where next token is actually BOW
    # Following reference implementation: external/wt_decoding/postprocess_bi.py:18-24
    for i in range(seq_len - 1):
        # Get mask for positions where next token (i+1) is BOW
        next_is_bow = is_bow_token[:, i + 1]  # (batch_size,)

        if next_is_bow.any():
            # Get log probability that next token is BOW at position i+1
            # Reference: all_bprob[i+1] in postprocess_bi.py:20-21
            # Note: Reference uses log2, but we use natural log for consistency with existing models
            # The WT decoding correction logic is the same, only the units differ
            log_prob_bow_at_i_plus_1 = torch.log(prob_next_is_bow[:, i + 1] + 1e-10)

            # Only apply correction where next token is actually BOW
            # Reference implementation:
            #   all_llmsurp[i] -= all_bprob[i+1]
            #   all_llmsurp[i+1] += all_bprob[i+1]
            surprisal_wt[:, i] = torch.where(
                next_is_bow,
                surprisal_wt[:, i] - log_prob_bow_at_i_plus_1,
                surprisal_wt[:, i]
            )
            surprisal_wt[:, i + 1] = torch.where(
                next_is_bow,
                surprisal_wt[:, i + 1] + log_prob_bow_at_i_plus_1,
                surprisal_wt[:, i + 1]
            )

    return surprisal_wt, loss, log_probs
