"""
Simplified loss computation for reading time prediction.

Core functionality only:
- MSE loss between predicted and actual reading times
- ROI exclusion from regression coefficient computation
- Filler regression support for evaluation
"""

import logging
from typing import Dict, Optional, Tuple, Any

import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

from models.surprisal import compute_surprisal
from models.loss_utils import solve_beta, compute_kl_full

logger = logging.getLogger(__name__)


def compute_regression_coefficients(
    surprisal: torch.Tensor,
    reading_times: torch.Tensor,
    word_lengths: torch.Tensor,
    word_frequencies: torch.Tensor,
    word_positions: Optional[torch.Tensor] = None,
    use_frequency_spillover: bool = False,
    use_surprisal_spillover: bool = False,
    use_length_spillover: bool = False,
    word_frequencies_prev1: Optional[torch.Tensor] = None,
    word_frequencies_prev2: Optional[torch.Tensor] = None,
    surprisal_prev1: Optional[torch.Tensor] = None,
    surprisal_prev2: Optional[torch.Tensor] = None,
    word_lengths_prev1: Optional[torch.Tensor] = None,
    word_lengths_prev2: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute regression coefficients from features.

    NOTE: All input features should be pre-filtered (e.g., ROI exclusion)
    before calling this function.

    Args:
        surprisal: Surprisal values (pre-filtered)
        reading_times: Target reading times (pre-filtered)
        word_lengths: Word length features (pre-filtered)
        word_frequencies: Word frequency features (pre-filtered)
        word_positions: Word position features (pre-filtered, optional)
        use_*_spillover: Whether to use spillover features
        *_prev1/2: Spillover feature values (pre-filtered)
    Returns:
        Dictionary of regression coefficients

    Raises:
        ValueError: If insufficient data for regression
    """
    # Build feature list
    features = [surprisal, word_lengths, word_frequencies]
    coeff_names = ["beta_s", "beta_l", "beta_f"]

    # Add word positions if available
    if word_positions is not None:
        features.append(word_positions)
        coeff_names.append("beta_pos")

    # Add spillover features if requested
    if use_frequency_spillover and word_frequencies_prev1 is not None:
        features.append(word_frequencies_prev1)
        coeff_names.append("beta_f_prev1")
        if word_frequencies_prev2 is not None:
            features.append(word_frequencies_prev2)
            coeff_names.append("beta_f_prev2")

    if use_surprisal_spillover and surprisal_prev1 is not None:
        features.append(surprisal_prev1)
        coeff_names.append("beta_s_prev1")
        if surprisal_prev2 is not None:
            features.append(surprisal_prev2)
            coeff_names.append("beta_s_prev2")

    if use_length_spillover and word_lengths_prev1 is not None:
        features.append(word_lengths_prev1)
        coeff_names.append("beta_l_prev1")
        if word_lengths_prev2 is not None:
            features.append(word_lengths_prev2)
            coeff_names.append("beta_l_prev2")

    # Add intercept
    ones = torch.ones_like(surprisal)
    features.append(ones)
    coeff_names.append("beta_0")

    # Stack features into design matrix
    X = torch.stack(features, dim=1)

    # Solve for coefficients
    if X.shape[0] > X.shape[1]:  # Need more samples than features
        beta = solve_beta(X, reading_times)
    else:
        # Not enough data for regression - stop immediately
        raise ValueError(
            f"Insufficient data for regression: {X.shape[0]} samples, "
            f"{X.shape[1]} features. Need at least {X.shape[1]} samples. "
            f"This often happens with aggressive ROI exclusion or very small batches. "
            f"Check your ROI exclusion settings or increase batch size."
        )

    # Create coefficient dictionary
    coefficients = {}
    for i, name in enumerate(coeff_names):
        coefficients[name] = beta[i]

    return coefficients


def compute_predicted_reading_times(
    surprisal: torch.Tensor,
    word_lengths: torch.Tensor,
    word_frequencies: torch.Tensor,
    coefficients: Dict[str, torch.Tensor],
    word_positions: Optional[torch.Tensor] = None,
    use_frequency_spillover: bool = False,
    use_surprisal_spillover: bool = False,
    use_length_spillover: bool = False,
    word_frequencies_prev1: Optional[torch.Tensor] = None,
    word_frequencies_prev2: Optional[torch.Tensor] = None,
    surprisal_prev1: Optional[torch.Tensor] = None,
    surprisal_prev2: Optional[torch.Tensor] = None,
    word_lengths_prev1: Optional[torch.Tensor] = None,
    word_lengths_prev2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute predicted reading times using regression coefficients.

    Args:
        surprisal, word_lengths, word_frequencies: Base features
        coefficients: Dictionary of regression coefficients
        use_*_spillover: Whether to use spillover features
        *_prev1/2: Spillover feature values

    Returns:
        Predicted reading times
    """
    # Start with base prediction
    pred = (coefficients.get("beta_s", 0) * surprisal +
            coefficients.get("beta_l", 0) * word_lengths +
            coefficients.get("beta_f", 0) * word_frequencies +
            coefficients.get("beta_0", 0))

    # Add word position if available
    if word_positions is not None and "beta_pos" in coefficients:
        pred = pred + coefficients["beta_pos"] * word_positions

    # Add spillover contributions if available
    if use_frequency_spillover:
        if word_frequencies_prev1 is not None and "beta_f_prev1" in coefficients:
            pred = pred + coefficients["beta_f_prev1"] * word_frequencies_prev1
        if word_frequencies_prev2 is not None and "beta_f_prev2" in coefficients:
            pred = pred + coefficients["beta_f_prev2"] * word_frequencies_prev2

    if use_surprisal_spillover:
        if surprisal_prev1 is not None and "beta_s_prev1" in coefficients:
            pred = pred + coefficients["beta_s_prev1"] * surprisal_prev1
        if surprisal_prev2 is not None and "beta_s_prev2" in coefficients:
            pred = pred + coefficients["beta_s_prev2"] * surprisal_prev2

    if use_length_spillover:
        if word_lengths_prev1 is not None and "beta_l_prev1" in coefficients:
            pred = pred + coefficients["beta_l_prev1"] * word_lengths_prev1
        if word_lengths_prev2 is not None and "beta_l_prev2" in coefficients:
            pred = pred + coefficients["beta_l_prev2"] * word_lengths_prev2

    return pred


def compute_coefficient_regularization(
    batch_coefficients: Dict[str, torch.Tensor],
    target_coefficients: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute L2 regularization loss between batch and target coefficients.

    This prevents batch coefficients from deviating too far from target coefficients,
    maintaining baseline linguistic effects while allowing garden-path specialization.

    Args:
        batch_coefficients: Coefficients computed from current batch
        target_coefficients: Reference coefficients (filler, baseline, or combined)

    Returns:
        L2 distance between coefficient vectors
    """
    l2_loss = torch.tensor(0.0, device=batch_coefficients["beta_0"].device)
    n_coeffs = 0

    # Compute L2 for each matching coefficient
    for name in batch_coefficients.keys():
        if name in target_coefficients:
            batch_val = batch_coefficients[name]
            target_val = target_coefficients[name]

            # Ensure both are on the same device
            if target_val.device != batch_val.device:
                target_val = target_val.to(batch_val.device)

            # Add squared difference
            l2_loss = l2_loss + (batch_val - target_val) ** 2
            n_coeffs += 1

    # Take square root for L2 norm (optional, could also use sum of squares)
    if n_coeffs > 0:
        l2_loss = torch.sqrt(l2_loss)

    return l2_loss


def compute_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch: Dict[str, Any],
    loss_type: str = "mse",
    reduction: str = "mean",
    mask_zero_reading_times: bool = True,
    mask_zero_freqs: bool = True,
    mask_last_region: bool = True,
    kl_weight: float = 0.0,
    kl_variant: str = "abs",
    model_ref: Optional[PreTrainedModel] = None,
    # KL masking controls
    kl_exclude_roi: bool = False,
    kl_mask_zero_reading_times: bool = False,
    kl_mask_zero_freqs: bool = False,
    kl_mask_last_region: bool = False,
    # ROI exclusion
    exclude_roi_from_regression: bool = False,
    roi_exclusion_values: Optional[list] = None,
    # Spillover features
    use_frequency_spillover: bool = False,
    use_surprisal_spillover: bool = False,
    use_length_spillover: bool = False,
    # WT (Whitespace-Trailing) decoding
    use_wt_decoding: bool = False,
    # Global coefficients (e.g., from filler regression)
    use_global_coefficients: bool = False,
    global_coefficients: Optional[Dict[str, torch.Tensor]] = None,
    # ROI-only loss
    use_roi_only_loss: bool = False,
    roi_values_for_loss: Optional[list] = None,
    invert_roi_loss_mask: bool = False,  # If True, use all ROIs EXCEPT roi_values_for_loss
    # Baseline (untrained model) coefficient regularization
    use_baseline_coefficient_regularization: bool = False,
    baseline_coef_regularization_weight: float = 1.0,
    baseline_coefficients: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Any, Any, Any, torch.Tensor, Dict]:
    """
    Simplified loss computation.

    Args:
        model: Language model
        tokenizer: Tokenizer
        batch: Batch data
        loss_type: Type of loss ("mse")
        reduction: Reduction method ("mean", "none")
        mask_zero_reading_times: Whether to mask zero RTs
        mask_zero_freqs: Whether to mask zero frequencies
        mask_last_region: Whether to mask the last word in each sentence
        kl_weight: KL divergence weight (if > 0)
        kl_variant: KL variant ("abs", "full")
        model_ref: Reference model for KL
        exclude_roi_from_regression: Whether to exclude ROI from regression
        roi_exclusion_values: ROI values to exclude
        use_*_spillover: Whether to use spillover features
        use_global_coefficients: Whether to use pre-computed global coefficients
        global_coefficients: Pre-computed regression coefficients

    Returns:
        Tuple of (loss, coefficients, ppl, ce_loss, batch_metrics, kl_loss, monitoring_data)
    """
    device = model.device

    # Extract batch data (input_ids and attention_mask stay in batch for compute_surprisal)
    word_ids = batch.get("word_ids").to(device)  # (batch_size, seq_len)
    reading_times = batch.get("reading_times").to(device)  # (batch_size, seq_len)
    word_lengths = batch.get("word_lengths").to(device)  # (batch_size, seq_len)
    word_frequencies = batch.get("log_unigram_frequencies").to(device)  # (batch_size, seq_len)
    word_positions = batch.get("word_positions")
    if word_positions is not None:
        word_positions = word_positions.to(device)  # (batch_size, seq_len)

    # Zero frequency masks for filtering
    zero_freq_mask = batch.get("zero_freq_mask")
    zero_freq_mask_prev1 = batch.get("zero_freq_mask_prev1")
    zero_freq_mask_prev2 = batch.get("zero_freq_mask_prev2")

    if zero_freq_mask is not None:
        zero_freq_mask = zero_freq_mask.to(device)  # (batch_size, seq_len)
    if zero_freq_mask_prev1 is not None:
        zero_freq_mask_prev1 = zero_freq_mask_prev1.to(device)  # (batch_size, seq_len)
    if zero_freq_mask_prev2 is not None:
        zero_freq_mask_prev2 = zero_freq_mask_prev2.to(device)  # (batch_size, seq_len)

    # Optional spillover features
    word_frequencies_prev1 = batch.get("log_unigram_frequencies_prev1")
    word_frequencies_prev2 = batch.get("log_unigram_frequencies_prev2")
    word_lengths_prev1 = batch.get("word_lengths_prev1")
    word_lengths_prev2 = batch.get("word_lengths_prev2")

    if word_frequencies_prev1 is not None:
        word_frequencies_prev1 = word_frequencies_prev1.to(device)  # (batch_size, seq_len)
    if word_frequencies_prev2 is not None:
        word_frequencies_prev2 = word_frequencies_prev2.to(device)  # (batch_size, seq_len)
    if word_lengths_prev1 is not None:
        word_lengths_prev1 = word_lengths_prev1.to(device)  # (batch_size, seq_len)
    if word_lengths_prev2 is not None:
        word_lengths_prev2 = word_lengths_prev2.to(device)  # (batch_size, seq_len)

    # ROI information for exclusion
    roi = batch.get("roi")
    if roi is not None:
        roi = roi.to(device)  # (batch_size, seq_len)

    # Compute surprisal (subword level) with optional WT decoding
    if use_wt_decoding:
        from models.surprisal import compute_surprisal_with_wt_decoding
        surprisal_subword, ce_loss, log_probs = compute_surprisal_with_wt_decoding(batch, model, tokenizer, word_ids)
    else:
        surprisal_subword, ce_loss, log_probs = compute_surprisal(batch, model, tokenizer, word_ids)
    # surprisal_subword: (batch_size, seq_len), ce_loss: scalar, log_probs: (batch_size, seq_len+1, vocab_size)

    # IMPORTANT: Aggregate subword surprisal to word level
    # This is critical for matching word-level reading times
    mask_word_ids = word_ids != -1  # (batch_size, seq_len)
    masked_surprisal_subword = surprisal_subword * mask_word_ids  # (batch_size, seq_len)
    masked_word_ids = word_ids * mask_word_ids  # (batch_size, seq_len)

    # Calculate perplexity from subword surprisal (before aggregation)
    ppl = torch.exp(torch.sum(masked_surprisal_subword) / torch.sum(mask_word_ids))  # scalar

    # Sum surprisal for tokens belonging to same word
    surprisal = torch.zeros_like(surprisal_subword)  # (batch_size, seq_len)
    surprisal.scatter_add_(1, masked_word_ids, masked_surprisal_subword)  # (batch_size, seq_len)

    # Compute spillover surprisal if needed (at word level)
    surprisal_prev1 = None
    surprisal_prev2 = None
    if use_surprisal_spillover:
        # Shift for spillover at word level
        surprisal_prev1 = torch.zeros_like(surprisal)  # (batch_size, seq_len)
        surprisal_prev2 = torch.zeros_like(surprisal)  # (batch_size, seq_len)

        # Shift within each sequence (vectorized operation)
        surprisal_prev1[:, 1:] = surprisal[:, :-1]  # (batch_size, seq_len)
        surprisal_prev2[:, 2:] = surprisal[:, :-2]  # (batch_size, seq_len)

    # Create masks
    valid_mask = (word_ids != -1)  # (batch_size, seq_len)
    if mask_zero_reading_times:
        valid_mask = valid_mask & (reading_times > 0)
    if mask_zero_freqs and zero_freq_mask is not None:
        # Use zero_freq_mask (0 means zero frequency, should be masked)
        valid_mask = valid_mask & (zero_freq_mask != 0)
        # Also mask spillover features with zero frequency
        if use_frequency_spillover:
            if zero_freq_mask_prev1 is not None:
                valid_mask = valid_mask & (zero_freq_mask_prev1 != 0)
            if zero_freq_mask_prev2 is not None:
                valid_mask = valid_mask & (zero_freq_mask_prev2 != 0)

    # Mask last region if requested
    if mask_last_region:
        # For each sentence, find the last valid word position and mask it
        for batch_idx in range(word_ids.shape[0]):
            # Find the last valid position (rightmost True in valid_mask)
            valid_positions = torch.where(valid_mask[batch_idx])[0]
            if len(valid_positions) > 0:
                last_position = valid_positions[-1]
                valid_mask[batch_idx, last_position] = False

    # ROI exclusion mask
    exclude_mask = None
    if exclude_roi_from_regression and roi is not None and roi_exclusion_values:
        exclude_mask = torch.zeros_like(valid_mask, dtype=torch.bool)  # (batch_size, seq_len)
        for roi_val in roi_exclusion_values:
            exclude_mask = exclude_mask | (roi == roi_val)  # (batch_size, seq_len)
        # Combine with valid mask for regression
        regression_mask = valid_mask & ~exclude_mask  # (batch_size, seq_len)
    else:
        regression_mask = valid_mask  # (batch_size, seq_len)

    # Flatten all features (2D -> 1D by selecting valid positions)
    surprisal_flat = surprisal[valid_mask]  # (n_valid,)
    reading_times_flat = reading_times[valid_mask]  # (n_valid,)
    word_lengths_flat = word_lengths[valid_mask]  # (n_valid,)
    word_frequencies_flat = word_frequencies[valid_mask]  # (n_valid,)
    word_positions_flat = (word_positions[valid_mask]  # (n_valid,)
                          if word_positions is not None else None)

    # Flatten spillover features if available
    if use_frequency_spillover and word_frequencies_prev1 is not None:
        word_frequencies_prev1_flat = word_frequencies_prev1[valid_mask]
        word_frequencies_prev2_flat = (word_frequencies_prev2[valid_mask]
                                      if word_frequencies_prev2 is not None else None)
    else:
        word_frequencies_prev1_flat = None
        word_frequencies_prev2_flat = None

    if use_surprisal_spillover and surprisal_prev1 is not None:
        surprisal_prev1_flat = surprisal_prev1[valid_mask]
        surprisal_prev2_flat = (surprisal_prev2[valid_mask]
                               if surprisal_prev2 is not None else None)
    else:
        surprisal_prev1_flat = None
        surprisal_prev2_flat = None

    if use_length_spillover and word_lengths_prev1 is not None:
        word_lengths_prev1_flat = word_lengths_prev1[valid_mask]
        word_lengths_prev2_flat = (word_lengths_prev2[valid_mask]
                                  if word_lengths_prev2 is not None else None)
    else:
        word_lengths_prev1_flat = None
        word_lengths_prev2_flat = None

    # Use global coefficients if provided, otherwise compute from batch
    if use_global_coefficients and global_coefficients is not None:
        # Use pre-computed global coefficients (e.g., from filler regression)
        coefficients = global_coefficients
    elif exclude_roi_from_regression and exclude_mask is not None:
        # Apply ROI exclusion by filtering the flattened data
        regression_mask_flat = regression_mask[valid_mask]  # (n_valid,)
        coefficients = compute_regression_coefficients(
                surprisal_flat[regression_mask_flat],
                reading_times_flat[regression_mask_flat],
                word_lengths_flat[regression_mask_flat],
                word_frequencies_flat[regression_mask_flat],
                word_positions=(word_positions_flat[regression_mask_flat]
                               if word_positions_flat is not None else None),
                use_frequency_spillover=use_frequency_spillover,
                use_surprisal_spillover=use_surprisal_spillover,
                use_length_spillover=use_length_spillover,
                word_frequencies_prev1=(word_frequencies_prev1_flat[regression_mask_flat]
                                       if word_frequencies_prev1_flat is not None else None),
                word_frequencies_prev2=(word_frequencies_prev2_flat[regression_mask_flat]
                                       if word_frequencies_prev2_flat is not None else None),
                surprisal_prev1=(surprisal_prev1_flat[regression_mask_flat]
                                if surprisal_prev1_flat is not None else None),
                surprisal_prev2=(surprisal_prev2_flat[regression_mask_flat]
                                if surprisal_prev2_flat is not None else None),
                word_lengths_prev1=(word_lengths_prev1_flat[regression_mask_flat]
                                   if word_lengths_prev1_flat is not None else None),
                word_lengths_prev2=(word_lengths_prev2_flat[regression_mask_flat]
                                   if word_lengths_prev2_flat is not None else None),
            )
    else:
        # No ROI exclusion - use all valid data
        coefficients = compute_regression_coefficients(
            surprisal_flat,
            reading_times_flat,
            word_lengths_flat,
            word_frequencies_flat,
            word_positions=word_positions_flat,
            use_frequency_spillover=use_frequency_spillover,
            use_surprisal_spillover=use_surprisal_spillover,
            use_length_spillover=use_length_spillover,
            word_frequencies_prev1=word_frequencies_prev1_flat,
            word_frequencies_prev2=word_frequencies_prev2_flat,
            surprisal_prev1=surprisal_prev1_flat,
            surprisal_prev2=surprisal_prev2_flat,
            word_lengths_prev1=word_lengths_prev1_flat,
            word_lengths_prev2=word_lengths_prev2_flat,
        )

    # Compute predicted reading times
    predicted_rts = compute_predicted_reading_times(
        surprisal_flat,
        word_lengths_flat,
        word_frequencies_flat,
        coefficients,
        word_positions=word_positions_flat,
        use_frequency_spillover=use_frequency_spillover,
        use_surprisal_spillover=use_surprisal_spillover,
        use_length_spillover=use_length_spillover,
        word_frequencies_prev1=word_frequencies_prev1_flat,
        word_frequencies_prev2=word_frequencies_prev2_flat,
        surprisal_prev1=surprisal_prev1_flat,
        surprisal_prev2=surprisal_prev2_flat,
        word_lengths_prev1=word_lengths_prev1_flat,
        word_lengths_prev2=word_lengths_prev2_flat,
    )

    # Compute loss
    if loss_type == "mse":
        # Apply ROI-only mask if enabled
        if use_roi_only_loss and roi_values_for_loss is not None and 'roi' in batch:
            # Convert ROI to flat tensor matching the flattened predictions
            roi_flat = torch.zeros(predicted_rts.shape[0], dtype=torch.long, device=device)

            # Get ROI values for valid positions
            if isinstance(batch['roi'], list):
                # Convert list of numpy arrays to tensor
                roi_tensor = torch.stack([torch.tensor(r, device=device) for r in batch['roi']])
            else:
                roi_tensor = batch['roi'].to(device)

            # Flatten ROI values for valid positions only
            roi_flat = roi_tensor[valid_mask]

            # Create mask for ROI positions we want to use for loss
            roi_loss_mask = torch.zeros_like(roi_flat, dtype=torch.bool)
            for roi_value in roi_values_for_loss:
                roi_loss_mask = roi_loss_mask | (roi_flat == roi_value)

            # Invert mask if requested (use all ROIs EXCEPT specified ones)
            if invert_roi_loss_mask:
                roi_loss_mask = ~roi_loss_mask

            if roi_loss_mask.any():
                # Use only ROI positions for loss
                loss = torch.nn.functional.mse_loss(
                    predicted_rts[roi_loss_mask],
                    reading_times_flat[roi_loss_mask],
                    reduction=reduction
                )
            else:
                # Fallback to regular loss if no ROI positions found
                loss = torch.nn.functional.mse_loss(predicted_rts, reading_times_flat, reduction=reduction)
        else:
            loss = torch.nn.functional.mse_loss(predicted_rts, reading_times_flat, reduction=reduction)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Add KL divergence if requested
    kl_loss = torch.tensor(0.0, device=device)
    if kl_weight > 0 and model_ref is not None:
        # Create basic KL mask: exclude only padding positions
        kl_mask = word_ids != -1  # (batch_size, seq_len)

        # Apply additional KL masking based on configuration
        if kl_mask_zero_reading_times:
            kl_mask = kl_mask & (reading_times > 0)

        if kl_mask_zero_freqs and zero_freq_mask is not None:
            kl_mask = kl_mask & (zero_freq_mask != 0)
            # Also mask spillover features with zero frequency if needed
            if use_frequency_spillover:
                if zero_freq_mask_prev1 is not None:
                    kl_mask = kl_mask & (zero_freq_mask_prev1 != 0)
                if zero_freq_mask_prev2 is not None:
                    kl_mask = kl_mask & (zero_freq_mask_prev2 != 0)

        if kl_mask_last_region:
            # For each sentence, find the last valid word position and mask it
            for batch_idx in range(word_ids.shape[0]):
                valid_positions = torch.where(kl_mask[batch_idx])[0]
                if len(valid_positions) > 0:
                    last_position = valid_positions[-1]
                    kl_mask[batch_idx, last_position] = False

        if kl_exclude_roi and roi is not None and roi_exclusion_values:
            # Exclude ROI positions from KL loss
            roi_mask = torch.zeros_like(kl_mask, dtype=torch.bool)
            for roi_val in roi_exclusion_values:
                roi_mask = roi_mask | (roi == roi_val)
            kl_mask = kl_mask & ~roi_mask

        if kl_variant == "full":
            # Full KL divergence using log probabilities
            with torch.no_grad():
                if use_wt_decoding:
                    from models.surprisal import compute_surprisal_with_wt_decoding
                    _, _, ref_logprobs = compute_surprisal_with_wt_decoding(batch, model_ref, tokenizer, word_ids)
                else:
                    _, _, ref_logprobs = compute_surprisal(batch, model_ref, tokenizer, word_ids)

            # Shift mask for subword-level KL (align with log_probs indexing)
            batch_size = word_ids.shape[0]
            full_kl_mask = torch.cat(
                [torch.zeros((batch_size, 1), dtype=torch.bool, device=device), kl_mask[:, :-1]],
                dim=1,
            )

            # Slice log probs (remove last position)
            log_probs_sliced = log_probs[:, :-1, :]
            ref_logprobs_sliced = ref_logprobs[:, :-1, :]

            # Compute KL divergence
            kl_full = compute_kl_full(
                log_probs_sliced[full_kl_mask].unsqueeze(0),
                ref_logprobs_sliced[full_kl_mask].unsqueeze(0),
            )
            kl_loss = kl_full.mean()

        else:  # kl_variant == "abs" or default
            # Word-level absolute divergence
            with torch.no_grad():
                if use_wt_decoding:
                    from models.surprisal import compute_surprisal_with_wt_decoding
                    ref_surprisal_subword, _, _ = compute_surprisal_with_wt_decoding(batch, model_ref, tokenizer, word_ids)
                else:
                    ref_surprisal_subword, _, _ = compute_surprisal(batch, model_ref, tokenizer, word_ids)
                # Aggregate reference surprisal to word level (same as target)
                masked_ref_surprisal_subword = ref_surprisal_subword * mask_word_ids
                ref_surprisal = torch.zeros_like(ref_surprisal_subword)
                ref_surprisal.scatter_add_(1, masked_word_ids, masked_ref_surprisal_subword)

            # Use kl_mask (not valid_mask) to exclude only padding
            kl_loss = torch.abs(surprisal[kl_mask] - ref_surprisal[kl_mask]).mean()

        loss = loss + kl_weight * kl_loss

    # Add baseline coefficient regularization if requested
    baseline_coef_reg_loss = torch.tensor(0.0, device=device)

    # Only apply regularization if using batch coefficients (not global coefficients)
    if not use_global_coefficients:
        # Baseline coefficient regularization
        if use_baseline_coefficient_regularization and baseline_coefficients is not None:
            # Regularize batch coefficients to baseline (untrained) model coefficients
            baseline_coef_reg_loss = compute_coefficient_regularization(
                batch_coefficients=coefficients,
                target_coefficients=baseline_coefficients,
            )
            loss = loss + baseline_coef_regularization_weight * baseline_coef_reg_loss

    # Monitoring data including garden-path differences
    monitoring_data = {
        "n_samples": valid_mask.sum().item(),
        "mean_loss": loss.item() if loss.numel() == 1 else loss.mean().item(),
        "baseline_coef_reg_loss": baseline_coef_reg_loss.item() if baseline_coef_reg_loss.numel() == 1 else 0.0,
    }

    # Add garden-path monitoring if applicable
    if "construction" in batch and "pair_id" in batch and "ambiguity" in batch:
        # Reshape predicted RTs back to 2D for monitoring
        # Create a 2D tensor and fill in the valid positions
        predicted_rts_2d = torch.zeros_like(reading_times)
        predicted_rts_2d[valid_mask] = predicted_rts

        gp_monitoring = monitor_garden_path_differences(
            batch=batch,
            reading_times=reading_times,
            predicted_rts=predicted_rts_2d,
            valid_mask=valid_mask,
            device=device,
            surprisal=surprisal,  # Pass word-level surprisal
        )
        monitoring_data.update(gp_monitoring)

    # Return tuple matching expected format
    return (
        loss,
        coefficients,
        ppl,
        ce_loss,
        None,  # batch_metrics (not used in simplified version)
        kl_loss,
        monitoring_data,
    )


def monitor_garden_path_differences(
    batch: Dict[str, torch.Tensor],
    reading_times: torch.Tensor,  # (batch_size, seq_len)
    predicted_rts: torch.Tensor,   # (batch_size, seq_len)
    valid_mask: torch.Tensor,     # (batch_size, seq_len)
    device: torch.device,
    surprisal: torch.Tensor = None,  # (batch_size, seq_len) - word-level surprisal
) -> Dict[str, float]:
    """
    Monitor differences between ambiguous and unambiguous conditions for critical regions.

    Args:
        batch: Batch data containing construction, pair_id, ambiguity, roi
        reading_times: Actual reading times (batch_size, seq_len)
        predicted_rts: Predicted reading times from regression (batch_size, seq_len)
        valid_mask: Valid word mask (batch_size, seq_len)
        device: Computation device
        surprisal: Word-level surprisal values (optional)

    Returns:
        Dictionary with monitoring metrics for each construction and ROI
    """
    monitoring_results = {}

    # Extract metadata (custom_collate_fn returns lists of numpy arrays for string data)
    import numpy as np

    construction = batch["construction"]  # List of numpy arrays
    pair_id = batch["pair_id"]            # List of numpy arrays
    ambiguity = batch["ambiguity"]        # List of numpy arrays
    roi = batch.get("roi", None)          # Tensor (batch_size, seq_len)

    if roi is None:
        return monitoring_results

    # Stack numpy arrays for processing
    construction = np.stack(construction)
    pair_id = np.stack(pair_id)
    ambiguity = np.stack(ambiguity)

    # Get unique values per sentence (since they're repeated across tokens)
    # Use first valid token position for each sentence
    batch_size = len(construction)
    sentence_construction = []
    sentence_pair_id = []
    sentence_ambiguity = []

    for i in range(batch_size):
        valid_idx = valid_mask[i].nonzero(as_tuple=True)[0]
        if len(valid_idx) > 0:
            first_idx = valid_idx[0].item()
            # Access numpy arrays
            sentence_construction.append(construction[i][first_idx])
            sentence_pair_id.append(pair_id[i][first_idx])
            sentence_ambiguity.append(ambiguity[i][first_idx])
        else:
            sentence_construction.append("PAD")
            sentence_pair_id.append(-1)
            sentence_ambiguity.append("PAD")

    # Define critical ROIs (0, 1, 2 are typically critical and spillover positions)
    critical_rois = [0, 1, 2]

    # Get unique constructions (excluding PAD)
    unique_constructions = set(sentence_construction) - {"PAD"}

    for construction_type in unique_constructions:
        # Find all sentences of this construction type
        construction_mask = np.array([c == construction_type for c in sentence_construction])
        construction_indices = np.where(construction_mask)[0]

        if len(construction_indices) == 0:
            continue

        # Get unique pair IDs for this construction
        construction_pair_ids = [sentence_pair_id[i] for i in construction_indices]
        unique_pairs = set(construction_pair_ids) - {-1}

        # For each ROI in critical region
        for roi_value in critical_rois:
            roi_diffs_actual = []
            roi_diffs_predicted = []
            roi_surprisal_amb = []
            roi_surprisal_unamb = []

            for pair in unique_pairs:
                # Find ambiguous and unambiguous sentences for this pair
                amb_idx = None
                unamb_idx = None

                for idx in construction_indices:
                    if sentence_pair_id[idx] == pair:
                        if sentence_ambiguity[idx] == "Amb":
                            amb_idx = idx
                        elif sentence_ambiguity[idx] == "Unamb":
                            unamb_idx = idx

                # If we have both conditions for this pair
                if amb_idx is not None and unamb_idx is not None:
                    # Find tokens with this ROI value
                    amb_roi_mask = (roi[amb_idx] == roi_value) & valid_mask[amb_idx]
                    unamb_roi_mask = (roi[unamb_idx] == roi_value) & valid_mask[unamb_idx]

                    # Get mean RT for this ROI
                    if amb_roi_mask.any() and unamb_roi_mask.any():
                        amb_rt_actual = reading_times[amb_idx][amb_roi_mask].mean().item()
                        unamb_rt_actual = reading_times[unamb_idx][unamb_roi_mask].mean().item()

                        amb_rt_pred = predicted_rts[amb_idx][amb_roi_mask].mean().item()
                        unamb_rt_pred = predicted_rts[unamb_idx][unamb_roi_mask].mean().item()

                        # Calculate differences (Amb - Unamb)
                        roi_diffs_actual.append(amb_rt_actual - unamb_rt_actual)
                        roi_diffs_predicted.append(amb_rt_pred - unamb_rt_pred)

                        # Get surprisal values if available
                        if surprisal is not None:
                            amb_surp = surprisal[amb_idx][amb_roi_mask].mean().item()
                            unamb_surp = surprisal[unamb_idx][unamb_roi_mask].mean().item()
                            roi_surprisal_amb.append(amb_surp)
                            roi_surprisal_unamb.append(unamb_surp)

            # Store mean differences for this construction and ROI
            if roi_diffs_actual:
                key_actual = f"gp_{construction_type}_roi{roi_value}_diff_actual"
                key_predicted = f"gp_{construction_type}_roi{roi_value}_diff_predicted"

                monitoring_results[key_actual] = np.mean(roi_diffs_actual)
                monitoring_results[key_predicted] = np.mean(roi_diffs_predicted)

                # Store surprisal values if available
                if roi_surprisal_amb and roi_surprisal_unamb:
                    monitoring_results[f"gp_{construction_type}_roi{roi_value}_surprisal_amb"] = np.mean(roi_surprisal_amb)
                    monitoring_results[f"gp_{construction_type}_roi{roi_value}_surprisal_unamb"] = np.mean(roi_surprisal_unamb)
                    monitoring_results[f"gp_{construction_type}_roi{roi_value}_surprisal_diff"] = np.mean(roi_surprisal_amb) - np.mean(roi_surprisal_unamb)

                # Also store the count of pairs used
                monitoring_results[f"gp_{construction_type}_roi{roi_value}_n_pairs"] = len(roi_diffs_actual)

    return monitoring_results
