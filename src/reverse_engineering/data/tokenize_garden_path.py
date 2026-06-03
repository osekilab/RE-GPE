"""
Tokenization module specifically for garden-path data.
Garden-path data already contains pre-computed linguistic features,
so this module preserves those features instead of recalculating them.
"""

from typing import Optional

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer


def tokenize_garden_path_texts(
    texts: list[list[str]],
    reading_times: list[list[float]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    # Pre-computed features from garden-path CSV
    word_lengths: Optional[list[list[int]]] = None,
    log_frequencies: Optional[list[list[float]]] = None,
    log_frequencies_prev1: Optional[list[list[float]]] = None,
    log_frequencies_prev2: Optional[list[list[float]]] = None,
    word_lengths_prev1: Optional[list[list[float]]] = None,
    word_lengths_prev2: Optional[list[list[float]]] = None,
    word_positions: Optional[list[list[int]]] = None,
    # ROI and metadata
    roi_data: Optional[list[list[int]]] = None,
    pair_id_data: Optional[list[str]] = None,
    construction_data: Optional[list[str]] = None,
    ambiguity_data: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    """
    Tokenizes garden-path texts while preserving pre-computed features.

    Unlike the general tokenize_texts function, this version:
    1. Uses pre-computed word lengths instead of recalculating
    2. Uses pre-computed log frequencies instead of looking up in wordfreq
    3. Uses pre-computed spillover features (freq_prev1, freq_prev2)
    4. Preserves the exact feature values from the garden-path dataset

    Args:
        texts: List of lists of words
        reading_times: List of lists of reading times
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        word_lengths: Pre-computed word lengths from garden-path data
        log_frequencies: Pre-computed log frequencies from garden-path data
        log_frequencies_prev1: Pre-computed lag-1 frequencies
        log_frequencies_prev2: Pre-computed lag-2 frequencies
        word_lengths_prev1: Pre-computed lag-1 word lengths
        word_lengths_prev2: Pre-computed lag-2 word lengths
        word_positions: Pre-computed word positions
        roi_data: ROI values for each word
        pair_id_data: Garden-path pair IDs
        construction_data: Construction types (MVRR, NPS, NPZ)
        ambiguity_data: Ambiguity conditions (Amb, Unamb)

    Returns:
        Dictionary with tokenized data and preserved features
    """
    tokenized_texts = []
    token_word_ids = []
    attention_masks = []
    all_word_lengths = []
    all_log_frequencies = []
    all_reading_times = []
    all_words = []
    all_zero_freqs = []
    all_word_positions = []
    # Spillover features
    all_log_freqs_prev1 = []
    all_zero_freqs_prev1 = []
    all_log_freqs_prev2 = []
    all_zero_freqs_prev2 = []
    all_word_lengths_prev1 = []
    all_word_lengths_prev2 = []
    all_roi_values = []
    # Garden-path metadata
    all_pair_ids = []
    all_constructions = []
    all_ambiguities = []

    for sent_index, words in enumerate(texts):
        # Skip short sentences
        if len(words) <= 3:
            continue
        elif len(words) > max_length - 1:
            print("Skipping sentence with more than max_length words -", len(words))
            continue
        # Exclude if contains nan
        elif any([pd.isna(word) for word in words]):
            continue

        joined_sentence = " ".join(words)
        tokens = tokenizer(
            joined_sentence,
            padding="max_length",
            truncation=False,
            max_length=max_length,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )

        ids = np.array(tokens["input_ids"])
        unpadded_ids = ids[ids != tokenizer.pad_token_id]
        if len(unpadded_ids) > max_length:
            print("Skipping sentence with more than max_length tokens -", len(unpadded_ids))
            continue

        word_ids = np.array(tokens.word_ids())
        offset_mapping = tokens["offset_mapping"]
        offset_words = [joined_sentence[start:end] for start, end in offset_mapping]

        # Correct word IDs (same logic as original tokenize_texts)
        corrected_word_ids = []
        corrected_words = []
        words_index = 0
        offset_index = 0

        try:
            while words_index < len(words):
                word = words[words_index]
                if words_index > 0:
                    word = " " + word
                if word == offset_words[offset_index]:
                    # If the word matches the offset word, keep it as is
                    corrected_word_ids.append(words_index)
                    corrected_words.append(word.lstrip())
                    words_index += 1
                    offset_index += 1
                else:
                    # If the word does not match, merge until it matches
                    target_word = word
                    merged_word = offset_words[offset_index]
                    merged_word_id = [words_index]
                    offset_index += 1
                    while target_word != merged_word:
                        merged_word += offset_words[offset_index]
                        merged_word_id.append(words_index)
                        offset_index += 1
                    corrected_words.append(merged_word.lstrip())
                    corrected_word_ids.extend(merged_word_id)
                    words_index += 1

            assert corrected_words == words, (
                "Corrected words do not match the original words."
            )
            assert max(corrected_word_ids) + 1 == len(words), (
                "Highest index in corrected word IDs does not match the number of words."
            )
            assert len(corrected_word_ids) == len(word_ids[word_ids != None]), (
                "Length mismatch between corrected word IDs and word IDs."
            )
            assert len(reading_times[sent_index]) == max(corrected_word_ids) + 1, (
                "Reading times length does not match the number of words."
            )

            # Verify tokenization
            ids_checker = {}
            for index, word_id in enumerate(corrected_word_ids):
                if word_id not in ids_checker:
                    ids_checker[word_id] = []
                ids_checker[word_id].append(tokens["input_ids"][index])
                # dict from word_id to token IDs
            for word_id, word_ids_list in ids_checker.items():
                if words[word_id].strip() != tokenizer.decode(word_ids_list).strip():
                    print("Mismatch between word and tokenized word.")
                    print("Word: ", words[word_id].strip())
                    print("Tokenized word: ", tokenizer.decode(word_ids_list).strip())
                    raise ValueError("Mismatch between word and tokenized word.")

        except Exception as e:
            print("Encountered an error while correcting word IDs.: ", e)
            print("Skipping sentence: ", words)
            continue

        corrected_word_ids = np.pad(
            corrected_word_ids,
            (0, max_length - len(corrected_word_ids)),
            "constant",
            constant_values=-1,
        )

        # Use pre-computed features from garden-path data
        # Word lengths
        if word_lengths is not None:
            current_lengths = word_lengths[sent_index]
            assert len(current_lengths) == len(words), (
                f"Word lengths mismatch: {len(current_lengths)} != {len(words)}"
            )
        else:
            # Fallback to computing if not provided
            current_lengths = [len(word) for word in words]

        # Log frequencies (already computed in garden-path data as 'freq')
        if log_frequencies is not None:
            current_log_freqs = log_frequencies[sent_index]
            assert len(current_log_freqs) == len(words), (
                f"Log frequencies mismatch: {len(current_log_freqs)} != {len(words)}"
            )
        else:
            # Should not happen for garden-path data
            raise ValueError("Log frequencies must be provided for garden-path data")

        # Create zero frequency mask from log frequencies
        # When frequency=0, we use 1e-10 as substitute, so freq = -log2(1e-10) ≈ 33.219
        # Any value >= this threshold indicates zero frequency
        ZERO_FREQ_THRESHOLD = -np.log2(1e-10)  # ≈ 33.219
        zero_freqs = np.where(np.array(current_log_freqs) >= ZERO_FREQ_THRESHOLD, 0, 1).tolist()

        # Word positions
        if word_positions is not None:
            current_positions = word_positions[sent_index]
            # Garden-path positions are 1-indexed, convert to 0-indexed
            current_positions = [p - 1 for p in current_positions]
        else:
            current_positions = list(range(len(words)))

        # Spillover features - use pre-computed values
        if log_frequencies_prev1 is not None:
            current_log_freqs_prev1 = log_frequencies_prev1[sent_index]
            # Replace NaN with -1.0 (missing value indicator)
            current_log_freqs_prev1 = [
                -1.0 if pd.isna(x) else x for x in current_log_freqs_prev1
            ]
        else:
            # Compute from current frequencies (fallback)
            current_log_freqs_prev1 = [-1.0] + current_log_freqs[:-1]

        if log_frequencies_prev2 is not None:
            current_log_freqs_prev2 = log_frequencies_prev2[sent_index]
            # Replace NaN with -1.0 (missing value indicator)
            current_log_freqs_prev2 = [
                -1.0 if pd.isna(x) else x for x in current_log_freqs_prev2
            ]
        else:
            # Compute from current frequencies (fallback)
            if len(current_log_freqs) > 1:
                current_log_freqs_prev2 = [-1.0, -1.0] + current_log_freqs[:-2]
            else:
                current_log_freqs_prev2 = [-1.0] * len(current_log_freqs)

        # Create zero frequency masks for spillover features
        # Use the same threshold as above for consistency
        zero_freqs_prev1 = [0 if x == -1.0 or x >= ZERO_FREQ_THRESHOLD else 1 for x in current_log_freqs_prev1]
        zero_freqs_prev2 = [0 if x == -1.0 or x >= ZERO_FREQ_THRESHOLD else 1 for x in current_log_freqs_prev2]

        # Process word length spillover features
        if word_lengths_prev1 is not None:
            current_lengths_prev1 = word_lengths_prev1[sent_index]
            # Replace NaN with -1 (missing value indicator)
            current_lengths_prev1 = [
                -1 if pd.isna(x) else int(x) for x in current_lengths_prev1
            ]
        else:
            # Compute from current lengths (fallback)
            current_lengths_prev1 = [-1] + current_lengths[:-1]

        if word_lengths_prev2 is not None:
            current_lengths_prev2 = word_lengths_prev2[sent_index]
            # Replace NaN with -1 (missing value indicator)
            current_lengths_prev2 = [
                -1 if pd.isna(x) else int(x) for x in current_lengths_prev2
            ]
        else:
            # Compute from current lengths (fallback)
            if len(current_lengths) > 1:
                current_lengths_prev2 = [-1, -1] + current_lengths[:-2]
            else:
                current_lengths_prev2 = [-1] * len(current_lengths)

        # Pad all features
        token_word_ids.append(corrected_word_ids)
        tokenized_texts.append(tokens["input_ids"])
        attention_masks.append(tokens["attention_mask"])

        all_word_lengths.append(
            np.pad(
                current_lengths,
                (0, max_length - len(current_lengths)),
                "constant",
                constant_values=-1
            )
        )

        all_log_frequencies.append(
            np.pad(
                current_log_freqs,
                (0, max_length - len(current_log_freqs)),
                "constant",
                constant_values=-1,
            )
        )

        all_zero_freqs.append(
            np.pad(
                zero_freqs,
                (0, max_length - len(zero_freqs)),
                "constant",
                constant_values=0,
            )
        )

        all_word_positions.append(
            np.pad(
                current_positions,
                (0, max_length - len(current_positions)),
                "constant",
                constant_values=-1,
            )
        )

        # Spillover features
        all_log_freqs_prev1.append(
            np.pad(
                current_log_freqs_prev1,
                (0, max_length - len(current_log_freqs_prev1)),
                "constant",
                constant_values=-1.0,
            )
        )

        all_zero_freqs_prev1.append(
            np.pad(
                zero_freqs_prev1,
                (0, max_length - len(zero_freqs_prev1)),
                "constant",
                constant_values=0,
            )
        )

        all_log_freqs_prev2.append(
            np.pad(
                current_log_freqs_prev2,
                (0, max_length - len(current_log_freqs_prev2)),
                "constant",
                constant_values=-1.0,
            )
        )

        all_zero_freqs_prev2.append(
            np.pad(
                zero_freqs_prev2,
                (0, max_length - len(zero_freqs_prev2)),
                "constant",
                constant_values=0,
            )
        )

        # Word length spillover features
        all_word_lengths_prev1.append(
            np.pad(
                current_lengths_prev1,
                (0, max_length - len(current_lengths_prev1)),
                "constant",
                constant_values=-1,
            )
        )

        all_word_lengths_prev2.append(
            np.pad(
                current_lengths_prev2,
                (0, max_length - len(current_lengths_prev2)),
                "constant",
                constant_values=-1,
            )
        )

        # Process reading times
        current_reading_times = reading_times[sent_index]
        assert len(current_reading_times) == len(words), (
            "Reading times length does not match the number of words."
        )

        all_reading_times.append(
            np.pad(
                current_reading_times,
                (0, max_length - len(current_reading_times)),
                "constant",
                constant_values=-1,
            )
        )

        # Process ROI data if available (keep at word-level)
        if roi_data is not None:
            current_roi = roi_data[sent_index]
            assert len(current_roi) == len(words), (
                f"ROI data length ({len(current_roi)}) does not match number of words ({len(words)})"
            )

            # Replace NaN with -100 before padding
            # (torch.long() converts NaN to 0, which causes incorrect ROI matching)
            current_roi_clean = np.array([x if not np.isnan(x) else -100 for x in current_roi])

            # Keep ROI at word-level, just pad to max_length
            roi_padded = np.pad(
                current_roi_clean,
                (0, max_length - len(current_roi_clean)),
                "constant",
                constant_values=-100,  # Pad value for ROI (well outside actual range of -8 to 9)
            )
            all_roi_values.append(roi_padded)

        # Process garden-path metadata if available (keep at word-level)
        if pair_id_data is not None:
            current_pair_ids = pair_id_data[sent_index]
            if not isinstance(current_pair_ids, list):
                current_pair_ids = [current_pair_ids] * len(words)

            # Keep pair_id at word-level, manually pad to max_length
            padding_length = max_length - len(current_pair_ids)
            pair_ids_padded = list(current_pair_ids) + ["PAD"] * padding_length
            all_pair_ids.append(np.array(pair_ids_padded, dtype=object))

        if construction_data is not None:
            current_constructions = construction_data[sent_index]
            if not isinstance(current_constructions, list):
                current_constructions = [current_constructions] * len(words)

            # Keep construction at word-level, manually pad to max_length
            padding_length = max_length - len(current_constructions)
            constructions_padded = list(current_constructions) + ["PAD"] * padding_length
            all_constructions.append(np.array(constructions_padded, dtype=object))

        if ambiguity_data is not None:
            current_ambiguities = ambiguity_data[sent_index]
            if not isinstance(current_ambiguities, list):
                current_ambiguities = [current_ambiguities] * len(words)

            # Keep ambiguity at word-level, manually pad to max_length
            padding_length = max_length - len(current_ambiguities)
            ambiguities_padded = list(current_ambiguities) + ["PAD"] * padding_length
            all_ambiguities.append(np.array(ambiguities_padded, dtype=object))

        # Save words for logging
        all_words.append(
            np.pad(words, (0, max_length - len(words)), "constant", constant_values=-1)
        )

    # Create result dictionary
    result = {
        "input_ids": np.asarray(tokenized_texts, dtype=np.int64),
        "attention_mask": np.asarray(attention_masks, dtype=np.int64),
        "reading_times": np.asarray(all_reading_times, dtype=np.float32),
        "word_ids": np.asarray(token_word_ids, dtype=np.int64),
        "word_lengths": np.asarray(all_word_lengths, dtype=np.int64),
        "log_unigram_frequencies": np.asarray(all_log_frequencies, dtype=np.float32),
        "zero_freq_mask": np.asarray(all_zero_freqs, dtype=np.int64),
        "word_positions": np.asarray(all_word_positions, dtype=np.int64),
        "words": np.asarray(all_words, dtype=object),
        # Spillover features
        "log_unigram_frequencies_prev1": np.asarray(
            all_log_freqs_prev1, dtype=np.float32
        ),
        "zero_freq_mask_prev1": np.asarray(all_zero_freqs_prev1, dtype=np.int64),
        "log_unigram_frequencies_prev2": np.asarray(
            all_log_freqs_prev2, dtype=np.float32
        ),
        "zero_freq_mask_prev2": np.asarray(all_zero_freqs_prev2, dtype=np.int64),
        # Word length spillover features
        "word_lengths_prev1": np.asarray(all_word_lengths_prev1, dtype=np.int64),
        "word_lengths_prev2": np.asarray(all_word_lengths_prev2, dtype=np.int64),
    }

    # Add optional data if available
    if roi_data is not None:
        result["roi"] = np.asarray(all_roi_values, dtype=np.int64)

    if pair_id_data is not None:
        result["pair_id"] = np.asarray(all_pair_ids, dtype=object)
    if construction_data is not None:
        result["construction"] = np.asarray(all_constructions, dtype=object)
    if ambiguity_data is not None:
        result["ambiguity"] = np.asarray(all_ambiguities, dtype=object)

    return result
