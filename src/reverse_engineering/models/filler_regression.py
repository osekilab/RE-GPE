"""
Filler data regression coefficient computation for garden-path training.

This module computes regression coefficients using filler data,
following the same pattern as SPR evaluation.
Fits one regression model on the entire filler dataset.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from data.tokenize_garden_path import tokenize_garden_path_texts
from data.load_spr import SPRDataset
from models.surprisal import compute_surprisal, compute_surprisal_with_wt_decoding
from models.loss import compute_regression_coefficients

logger = logging.getLogger(__name__)


def load_filler_data(
    tokenizer: PreTrainedTokenizer,
    config: dict,
    device: torch.device,
    batch_size: int = 8,
    data_path: Optional[str] = None
) -> Tuple[Optional[DataLoader], Optional[Dict]]:
    """
    Load filler data for regression coefficient computation.

    Args:
        tokenizer: Tokenizer for the model
        config: Configuration dictionary
        device: Device to use
        batch_size: Batch size for data loading
        data_path: Explicit path to filler data file

    Returns:
        Tuple of (filler_data_loader, metadata)
    """
    # Use explicit path if provided, otherwise use default
    if data_path:
        filler_path = Path(data_path)
    else:
        # Default path relative to src/reverse_engineering
        filler_path = Path("../../src/garden_path_cross_validation/folds/fillers_processed.csv")

    if not filler_path.exists():
        logger.warning(f"Filler data not found at {filler_path}")
        return None, None

    logger.info(f"Loading filler data from {filler_path}")
    filler_df = pd.read_csv(filler_path)

    # URL decode words (same as SPR datasets)
    if "word" in filler_df.columns:
        filler_df["word"] = filler_df["word"].str.replace("%2C", ",")

    logger.info(f"Loaded {len(filler_df)} filler words")
    logger.info(f"Number of filler sentences: {filler_df['item'].nunique()}")

    # Group by sentence (item)
    texts = filler_df.groupby("item")["word"].apply(list).tolist()
    reading_times = filler_df.groupby("item")["reading_time"].apply(list).tolist()

    # Extract pre-computed features (same as SPR datasets)
    word_lengths = None
    log_frequencies = None
    log_frequencies_prev1 = None
    log_frequencies_prev2 = None
    word_lengths_prev1 = None
    word_lengths_prev2 = None
    word_positions = None

    if "length" in filler_df.columns:
        word_lengths = filler_df.groupby("item")["length"].apply(list).tolist()

    if "freq" in filler_df.columns:
        log_frequencies = filler_df.groupby("item")["freq"].apply(list).tolist()

    if "freq_prev1" in filler_df.columns:
        log_frequencies_prev1 = filler_df.groupby("item")["freq_prev1"].apply(list).tolist()

    if "freq_prev2" in filler_df.columns:
        log_frequencies_prev2 = filler_df.groupby("item")["freq_prev2"].apply(list).tolist()

    if "length_prev1" in filler_df.columns:
        word_lengths_prev1 = filler_df.groupby("item")["length_prev1"].apply(list).tolist()

    if "length_prev2" in filler_df.columns:
        word_lengths_prev2 = filler_df.groupby("item")["length_prev2"].apply(list).tolist()

    # Note: filler data uses "word_position" not "position" like some SPR datasets
    if "word_position" in filler_df.columns:
        word_positions = filler_df.groupby("item")["word_position"].apply(list).tolist()

    max_length = config["training_kwargs"]["filler_max_length"]

    data_dict = tokenize_garden_path_texts(
        texts,
        reading_times,
        tokenizer,
        max_length=max_length,
        word_lengths=word_lengths,
        log_frequencies=log_frequencies,
        log_frequencies_prev1=log_frequencies_prev1,
        log_frequencies_prev2=log_frequencies_prev2,
        word_lengths_prev1=word_lengths_prev1,
        word_lengths_prev2=word_lengths_prev2,
        word_positions=word_positions,
        # Filler data doesn't have garden-path specific fields
        roi_data=None,
        pair_id_data=None,
        construction_data=None,
        ambiguity_data=None,
    )

    dataset = SPRDataset(data_dict)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return data_loader, {"total_words": len(filler_df), "total_sentences": filler_df['item'].nunique()}


def compute_filler_coefficients(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    config: dict,
    use_frequency_spillover: bool = True,
    use_surprisal_spillover: bool = True,
    use_length_spillover: bool = True,
    use_word_position: bool = True,
    mask_zero_reading_times: bool = True,
    mask_zero_freqs: bool = True,
    mask_last_region: bool = True,
    use_wt_decoding: bool = False,
    logger=None,
    step: int = None,
    data_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute regression coefficients using filler data with current model's surprisal.

    Following the same pattern as SPR evaluation:
    1. Load and tokenize filler data
    2. Compute surprisal using current model
    3. Fit regression on entire dataset
    4. Return coefficients

    Args:
        model: Current language model for surprisal computation
        tokenizer: Tokenizer
        device: Computation device
        config: Configuration dictionary
        use_frequency_spillover: Include frequency spillover features
        use_surprisal_spillover: Include surprisal spillover features
        use_length_spillover: Include length spillover features
        use_word_position: Include word position feature
        mask_zero_reading_times: Mask zero reading times
        mask_zero_freqs: Mask zero frequencies
        mask_last_region: Mask the last word in each sentence
        logger: Logger instance
        step: Current training step
        data_path: Explicit path to filler data file

    Returns:
        Dictionary of regression coefficients
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Load filler data
    filler_loader, metadata = load_filler_data(
        tokenizer=tokenizer,
        config=config,
        device=device,
        batch_size=8,
        data_path=data_path
    )

    if filler_loader is None:
        logger.warning("No filler data available")
        return {}

    logger.info(f"Processing {metadata['total_sentences']} filler sentences")

    # Collect all features across batches
    all_surprisal = []
    all_reading_times = []
    all_word_lengths = []
    all_word_frequencies = []
    all_word_positions = []
    all_freq_prev1 = []
    all_freq_prev2 = []
    all_surprisal_prev1 = []
    all_surprisal_prev2 = []
    all_length_prev1 = []
    all_length_prev2 = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(filler_loader):
            # Move batch to device
            batch = {
                "input_ids": batch_data["input_ids"].to(device),
                "attention_mask": batch_data["attention_mask"].to(device),
            }

            # Get features from batch
            word_ids = batch_data["word_ids"].to(device)
            reading_times = batch_data["reading_times"].to(device)
            word_lengths = batch_data["word_lengths"].to(device)
            word_frequencies = batch_data["log_unigram_frequencies"].to(device)

            # Compute surprisal (subword level, with WT decoding if enabled)
            if use_wt_decoding:
                surprisal_subword, _, _ = compute_surprisal_with_wt_decoding(batch, model, tokenizer, word_ids)
            else:
                surprisal_subword, _, _ = compute_surprisal(batch, model, tokenizer, word_ids)

            # Aggregate to word level (same as loss.py)
            mask_word_ids = word_ids != -1
            masked_surprisal_subword = surprisal_subword * mask_word_ids
            masked_word_ids = word_ids * mask_word_ids

            surprisal_word = torch.zeros_like(surprisal_subword)
            surprisal_word.scatter_add_(1, masked_word_ids, masked_surprisal_subword)

            # Process each sentence in the batch
            for idx in range(batch["input_ids"].shape[0]):
                # Create valid mask for this sentence
                valid_mask = (word_ids[idx] != -1) & (reading_times[idx] != -1)
                if mask_zero_reading_times:
                    valid_mask = valid_mask & (reading_times[idx] > 0)
                if mask_zero_freqs:
                    zero_freq_mask = batch_data["zero_freq_mask"][idx].to(device)
                    valid_mask = valid_mask & (zero_freq_mask != 0)
                    # Also mask spillover features with zero frequency
                    if use_frequency_spillover:
                        zero_freq_mask_prev1 = batch_data["zero_freq_mask_prev1"][idx].to(device)
                        valid_mask = valid_mask & (zero_freq_mask_prev1 != 0)
                        zero_freq_mask_prev2 = batch_data["zero_freq_mask_prev2"][idx].to(device)
                        valid_mask = valid_mask & (zero_freq_mask_prev2 != 0)

                # Mask last region if requested
                if mask_last_region:
                    # Find the last valid position and mask it
                    valid_positions = torch.where(valid_mask)[0]
                    if len(valid_positions) > 0:
                        last_position = valid_positions[-1]
                        valid_mask[last_position] = False

                # Flatten valid features
                all_surprisal.append(surprisal_word[idx][valid_mask])
                all_reading_times.append(reading_times[idx][valid_mask])
                all_word_lengths.append(word_lengths[idx][valid_mask])
                all_word_frequencies.append(word_frequencies[idx][valid_mask])

                # Word positions
                if use_word_position:
                    word_positions = batch_data["word_positions"][idx].to(device)
                    all_word_positions.append(word_positions[valid_mask])

                # Frequency spillover (from pre-computed data)
                if use_frequency_spillover:
                    freq_prev1 = batch_data["log_unigram_frequencies_prev1"][idx].to(device)
                    freq_prev2 = batch_data["log_unigram_frequencies_prev2"][idx].to(device)
                    all_freq_prev1.append(freq_prev1[valid_mask])
                    all_freq_prev2.append(freq_prev2[valid_mask])

                # Surprisal spillover (compute from surprisal_word, before flattening)
                if use_surprisal_spillover:
                    surprisal_prev1_2d = torch.zeros_like(surprisal_word[idx:idx+1])
                    surprisal_prev2_2d = torch.zeros_like(surprisal_word[idx:idx+1])
                    surprisal_prev1_2d[:, 1:] = surprisal_word[idx:idx+1, :-1]
                    surprisal_prev2_2d[:, 2:] = surprisal_word[idx:idx+1, :-2]
                    all_surprisal_prev1.append(surprisal_prev1_2d[0][valid_mask])
                    all_surprisal_prev2.append(surprisal_prev2_2d[0][valid_mask])

                # Length spillover (from pre-computed data)
                if use_length_spillover and "word_lengths_prev1" in batch_data:
                    length_prev1 = batch_data["word_lengths_prev1"][idx].to(device)
                    length_prev2 = batch_data["word_lengths_prev2"][idx].to(device)
                    all_length_prev1.append(length_prev1[valid_mask])
                    all_length_prev2.append(length_prev2[valid_mask])

    # Concatenate all features
    all_surprisal = torch.cat(all_surprisal)
    all_reading_times = torch.cat(all_reading_times)
    all_word_lengths = torch.cat(all_word_lengths)
    all_word_frequencies = torch.cat(all_word_frequencies)

    # Optional features
    word_positions_concat = torch.cat(all_word_positions) if all_word_positions else None
    freq_prev1_concat = torch.cat(all_freq_prev1) if all_freq_prev1 else None
    freq_prev2_concat = torch.cat(all_freq_prev2) if all_freq_prev2 else None
    surprisal_prev1_concat = torch.cat(all_surprisal_prev1) if all_surprisal_prev1 else None
    surprisal_prev2_concat = torch.cat(all_surprisal_prev2) if all_surprisal_prev2 else None
    length_prev1_concat = torch.cat(all_length_prev1) if all_length_prev1 else None
    length_prev2_concat = torch.cat(all_length_prev2) if all_length_prev2 else None

    # Compute regression coefficients (same as loss.py)
    coefficients = compute_regression_coefficients(
        surprisal=all_surprisal,
        reading_times=all_reading_times,
        word_lengths=all_word_lengths,
        word_frequencies=all_word_frequencies,
        word_positions=word_positions_concat,
        use_frequency_spillover=use_frequency_spillover and freq_prev1_concat is not None,
        use_surprisal_spillover=use_surprisal_spillover and surprisal_prev1_concat is not None,
        use_length_spillover=use_length_spillover and length_prev1_concat is not None,
        word_frequencies_prev1=freq_prev1_concat,
        word_frequencies_prev2=freq_prev2_concat,
        surprisal_prev1=surprisal_prev1_concat,
        surprisal_prev2=surprisal_prev2_concat,
        word_lengths_prev1=length_prev1_concat,
        word_lengths_prev2=length_prev2_concat,
    )

    logger.info(f"Computed filler coefficients with {len(all_reading_times)} valid words")
    if step is not None:
        logger.info(f"Step {step} - Filler regression coefficients:")
    for name, value in coefficients.items():
        logger.info(f"  {name}: {value.item():.4f}")

    return coefficients
