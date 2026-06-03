"""
Unified corpus evaluation module for all reading time data.

This module provides a single interface for evaluating models on any corpus
with reading time data (SPR datasets, eye-tracking corpora, filler data, etc.)
using delta log-likelihood approach.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.load_spr import SPRDataset, process_spr_corpus
from models.loss import compute_regression_coefficients, compute_predicted_reading_times
from models.surprisal import compute_surprisal, compute_surprisal_with_wt_decoding

logger = logging.getLogger(__name__)


def evaluate_corpus(
    model: torch.nn.Module,
    tokenizer,
    corpus_name: str,
    corpus_path: str,
    config: dict,
    device: torch.device,
    use_wt_decoding: bool = False,
    external_coefficients: Optional[Dict[str, torch.Tensor]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Evaluate model on any corpus with reading time data.

    This is a unified function for evaluating SPR datasets, eye-tracking corpora,
    filler data, or any other corpus with the same format.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        corpus_name: Name of the corpus (for logging)
        corpus_path: Path to the corpus CSV file
        config: Configuration dictionary
        device: Device to use
        use_wt_decoding: Whether to use WT decoding for surprisal computation
        external_coefficients: Optional pre-computed coefficients (e.g., from training data or filler).
                              If provided, these will be used instead of computing from corpus data.
        logger: Logger instance

    Returns:
        Dictionary with evaluation results including delta log-likelihood and coefficients
    """
    if logger:
        logger.info(f"Evaluating on {corpus_name.upper()} corpus...")

    # Check if corpus file exists
    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        if logger:
            logger.warning(f"{corpus_name.upper()} corpus not found at {corpus_path}")
        return {
            "error": f"Corpus file not found: {corpus_path}",
            "delta_llh": None,
        }

    try:
        # Load corpus data
        corpus_df = pd.read_csv(corpus_path)
        if logger:
            logger.info(f"Loaded {len(corpus_df)} words from {corpus_name.upper()}")

        # Handle different column naming conventions
        if "item" in corpus_df.columns and "sentence_num" not in corpus_df.columns:
            corpus_df["sentence_num"] = corpus_df["item"]
        if "word_position" in corpus_df.columns and "position" not in corpus_df.columns:
            corpus_df["position"] = corpus_df["word_position"]

        # Filter out invalid reading times (NaN, negative, zero)
        valid_mask = (
            corpus_df["reading_time"].notna() &
            (corpus_df["reading_time"] > 0)
        )
        valid_df = corpus_df[valid_mask].copy()
        if logger:
            logger.info(
                f"Using {len(valid_df)} words with valid reading times "
                f"({len(valid_df)/len(corpus_df):.1%})"
            )

        # Save processed data to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            corpus_df.to_csv(tmp.name, index=False)
            temp_corpus_path = tmp.name

        # Prepare config for SPR processing pipeline
        temp_config = config.copy()
        temp_config["training_kwargs"] = config.get("training_kwargs", {}).copy()

        # Override data path and settings
        temp_config["training_kwargs"]["eval_data_path"] = temp_corpus_path
        temp_config["training_kwargs"]["eval_dataset_target"] = "reading_time"

        # Process corpus using unified SPR loader
        data_dict = process_spr_corpus(
            tokenizer=tokenizer,
            config=temp_config,
            spr_dataset_name=corpus_name,
            evaluate=True,
        )

        # Create dataset and dataloader
        dataset = SPRDataset(data_dict)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )

        # Collect features for regression (following SPR evaluation pattern)
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

        # Get configuration parameters
        training_kwargs = config.get("training_kwargs", {})

        # Auto-detect spillover features from data (overrides config)
        # Check if first batch has spillover columns to determine availability
        first_batch_iter = iter(dataloader)
        first_batch = next(first_batch_iter)
        has_freq_spillover = (
            "log_unigram_frequencies_prev1" in first_batch and
            "log_unigram_frequencies_prev2" in first_batch
        )
        has_length_spillover = (
            "word_lengths_prev1" in first_batch and
            "word_lengths_prev2" in first_batch
        )

        # Recreate dataloader after consuming one batch for inspection
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )

        # Use spillover features if available in data, regardless of config
        # This ensures consistency with process_spr_corpus behavior
        use_frequency_spillover = has_freq_spillover
        use_length_spillover = has_length_spillover
        # Surprisal spillover is always computed from surprisal, so use config default
        use_surprisal_spillover = training_kwargs.get(
            "use_surprisal_spillover", True  # Default to True for consistency
        )
        use_word_position = training_kwargs.get("use_word_position", True)

        if logger:
            logger.info(f"Spillover features detected in data:")
            logger.info(f"  Frequency spillover: {has_freq_spillover}")
            logger.info(f"  Length spillover: {has_length_spillover}")
            logger.info(f"  Using frequency spillover: {use_frequency_spillover}")
            logger.info(f"  Using surprisal spillover: {use_surprisal_spillover}")
            logger.info(f"  Using length spillover: {use_length_spillover}")
        # Read masking parameters from config (consistent with evaluate_fold.py)
        mask_zero_reading_times = training_kwargs.get("mask_zero_reading_times", True)
        mask_zero_freqs = training_kwargs.get("mask_zero_freqs", True)
        mask_last_region = training_kwargs.get("mask_last_region", True)

        model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc=f"Processing {corpus_name.upper()}",
                     leave=False)
            ):
                # Save original batch for spillover extraction
                original_batch = {k: v for k, v in batch.items()}

                # Move batch to device
                batch = {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in batch.items()
                }

                # Get word_ids for surprisal computation
                word_ids = batch.get("word_ids").to(device)

                # Compute surprisal (subword level, with WT decoding if enabled)
                if use_wt_decoding:
                    surprisal_subword, _, _ = compute_surprisal_with_wt_decoding(
                        batch, model, tokenizer, word_ids
                    )
                else:
                    surprisal_subword, _, _ = compute_surprisal(
                        batch, model, tokenizer, word_ids
                    )

                # Aggregate subword surprisal to word level (from natural_stories)
                mask_word_ids = word_ids != -1
                masked_surprisal_subword = surprisal_subword * mask_word_ids
                masked_word_ids = word_ids * mask_word_ids

                # Sum surprisal for tokens belonging to same word
                surprisal_word = torch.zeros_like(surprisal_subword)
                surprisal_word.scatter_add_(
                    1, masked_word_ids, masked_surprisal_subword
                )

                # Extract features from batch
                reading_times = batch.get("reading_times").cpu().numpy()
                word_lengths = batch.get("word_lengths").cpu().numpy()
                word_frequencies = batch.get(
                    "log_unigram_frequencies"
                ).cpu().numpy()

                # Create mask for valid words
                valid_mask = (
                    (word_ids.cpu().numpy() != -1) &
                    (reading_times != -1)
                )
                if mask_zero_reading_times:
                    valid_mask = valid_mask & (reading_times > 0)
                if mask_zero_freqs:
                    zero_freq_mask = batch.get("zero_freq_mask")
                    if zero_freq_mask is not None:
                        valid_mask = valid_mask & (
                            zero_freq_mask.cpu().numpy() != 0
                        )
                        # Also mask spillover features with zero frequency
                        if use_frequency_spillover:
                            zero_freq_mask_prev1 = batch.get(
                                "zero_freq_mask_prev1"
                            )
                            if zero_freq_mask_prev1 is not None:
                                valid_mask = valid_mask & (
                                    zero_freq_mask_prev1.cpu().numpy() != 0
                                )
                            zero_freq_mask_prev2 = batch.get(
                                "zero_freq_mask_prev2"
                            )
                            if zero_freq_mask_prev2 is not None:
                                valid_mask = valid_mask & (
                                    zero_freq_mask_prev2.cpu().numpy() != 0
                                )

                # Mask last region if requested (same as natural_stories_evaluation.py)
                if mask_last_region:
                    # For each sentence in the batch, find the last valid word position and mask it
                    for batch_idx in range(valid_mask.shape[0]):
                        # Find the last valid position (rightmost True in valid_mask)
                        valid_positions = np.where(valid_mask[batch_idx])[0]
                        if len(valid_positions) > 0:
                            last_position = valid_positions[-1]
                            valid_mask[batch_idx, last_position] = False

                # Mask ROI regions if data has 'roi' field (garden-path data)
                # For training data evaluation, exclude ROI positions from regression fitting
                if "roi" in batch and config.get("training_kwargs", {}).get("exclude_roi_from_regression", False):
                    roi_values = batch["roi"].cpu().numpy()
                    roi_exclusion_values = config.get("training_kwargs", {}).get("roi_exclusion_values", [0, 1, 2])

                    # Create ROI mask (False for excluded ROI values)
                    roi_mask = np.ones_like(valid_mask, dtype=bool)
                    for roi_val in roi_exclusion_values:
                        roi_mask = roi_mask & (roi_values != roi_val)

                    # Apply ROI mask to valid_mask
                    valid_mask = valid_mask & roi_mask

                # Flatten and mask features
                surprisal_flat = surprisal_word.cpu().numpy()[valid_mask]
                rt_flat = reading_times[valid_mask]
                length_flat = word_lengths[valid_mask]
                freq_flat = word_frequencies[valid_mask]

                # Append to collections
                all_surprisal.append(surprisal_flat)
                all_reading_times.append(rt_flat)
                all_word_lengths.append(length_flat)
                all_word_frequencies.append(freq_flat)

                # Extract word positions if available
                if use_word_position and "word_positions" in original_batch:
                    word_pos = original_batch["word_positions"].cpu().numpy()
                    word_pos_flat = word_pos[valid_mask]
                    all_word_positions.append(word_pos_flat)
                else:
                    # Create dummy positions if not using
                    all_word_positions.append(np.zeros_like(surprisal_flat))

                # Extract spillover features if available
                if (use_frequency_spillover and
                    "log_unigram_frequencies_prev1" in original_batch):
                    freq_prev1 = original_batch[
                        "log_unigram_frequencies_prev1"
                    ].cpu().numpy()
                    freq_prev2 = original_batch.get(
                        "log_unigram_frequencies_prev2",
                        torch.zeros_like(
                            original_batch["log_unigram_frequencies_prev1"]
                        )
                    ).cpu().numpy()

                    freq_prev1_flat = freq_prev1[valid_mask]
                    freq_prev2_flat = freq_prev2[valid_mask]

                    all_freq_prev1.append(freq_prev1_flat)
                    all_freq_prev2.append(freq_prev2_flat)
                    has_spillover_features = True

                if use_surprisal_spillover:
                    # Compute surprisal spillover BEFORE flattening
                    surprisal_prev1_2d = torch.zeros_like(surprisal_word)
                    surprisal_prev2_2d = torch.zeros_like(surprisal_word)

                    # Shift within each sequence
                    surprisal_prev1_2d[:, 1:] = surprisal_word[:, :-1]
                    surprisal_prev2_2d[:, 2:] = surprisal_word[:, :-2]

                    # Now flatten using the same valid_mask
                    surprisal_prev1_flat = (
                        surprisal_prev1_2d.cpu().numpy()[valid_mask]
                    )
                    surprisal_prev2_flat = (
                        surprisal_prev2_2d.cpu().numpy()[valid_mask]
                    )

                    all_surprisal_prev1.append(surprisal_prev1_flat)
                    all_surprisal_prev2.append(surprisal_prev2_flat)
                    has_spillover_features = True

                if (use_length_spillover and
                    "word_lengths_prev1" in original_batch):
                    length_prev1 = original_batch[
                        "word_lengths_prev1"
                    ].cpu().numpy()
                    length_prev2 = original_batch.get(
                        "word_lengths_prev2",
                        torch.zeros_like(original_batch["word_lengths_prev1"])
                    ).cpu().numpy()

                    length_prev1_flat = length_prev1[valid_mask]
                    length_prev2_flat = length_prev2[valid_mask]

                    all_length_prev1.append(length_prev1_flat)
                    all_length_prev2.append(length_prev2_flat)
                    has_spillover_features = True

        # Clean up temp file
        import os
        try:
            os.unlink(temp_corpus_path)
        except Exception:
            pass

        # Concatenate all features
        if all_surprisal:
            all_surprisal = np.concatenate(all_surprisal)
            all_reading_times = np.concatenate(all_reading_times)
            all_word_lengths = np.concatenate(all_word_lengths)
            all_word_frequencies = np.concatenate(all_word_frequencies)
            if len(all_word_positions) > 0:
                all_word_positions = np.concatenate(all_word_positions)
            else:
                all_word_positions = None

            # Concatenate spillover features if available
            if has_spillover_features:
                if len(all_freq_prev1) > 0:
                    all_freq_prev1 = np.concatenate(all_freq_prev1)
                    all_freq_prev2 = np.concatenate(all_freq_prev2)
                else:
                    all_freq_prev1 = None
                    all_freq_prev2 = None
                if len(all_surprisal_prev1) > 0:
                    all_surprisal_prev1 = np.concatenate(all_surprisal_prev1)
                    all_surprisal_prev2 = np.concatenate(all_surprisal_prev2)
                else:
                    all_surprisal_prev1 = None
                    all_surprisal_prev2 = None
                if len(all_length_prev1) > 0:
                    all_length_prev1 = np.concatenate(all_length_prev1)
                    all_length_prev2 = np.concatenate(all_length_prev2)
                else:
                    all_length_prev1 = None
                    all_length_prev2 = None
            else:
                all_freq_prev1 = None
                all_freq_prev2 = None
                all_surprisal_prev1 = None
                all_surprisal_prev2 = None
                all_length_prev1 = None
                all_length_prev2 = None

            # Convert numpy arrays to torch tensors for regression
            device_reg = torch.device("cpu")  # Regression on CPU
            surprisal_tensor = torch.tensor(
                all_surprisal, dtype=torch.float32, device=device_reg
            )
            reading_times_tensor = torch.tensor(
                all_reading_times, dtype=torch.float32, device=device_reg
            )
            word_lengths_tensor = torch.tensor(
                all_word_lengths, dtype=torch.float32, device=device_reg
            )
            word_frequencies_tensor = torch.tensor(
                all_word_frequencies, dtype=torch.float32, device=device_reg
            )

            # Convert optional features
            word_positions_tensor = (
                torch.tensor(all_word_positions, dtype=torch.float32,
                            device=device_reg)
                if all_word_positions is not None else None
            )
            freq_prev1_tensor = (
                torch.tensor(all_freq_prev1, dtype=torch.float32,
                            device=device_reg)
                if all_freq_prev1 is not None else None
            )
            freq_prev2_tensor = (
                torch.tensor(all_freq_prev2, dtype=torch.float32,
                            device=device_reg)
                if all_freq_prev2 is not None else None
            )
            surprisal_prev1_tensor = (
                torch.tensor(all_surprisal_prev1, dtype=torch.float32,
                            device=device_reg)
                if all_surprisal_prev1 is not None else None
            )
            surprisal_prev2_tensor = (
                torch.tensor(all_surprisal_prev2, dtype=torch.float32,
                            device=device_reg)
                if all_surprisal_prev2 is not None else None
            )
            length_prev1_tensor = (
                torch.tensor(all_length_prev1, dtype=torch.float32,
                            device=device_reg)
                if all_length_prev1 is not None else None
            )
            length_prev2_tensor = (
                torch.tensor(all_length_prev2, dtype=torch.float32,
                            device=device_reg)
                if all_length_prev2 is not None else None
            )

            # Fit baseline model (without surprisal)
            zero_surprisal = torch.zeros_like(surprisal_tensor)

            # If using external coefficients, use them for baseline too
            # (only difference is surprisal=0 vs actual surprisal)
            if external_coefficients is not None:
                # Use same external coefficients for fair comparison
                baseline_coeffs = {}
                for key, value in external_coefficients.items():
                    if isinstance(value, torch.Tensor):
                        baseline_coeffs[key] = value.cpu()
                    else:
                        baseline_coeffs[key] = value
                # Set surprisal coefficients to 0 for baseline
                baseline_coeffs["beta_s"] = torch.tensor(0.0)
                if "beta_s_prev1" in baseline_coeffs:
                    baseline_coeffs["beta_s_prev1"] = torch.tensor(0.0)
                if "beta_s_prev2" in baseline_coeffs:
                    baseline_coeffs["beta_s_prev2"] = torch.tensor(0.0)
            else:
                # Compute baseline coefficients from corpus data
                baseline_coeffs = compute_regression_coefficients(
                    surprisal=zero_surprisal,
                    reading_times=reading_times_tensor,
                    word_lengths=word_lengths_tensor,
                    word_frequencies=word_frequencies_tensor,
                    word_positions=word_positions_tensor,
                    use_frequency_spillover=(
                        use_frequency_spillover and freq_prev1_tensor is not None
                    ),
                    use_surprisal_spillover=False,
                    use_length_spillover=(
                        use_length_spillover and length_prev1_tensor is not None
                    ),
                    word_frequencies_prev1=freq_prev1_tensor,
                    word_frequencies_prev2=freq_prev2_tensor,
                    surprisal_prev1=None,
                    surprisal_prev2=None,
                    word_lengths_prev1=length_prev1_tensor,
                    word_lengths_prev2=length_prev2_tensor,
                )

            # Compute baseline predictions
            # Note: For surprisal spillover, we use zero surprisal
            zero_surprisal_prev1 = (
                torch.zeros_like(surprisal_prev1_tensor)
                if surprisal_prev1_tensor is not None
                else None
            )
            zero_surprisal_prev2 = (
                torch.zeros_like(surprisal_prev2_tensor)
                if surprisal_prev2_tensor is not None
                else None
            )

            baseline_pred = compute_predicted_reading_times(
                surprisal=zero_surprisal,
                word_lengths=word_lengths_tensor,
                word_frequencies=word_frequencies_tensor,
                coefficients=baseline_coeffs,
                word_positions=word_positions_tensor,
                use_frequency_spillover=(
                    use_frequency_spillover and freq_prev1_tensor is not None
                ),
                use_surprisal_spillover=(
                    use_surprisal_spillover and surprisal_prev1_tensor is not None
                ),
                use_length_spillover=(
                    use_length_spillover and length_prev1_tensor is not None
                ),
                word_frequencies_prev1=freq_prev1_tensor,
                word_frequencies_prev2=freq_prev2_tensor,
                surprisal_prev1=zero_surprisal_prev1,
                surprisal_prev2=zero_surprisal_prev2,
                word_lengths_prev1=length_prev1_tensor,
                word_lengths_prev2=length_prev2_tensor,
            ).cpu().numpy()

            # Fit target model with surprisal (or use external coefficients)
            if external_coefficients is not None:
                # Use provided external coefficients (e.g., from training data or filler)
                # Move coefficients to CPU for regression (matching other tensors)
                target_coeffs = {}
                for key, value in external_coefficients.items():
                    if isinstance(value, torch.Tensor):
                        target_coeffs[key] = value.cpu()
                    else:
                        target_coeffs[key] = value
                if logger:
                    logger.info(f"Using external coefficients for {corpus_name.upper()}")
            else:
                # Compute coefficients from corpus data
                target_coeffs = compute_regression_coefficients(
                    surprisal=surprisal_tensor,
                    reading_times=reading_times_tensor,
                    word_lengths=word_lengths_tensor,
                    word_frequencies=word_frequencies_tensor,
                    word_positions=word_positions_tensor,
                    use_frequency_spillover=(
                        use_frequency_spillover and freq_prev1_tensor is not None
                    ),
                    use_surprisal_spillover=(
                        use_surprisal_spillover and surprisal_prev1_tensor is not None
                    ),
                    use_length_spillover=(
                        use_length_spillover and length_prev1_tensor is not None
                    ),
                    word_frequencies_prev1=freq_prev1_tensor,
                    word_frequencies_prev2=freq_prev2_tensor,
                    surprisal_prev1=surprisal_prev1_tensor,
                    surprisal_prev2=surprisal_prev2_tensor,
                    word_lengths_prev1=length_prev1_tensor,
                    word_lengths_prev2=length_prev2_tensor,
                )

            # Compute target predictions
            target_pred = compute_predicted_reading_times(
                surprisal=surprisal_tensor,
                word_lengths=word_lengths_tensor,
                word_frequencies=word_frequencies_tensor,
                coefficients=target_coeffs,
                word_positions=word_positions_tensor,
                use_frequency_spillover=(
                    use_frequency_spillover and freq_prev1_tensor is not None
                ),
                use_surprisal_spillover=(
                    use_surprisal_spillover and surprisal_prev1_tensor is not None
                ),
                use_length_spillover=(
                    use_length_spillover and length_prev1_tensor is not None
                ),
                word_frequencies_prev1=freq_prev1_tensor,
                word_frequencies_prev2=freq_prev2_tensor,
                surprisal_prev1=surprisal_prev1_tensor,
                surprisal_prev2=surprisal_prev2_tensor,
                word_lengths_prev1=length_prev1_tensor,
                word_lengths_prev2=length_prev2_tensor,
            ).cpu().numpy()

            # MSE
            mse_baseline = np.mean((all_reading_times - baseline_pred) ** 2)
            mse_target = np.mean((all_reading_times - target_pred) ** 2)

            # Log-likelihood (assuming Gaussian noise)
            var_baseline = np.var(all_reading_times - baseline_pred)
            var_target = np.var(all_reading_times - target_pred)

            # Avoid division by zero
            var_baseline = max(float(var_baseline), 1e-10)
            var_target = max(float(var_target), 1e-10)

            # Compute log-likelihoods
            llh_baseline = -0.5 * np.log(2 * np.pi * var_baseline) - (
                (all_reading_times - baseline_pred) ** 2
            ) / (2 * var_baseline)
            llh_target = -0.5 * np.log(2 * np.pi * var_target) - (
                (all_reading_times - target_pred) ** 2
            ) / (2 * var_target)

            mean_llh_baseline = np.mean(llh_baseline)
            mean_llh_target = np.mean(llh_target)

            # Delta log-likelihood
            delta_llh = mean_llh_target - mean_llh_baseline

            # Create coefficient dictionary from target_coeffs
            coefficients = {}
            for key, value in target_coeffs.items():
                coefficients[key] = value.item()

            if logger:
                logger.info(f"{corpus_name.upper()} evaluation complete:")
                logger.info(f"  Words analyzed: {len(all_reading_times)}")
                logger.info(f"  LLH Baseline: {mean_llh_baseline:.6f}")
                logger.info(f"  LLH Target: {mean_llh_target:.6f}")
                logger.info(f"  Delta LLH: {delta_llh:.6f}")

                # If using external coefficients, only show surprisal coefficient
                # Otherwise, show all coefficients (corpus-specific)
                if external_coefficients is not None:
                    logger.info(f"  Surprisal coefficient (from training): {coefficients.get('beta_s', 0):.4f}")
                else:
                    logger.info(f"  Coefficients (corpus-specific):")
                    logger.info(f"    Beta_s (surprisal): {coefficients.get('beta_s', 0):.4f}")
                    logger.info(f"    Beta_l (length): {coefficients.get('beta_l', 0):.4f}")
                    logger.info(f"    Beta_f (frequency): {coefficients.get('beta_f', 0):.4f}")
                    if 'beta_pos' in coefficients:
                        logger.info(f"    Beta_pos (position): {coefficients.get('beta_pos', 0):.4f}")
                    if 'beta_f_prev1' in coefficients:
                        logger.info(f"    Beta_f_prev1 (freq lag-1): {coefficients.get('beta_f_prev1', 0):.4f}")
                        logger.info(f"    Beta_f_prev2 (freq lag-2): {coefficients.get('beta_f_prev2', 0):.4f}")
                    if 'beta_s_prev1' in coefficients:
                        logger.info(f"    Beta_s_prev1 (surp lag-1): {coefficients.get('beta_s_prev1', 0):.4f}")
                        logger.info(f"    Beta_s_prev2 (surp lag-2): {coefficients.get('beta_s_prev2', 0):.4f}")
                    if 'beta_l_prev1' in coefficients:
                        logger.info(f"    Beta_l_prev1 (length lag-1): {coefficients.get('beta_l_prev1', 0):.4f}")
                        logger.info(f"    Beta_l_prev2 (length lag-2): {coefficients.get('beta_l_prev2', 0):.4f}")
                    logger.info(f"    Beta_0 (intercept): {coefficients.get('beta_0', 0):.4f}")

            return {
                "corpus": corpus_name,
                "n_words": len(all_reading_times),
                "delta_llh": float(delta_llh),
                "coefficients": coefficients,
                "mse_full": float(mse_target),
                "mse_baseline": float(mse_baseline),
            }
        else:
            if logger:
                logger.warning(f"No valid data for {corpus_name.upper()} regression")
            return {
                "corpus": corpus_name,
                "error": "No valid data for regression",
                "delta_llh": None,
            }

    except Exception as e:
        if logger:
            logger.error(f"Error evaluating {corpus_name.upper()}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return {
            "corpus": corpus_name,
            "error": str(e),
            "delta_llh": None,
        }


def evaluate_multiple_corpora(
    model: torch.nn.Module,
    tokenizer,
    corpora_list: List[Tuple[str, str]],
    config: dict,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Evaluate model on multiple corpora.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        corpora_list: List of (corpus_name, corpus_path) tuples
        config: Configuration dictionary
        device: Device to use
        logger: Logger instance

    Returns:
        Dictionary with results for each corpus
    """
    results = {}

    for corpus_name, corpus_path in corpora_list:
        corpus_results = evaluate_corpus(
            model=model,
            tokenizer=tokenizer,
            corpus_name=corpus_name,
            corpus_path=corpus_path,
            config=config,
            device=device,
            logger=logger,
        )
        results[corpus_name] = corpus_results

    return results
