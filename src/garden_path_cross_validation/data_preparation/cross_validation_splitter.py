#!/usr/bin/env python3
"""
Create cross-validation data splits for garden-path fine-tuning.

This script:
1. Takes 24 garden-path items and reserves 1 for final evaluation
2. Splits remaining 23 items into train:test = 6:4 (approximately 14:9 items)
3. Converts data to Natural Stories format for reverse-engineering training
4. Generates subject-averaged reading times in the appropriate format

Usage:
    python create_cross_validation_data.py [--fold-id 0] [--output-dir DIR]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from wordfreq import word_frequency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def calculate_frequency_features(word: str, prev_words: list[str]) -> dict:
    """Calculate frequency-related features for a word."""
    # Clean word by removing punctuation (matching fillers_processor.py)
    clean_word = word.replace("%2C", "").replace(".", "")

    # Current word frequency (log2 transformed, similar to Natural Stories)
    frequency = word_frequency(clean_word, "en", wordlist="best")
    freq = -np.log2(frequency if frequency > 0 else 1e-10)

    # Previous word frequencies
    if len(prev_words) > 0:
        clean_prev1 = prev_words[0].replace("%2C", "").replace(".", "")
        freq_prev1_val = word_frequency(clean_prev1, "en", wordlist="best")
        freq_prev1 = -np.log2(freq_prev1_val if freq_prev1_val > 0 else 1e-10)
    else:
        freq_prev1 = np.nan

    if len(prev_words) > 1:
        clean_prev2 = prev_words[1].replace("%2C", "").replace(".", "")
        freq_prev2_val = word_frequency(clean_prev2, "en", wordlist="best")
        freq_prev2 = -np.log2(freq_prev2_val if freq_prev2_val > 0 else 1e-10)
    else:
        freq_prev2 = np.nan

    return {"freq": freq, "freq_prev1": freq_prev1, "freq_prev2": freq_prev2}


def calculate_length_features(word: str, prev_words: list[str]) -> dict:
    """Calculate length-related features for a word."""
    # Clean word by replacing %2C with comma for accurate length (matching fillers_processor.py)
    clean_word = word.replace("%2C", ",")

    return {
        "length": len(clean_word),
        "length_prev1": len(prev_words[0].replace("%2C", ",")) if len(prev_words) > 0 else np.nan,
        "length_prev2": len(prev_words[1].replace("%2C", ",")) if len(prev_words) > 1 else np.nan,
    }


def convert_to_natural_stories_format(
    df: pd.DataFrame, item_list: list[int], split_name: str, construction_idx: int = 0
) -> pd.DataFrame:
    """
    Convert garden-path data to Natural Stories format with unified cross-phenomenon support.

    Args:
        df: Input DataFrame with garden-path data
        item_list: List of item IDs to include
        split_name: Name of the split (train/eval/test)
        construction_idx: Index of construction (0=MVRR, 1=NPS, 2=NPZ) for unique pair_id generation

    Returns:
        DataFrame in Natural Stories format optimized for CrossPhenomenonBatchSampler
    """
    # Filter to specified items
    filtered_df = df[df["item"].isin(item_list)].copy()
    logger.info(f"Creating {split_name} data from {len(filtered_df)} word observations")

    natural_stories_data = []

    # Start global sentence ID based on construction to ensure uniqueness
    # MVRR: 1-48, NPS: 101-148, NPZ: 201-248
    global_sentence_id = (construction_idx * 100) + 1

    # Group by item and sentence type to process each sentence separately
    for (item_id, _sentence_type), group in filtered_df.groupby(["item", "Type"]):
        # Sort by word position to maintain sentence order
        sentence_data = group.sort_values("WordPosition")

        # Extract words to calculate features
        words = sentence_data["EachWord"].tolist()

        for idx, (_, row) in enumerate(sentence_data.iterrows()):
            word = row["EachWord"]
            position = row["WordPosition"]

            # Get previous words for spillover features
            prev_words = []
            if idx > 0:
                prev_words.append(words[idx - 1])
            if idx > 1:
                prev_words.append(words[idx - 2])

            # Calculate features
            freq_features = calculate_frequency_features(word, prev_words)
            length_features = calculate_length_features(word, prev_words)

            # Create Natural Stories format row
            ns_row = {
                "sentence_num": global_sentence_id,  # Use global sentence id for grouping
                "word": word,
                "reading_time": row["RT_mean"],
                "reading_time_sd": row["RT_std"],
                "length": length_features["length"],
                "freq": freq_features["freq"],
                "position": position,
                "freq_prev1": freq_features["freq_prev1"],
                "freq_prev2": freq_features["freq_prev2"],
                "length_prev1": length_features["length_prev1"],
                "length_prev2": length_features["length_prev2"],
                "item": int(item_id),  # Used for pairing sentences in batch samplers
                "nItem": len(item_list),
                "roi": int(row["ROI"]),  # Add ROI information for loss weighting
                # Metadata columns
                "Type": row["Type"],  # e.g., MVRR_AMB, MVRR_UNAMB
                "construction": row["CONSTRUCTION"],  # MVRR, NPS, or NPZ
                "ambiguity": row["AMBIG"],  # Amb or Unamb
            }

            natural_stories_data.append(ns_row)

        global_sentence_id += 1

    # Convert to DataFrame
    ns_df = pd.DataFrame(natural_stories_data)

    # Handle empty data case
    if len(ns_df) == 0:
        logger.warning(f"No data found for {split_name} - items may have been excluded during preprocessing")
        # Return empty DataFrame with expected columns
        empty_df = pd.DataFrame(columns=[
            "index", "sentence_num", "word", "reading_time", "reading_time_sd",
            "length", "freq", "position", "freq_prev1", "freq_prev2",
            "length_prev1", "length_prev2", "item", "nItem", "roi",
            "Type", "construction", "ambiguity"
        ])
        return empty_df

    # Add index column as the first column (simple row ID)
    ns_df.insert(0, 'index', range(len(ns_df)))

    logger.info(
        f"Converted to {len(ns_df)} words across {ns_df['sentence_num'].nunique()} sentences"
    )
    return ns_df


class CrossValidationSplitter:
    """Wrapper class for cross-validation splitting functionality."""

    def create_folds(self, num_folds: int = 24, output_dir: str = None):
        """Create unified cross-validation folds containing all constructions."""
        import json

        supported_constructions = ["MVRR", "NPS", "NPZ"]
        logger.info(f"Creating {num_folds} unified folds for all constructions: {supported_constructions}")

        if output_dir is None:
            output_dir = Path("../folds")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Load subject-averaged data
        try:
            subject_data_path = Path(
                "src/garden_path_cross_validation/subject_averages/word_averaged_reading_times.csv"
            )
            if not subject_data_path.exists():
                logger.error(f"Subject-averaged data not found: {subject_data_path}")
                logger.info("Please run 'process-human-data' command first")
                return

            logger.info(f"Loading subject-averaged data from {subject_data_path}")
            subject_data = pd.read_csv(subject_data_path)

            # Get available items for each construction (should be 1-24 for each)
            construction_items = {}
            for construction in supported_constructions:
                construction_data = subject_data[
                    subject_data["CONSTRUCTION"] == construction
                ]
                available_items = sorted(
                    [int(item) for item in construction_data["item"].unique()]
                )
                construction_items[construction] = available_items
                logger.info(f"Available items for {construction}: {available_items}")

                if len(available_items) != num_folds:
                    logger.warning(
                        f"Expected {num_folds} items but found {len(available_items)} for {construction}"
                    )

        except Exception as e:
            logger.error(f"Error loading subject data: {e}")
            return

        # Find the common set of available items across all constructions
        common_items = set(construction_items[supported_constructions[0]])
        for construction in supported_constructions[1:]:
            common_items = common_items.intersection(set(construction_items[construction]))

        common_items = sorted(list(common_items))
        logger.info(f"Common items across all constructions: {common_items}")

        if len(common_items) < num_folds:
            logger.warning(f"Only {len(common_items)} items available, creating {len(common_items)} folds instead of {num_folds}")
            num_folds = len(common_items)

        logger.info(f"Creating {num_folds} unified folds in {output_dir}")

        fold_counter = 0
        for test_item_id in common_items[:num_folds]:
            # Use leave-one-out: all other items for training
            train_item_ids = [item for item in common_items if item != test_item_id]

            fold_dir = output_dir / f"fold_{fold_counter}"
            fold_dir.mkdir(exist_ok=True)

            # Create combined data from all constructions
            all_train_data = []
            all_test_data = []

            has_valid_data = True

            for construction_idx, construction in enumerate(supported_constructions):
                construction_data = subject_data[
                    subject_data["CONSTRUCTION"] == construction
                ]

                # Check if test item exists for this construction
                if test_item_id not in construction_items[construction]:
                    logger.warning(f"Test item {test_item_id} not available for {construction} in fold {fold_counter}")
                    has_valid_data = False
                    continue

                # Filter train_item_ids to only include items available for this construction
                valid_train_items = [item for item in train_item_ids if item in construction_items[construction]]

                # Create splits for this construction
                train_data = convert_to_natural_stories_format(
                    construction_data, valid_train_items, f"train_{construction}", construction_idx
                )
                test_data = convert_to_natural_stories_format(
                    construction_data, [test_item_id], f"test_{construction}", construction_idx
                )

                all_train_data.append(train_data)
                all_test_data.append(test_data)

            if not has_valid_data:
                logger.warning(f"Skipping fold {fold_counter} due to missing test data")
                fold_counter += 1
                continue

            # Combine all constructions
            combined_train = pd.concat(all_train_data, ignore_index=True)
            combined_test = pd.concat(all_test_data, ignore_index=True)

            # Save combined data files
            train_output = fold_dir / "garden_path_train.csv"
            test_output = fold_dir / "garden_path_test.csv"

            combined_train.to_csv(train_output, index=False)
            combined_test.to_csv(test_output, index=False)

            logger.info(
                f"Fold {fold_counter}: train={len(combined_train)} words, test={len(combined_test)} words"
            )

            # Save split configuration
            split_info = {
                "fold_id": fold_counter,
                "dataset": "unified",
                "constructions": supported_constructions,
                "split_strategy": "23:1 (leave-one-out)",
                "num_folds": num_folds,
                "test_item": test_item_id,
                "train_items": train_item_ids,
                "test_items": [test_item_id],
                "train_count": len(train_item_ids),
                "test_count": 1,
                "constructions_included": supported_constructions,
            }

            split_output = fold_dir / "split_info.json"
            with open(split_output, "w") as f:
                json.dump(split_info, f, indent=2)

            fold_counter += 1

        logger.info(f"Created {fold_counter} unified folds successfully (requested: {num_folds})")

