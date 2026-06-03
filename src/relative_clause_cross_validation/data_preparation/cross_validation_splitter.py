#!/usr/bin/env python3
"""
Create cross-validation data splits for Relative Clause fine-tuning.

This script creates 24-fold leave-one-out cross-validation splits where:
- Each fold uses 23 item pairs for training
- 1 item pair (RC_Subj + RC_Obj) for testing

Usage:
    python cross_validation_splitter.py [--num-folds 24] [--output-dir DIR]
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
    # Clean word by removing punctuation
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
    # Clean word by replacing %2C with comma for accurate length
    clean_word = word.replace("%2C", ",")

    return {
        "length": len(clean_word),
        "length_prev1": len(prev_words[0].replace("%2C", ",")) if len(prev_words) > 0 else np.nan,
        "length_prev2": len(prev_words[1].replace("%2C", ",")) if len(prev_words) > 1 else np.nan,
    }


def convert_to_training_format(
    df: pd.DataFrame, item_list: list[int], split_name: str
) -> pd.DataFrame:
    """
    Convert Relative Clause data to training format.

    Args:
        df: Input DataFrame with RC data
        item_list: List of item IDs to include
        split_name: Name of the split (train/test)

    Returns:
        DataFrame in training format
    """
    # Filter to specified items
    filtered_df = df[df["item"].isin(item_list)].copy()
    logger.info(f"Creating {split_name} data from {len(filtered_df)} word observations")

    training_data = []

    # Start global sentence ID
    global_sentence_id = 1

    # Group by item and sentence type to process each sentence separately
    for (item_id, sentence_type), group in filtered_df.groupby(["item", "Type"]):
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

            # Create training format row
            train_row = {
                "sentence_num": global_sentence_id,
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
                "item": int(item_id),
                "nItem": len(item_list),
                "roi": row["ROI"] if not pd.isna(row["ROI"]) else np.nan,  # Preserve NA values
                # Metadata columns
                "Type": row["Type"],  # RC_Subj or RC_Obj
                "construction": row["CONSTRUCTION"],  # RelativeClause
                "ambiguity": row["AMBIG"],  # Amb or Unamb
            }

            training_data.append(train_row)

        global_sentence_id += 1

    # Convert to DataFrame
    train_df = pd.DataFrame(training_data)

    # Handle empty data case
    if len(train_df) == 0:
        logger.warning(f"No data found for {split_name}")
        empty_df = pd.DataFrame(columns=[
            "index", "sentence_num", "word", "reading_time", "reading_time_sd",
            "length", "freq", "position", "freq_prev1", "freq_prev2",
            "length_prev1", "length_prev2", "item", "nItem", "roi",
            "Type", "construction", "ambiguity"
        ])
        return empty_df

    # Add index column as the first column
    train_df.insert(0, 'index', range(len(train_df)))

    logger.info(
        f"Converted to {len(train_df)} words across {train_df['sentence_num'].nunique()} sentences"
    )
    return train_df


class CrossValidationSplitter:
    """Wrapper class for cross-validation splitting functionality."""

    def create_folds(self, num_folds: int = 24, output_dir: str = None):
        """
        Create 24-fold leave-one-out cross-validation splits.

        Each fold:
        - Train: 23 item pairs (46 sentences: 23 RC_Subj + 23 RC_Obj)
        - Test: 1 item pair (2 sentences: 1 RC_Subj + 1 RC_Obj)
        """
        logger.info(f"Creating {num_folds} leave-one-out folds for Relative Clause data")

        if output_dir is None:
            output_dir = Path("src/relative_clause_cross_validation/folds")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Load subject-averaged data
        subject_data_path = Path(
            "src/relative_clause_cross_validation/subject_averages/word_averaged_reading_times.csv"
        )

        if not subject_data_path.exists():
            logger.error(f"Subject-averaged data not found: {subject_data_path}")
            logger.info("Please run 'process-human-data' command first")
            return

        logger.info(f"Loading subject-averaged data from {subject_data_path}")
        subject_data = pd.read_csv(subject_data_path)

        # Get all items (should be 25-48 for RelativeClauseSet)
        all_items = sorted(subject_data["item"].unique())
        logger.info(f"Found {len(all_items)} unique items: {all_items}")

        if len(all_items) != 24:
            logger.warning(f"Expected 24 items, found {len(all_items)}")

        # Create folds metadata
        folds_metadata = {
            "num_folds": len(all_items),
            "total_items": len(all_items),
            "item_range": f"{min(all_items)}-{max(all_items)}",
            "folds": []
        }

        # Create each fold
        for fold_idx in range(len(all_items)):
            test_item = all_items[fold_idx]
            train_items = [item for item in all_items if item != test_item]

            logger.info(f"\nCreating fold {fold_idx}:")
            logger.info(f"  Test item: {test_item}")
            logger.info(f"  Train items: {len(train_items)} items")

            # Create fold directory
            fold_dir = output_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(exist_ok=True)

            # Convert to training format
            train_df = convert_to_training_format(subject_data, train_items, "train")
            test_df = convert_to_training_format(subject_data, [test_item], "test")

            # Save to CSV
            train_path = fold_dir / "rc_train.csv"
            test_path = fold_dir / "rc_test.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info(f"  Saved train data: {len(train_df)} words, {train_df['sentence_num'].nunique()} sentences")
            logger.info(f"  Saved test data: {len(test_df)} words, {test_df['sentence_num'].nunique()} sentences")

            # Record fold metadata
            fold_metadata = {
                "fold_id": fold_idx,
                "test_item": int(test_item),
                "train_items": [int(i) for i in train_items],
                "train_sentences": int(train_df["sentence_num"].nunique()),
                "test_sentences": int(test_df["sentence_num"].nunique()),
                "train_words": len(train_df),
                "test_words": len(test_df),
            }
            folds_metadata["folds"].append(fold_metadata)

        # Save overall metadata
        metadata_path = output_dir / "folds_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(folds_metadata, f, indent=2)

        logger.info(f"\nAll {len(all_items)} folds created successfully")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create cross-validation folds for Relative Clause data"
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=24,
        help="Number of folds to create (default: 24)",
    )
    parser.add_argument(
        "--output-dir",
        default="src/relative_clause_cross_validation/folds",
        help="Output directory for folds",
    )

    args = parser.parse_args()

    splitter = CrossValidationSplitter()
    splitter.create_folds(num_folds=args.num_folds, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
