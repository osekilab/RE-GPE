#!/usr/bin/env python3
"""
Process filler sentences data for regression model fitting.

This module processes Fillers.csv to prepare data for fitting a regression model
that will be used to predict reading times for garden-path sentences.

The filler sentences are used because:
1. They provide natural reading time patterns without garden-path effects
2. They have sufficient data for robust regression fitting
3. They can be used to compute surprisal values without gradient computation
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from wordfreq import word_frequency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FillersProcessor:
    """Process filler sentences for regression model fitting."""

    def __init__(self):
        """Initialize the processor."""
        self.filler_data = None
        self.processed_data = None

    def load_fillers_data(self, file_path: str) -> pd.DataFrame:
        """
        Load Fillers.csv data.

        Args:
            file_path: Path to Fillers.csv file

        Returns:
            DataFrame with filler sentences data
        """
        logger.info(f"Loading filler data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows of filler data")

        # Store for later use
        self.filler_data = df
        return df

    def apply_exclusion_criteria(self, df: pd.DataFrame, min_rt: float = 100, max_rt: float = 3000) -> pd.DataFrame:
        """
        Apply exclusion criteria similar to garden-path data processing.

        Excludes:
        - Participants with low accuracy (AccLow == "yes")
        - Non-English speaking participants (nonEng == "yes")
        - Missing RT values
        - Individual RTs outside the range [min_rt, max_rt] ms

        Args:
            df: Input DataFrame
            min_rt: Minimum acceptable RT in ms (default: 100)
            max_rt: Maximum acceptable RT in ms (default: 3000)

        Returns:
            Filtered DataFrame
        """
        initial_count = len(df)

        # Step 1: Apply participant-level exclusion criteria
        df_filtered = df[
            (df["AccLow"] == "no") &
            (df["nonEng"] == "no")
        ].copy()

        participant_excluded = initial_count - len(df_filtered)
        logger.info(
            f"Participant-level exclusions: {participant_excluded} rows ({participant_excluded / initial_count * 100:.1f}%)"
        )

        # Step 2: Convert RT to numeric and apply RT-based filtering
        df_filtered["RT"] = pd.to_numeric(df_filtered["RT"], errors="coerce")

        # Count RTs outside acceptable range before filtering
        too_fast = (df_filtered["RT"] < min_rt).sum()
        too_slow = (df_filtered["RT"] > max_rt).sum()
        missing_rt = df_filtered["RT"].isna().sum()

        # Apply RT filtering
        pre_rt_filter = len(df_filtered)
        df_filtered = df_filtered[
            (df_filtered["RT"] >= min_rt) &
            (df_filtered["RT"] <= max_rt)
        ].copy()

        rt_excluded = pre_rt_filter - len(df_filtered)

        logger.info(
            f"RT-based exclusions: {rt_excluded} rows ({rt_excluded / pre_rt_filter * 100:.1f}%)"
        )
        logger.info(f"  - Too fast (<{min_rt}ms): {too_fast}")
        logger.info(f"  - Too slow (>{max_rt}ms): {too_slow}")
        logger.info(f"  - Missing RT: {missing_rt}")

        total_excluded = initial_count - len(df_filtered)
        logger.info(
            f"Total excluded: {total_excluded} rows ({total_excluded / initial_count * 100:.1f}%)"
        )
        logger.info(f"Remaining {len(df_filtered)} rows for analysis")

        return df_filtered


    def calculate_subject_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate subject-averaged reading times for each word position.

        Args:
            df: Filtered filler data

        Returns:
            DataFrame with averaged reading times per word position
        """
        logger.info("Calculating subject-averaged reading times...")

        # Group by sentence structure elements
        averaged_df = df.groupby(
            ["item", "Type", "WordPosition", "EachWord", "Sentence"]
        ).agg({
            "RT": ["mean", "std", "count"]
        }).reset_index()

        # Flatten column names
        averaged_df.columns = [
            "item", "type", "word_position", "word", "sentence",
            "reading_time", "reading_time_sd", "n_subjects"
        ]

        logger.info(f"Calculated averages for {len(averaged_df)} unique word positions")
        logger.info(f"Average subjects per word: {averaged_df['n_subjects'].mean():.1f}")

        return averaged_df

    def add_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add baseline features to subject-averaged data.

        Args:
            df: DataFrame with subject-averaged reading times

        Returns:
            DataFrame with added baseline features
        """
        logger.info("Adding baseline features...")

        df = df.copy()

        # Add word length
        df["length"] = df["word"].apply(lambda w: len(w.replace("%2C", ",")))

        # Add word frequency (log-transformed)
        def compute_log_frequency(word):
            clean_word = word.replace("%2C", "").replace(".", "")
            frequency = word_frequency(clean_word, "en", wordlist="best")
            return -np.log2(frequency if frequency > 0 else 1e-10)

        df["freq"] = df["word"].apply(compute_log_frequency)

        logger.info("Added baseline features: length, freq")

        return df

    def add_spillover_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add spillover features within each sentence.

        Args:
            df: DataFrame with baseline features

        Returns:
            DataFrame with added spillover features
        """
        logger.info("Adding spillover features...")

        df = df.copy()

        # Initialize spillover columns
        spillover_features = ["freq", "length"]
        for feature in spillover_features:
            df[f"{feature}_prev1"] = np.nan
            df[f"{feature}_prev2"] = np.nan

        # Process each sentence separately
        for (item_id, filler_type), group in df.groupby(["item", "type"]):
            # Sort by word position to ensure correct order
            group = group.sort_values("word_position")
            group_indices = group.index

            for feature in spillover_features:
                feature_values = group[feature].values

                # Add lag-1 features (skip first word)
                for i in range(1, len(feature_values)):
                    current_idx = group_indices[i]
                    df.loc[current_idx, f"{feature}_prev1"] = feature_values[i - 1]

                # Add lag-2 features (skip first two words)
                for i in range(2, len(feature_values)):
                    current_idx = group_indices[i]
                    df.loc[current_idx, f"{feature}_prev2"] = feature_values[i - 2]

        # Log NaN counts for validation
        nan_counts = {}
        for feature in spillover_features:
            nan_counts[f"{feature}_prev1"] = df[f"{feature}_prev1"].isna().sum()
            nan_counts[f"{feature}_prev2"] = df[f"{feature}_prev2"].isna().sum()

        logger.info(f"Added spillover features with NaN counts: {nan_counts}")

        return df

    def process_fillers_for_regression(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process filler sentences to create regression-ready dataset.

        Args:
            df: Filtered filler data

        Returns:
            DataFrame with features and reading times for regression
        """
        logger.info("Processing filler sentences for regression model")

        # Step 1: Calculate subject averages
        averaged_df = self.calculate_subject_averages(df)

        # Step 2: Add baseline features
        averaged_df = self.add_baseline_features(averaged_df)

        # Step 3: Add spillover features
        averaged_df = self.add_spillover_features(averaged_df)

        # Reorder columns for clarity
        column_order = [
            "item", "type", "word_position", "word",
            "reading_time", "reading_time_sd", "n_subjects",
            "freq", "length", "freq_prev1", "freq_prev2",
            "length_prev1", "length_prev2", "sentence"
        ]
        averaged_df = averaged_df[column_order]

        self.processed_data = averaged_df
        return averaged_df

    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """
        Save processed filler data.

        Args:
            df: Processed DataFrame
            output_path: Path to save the data
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed filler data to {output_file}")

        # Print summary statistics
        print("\n=== Filler Data Summary ===")
        print(f"Total words: {len(df)}")
        print(f"Unique items: {df['item'].nunique()}")
        print(f"Average RT: {df['reading_time'].mean():.0f}ms")
        print(f"RT range: {df['reading_time'].min():.0f}-{df['reading_time'].max():.0f}ms")
        print(f"Average word length: {df['length'].mean():.1f}")
        print(f"Average word frequency: {df['freq'].mean():.2f}")

    def process(self, input_path: str, output_path: Optional[str] = None,
                min_rt: float = 100, max_rt: float = 3000) -> pd.DataFrame:
        """
        Main processing pipeline for filler data.

        Args:
            input_path: Path to Fillers.csv
            output_path: Optional path to save processed data
            min_rt: Minimum acceptable RT in ms (default: 100)
            max_rt: Maximum acceptable RT in ms (default: 3000)

        Returns:
            Processed DataFrame ready for regression fitting
        """
        logger.info(f"Processing fillers with RT filtering range: {min_rt}-{max_rt}ms")

        # Load data
        df = self.load_fillers_data(input_path)

        # Apply exclusions with RT filtering
        df_clean = self.apply_exclusion_criteria(df, min_rt=min_rt, max_rt=max_rt)

        # Process for regression
        df_processed = self.process_fillers_for_regression(df_clean)

        # Save if output path provided
        if output_path:
            self.save_processed_data(df_processed, output_path)

        return df_processed


def main():
    """CLI interface for filler data processing."""
    parser = argparse.ArgumentParser(
        description="Process filler sentences for regression model fitting"
    )
    parser.add_argument(
        "--input-path",
        default="data/Fillers.csv",
        help="Path to Fillers.csv file (default: data/Fillers.csv)"
    )
    parser.add_argument(
        "--output-path",
        default="src/garden_path_cross_validation/folds/fillers_processed.csv",
        help="Path to save processed data (default: src/garden_path_cross_validation/folds/fillers_processed.csv)"
    )

    args = parser.parse_args()

    # Create processor and run
    processor = FillersProcessor()
    processed_data = processor.process(args.input_path, args.output_path)

    print(f"\nProcessing complete. Data saved to: {args.output_path}")


if __name__ == "__main__":
    main()
