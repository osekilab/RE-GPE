"""
Natural Stories corpus preprocessing for reverse-engineering methodology.

This module processes the Natural Stories self-paced reading data to create
datasets compatible with the reverse-engineering-the-reader pipeline.
"""

import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from wordfreq import word_frequency

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NaturalStoriesProcessor:
    """
    Processes Natural Stories corpus for psycholinguistic modeling.

    Implements the preprocessing pipeline to convert Natural Stories self-paced
    reading data into the format required for reverse-engineering methodology,
    including baseline features and spillover effects.
    """

    def __init__(self, naturalstories_path: str):
        """
        Initialize the processor with path to Natural Stories data.

        Args:
            naturalstories_path: Path to external/naturalstories directory
        """
        self.naturalstories_path = Path(naturalstories_path)
        self.rts_file = (
            self.naturalstories_path / "naturalstories_RTS" / "processed_RTs.tsv"
        )
        self.wordinfo_file = (
            self.naturalstories_path / "naturalstories_RTS" / "processed_wordinfo.tsv"
        )
        self.words_file = self.naturalstories_path / "words.tsv"
        # Universal Dependencies parse, aligned to the RT zones, used to derive
        # per-word sentence position (replaces the previously used private file).
        self.parse_file = (
            self.naturalstories_path / "parses" / "ud" / "stories-aligned.conllx"
        )

        # Validate file existence
        for file_path in [
            self.rts_file,
            self.wordinfo_file,
            self.words_file,
            self.parse_file,
        ]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

        logger.info(f"Initialized processor with data from {naturalstories_path}")

    def load_and_sort_data(self) -> pd.DataFrame:
        """
        Load Natural Stories wordinfo data and sort by item -> zone.

        Returns:
            DataFrame with columns: word, zone, item, nItem, meanItemRT, etc.
            Sorted by item (story number) then zone (position within story)
        """
        logger.info("Loading Natural Stories wordinfo data...")

        # Load processed_wordinfo.tsv
        df = pd.read_csv(self.wordinfo_file, sep="\t")

        # Sort by item (story) then zone (position)
        df_sorted = df.sort_values(["item", "zone"]).reset_index(drop=True)

        logger.info(
            f"Loaded {len(df_sorted)} words from {df_sorted['item'].nunique()} stories"
        )

        return df_sorted

    def add_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add baseline regression features: frequency, length, word_position.

        Args:
            df: DataFrame with Natural Stories data

        Returns:
            DataFrame with added features: freq, length, word_position
        """
        logger.info("Adding baseline features...")

        df = df.copy()

        # Add word length (character count)
        df["length"] = df["word"].str.len()

        # Add log word frequency using wordfreq library (matching reverse-engineering-the-reader)
        logger.info("Computing word frequencies...")

        def _compute_log_frequency(word):
            frequency = word_frequency(word, "en", wordlist="best")
            return -math.log2(frequency if frequency > 0 else 1e-10)

        df["freq"] = df["word"].apply(_compute_log_frequency)

        # Note: word_position will be added later based on proper sentence structure

        logger.info("Added baseline features: length, freq")

        return df

    def add_spillover_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add spillover features (lag-1 and lag-2) at sentence level.

        Spillover features are calculated within sentences only:
        - First word of each sentence: prev1 and prev2 are NaN
        - Second word of each sentence: prev2 is NaN
        - Other words: have both prev1 and prev2 from within the same sentence

        Args:
            df: DataFrame with baseline features and sentence structure

        Returns:
            DataFrame with added spillover features: freq_prev1, freq_prev2,
            length_prev1, length_prev2
        """
        logger.info("Adding spillover features (sentence-level)...")

        df = df.copy()

        # Features to create spillover for
        spillover_features = ["freq", "length"]

        # Initialize spillover columns with NaN
        for feature in spillover_features:
            df[f"{feature}_prev1"] = np.nan
            df[f"{feature}_prev2"] = np.nan

        # Process each sentence separately to handle sentence boundaries
        for sentence_id in df["sentence_num"].unique():
            if pd.isna(sentence_id):
                continue

            sentence_mask = df["sentence_num"] == sentence_id
            sentence_data = df[sentence_mask].copy()

            # Sort by position within sentence to ensure correct order
            sentence_data = sentence_data.sort_values("position")
            sentence_indices = sentence_data.index

            for feature in spillover_features:
                sentence_feature_values = sentence_data[feature].values

                # Create lag-1 (previous word) features
                # Skip first word (position=1) - it should remain NaN
                for i in range(1, len(sentence_feature_values)):
                    current_idx = sentence_indices[i]
                    prev_value = sentence_feature_values[i - 1]
                    df.loc[current_idx, f"{feature}_prev1"] = prev_value

                # Create lag-2 (two words back) features
                # Skip first two words (positions 1 and 2) - they should remain NaN
                for i in range(2, len(sentence_feature_values)):
                    current_idx = sentence_indices[i]
                    prev2_value = sentence_feature_values[i - 2]
                    df.loc[current_idx, f"{feature}_prev2"] = prev2_value

        # Count NaN values for validation
        nan_counts = {}
        for feature in spillover_features:
            nan_counts[f"{feature}_prev1"] = df[f"{feature}_prev1"].isna().sum()
            nan_counts[f"{feature}_prev2"] = df[f"{feature}_prev2"].isna().sum()

        # Also count by position for verification
        position_1_count = (df["position"] == 1).sum()
        position_2_count = (df["position"] == 2).sum()

        logger.info(f"Added spillover features with NaN counts: {nan_counts}")
        logger.info(
            f"Expected NaN counts: prev1≥{position_1_count} (pos=1), prev2≥{position_1_count + position_2_count} (pos=1,2)"
        )

        return df

    def apply_exclusion_criteria(
        self, df: pd.DataFrame, apply_outlier_removal: bool = False
    ) -> pd.DataFrame:
        """
        No exclusion criteria applied - keep all data as-is.

        All words and reading times are retained without any filtering
        to maintain complete data integrity for analysis.

        Args:
            df: DataFrame with all features
            apply_outlier_removal: Ignored - no filtering is applied

        Returns:
            DataFrame unchanged
        """
        logger.info("No exclusion criteria applied - keeping all data")

        # Count statistics for logging only
        total_count = len(df)
        missing_count = df["meanItemRT"].isna().sum()
        zero_count = (df["meanItemRT"] == 0).sum() if "meanItemRT" in df.columns else 0

        logger.info(
            f"Data statistics: {total_count} words total, "
            f"{missing_count} missing reading times, "
            f"{zero_count} zero reading times"
        )

        logger.info(f"Final dataset: {total_count} words (all retained)")

        return df

    def load_position_data(self) -> pd.DataFrame:
        """
        Derive per-word sentence position from the public Natural Stories
        Universal Dependencies parse (``parses/ud/stories-aligned.conllx``).

        Each token line carries ``TokenId=item.zone(.subtoken)`` which maps it
        to the Natural Stories ``(item, zone)``; blank lines delimit sentences.
        Sub-tokens of one reading zone (e.g. a word and its trailing
        punctuation) are collapsed back to a single zone, and the distinct
        zones within a sentence are numbered ``1..N`` to give the within-sentence
        position. This reproduces the sentence id / position annotation used in
        the regression while relying only on the public corpus.

        Returns:
            DataFrame with columns ``word, position, sentence_num, item, zone``
        """
        logger.info(f"Loading position data from {self.parse_file}...")

        records = []
        sentence_num = 0
        with open(self.parse_file) as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip() == "":
                    # Blank line separates sentences.
                    sentence_num += 1
                    continue
                fields = line.split("\t")
                if len(fields) < 10:
                    continue
                match = re.search(r"TokenId=(\d+)\.(\d+)", fields[9])
                if not match:
                    continue
                item, zone = int(match.group(1)), int(match.group(2))
                records.append((sentence_num, item, zone, fields[1]))

        conll = pd.DataFrame(
            records, columns=["sentence_num", "item", "zone", "word"]
        )

        # Collapse sub-tokens to one row per (item, zone), keeping the sentence
        # in which the zone first appears.
        position_data = (
            conll.groupby(["item", "zone"], sort=False)
            .agg(sentence_num=("sentence_num", "min"), word=("word", "first"))
            .reset_index()
        )

        # Within-sentence position: rank distinct zones by reading order.
        position_data = position_data.sort_values(
            ["item", "sentence_num", "zone"]
        ).reset_index(drop=True)
        position_data["position"] = (
            position_data.groupby(["item", "sentence_num"]).cumcount() + 1
        )

        position_data = position_data[
            ["word", "position", "sentence_num", "item", "zone"]
        ]

        logger.info(f"Loaded position data for {len(position_data)} words")
        logger.info(f"Sentences: {position_data['sentence_num'].nunique()}")
        logger.info(
            f"Position range: {position_data['position'].min()}-{position_data['position'].max()}"
        )

        return position_data

    def add_sentence_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add proper sentence structure using position data.

        Args:
            df: DataFrame with baseline and spillover features

        Returns:
            DataFrame with sentence_num and position columns added
        """
        logger.info("Adding sentence structure...")

        # Load position data
        position_df = self.load_position_data()

        # Merge with main dataframe (drop duplicate word column from position_df)
        position_df_clean = position_df.drop(columns=["word"])
        df_with_structure = df.merge(position_df_clean, on=["item", "zone"], how="left")

        # Handle any missing position data
        missing_position = df_with_structure["position"].isna().sum()
        if missing_position > 0:
            logger.warning(
                f"Missing position data for {missing_position} words"
            )
            df_with_structure["position"] = df_with_structure["position"].fillna(
                0
            )
            df_with_structure["sentence_num"] = df_with_structure[
                "sentence_num"
            ].fillna(0)

        # Create unique sentence numbering across the entire dataset
        # Overwrite sentence_num with globally unique IDs
        df_with_structure["sentence_num"] = df_with_structure.groupby(
            ["item", "sentence_num"]
        ).ngroup()

        logger.info(
            f"Added sentence structure: {df_with_structure['sentence_num'].nunique()} sentences"
        )
        logger.info(
            f"Position range: {df_with_structure['position'].min()}-{df_with_structure['position'].max()}"
        )

        return df_with_structure

    def format_for_reverse_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format data to match reverse-engineering-the-reader expected format.

        Args:
            df: Processed DataFrame

        Returns:
            DataFrame formatted for reverse-engineering pipeline
        """
        logger.info("Formatting for reverse-engineering pipeline...")

        # Add index column as the first column
        df_formatted = df.copy()
        df_formatted["index"] = range(len(df_formatted))

        # Rename columns to match expected format
        df_formatted = df_formatted.rename(
            columns={
                "meanItemRT": "reading_time",  # Primary reading time measure
                "sdItemRT": "reading_time_sd",
            }
        )

        # Select final columns
        final_columns = [
            "index",
            "sentence_num",
            "word",
            "reading_time",
            "reading_time_sd",
            "length",
            "freq",
            "position",  # position instead of word_position
            "freq_prev1",
            "freq_prev2",
            "length_prev1",
            "length_prev2",
            "item",
            "zone",
            "nItem",  # Keep original identifiers
        ]

        df_final = df_formatted[final_columns].copy()

        logger.info(f"Final dataset shape: {df_final.shape}")

        return df_final

    def aggregate_sentences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate words into sentences for train/test splitting.

        Following reverse-engineering methodology: group by sentence and join words.

        Args:
            df: DataFrame with processed features

        Returns:
            DataFrame with aggregated sentences
        """
        logger.info("Aggregating sentences for train/test splitting...")

        # Use sentence_num for unique sentence identification
        aggregated = (
            df.groupby("sentence_num")["word"].apply(" ".join).reset_index()
        )

        logger.info(f"Aggregated {len(aggregated)} unique sentences")

        return aggregated

    def remove_duplicates_within_dataset(
        self, aggregated_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Remove duplicate sentences within the dataset.

        Args:
            aggregated_df: DataFrame with aggregated sentences

        Returns:
            DataFrame with duplicates removed
        """
        logger.info("Removing duplicate sentences within dataset...")

        initial_count = len(aggregated_df)
        unique_sentences = aggregated_df.drop_duplicates(subset="word").reset_index(
            drop=True
        )
        removed_count = initial_count - len(unique_sentences)

        logger.info(
            f"Removed {removed_count} duplicate sentences ({removed_count / initial_count:.1%})"
        )

        return unique_sentences

    def split_sentences(
        self,
        aggregated_df: pd.DataFrame,
        test_size: float = 0.4,
        random_state: int = 12321,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split sentences into train and test sets.

        Args:
            aggregated_df: DataFrame with unique aggregated sentences
            test_size: Proportion of sentences for test set (default: 0.4)
            random_state: Random seed for reproducibility (default: 12321)

        Returns:
            Tuple of (train_sentences, test_sentences)
        """
        logger.info(
            f"Splitting sentences into train/test ({1 - test_size:.0%}/{test_size:.0%})..."
        )

        unique_sentence_ids = aggregated_df["sentence_num"].unique()
        train_ids, test_ids = train_test_split(
            unique_sentence_ids, test_size=test_size, random_state=random_state
        )

        train_sentences = aggregated_df[
            aggregated_df["sentence_num"].isin(train_ids)
        ].reset_index(drop=True)

        test_sentences = aggregated_df[
            aggregated_df["sentence_num"].isin(test_ids)
        ].reset_index(drop=True)

        logger.info(f"Train sentences: {len(train_sentences)}")
        logger.info(f"Test sentences: {len(test_sentences)}")

        return train_sentences, test_sentences

    def remove_duplicates_across_datasets(
        self, train_sentences: pd.DataFrame, test_sentences: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove sentences that appear in both train and test sets.

        Following reverse-engineering methodology: remove from test set only.

        Args:
            train_sentences: Training sentences
            test_sentences: Test sentences

        Returns:
            Tuple of (cleaned_train, cleaned_test)
        """
        logger.info("Removing cross-dataset duplicates...")

        initial_test_count = len(test_sentences)
        test_cleaned = test_sentences[
            ~test_sentences["word"].isin(train_sentences["word"])
        ].reset_index(drop=True)

        removed_count = initial_test_count - len(test_cleaned)
        logger.info(
            f"Removed {removed_count} sentences from test set ({removed_count / initial_test_count:.1%})"
        )

        return train_sentences, test_cleaned

    def deaggregate_sentences(
        self, sentence_list: pd.DataFrame, original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert aggregated sentences back to word-level format.

        Args:
            sentence_list: DataFrame with selected sentence IDs
            original_df: Original word-level DataFrame

        Returns:
            Word-level DataFrame for selected sentences
        """
        logger.info("Converting back to word-level format...")

        selected_data = original_df[
            original_df["sentence_num"].isin(
                sentence_list["sentence_num"]
            )
        ].reset_index(drop=True)

        logger.info(
            f"Selected {len(selected_data)} words from {len(sentence_list)} sentences"
        )

        return selected_data

    def _check_overlap(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """
        Check if there are overlapping sentences between train and test sets.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame

        Returns:
            True if no overlap, False if overlap detected
        """
        overlap = pd.merge(train_df, test_df, on=["sentence_num", "word"])
        return len(overlap) == 0

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        output_dir: str | None = None,
        test_size: float = 0.4,
        random_state: int = 12321,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split following reverse-engineering methodology.

        Args:
            df: Processed DataFrame
            output_dir: Optional directory to save split data
            test_size: Proportion for test set (default: 0.4)
            random_state: Random seed (default: 12321)

        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(
            "Creating train/test split following reverse-engineering methodology..."
        )

        # Step 1: Aggregate sentences
        aggregated = self.aggregate_sentences(df)

        # Step 2: Remove duplicates within dataset
        unique_sentences = self.remove_duplicates_within_dataset(aggregated)

        # Step 3: Split sentences
        train_sentences, test_sentences = self.split_sentences(
            unique_sentences, test_size=test_size, random_state=random_state
        )

        # Step 4: Remove cross-dataset duplicates
        train_sentences, test_sentences = self.remove_duplicates_across_datasets(
            train_sentences, test_sentences
        )

        # Step 5: Convert back to word-level format
        train_df = self.deaggregate_sentences(train_sentences, df)
        test_df = self.deaggregate_sentences(test_sentences, df)

        # Step 6: Verify no overlap
        no_overlap = self._check_overlap(train_df, test_df)
        logger.info(
            f"Overlap check: {'No overlap' if no_overlap else 'Overlap detected'}"
        )

        if not no_overlap:
            logger.warning("Overlap detected between train and test sets!")

        # Step 7: Save if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            train_file = output_path / "natural_stories_train.csv"
            test_file = output_path / "natural_stories_test.csv"

            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)

            logger.info(f"Saved train data to {train_file}")
            logger.info(f"Saved test data to {test_file}")

        logger.info(
            f"Final split - Train: {len(train_df)} words, Test: {len(test_df)} words"
        )

        return train_df, test_df

    def process_full_pipeline(self, output_path: str | None = None) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.

        Args:
            output_path: Optional path to save processed data

        Returns:
            Fully processed DataFrame ready for reverse-engineering
        """
        logger.info("Starting full preprocessing pipeline...")

        # Step 1: Load and sort data
        df = self.load_and_sort_data()

        # Step 2: Add baseline features
        df = self.add_baseline_features(df)

        # Step 3: Apply exclusion criteria (minimal by default)
        df = self.apply_exclusion_criteria(df, apply_outlier_removal=False)

        # Step 4: Add proper sentence structure with position
        df = self.add_sentence_structure(df)

        # Step 5: Add spillover features (must be after sentence structure)
        df = self.add_spillover_features(df)

        # Step 6: Format for reverse-engineering
        df_final = self.format_for_reverse_engineering(df)

        # Save if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df_final.to_csv(output_file, index=False)
            logger.info(f"Saved processed data to {output_file}")

        logger.info("Preprocessing pipeline completed successfully!")

        return df_final


def main():
    """Example usage of the Natural Stories processor."""

    # Initialize processor
    processor = NaturalStoriesProcessor("external/naturalstories")

    # Run preprocessing pipeline
    processed_data = processor.process_full_pipeline(
        "data/natural_stories_processed.csv"
    )

    # Display summary statistics
    print("\nDataset Summary:")
    print(f"Total words: {len(processed_data)}")
    print(f"Stories: {processed_data['item'].nunique()}")
    print(f"Sentences: {processed_data['sentence_num'].nunique()}")
    print(f"Mean reading time: {processed_data['reading_time'].mean():.2f} ms")
    print("Sentence structure:")
    print(f"  Sentences: {processed_data['sentence_num'].nunique()}")
    print(
        f"  Position range: {processed_data['position'].min()}-{processed_data['position'].max()}"
    )
    print("Spillover coverage:")
    print(f"  freq_prev1 non-null: {processed_data['freq_prev1'].notna().sum()}")
    print(f"  freq_prev2 non-null: {processed_data['freq_prev2'].notna().sum()}")


if __name__ == "__main__":
    main()
