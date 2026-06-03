"""
UCL eyetracking corpus preprocessing for reverse-engineering methodology.

This module processes the UCL eyetracking data to create
datasets compatible with the reverse-engineering-the-reader pipeline.
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from wordfreq import word_frequency

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UCLProcessor:
    """
    Processes UCL eyetracking corpus for psycholinguistic modeling.

    Implements the preprocessing pipeline to convert UCL eyetracking data
    into the format required for reverse-engineering methodology,
    including baseline features and spillover effects.
    """

    def __init__(self, ucl_data_path: str = "data/ucl"):
        """
        Initialize the processor with path to UCL data.

        Args:
            ucl_data_path: Path to UCL data directory
        """
        self.ucl_data_path = Path(ucl_data_path)
        self.eyetracking_file = self.ucl_data_path / "eyetracking.RT.txt"
        self.selfpaced_file = self.ucl_data_path / "selfpacedreading.RT.txt"

        # Validate file existence
        for file_path in [self.eyetracking_file]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

        logger.info(f"Initialized UCL processor with data from {ucl_data_path}")

    def load_eyetracking_data(self, use_measure: str = "RTfirstpass") -> pd.DataFrame:
        """
        Load UCL eyetracking data and aggregate by sentence and word position.

        Args:
            use_measure: Which RT measure to use (RTfirstfix, RTfirstpass, RTrightbound, RTgopast)

        Returns:
            DataFrame with columns: sent_nr, word_pos, word, reading_time, reading_time_sd
        """
        logger.info(f"Loading UCL eyetracking data using measure: {use_measure}...")

        # Load eyetracking data
        # Keep "None" as string, not as NaN
        df = pd.read_csv(self.eyetracking_file, sep="\t", keep_default_na=False, na_values=[''])

        # Group by sentence and word position to get mean and std
        aggregated = df.groupby(["sent_nr", "word_pos"], as_index=False, sort=False).agg({
            use_measure: ["mean", "std"],
            "word": "max"  # Use max to be consistent with reference code
        })

        # Flatten column names
        aggregated.columns = ["sent_nr", "word_pos", "reading_time", "reading_time_sd", "word"]

        # Strip whitespace from words
        aggregated["word"] = aggregated["word"].str.strip()

        # Reorder columns
        aggregated = aggregated[["sent_nr", "word_pos", "word", "reading_time", "reading_time_sd"]]

        # Sort by sentence and word position
        aggregated = aggregated.sort_values(["sent_nr", "word_pos"]).reset_index(drop=True)

        logger.info(
            f"Loaded {len(aggregated)} words from {aggregated['sent_nr'].nunique()} sentences"
        )

        return aggregated

    def load_selfpaced_data(self) -> pd.DataFrame:
        """
        Load UCL self-paced reading data and aggregate by sentence and word position.

        Returns:
            DataFrame with columns: sent_nr, word_pos, word, reading_time, reading_time_sd
        """
        logger.info("Loading UCL self-paced reading data...")

        # Load self-paced reading data
        # Keep "None" as string, not as NaN
        df = pd.read_csv(self.selfpaced_file, sep="\t", keep_default_na=False, na_values=[''])

        # Group by sentence and word position to get mean and std
        aggregated = df.groupby(["sent_nr", "word_pos"], as_index=False, sort=False).agg({
            "RT": ["mean", "std"],
            "word": "max"  # Use max to be consistent with reference code
        })

        # Flatten column names
        aggregated.columns = ["sent_nr", "word_pos", "reading_time", "reading_time_sd", "word"]

        # Strip whitespace from words
        aggregated["word"] = aggregated["word"].str.strip()

        # Reorder columns
        aggregated = aggregated[["sent_nr", "word_pos", "word", "reading_time", "reading_time_sd"]]

        # Sort by sentence and word position
        aggregated = aggregated.sort_values(["sent_nr", "word_pos"]).reset_index(drop=True)

        logger.info(
            f"Loaded {len(aggregated)} words from {aggregated['sent_nr'].nunique()} sentences"
        )

        return aggregated

    def add_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add baseline regression features: frequency, length, position.

        Args:
            df: DataFrame with UCL data

        Returns:
            DataFrame with added features: freq, length, position
        """
        logger.info("Adding baseline features...")

        df = df.copy()

        # Add word length (character count)
        df["length"] = df["word"].str.len()

        # Add log word frequency using wordfreq library
        logger.info("Computing word frequencies...")

        def _compute_log_frequency(word):
            # Now "None" should be preserved as a string
            frequency = word_frequency(str(word), "en", wordlist="best")
            return -math.log2(frequency if frequency > 0 else 1e-10)

        df["freq"] = df["word"].apply(_compute_log_frequency)

        # Add position (word_pos is already position within sentence)
        df["position"] = df["word_pos"]

        logger.info("Added baseline features: length, freq, position")

        return df

    def add_spillover_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add spillover features (lag-1 and lag-2) at sentence level.

        Spillover features are calculated within sentences only:
        - First word of each sentence: prev1 and prev2 are NaN
        - Second word of each sentence: prev2 is NaN
        - Other words: have both prev1 and prev2 from within the same sentence

        Args:
            df: DataFrame with baseline features

        Returns:
            DataFrame with added spillover features
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
        for sent_nr in df["sent_nr"].unique():
            sentence_mask = df["sent_nr"] == sent_nr
            sentence_data = df[sentence_mask].copy()

            # Sort by word position within sentence
            sentence_data = sentence_data.sort_values("word_pos")
            sentence_indices = sentence_data.index

            for feature in spillover_features:
                sentence_feature_values = sentence_data[feature].values

                # Create lag-1 (previous word) features
                for i in range(1, len(sentence_feature_values)):
                    current_idx = sentence_indices[i]
                    prev_value = sentence_feature_values[i - 1]
                    df.loc[current_idx, f"{feature}_prev1"] = prev_value

                # Create lag-2 (two words back) features
                for i in range(2, len(sentence_feature_values)):
                    current_idx = sentence_indices[i]
                    prev2_value = sentence_feature_values[i - 2]
                    df.loc[current_idx, f"{feature}_prev2"] = prev2_value

        # Count NaN values for validation
        nan_counts = {}
        for feature in spillover_features:
            nan_counts[f"{feature}_prev1"] = df[f"{feature}_prev1"].isna().sum()
            nan_counts[f"{feature}_prev2"] = df[f"{feature}_prev2"].isna().sum()

        logger.info(f"Added spillover features with NaN counts: {nan_counts}")

        return df

    def create_sentence_numbering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create unique sentence numbering for the entire dataset.

        Args:
            df: DataFrame with sentence-level data

        Returns:
            DataFrame with unique sentence_num column
        """
        logger.info("Creating unique sentence numbering...")

        df = df.copy()

        # Create unique sentence IDs
        unique_sentences = df["sent_nr"].unique()
        sent_mapping = {sent: idx for idx, sent in enumerate(unique_sentences)}
        df["sentence_num"] = df["sent_nr"].map(sent_mapping)

        logger.info(f"Created {len(unique_sentences)} unique sentence IDs")

        return df

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

        # For compatibility, add dummy columns if not present
        if "item" not in df_formatted.columns:
            # Use sent_nr as item (story/stimulus identifier)
            df_formatted["item"] = df_formatted["sent_nr"]

        if "zone" not in df_formatted.columns:
            # Use word_pos as zone (position within item)
            df_formatted["zone"] = df_formatted["word_pos"]

        if "nItem" not in df_formatted.columns:
            # Count unique items
            df_formatted["nItem"] = df_formatted["item"].nunique()

        # Select final columns matching Natural Stories format
        final_columns = [
            "index",
            "sentence_num",
            "word",
            "reading_time",
            "reading_time_sd",
            "length",
            "freq",
            "position",
            "freq_prev1",
            "freq_prev2",
            "length_prev1",
            "length_prev2",
            "item",
            "zone",
            "nItem"
        ]

        df_final = df_formatted[final_columns].copy()

        logger.info(f"Final dataset shape: {df_final.shape}")

        return df_final

    def aggregate_sentences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate words into sentences for train/test splitting.

        Args:
            df: DataFrame with processed features

        Returns:
            DataFrame with aggregated sentences
        """
        logger.info("Aggregating sentences for train/test splitting...")

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
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split sentences into train and test sets.

        Args:
            aggregated_df: DataFrame with unique aggregated sentences
            test_size: Proportion of sentences for test set
            random_state: Random seed for reproducibility

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
            original_df["sentence_num"].isin(sentence_list["sentence_num"])
        ].reset_index(drop=True)

        logger.info(
            f"Selected {len(selected_data)} words from {len(sentence_list)} sentences"
        )

        return selected_data

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        output_dir: str | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split following reverse-engineering methodology.

        Args:
            df: Processed DataFrame
            output_dir: Optional directory to save split data
            test_size: Proportion for test set
            random_state: Random seed

        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Creating train/test split...")

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

        # Step 6: Save if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            train_file = output_path / "ucl_eyetracking_train.csv"
            test_file = output_path / "ucl_eyetracking_test.csv"

            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)

            logger.info(f"Saved train data to {train_file}")
            logger.info(f"Saved test data to {test_file}")

        logger.info(
            f"Final split - Train: {len(train_df)} words, Test: {len(test_df)} words"
        )

        return train_df, test_df

    def process_full_pipeline(
        self,
        output_path: str | None = None,
        use_eyetracking: bool = True,
        et_measure: str = "RTfirstpass",
        create_split: bool = False,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete preprocessing pipeline.

        Args:
            output_path: Optional path to save processed data
            use_eyetracking: If True, use eyetracking data; if False, use self-paced reading
            et_measure: Which eyetracking measure to use (if use_eyetracking=True)
            create_split: If True, create train/test split
            test_size: Proportion for test set (if create_split=True)
            random_state: Random seed (if create_split=True)

        Returns:
            If create_split=False: Fully processed DataFrame
            If create_split=True: Tuple of (train_df, test_df)
        """
        logger.info("Starting full preprocessing pipeline...")

        # Step 1: Load data
        if use_eyetracking:
            df = self.load_eyetracking_data(use_measure=et_measure)
        else:
            df = self.load_selfpaced_data()

        # Step 2: Add baseline features
        df = self.add_baseline_features(df)

        # Step 3: Create unique sentence numbering
        df = self.create_sentence_numbering(df)

        # Step 4: Add spillover features
        df = self.add_spillover_features(df)

        # Step 5: Format for reverse-engineering
        df_final = self.format_for_reverse_engineering(df)

        # Step 6: Create train/test split if requested
        if create_split:
            train_df, test_df = self.create_train_test_split(
                df_final,
                output_dir=Path(output_path).parent if output_path else None,
                test_size=test_size,
                random_state=random_state
            )

            logger.info("Preprocessing pipeline completed successfully!")
            return train_df, test_df
        else:
            # Save if output path provided
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                df_final.to_csv(output_file, index=False)
                logger.info(f"Saved processed data to {output_file}")

            logger.info("Preprocessing pipeline completed successfully!")
            return df_final


def main():
    """Example usage of the UCL processor."""

    # Initialize processor
    processor = UCLProcessor("data/ucl")

    # Option 1: Process full dataset without split
    processed_data = processor.process_full_pipeline(
        output_path="data/ucl_eyetracking_processed.csv",
        use_eyetracking=True,
        et_measure="RTfirstpass",
        create_split=False
    )

    # Display summary statistics
    print("\nDataset Summary:")
    print(f"Total words: {len(processed_data)}")
    print(f"Sentences: {processed_data['sentence_num'].nunique()}")
    print(f"Mean reading time: {processed_data['reading_time'].mean():.2f} ms")
    print(f"Reading time range: [{processed_data['reading_time'].min():.2f}, {processed_data['reading_time'].max():.2f}] ms")
    print("\nSpillover coverage:")
    print(f"  freq_prev1 non-null: {processed_data['freq_prev1'].notna().sum()} / {len(processed_data)}")
    print(f"  freq_prev2 non-null: {processed_data['freq_prev2'].notna().sum()} / {len(processed_data)}")

    # Option 2: Process with train/test split
    train_df, test_df = processor.process_full_pipeline(
        output_path="data/ucl_eyetracking_processed.csv",
        use_eyetracking=True,
        et_measure="RTfirstpass",
        create_split=True,
        test_size=0.2,
        random_state=42
    )

    print("\nTrain/Test Split Summary:")
    print(f"Train: {len(train_df)} words, {train_df['sentence_num'].nunique()} sentences")
    print(f"Test: {len(test_df)} words, {test_df['sentence_num'].nunique()} sentences")


if __name__ == "__main__":
    main()
