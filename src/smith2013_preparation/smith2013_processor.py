"""
Smith 2013 eyetracking corpus preprocessing for reverse-engineering methodology.

This module processes the Smith 2013 eyetracking data to create
datasets compatible with the reverse-engineering-the-reader pipeline.
"""

import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from wordfreq import word_frequency

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Smith2013Processor:
    """
    Processes Smith 2013 eyetracking corpus for psycholinguistic modeling.

    Implements the preprocessing pipeline to convert Smith 2013 eyetracking data
    into the format required for reverse-engineering methodology,
    including baseline features and spillover effects.
    """

    def __init__(self, data_pkl_path: str = "data/data.pkl"):
        """
        Initialize the processor with path to data.pkl file.

        Args:
            data_pkl_path: Path to data.pkl file containing brown corpus data
        """
        self.data_file = Path(data_pkl_path)

        # Validate file existence
        if not self.data_file.exists():
            raise FileNotFoundError(f"Required file not found: {self.data_file}")

        logger.info(f"Initialized Smith 2013 processor with data from {data_pkl_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load Brown corpus data from data.pkl file.

        Returns:
            DataFrame with words and reading times
        """
        logger.info("Loading Brown corpus data from data.pkl...")

        # Load pickle file
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)

        if 'brown' not in data:
            raise ValueError("No 'brown' key found in data.pkl")

        brown_data = data['brown']

        # Extract sentences and self-paced reading times
        sentences = brown_data['sent']
        fp_times = brown_data['fp']

        logger.info(f"Found {len(sentences)} sentences")
        logger.info(f"Found {len(fp_times)} first-pass reading times")

        # Build word-level DataFrame
        rows = []
        fp_idx = 0

        for sent_idx, sentence in enumerate(sentences):
            for word_idx, word in enumerate(sentence):
                if word_idx == 0:
                    # First word of sentence - no reading time available
                    reading_time = -1  # Use -1 as marker for missing first word RT
                else:
                    # Get reading time from fp array
                    reading_time = fp_times[fp_idx]
                    fp_idx += 1

                rows.append({
                    'sentid': sent_idx,
                    'sentpos': word_idx,
                    'word': word,
                    'reading_time': reading_time,
                    'reading_time_sd': 0  # No SD available in this dataset
                })

        # Verify we used all fp values
        if fp_idx != len(fp_times):
            logger.warning(f"FP index mismatch: used {fp_idx}, total {len(fp_times)}")

        df = pd.DataFrame(rows)

        # Sort by sentence and position
        df = df.sort_values(['sentid', 'sentpos']).reset_index(drop=True)

        logger.info(f"Loaded {len(df)} words from {df['sentid'].nunique()} sentences")
        logger.info(f"Words with reading time: {(df['reading_time'] > 0).sum()}")
        logger.info(f"Words without reading time (first words): {(df['reading_time'] == -1).sum()}")

        return df

    def add_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add baseline regression features: frequency, length, position.

        Args:
            df: DataFrame with Smith 2013 data

        Returns:
            DataFrame with added features: freq, length, position
        """
        logger.info("Adding baseline features...")

        df = df.copy()

        # Calculate word length
        df["length"] = df["word"].str.len()

        # Add log word frequency using wordfreq library
        logger.info("Computing word frequencies...")

        def _compute_log_frequency(word):
            frequency = word_frequency(str(word), "en", wordlist="best")
            return -math.log2(frequency if frequency > 0 else 1e-10)

        df["freq"] = df["word"].apply(_compute_log_frequency)

        # Use sentpos as position
        df["position"] = df["sentpos"] + 1  # Convert to 1-based index

        # No surprisal data available in Brown corpus
        # Skipping surprisal features

        logger.info("Added baseline features: length, freq, position")

        return df

    def add_spillover_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add spillover features (lag-1 and lag-2) at sentence level.

        Args:
            df: DataFrame with baseline features

        Returns:
            DataFrame with added spillover features
        """
        logger.info("Adding spillover features (sentence-level)...")

        df = df.copy()

        # Features to create spillover for
        spillover_features = ["freq", "length"]
        if "surprisal" in df.columns:
            spillover_features.append("surprisal")

        # Initialize spillover columns with NaN
        for feature in spillover_features:
            df[f"{feature}_prev1"] = np.nan
            df[f"{feature}_prev2"] = np.nan

        # Process each sentence separately (grouped by sentid)
        for sentid, group in df.groupby("sentid"):
            # Sort by position within sentence
            group = group.sort_values("sentpos")
            group_indices = group.index

            for feature in spillover_features:
                feature_values = group[feature].values

                # Create lag-1 (previous word) features
                for i in range(1, len(feature_values)):
                    current_idx = group_indices[i]
                    prev_value = feature_values[i - 1]
                    df.loc[current_idx, f"{feature}_prev1"] = prev_value

                # Create lag-2 (two words back) features
                for i in range(2, len(feature_values)):
                    current_idx = group_indices[i]
                    prev2_value = feature_values[i - 2]
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

        # Use sentid directly as sentence_num
        df["sentence_num"] = df["sentid"]

        logger.info(f"Created {df['sentence_num'].nunique()} unique sentence IDs")

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

        # For compatibility, map columns appropriately
        # Use sentid as item (sentence identifier)
        df_formatted["item"] = df_formatted["sentid"]

        # Use sentpos as zone (position within sentence)
        df_formatted["zone"] = df_formatted["sentpos"]

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
            "length_prev2"
        ]

        # Add surprisal columns if available
        if "surprisal" in df_formatted.columns:
            # Insert surprisal after freq
            idx = final_columns.index("position")
            final_columns.insert(idx, "surprisal")
            if "surprisal_prev1" in df_formatted.columns:
                final_columns.insert(idx + 1, "surprisal_prev1")
            if "surprisal_prev2" in df_formatted.columns:
                final_columns.insert(idx + 2, "surprisal_prev2")

        # Add metadata columns
        final_columns.extend(["item", "zone", "nItem"])

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

        if removed_count > 0:
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
        if removed_count > 0:
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

            train_file = output_path / "smith2013_train.csv"
            test_file = output_path / "smith2013_test.csv"

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
        create_split: bool = False,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete preprocessing pipeline.

        Args:
            output_path: Optional path to save processed data
            create_split: If True, create train/test split
            test_size: Proportion for test set (if create_split=True)
            random_state: Random seed (if create_split=True)

        Returns:
            If create_split=False: Fully processed DataFrame
            If create_split=True: Tuple of (train_df, test_df)
        """
        logger.info("Starting full preprocessing pipeline...")

        # Step 1: Load and aggregate data
        df = self.load_data()

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
    """Example usage of the Smith 2013 processor."""

    # Initialize processor with data.pkl
    processor = Smith2013Processor("data/data.pkl")

    # Option 1: Process full dataset without split
    processed_data = processor.process_full_pipeline(
        output_path="data/smith2013_processed.csv",
        create_split=False
    )

    # Display summary statistics
    print("\nDataset Summary:")
    print(f"Total words: {len(processed_data)}")
    print(f"Sentences: {processed_data['sentence_num'].nunique()}")
    print(f"Mean reading time: {processed_data['reading_time'].mean():.2f} ms")
    print(f"Reading time range: [{processed_data['reading_time'].min():.2f}, {processed_data['reading_time'].max():.2f}] ms")

    if "surprisal" in processed_data.columns:
        print(f"Mean surprisal: {processed_data['surprisal'].mean():.2f}")

    print("\nSpillover coverage:")
    print(f"  freq_prev1 non-null: {processed_data['freq_prev1'].notna().sum()} / {len(processed_data)}")
    print(f"  freq_prev2 non-null: {processed_data['freq_prev2'].notna().sum()} / {len(processed_data)}")

    if "surprisal_prev1" in processed_data.columns:
        print(f"  surprisal_prev1 non-null: {processed_data['surprisal_prev1'].notna().sum()} / {len(processed_data)}")

    # Option 2: Process with train/test split
    train_df, test_df = processor.process_full_pipeline(
        output_path="data/smith2013_processed.csv",
        create_split=True,
        test_size=0.2,
        random_state=42
    )

    print("\nTrain/Test Split Summary:")
    print(f"Train: {len(train_df)} words, {train_df['sentence_num'].nunique()} sentences")
    print(f"Test: {len(test_df)} words, {test_df['sentence_num'].nunique()} sentences")


if __name__ == "__main__":
    main()
