#!/usr/bin/env python3
"""
Calculate subject-averaged reading times for each word in Garden-Path sentences.

This script processes the ClassicGardenPathSet.csv file to calculate subject-averaged
reading times for each word position, following the methodology from SAP_preprocessing.R.

The script:
1. Loads the ClassicGardenPathSet.csv data
2. Filters data following SAP exclusion criteria
3. Calculates subject-averaged reading times per word position
4. Outputs results aligned with items_ClassicGP.csv structure

Usage:
    python calculate_subject_averages.py [--input-path PATH] [--output-path PATH]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_classic_gp_data(file_path: str) -> pd.DataFrame:
    """Load ClassicGardenPathSet.csv data."""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows of data")
    return df


def apply_exclusion_criteria(df: pd.DataFrame, min_rt: float = 100, max_rt: float = 3000) -> pd.DataFrame:
    """
    Apply SAP exclusion criteria following SAP_preprocessing.R methodology,
    plus standard psycholinguistic RT filtering.

    Excludes:
    - Participants with low accuracy (AccLow == "yes")
    - Non-English speaking participants (nonEng == "yes")
    - Consecutive construction trials (consec == "yes")
    - Individual RTs outside the range [min_rt, max_rt] ms

    Args:
        df: Input dataframe
        min_rt: Minimum acceptable RT in ms (default: 100)
        max_rt: Maximum acceptable RT in ms (default: 3000)

    Returns:
        Filtered dataframe
    """
    initial_count = len(df)

    # Step 1: Apply participant-level exclusion criteria
    df_filtered = df[
        (df["AccLow"] == "no") & (df["nonEng"] == "no") & (df["consec"] == "no")
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


def detect_duplicate_word_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect items with duplicate word positions at the AGGREGATED level.

    After aggregating by word position, check if the same position appears
    with different words (which would indicate a true data problem).

    Returns:
        DataFrame with columns: CONSTRUCTION, item, AMBIG, has_duplicates
    """
    logger.info("Checking for duplicate word positions in aggregated data...")

    duplicate_items = []

    # First aggregate the data to get unique word positions
    # Group by construction, item, ambiguity, and word position
    aggregated = df.groupby(
        ["CONSTRUCTION", "item", "AMBIG", "Type", "WordPosition", "EachWord"]
    ).agg({"RT": "mean"}).reset_index()

    # Now check for duplicate positions with different words
    for (construction, item, ambig), group in aggregated.groupby(["CONSTRUCTION", "item", "AMBIG"]):
        # Check for duplicate word positions
        position_word_counts = group.groupby("WordPosition")["EachWord"].nunique()

        # If a position has more than one unique word, it's a problem
        problem_positions = position_word_counts[position_word_counts > 1]

        if len(problem_positions) > 0:
            # Get the actual duplicate words at each position
            duplicate_details = []
            for pos in problem_positions.index:
                words_at_pos = group[group["WordPosition"] == pos]["EachWord"].unique()
                duplicate_details.append(f"pos {pos}: {', '.join(words_at_pos)}")

            duplicate_items.append({
                "CONSTRUCTION": construction,
                "item": item,
                "AMBIG": ambig,
                "duplicate_positions": problem_positions.index.tolist(),
                "duplicate_details": "; ".join(duplicate_details)
            })

    if duplicate_items:
        dup_df = pd.DataFrame(duplicate_items)
        logger.warning(f"Found {len(dup_df)} items with TRUE duplicate word positions:")
        for _, row in dup_df.iterrows():
            logger.warning(f"  {row['CONSTRUCTION']} item {row['item']} ({row['AMBIG']}): {row['duplicate_details']}")

    return pd.DataFrame(duplicate_items) if duplicate_items else pd.DataFrame()


def calculate_word_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate subject-averaged reading times for each word position.

    Groups by:
    - item: sentence item number
    - Type: construction type (NPS_AMB, NPS_UAMB, NPZ_AMB, etc.)
    - WordPosition: position in sentence
    - EachWord: the actual word
    - ROI: region of interest relative to critical position

    Returns averaged RT for each unique word position across subjects.
    """
    logger.info("Calculating subject-averaged reading times per word position")

    # Filter to only Classic Garden-Path constructions
    classic_gp = df[df["CONSTRUCTION"].isin(["NPS", "NPZ", "MVRR"])].copy()
    logger.info(f"Filtering to Classic GP constructions: {len(classic_gp)} rows")

    # Convert RT to numeric, handling any non-numeric values
    classic_gp["RT"] = pd.to_numeric(classic_gp["RT"], errors="coerce")

    # Remove rows with missing RT values
    classic_gp = classic_gp.dropna(subset=["RT"])
    logger.info(f"After removing missing RT values: {len(classic_gp)} rows")

    # Detect and exclude items with duplicate word positions
    duplicates_df = detect_duplicate_word_positions(classic_gp)

    if not duplicates_df.empty:
        # Get unique items to exclude (both Amb and Unamb conditions of the same item)
        items_to_exclude = set()
        for _, row in duplicates_df.iterrows():
            items_to_exclude.add((row["CONSTRUCTION"], row["item"]))

        # Exclude problematic items
        initial_count = len(classic_gp)
        for construction, item in items_to_exclude:
            classic_gp = classic_gp[
                ~((classic_gp["CONSTRUCTION"] == construction) &
                  (classic_gp["item"] == item))
            ]

        excluded_count = initial_count - len(classic_gp)
        logger.info(f"Excluded {excluded_count} rows from {len(items_to_exclude)} items with duplicate positions")
        logger.info(f"Excluded items: {sorted(items_to_exclude)}")

    # Calculate averages grouped by sentence structure
    word_averages = (
        classic_gp.groupby(
            ["item", "Type", "WordPosition", "EachWord", "ROI", "CONSTRUCTION", "AMBIG"]
        )
        .agg(
            {
                "RT": ["mean", "std", "count"],
                "Sentence": "first",  # Keep sentence for reference
                "CriticalPosition": "first",  # Keep critical position
            }
        )
        .reset_index()
    )

    # Flatten column names
    word_averages.columns = [
        "item",
        "Type",
        "WordPosition",
        "EachWord",
        "ROI",
        "CONSTRUCTION",
        "AMBIG",
        "RT_mean",
        "RT_std",
        "RT_count",
        "Sentence",
        "CriticalPosition",
    ]

    # Add additional useful columns
    word_averages["RT_sem"] = word_averages["RT_std"] / np.sqrt(
        word_averages["RT_count"]
    )

    # Sort by item, type, and word position for easier viewing
    word_averages = word_averages.sort_values(["item", "Type", "WordPosition"])

    logger.info(f"Calculated averages for {len(word_averages)} unique word positions")

    return word_averages


def calculate_roi_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for key ROI regions (critical word and spillover).

    This follows the SAP methodology of focusing on:
    - ROI 0: Critical word (disambiguation point)
    - ROI 1: Spillover region 1
    - ROI 2: Spillover region 2
    - RTacross3words: Average of ROI 0,1,2
    """
    logger.info("Calculating ROI-based summary statistics")

    # Filter to relevant ROI regions and ensure numeric ROI
    df["ROI"] = pd.to_numeric(df["ROI"], errors="coerce")
    roi_data = df[df["ROI"].isin([0, 1, 2])].copy()

    if len(roi_data) == 0:
        logger.warning("No data found for ROI regions 0, 1, 2")
        return pd.DataFrame()

    # Calculate summary by item, condition, and ROI
    roi_summary = (
        roi_data.groupby(["item", "Type", "ROI", "CONSTRUCTION", "AMBIG"])
        .agg(
            {
                "RT": ["mean", "std", "count"],
                "RTacross3words": "first",  # This is pre-calculated in the data
                "Sentence": "first",
            }
        )
        .reset_index()
    )

    # Flatten column names
    roi_summary.columns = [
        "item",
        "Type",
        "ROI",
        "CONSTRUCTION",
        "AMBIG",
        "RT_mean",
        "RT_std",
        "RT_count",
        "RTacross3words",
        "Sentence",
    ]

    roi_summary["RT_sem"] = roi_summary["RT_std"] / np.sqrt(roi_summary["RT_count"])

    # Sort for easier viewing
    roi_summary = roi_summary.sort_values(["item", "Type", "ROI"])

    logger.info(f"Calculated ROI summary for {len(roi_summary)} combinations")

    return roi_summary


def main():
    parser = argparse.ArgumentParser(
        description="Calculate subject-averaged reading times"
    )
    parser.add_argument(
        "--input-path",
        default="data/ClassicGardenPathSet.csv",
        help="Path to ClassicGardenPathSet.csv file",
    )
    parser.add_argument(
        "--output-dir",
        default="src/garden_path_cross_validation/subject_averages/",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    gp_data = load_classic_gp_data(args.input_path)

    # Apply exclusion criteria
    filtered_data = apply_exclusion_criteria(gp_data)

    # Calculate word-by-word averages
    word_averages = calculate_word_averages(filtered_data)

    # Calculate ROI summary
    roi_summary = calculate_roi_summary(filtered_data)

    # Save results
    word_output = output_dir / "word_averaged_reading_times.csv"
    roi_output = output_dir / "roi_summary_reading_times.csv"

    word_averages.to_csv(word_output, index=False)
    roi_summary.to_csv(roi_output, index=False)

    logger.info(f"Word averages saved to: {word_output}")
    logger.info(f"ROI summary saved to: {roi_output}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total items processed: {word_averages['item'].nunique()}")
    print(f"Construction types: {sorted(word_averages['CONSTRUCTION'].unique())}")
    print(f"Condition types: {sorted(word_averages['AMBIG'].unique())}")
    print(
        f"Word positions covered: {word_averages['WordPosition'].min()}-{word_averages['WordPosition'].max()}"
    )
    print(
        f"Average RT range: {word_averages['RT_mean'].min():.0f}-{word_averages['RT_mean'].max():.0f}ms"
    )

    if len(roi_summary) > 0:
        print(f"\nROI Regions covered: {sorted(roi_summary['ROI'].unique())}")
        print(
            f"ROI RT range: {roi_summary['RT_mean'].min():.0f}-{roi_summary['RT_mean'].max():.0f}ms"
        )

    print(f"\nResults saved to: {output_dir}")


class HumanDataProcessor:
    """Wrapper class for human data processing functionality."""

    def process_data(self, input_path: str, output_path: str = None, min_rt: float = 100, max_rt: float = 3000):
        """
        Process human reading time data.

        Args:
            input_path: Path to input CSV file
            output_path: Output directory path (optional)
            min_rt: Minimum acceptable RT in ms (default: 100)
            max_rt: Maximum acceptable RT in ms (default: 3000)
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            output_path = input_file.parent / "subject_averages"

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing human data from: {input_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"RT filtering range: {min_rt}-{max_rt}ms")

        # Load and process data
        df = load_classic_gp_data(str(input_file))
        logger.info(f"Loaded {len(df)} raw observations")

        # Apply exclusions with RT filtering
        df_clean = apply_exclusion_criteria(df, min_rt=min_rt, max_rt=max_rt)
        logger.info(f"After exclusions: {len(df_clean)} observations")

        # Calculate averages
        word_averages = calculate_word_averages(df_clean)
        roi_summary = calculate_roi_summary(df_clean)

        # Save results
        word_avg_path = output_dir / "word_averaged_reading_times.csv"
        roi_summary_path = output_dir / "roi_summary_reading_times.csv"

        word_averages.to_csv(word_avg_path, index=False)
        roi_summary.to_csv(roi_summary_path, index=False)

        logger.info(f"Word averages saved to: {word_avg_path}")
        logger.info(f"ROI summary saved to: {roi_summary_path}")

        return word_averages, roi_summary


if __name__ == "__main__":
    main()
