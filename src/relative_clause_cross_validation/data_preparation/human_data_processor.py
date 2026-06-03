#!/usr/bin/env python3
"""
Calculate subject-averaged reading times for Relative Clause sentences.

This script processes the RelativeClauseSet.csv file to calculate subject-averaged
reading times for each word position.

The script:
1. Loads the RelativeClauseSet.csv data
2. Filters data following exclusion criteria
3. Calculates subject-averaged reading times per word position
4. Handles NA values in ROI field appropriately

Usage:
    python human_data_processor.py [--input-path PATH] [--output-path PATH]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_rc_data(file_path: str) -> pd.DataFrame:
    """Load RelativeClauseSet.csv data."""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows of data")
    return df


def apply_exclusion_criteria(df: pd.DataFrame, min_rt: float = 100, max_rt: float = 3000) -> pd.DataFrame:
    """
    Apply exclusion criteria.

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


def calculate_word_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate subject-averaged reading times for each word position.

    Groups by:
    - item: sentence item number
    - Type: construction type (RC_Subj, RC_Obj)
    - WordPosition: position in sentence
    - EachWord: the actual word
    - ROI: region of interest relative to critical position (may contain NA)

    Returns averaged RT for each unique word position across subjects.
    """
    logger.info("Calculating subject-averaged reading times per word position")

    # Filter to only Relative Clause constructions
    rc_data = df[df["Type"].isin(["RC_Subj", "RC_Obj"])].copy()
    logger.info(f"Filtering to RC constructions: {len(rc_data)} rows")

    # Convert RT to numeric, handling any non-numeric values
    rc_data["RT"] = pd.to_numeric(rc_data["RT"], errors="coerce")

    # Remove rows with missing RT values
    rc_data = rc_data.dropna(subset=["RT"])
    logger.info(f"After removing missing RT values: {len(rc_data)} rows")

    # Convert ROI to numeric (will create NaN for "NA" strings)
    rc_data["ROI"] = pd.to_numeric(rc_data["ROI"], errors="coerce")

    # Log ROI NA statistics
    roi_na_count = rc_data["ROI"].isna().sum()
    logger.info(f"ROI NA values: {roi_na_count} rows ({roi_na_count / len(rc_data) * 100:.1f}%)")

    # Calculate averages grouped by sentence structure
    # Note: ROI may be NaN for some words - this is expected
    # IMPORTANT: dropna=False to preserve ROI=NaN rows
    word_averages = (
        rc_data.groupby(
            ["item", "Type", "WordPosition", "EachWord", "ROI", "CONSTRUCTION", "AMBIG"],
            dropna=False  # Preserve ROI=NaN rows (non-critical words)
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

    # Log ROI range (excluding NA)
    roi_min = word_averages["ROI"].min()
    roi_max = word_averages["ROI"].max()
    logger.info(f"ROI range: {roi_min} to {roi_max}")

    return word_averages


def calculate_roi_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for key ROI regions (critical word and spillover).

    This focuses on:
    - ROI 0: Critical word (disambiguation point)
    - ROI 1: Spillover region 1
    - ROI 2: Spillover region 2

    Note: ROI with NA values are excluded from this summary.
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
        "Sentence",
    ]

    roi_summary["RT_sem"] = roi_summary["RT_std"] / np.sqrt(roi_summary["RT_count"])

    # Sort for easier viewing
    roi_summary = roi_summary.sort_values(["item", "Type", "ROI"])

    logger.info(f"Calculated ROI summary for {len(roi_summary)} combinations")

    return roi_summary


def main():
    parser = argparse.ArgumentParser(
        description="Calculate subject-averaged reading times for Relative Clause data"
    )
    parser.add_argument(
        "--input-path",
        default="data/RelativeClauseSet.csv",
        help="Path to RelativeClauseSet.csv file",
    )
    parser.add_argument(
        "--output-dir",
        default="src/relative_clause_cross_validation/subject_averages/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--min-rt",
        type=float,
        default=100,
        help="Minimum acceptable RT in ms (default: 100)",
    )
    parser.add_argument(
        "--max-rt",
        type=float,
        default=3000,
        help="Maximum acceptable RT in ms (default: 3000)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    rc_data = load_rc_data(args.input_path)

    # Apply exclusion criteria
    filtered_data = apply_exclusion_criteria(rc_data, min_rt=args.min_rt, max_rt=args.max_rt)

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
    print(f"Type categories: {sorted(word_averages['Type'].unique())}")
    print(
        f"Word positions covered: {word_averages['WordPosition'].min()}-{word_averages['WordPosition'].max()}"
    )
    print(
        f"Average RT range: {word_averages['RT_mean'].min():.0f}-{word_averages['RT_mean'].max():.0f}ms"
    )

    # ROI statistics (excluding NA)
    roi_values = word_averages[~word_averages["ROI"].isna()]["ROI"]
    if len(roi_values) > 0:
        print(f"\nROI range: {roi_values.min():.0f} to {roi_values.max():.0f}")
        print(f"ROI NA count: {word_averages['ROI'].isna().sum()} positions")

    if len(roi_summary) > 0:
        print(f"\nCritical ROI Regions (0, 1, 2):")
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
        output_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Processing human data from: {input_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"RT filtering range: {min_rt}-{max_rt}ms")

        # Load and process data
        df = load_rc_data(str(input_file))
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
