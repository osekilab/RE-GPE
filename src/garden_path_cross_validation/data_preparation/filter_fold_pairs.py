#!/usr/bin/env python3
"""
Filter training data to remove specific ambiguous/unambiguous pairs
when test critical words appear in training data at the SAME ROI position.

Cross-construction filtering with ROI-position matching:
- Checks ROI=0 words (disambiguation point) across all constructions
- Checks ROI=-3 words in Ambiguous sentences (garden-path trigger) across all constructions
- Only excludes if the word appears at the SAME ROI position
- Different ROI positions are allowed (e.g., test ROI=0 vs train ROI=-3 is OK)

Example:
- Test: MVRR ROI=0 "stopped" → Train: NPS ROI=0 "stopped" → EXCLUDE
- Test: MVRR ROI=0 "stopped" → Train: NPZ ROI=-3 "stopped" → OK (different ROI)

This is more conservative than removing entire folds - it only
removes the specific sentence pairs that have word overlap at the same ROI.
"""

import pandas as pd
from pathlib import Path
from typing import Set, Dict, List, Tuple
import logging
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def clean_word(word: str) -> str:
    """Remove punctuation and normalize word."""
    return re.sub(r'[%,\.;:\!\?]|%2C', '', word.lower()).strip()


def get_critical_words_by_roi(df: pd.DataFrame, construction: str, roi: int) -> Set[str]:
    """
    Get words at a specific ROI position for a construction.

    Args:
        df: DataFrame with garden-path data
        construction: Construction type (MVRR, NPS, NPZ)
        roi: ROI position to extract words from

    Returns:
        Set of cleaned words at the specified ROI
    """
    const_df = df[df['construction'] == construction]

    if roi == -3:
        # For ROI=-3, only get words from AMBIGUOUS sentences
        # This is where the garden-path trigger (usually a verb) appears
        amb_df = const_df[const_df['ambiguity'] == 'Amb']
        roi_df = amb_df[amb_df['roi'] == roi]
    else:
        # For other ROIs (like ROI=0), get from all sentences
        roi_df = const_df[const_df['roi'] == roi]

    return set(clean_word(w) for w in roi_df['word'].unique())


def get_critical_words(df: pd.DataFrame, construction: str) -> Dict[str, Set[str]]:
    """
    Get critical words for overlap checking.

    Args:
        df: DataFrame with garden-path data
        construction: Construction type (MVRR, NPS, NPZ)

    Returns:
        Dict with 'roi0_words' and 'roi_minus3_words'
    """
    return {
        'roi0_words': get_critical_words_by_roi(df, construction, 0),
        'roi_minus3_words': get_critical_words_by_roi(df, construction, -3)
    }


def get_construction_pairs(df: pd.DataFrame, construction: str) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    Get all ambiguous/unambiguous pairs for a construction.

    Args:
        df: DataFrame with garden-path data
        construction: Construction type

    Returns:
        Dict mapping item number to {'Amb': df, 'Unamb': df}
    """
    pairs = {}
    construction_df = df[df['construction'] == construction]

    for item in construction_df['item'].unique():
        item_df = construction_df[construction_df['item'] == item]
        pairs[item] = {
            'Amb': item_df[item_df['ambiguity'] == 'Amb'].copy(),
            'Unamb': item_df[item_df['ambiguity'] == 'Unamb'].copy()
        }

    return pairs


def find_overlapping_pairs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_construction: str
) -> Dict[str, List[int]]:
    """
    Find training pairs that have critical word overlap with test data.
    Checks cross-construction overlaps with ROI-position matching.

    NEW RULE: Only exclude if the word appears at the SAME ROI position
    across different constructions.

    Args:
        train_df: Training data
        test_df: Test data
        test_construction: Test construction type

    Returns:
        Dict mapping construction to list of item numbers to exclude from training
    """
    # Get test critical words by ROI
    test_roi0_words = get_critical_words_by_roi(test_df, test_construction, 0)
    test_roi_minus3_words = get_critical_words_by_roi(test_df, test_construction, -3)

    if not test_roi0_words and not test_roi_minus3_words:
        return {}

    # Track overlapping items by construction
    overlapping_items = {}

    # Check all constructions in training data
    for train_construction in train_df['construction'].unique():
        items_to_exclude = []

        # Get training words by ROI
        train_roi0_words = get_critical_words_by_roi(train_df, train_construction, 0)
        train_roi_minus3_words = get_critical_words_by_roi(train_df, train_construction, -3)

        # Check for ROI=0 overlap
        roi0_overlap = test_roi0_words & train_roi0_words

        # Check for ROI=-3 overlap
        roi_minus3_overlap = test_roi_minus3_words & train_roi_minus3_words

        if roi0_overlap or roi_minus3_overlap:
            # Find which items contain these overlapping words
            train_const_df = train_df[train_df['construction'] == train_construction]

            for item in train_const_df['item'].unique():
                item_df = train_const_df[train_const_df['item'] == item]

                # Check if this item contains any overlapping words at the critical ROIs
                item_has_overlap = False

                # Check ROI=0 overlap
                if roi0_overlap:
                    item_roi0_words = get_critical_words_by_roi(item_df, train_construction, 0)
                    if item_roi0_words & roi0_overlap:
                        item_has_overlap = True

                # Check ROI=-3 overlap
                if roi_minus3_overlap and not item_has_overlap:
                    item_roi_minus3_words = get_critical_words_by_roi(item_df, train_construction, -3)
                    if item_roi_minus3_words & roi_minus3_overlap:
                        item_has_overlap = True

                if item_has_overlap:
                    items_to_exclude.append(item)

        if items_to_exclude:
            overlapping_items[train_construction] = items_to_exclude

    return overlapping_items


def filter_fold_data(
    train_path: Path,
    test_path: Path,
    output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[int]]]:
    """
    Filter a single fold to remove overlapping pairs.
    Uses cross-construction filtering with ROI-position matching.

    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
        output_dir: Directory for filtered output

    Returns:
        Tuple of (filtered_train_df, test_df, excluded_items_by_construction)
    """
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Track excluded items by construction
    # This will accumulate all exclusions across all test constructions
    all_excluded_items = {}

    # Check each test construction against all training constructions
    test_constructions = test_df['construction'].unique()

    for test_construction in test_constructions:
        # Find overlapping pairs (returns dict of train_construction -> items to exclude)
        overlapping_items = find_overlapping_pairs(train_df, test_df, test_construction)

        # Merge results
        for train_construction, items in overlapping_items.items():
            if train_construction not in all_excluded_items:
                all_excluded_items[train_construction] = set()
            all_excluded_items[train_construction].update(items)

    # Convert sets to sorted lists
    excluded_items = {
        construction: sorted(list(items))
        for construction, items in all_excluded_items.items()
    }

    # Log exclusions
    for construction in sorted(excluded_items.keys()):
        items = excluded_items[construction]
        if items:
            items_str = ', '.join(str(int(item)) for item in items)
            logger.info(f"  {construction}: Excluding items [{items_str}] due to cross-construction ROI overlap")

    # Filter training data
    filtered_train_dfs = []
    train_constructions = train_df['construction'].unique()

    for construction in train_constructions:
        construction_df = train_df[train_df['construction'] == construction]

        if construction in excluded_items and excluded_items[construction]:
            # Exclude overlapping items
            keep_df = construction_df[~construction_df['item'].isin(excluded_items[construction])]
            filtered_train_dfs.append(keep_df)
        else:
            # Keep all pairs for this construction
            filtered_train_dfs.append(construction_df)

    # Initialize excluded_items for constructions with no exclusions
    for construction in train_constructions:
        if construction not in excluded_items:
            excluded_items[construction] = []

    # Combine filtered data
    if filtered_train_dfs:
        filtered_train_df = pd.concat(filtered_train_dfs, ignore_index=True)
        # Sort by construction first, then by index
        filtered_train_df = filtered_train_df.sort_values(['construction', 'index']).reset_index(drop=True)
    else:
        filtered_train_df = pd.DataFrame()

    return filtered_train_df, test_df, excluded_items


def process_all_folds(
    folds_dir: Path,
    output_base_dir: Path,
    max_folds: int = 24,
    show_examples: bool = True
):
    """
    Process all folds and create filtered versions.

    Args:
        folds_dir: Directory containing original folds
        output_base_dir: Base directory for filtered output
        max_folds: Maximum number of folds to process
    """
    # Create output directory
    output_dir = output_base_dir / "folds_filtered"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Filtering Garden-Path Folds - Cross-Construction ROI-Position Matching")
    logger.info("=" * 80)

    # Track statistics
    total_excluded = {
        'MVRR': [],
        'NPS': [],
        'NPZ': []
    }

    # Process each fold
    for fold_num in tqdm(range(max_folds), desc="Processing folds"):
        fold_dir = folds_dir / f"fold_{fold_num}"
        train_path = fold_dir / "garden_path_train.csv"
        test_path = fold_dir / "garden_path_test.csv"

        if not train_path.exists() or not test_path.exists():
            logger.warning(f"Skipping fold {fold_num}: missing data files")
            continue

        logger.info(f"\nFold {fold_num}:")

        # Create output directory for this fold
        fold_output_dir = output_dir / f"fold_{fold_num}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # Filter the fold
        filtered_train_df, test_df, excluded_items = filter_fold_data(
            train_path, test_path, fold_output_dir
        )

        # Save filtered data
        filtered_train_path = fold_output_dir / "garden_path_train.csv"
        filtered_test_path = fold_output_dir / "garden_path_test.csv"

        filtered_train_df.to_csv(filtered_train_path, index=False)
        test_df.to_csv(filtered_test_path, index=False)  # Test data unchanged

        # Track statistics
        for construction, items in excluded_items.items():
            total_excluded[construction].extend(items)

        # Log statistics for this fold
        orig_train_size = pd.read_csv(train_path).shape[0]
        filtered_train_size = filtered_train_df.shape[0]

        if orig_train_size != filtered_train_size:
            logger.info(f"  Training data: {orig_train_size} → {filtered_train_size} rows")
            logger.info(f"  Removed {orig_train_size - filtered_train_size} rows")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("Filtering Summary")
    logger.info("=" * 80)

    for construction in ['MVRR', 'NPS', 'NPZ']:
        excluded_items = set(total_excluded[construction])
        if excluded_items:
            # Convert to regular ints for display
            items_list = sorted([int(item) for item in excluded_items])
            logger.info(f"{construction}:")
            logger.info(f"  Total unique items excluded across all folds: {len(excluded_items)}")
            logger.info(f"  Items: {items_list}")

    logger.info(f"\nFiltered folds saved to: {output_dir}")

    # Create a simple summary file
    summary_path = output_dir / "filtering_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Fold Filtering Summary\n")
        f.write("=" * 50 + "\n\n")

        # Go through each fold and report what was excluded
        for fold_num in range(max_folds):
            fold_dir = folds_dir / f"fold_{fold_num}"
            filtered_fold_dir = output_dir / f"fold_{fold_num}"

            if not fold_dir.exists() or not filtered_fold_dir.exists():
                continue

            # Check if anything was excluded
            train_path = fold_dir / "garden_path_train.csv"
            filtered_train_path = filtered_fold_dir / "garden_path_train.csv"

            if train_path.exists() and filtered_train_path.exists():
                orig_df = pd.read_csv(train_path)
                filt_df = pd.read_csv(filtered_train_path)

                if len(orig_df) != len(filt_df):
                    f.write(f"Fold {fold_num}:\n")

                    # Check each construction
                    for construction in ['MVRR', 'NPS', 'NPZ']:
                        orig_items = set(orig_df[orig_df['construction'] == construction]['item'].unique())
                        filt_items = set(filt_df[filt_df['construction'] == construction]['item'].unique())
                        excluded = orig_items - filt_items

                        if excluded:
                            items_str = ', '.join(str(int(item)) for item in sorted(excluded))
                            f.write(f"  {construction}: excluded pair {items_str}\n")

                    f.write("\n")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter garden-path folds to remove ROI=0 word overlaps"
    )
    parser.add_argument(
        "--folds-dir",
        type=Path,
        default=Path("../../src/garden_path_cross_validation/folds"),
        help="Directory containing original folds"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../src/garden_path_cross_validation"),
        help="Base output directory for filtered folds"
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=24,
        help="Maximum number of folds to process"
    )

    args = parser.parse_args()

    # Process all folds
    process_all_folds(
        folds_dir=args.folds_dir,
        output_base_dir=args.output_dir,
        max_folds=args.max_folds
    )


if __name__ == "__main__":
    main()
