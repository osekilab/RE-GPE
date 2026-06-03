#!/usr/bin/env python3
"""
Filter training data to remove specific RC pairs when test critical words
appear in training data at the SAME ROI position.

ROI-position matching:
- Checks ROI=0, 2 words (critical word and spillover_2)
- ROI=1 is excluded from checking (too many common words like "the")
- Only excludes if the word appears at the SAME ROI position
- ROI=NA values are excluded from overlap checking (not critical regions)

Example:
- Test: RC_Obj ROI=0 "visited" → Train: RC_Subj ROI=0 "visited" → EXCLUDE
- Test: RC_Obj ROI=2 "created" → Train: RC_Subj ROI=2 "created" → EXCLUDE
- Test: RC_Obj ROI=1 "the" → Train: RC_Subj ROI=1 "the" → OK (ROI=1 not checked)
- Test: RC_Obj ROI=0 "visited" → Train: RC_Subj ROI=2 "visited" → OK (different ROI)
- Test: RC_Obj ROI=NA "created" → Train: anything → OK (NA not checked)
"""

import pandas as pd
from pathlib import Path
from typing import Set, Dict, List
import logging
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def clean_word(word: str) -> str:
    """Remove punctuation and normalize word."""
    return re.sub(r'[%,\.;:\!\?]|%2C', '', word.lower()).strip()


def get_critical_words_by_roi(df: pd.DataFrame, roi: int) -> Set[str]:
    """
    Get words at a specific ROI position.

    Args:
        df: DataFrame with RC data
        roi: ROI position to extract words from

    Returns:
        Set of cleaned words at the specified ROI
    """
    # Filter to specific ROI, excluding NA values
    roi_df = df[~pd.isna(df['roi']) & (df['roi'] == roi)]

    return set(clean_word(w) for w in roi_df['word'].unique())


def get_rc_pairs(df: pd.DataFrame) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    Get all RC_Subj/RC_Obj pairs.

    Args:
        df: DataFrame with RC data

    Returns:
        Dict mapping item number to {'Amb': df, 'Unamb': df}
    """
    pairs = {}

    for item in df['item'].unique():
        item_df = df[df['item'] == item]
        pairs[item] = {
            'Amb': item_df[item_df['ambiguity'] == 'Amb'].copy(),
            'Unamb': item_df[item_df['ambiguity'] == 'Unamb'].copy()
        }

    return pairs


def find_overlapping_pairs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> List[int]:
    """
    Find training pairs that have critical word overlap with test data.
    Only excludes if the word appears at the SAME ROI position.

    Checks ROI=0, 2 (critical word and spillover_2).
    ROI=1 is excluded from checking to avoid false positives from common words.

    Args:
        train_df: Training data
        test_df: Test data

    Returns:
        List of item numbers to exclude from training
    """
    # Check ROI=0, 2 (critical word and spillover_2)
    # ROI=1 excluded to avoid common words like "the"
    critical_rois = [0, 2]

    # Get test critical words by ROI (excluding NA)
    test_words_by_roi = {}
    for roi in critical_rois:
        test_words_by_roi[roi] = get_critical_words_by_roi(test_df, roi)

    # If no critical words in test data, nothing to exclude
    if not any(test_words_by_roi.values()):
        return []

    items_to_exclude = []

    # Get training words by ROI (excluding NA)
    train_words_by_roi = {}
    for roi in critical_rois:
        train_words_by_roi[roi] = get_critical_words_by_roi(train_df, roi)

    # Check for overlap at each ROI
    overlaps_by_roi = {}
    for roi in critical_rois:
        overlaps_by_roi[roi] = test_words_by_roi[roi] & train_words_by_roi[roi]

    # Find items with overlapping words at any critical ROI
    if any(overlaps_by_roi.values()):
        for item in train_df['item'].unique():
            item_df = train_df[train_df['item'] == item]

            # Check each ROI for overlap
            for roi in critical_rois:
                if overlaps_by_roi[roi]:
                    item_roi_words = get_critical_words_by_roi(item_df, roi)
                    if item_roi_words & overlaps_by_roi[roi]:
                        items_to_exclude.append(item)
                        break  # Item already marked for exclusion

    return items_to_exclude


def filter_fold(fold_dir: Path, output_dir: Path, fold_num: int) -> Dict:
    """
    Filter a single fold to remove critical word overlaps.

    Args:
        fold_dir: Input fold directory
        output_dir: Output fold directory
        fold_num: Fold number

    Returns:
        Dict with filtering statistics
    """
    train_path = fold_dir / "rc_train.csv"
    test_path = fold_dir / "rc_test.csv"

    if not train_path.exists() or not test_path.exists():
        logger.warning(f"Fold {fold_num}: Missing train or test file, skipping")
        return None

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Get test item for reporting
    test_item = test_df['item'].iloc[0] if len(test_df) > 0 else None

    # Find overlapping pairs
    items_to_exclude = find_overlapping_pairs(train_df, test_df)

    # Filter training data
    if items_to_exclude:
        train_filtered = train_df[~train_df['item'].isin(items_to_exclude)].copy()
        excluded_count = len(items_to_exclude)
    else:
        train_filtered = train_df.copy()
        excluded_count = 0

    # Reset index
    if len(train_filtered) > 0:
        train_filtered['index'] = range(len(train_filtered))

    # Save filtered data
    output_fold_dir = output_dir / f"fold_{fold_num}"
    output_fold_dir.mkdir(parents=True, exist_ok=True)

    train_filtered.to_csv(output_fold_dir / "rc_train.csv", index=False)
    # Test data remains unchanged
    test_df.to_csv(output_fold_dir / "rc_test.csv", index=False)

    # Statistics
    stats = {
        'fold': fold_num,
        'test_item': int(test_item) if test_item is not None else None,
        'original_train_items': int(train_df['item'].nunique()),
        'excluded_items': excluded_count,
        'filtered_train_items': int(train_filtered['item'].nunique()),
        'excluded_item_list': [int(i) for i in items_to_exclude],
        'original_train_words': len(train_df),
        'filtered_train_words': len(train_filtered),
    }

    return stats


def process_all_folds(
    folds_dir: Path = None,
    output_base_dir: Path = None,
    max_folds: int = None
):
    """
    Process all folds to remove critical word overlaps.

    Args:
        folds_dir: Directory containing original folds
        output_base_dir: Base directory for output (will create folds_filtered/)
        max_folds: Maximum number of folds to process (None for all)
    """
    if folds_dir is None:
        folds_dir = Path("src/relative_clause_cross_validation/folds")

    if output_base_dir is None:
        output_base_dir = Path("src/relative_clause_cross_validation")

    output_dir = output_base_dir / "folds_filtered"

    logger.info(f"Processing folds from: {folds_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Find all fold directories
    fold_dirs = sorted([d for d in folds_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])

    if max_folds is not None:
        fold_dirs = fold_dirs[:max_folds]

    logger.info(f"Found {len(fold_dirs)} folds to process")

    all_stats = []

    for fold_dir in tqdm(fold_dirs, desc="Filtering folds"):
        fold_num = int(fold_dir.name.split("_")[1])
        stats = filter_fold(fold_dir, output_dir, fold_num)

        if stats:
            all_stats.append(stats)

            if stats['excluded_items'] > 0:
                logger.info(
                    f"Fold {fold_num}: Excluded {stats['excluded_items']} items "
                    f"({stats['filtered_train_items']}/{stats['original_train_items']} remaining)"
                )

    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("FILTERING SUMMARY")
    logger.info("="*60)

    total_excluded = sum(s['excluded_items'] for s in all_stats)
    folds_with_exclusions = sum(1 for s in all_stats if s['excluded_items'] > 0)

    logger.info(f"Total folds processed: {len(all_stats)}")
    logger.info(f"Folds with exclusions: {folds_with_exclusions}")
    logger.info(f"Total items excluded: {total_excluded}")

    if all_stats:
        avg_train_items = sum(s['filtered_train_items'] for s in all_stats) / len(all_stats)
        logger.info(f"Average train items per fold: {avg_train_items:.1f}")

    # Save detailed statistics
    import json
    stats_path = output_dir / "filtering_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    logger.info(f"\nDetailed statistics saved to: {stats_path}")
    logger.info(f"Filtered folds saved to: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter RC folds to remove critical word overlaps"
    )
    parser.add_argument(
        "--folds-dir",
        type=Path,
        default=Path("src/relative_clause_cross_validation/folds"),
        help="Directory containing original folds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/relative_clause_cross_validation"),
        help="Base output directory (will create folds_filtered/)",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        help="Maximum number of folds to process (for testing)",
    )

    args = parser.parse_args()

    process_all_folds(
        folds_dir=args.folds_dir,
        output_base_dir=args.output_dir,
        max_folds=args.max_folds
    )


if __name__ == "__main__":
    main()
