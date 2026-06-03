#!/usr/bin/env python3
"""
CLI script for Natural Stories data preprocessing and train/test splitting.

This script provides a command-line interface for running the complete
Natural Stories preprocessing pipeline including train/test split generation.
"""

import argparse
import logging
import sys
from pathlib import Path

from natural_stories_processor import NaturalStoriesProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Natural Stories data preprocessing and splitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing without splitting
  python cli.py --naturalstories-path external/naturalstories --output-dir data/

  # Full pipeline with train/test split
  python cli.py --naturalstories-path external/naturalstories --output-dir data/ --create-split

  # Custom split ratio (70-30 instead of 60-40)
  python cli.py --naturalstories-path external/naturalstories --output-dir data/ --create-split --test-size 0.3

  # Custom random seed
  python cli.py --naturalstories-path external/naturalstories --output-dir data/ --create-split --random-state 42
        """,
    )

    # Required arguments
    parser.add_argument(
        "--naturalstories-path",
        required=True,
        help="Path to external/naturalstories directory",
    )

    parser.add_argument(
        "--output-dir", required=True, help="Output directory for processed data"
    )

    # Optional arguments
    parser.add_argument(
        "--create-split",
        action="store_true",
        help="Create train/test split following reverse-engineering methodology",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.4,
        help="Proportion of data for test set (default: 0.4)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=12321,
        help="Random seed for reproducible splits (default: 12321)",
    )

    parser.add_argument(
        "--outlier-removal",
        action="store_true",
        help="Apply 3σ outlier removal during preprocessing",
    )

    args = parser.parse_args()

    # Validate arguments
    if not (0.0 < args.test_size < 1.0):
        logger.error("test-size must be between 0 and 1")
        sys.exit(1)

    naturalstories_path = Path(args.naturalstories_path)
    if not naturalstories_path.exists():
        logger.error(f"Natural Stories path does not exist: {naturalstories_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize processor
        logger.info("Initializing Natural Stories processor...")
        processor = NaturalStoriesProcessor(str(naturalstories_path))

        # Run preprocessing pipeline
        logger.info("Running preprocessing pipeline...")
        processed_data = processor.process_full_pipeline()

        if args.create_split:
            # Create train/test split
            logger.info("Creating train/test split...")
            train_df, test_df = processor.create_train_test_split(
                processed_data,
                output_dir=str(output_dir),
                test_size=args.test_size,
                random_state=args.random_state,
            )

            # Display split statistics
            logger.info("=" * 50)
            logger.info("SPLIT SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Total words: {len(processed_data):,}")
            logger.info(
                f"Train words: {len(train_df):,} ({len(train_df) / len(processed_data):.1%})"
            )
            logger.info(
                f"Test words: {len(test_df):,} ({len(test_df) / len(processed_data):.1%})"
            )
            logger.info(
                f"Train sentences: {train_df['sentence_num'].nunique():,}"
            )
            logger.info(f"Test sentences: {test_df['sentence_num'].nunique():,}")
            logger.info("Output files:")
            logger.info(f"  - {output_dir}/natural_stories_train.csv")
            logger.info(f"  - {output_dir}/natural_stories_test.csv")

        else:
            # Save complete processed data
            output_file = output_dir / "natural_stories_processed.csv"
            processed_data.to_csv(output_file, index=False)
            logger.info(f"Saved complete processed data to {output_file}")

            # Display data statistics
            logger.info("=" * 50)
            logger.info("DATA SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Total words: {len(processed_data):,}")
            logger.info(f"Stories: {processed_data['item'].nunique()}")
            logger.info(
                f"Sentences: {processed_data['sentence_num'].nunique():,}"
            )
            logger.info(
                f"Mean reading time: {processed_data['reading_time'].mean():.2f} ms"
            )
            logger.info("Spillover coverage:")
            logger.info(
                f"  - freq_prev1 non-null: {processed_data['freq_prev1'].notna().sum():,}"
            )
            logger.info(
                f"  - freq_prev2 non-null: {processed_data['freq_prev2'].notna().sum():,}"
            )

        logger.info("Processing completed successfully!")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
