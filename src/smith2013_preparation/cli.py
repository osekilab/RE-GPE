"""
Command-line interface for Smith 2013 corpus preprocessing.
"""

import argparse
import logging
from pathlib import Path

from smith2013_processor import Smith2013Processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process Smith 2013 eyetracking corpus for reverse-engineering methodology"
    )

    parser.add_argument(
        "--smith2013-path",
        type=str,
        default="./data/data.pkl",
        help="Path to Smith 2013 data directory (default: ./data/data.pkl)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save processed data (default: data)",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default="smith2013_processed.csv",
        help="Output filename for processed data (default: smith2013_processed.csv)",
    )

    parser.add_argument(
        "--create-split",
        action="store_true",
        help="Create train/test split",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for test set (default: 0.2)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )

    args = parser.parse_args()

    # Initialize processor
    logger.info(f"Initializing Smith 2013 processor with data from {args.smith2013_path}")
    processor = Smith2013Processor(args.smith2013_path)

    # Prepare output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_filename

    # Process data
    logger.info("Starting data processing...")

    if args.create_split:
        logger.info("Creating train/test split...")
        train_df, test_df = processor.process_full_pipeline(
            output_path=str(output_path),
            create_split=True,
            test_size=args.test_size,
            random_state=args.random_state
        )

        # Summary statistics
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)

        # Train set summary
        logger.info("Train Set Summary:")
        logger.info(f"  Total words: {len(train_df)}")
        logger.info(f"  Sentences: {train_df['sentence_num'].nunique()}")
        logger.info(f"  Mean reading time: {train_df['reading_time'].mean():.2f} ms")
        logger.info(
            f"  Reading time range: [{train_df['reading_time'].min():.2f}, "
            f"{train_df['reading_time'].max():.2f}] ms"
        )

        if "surprisal" in train_df.columns:
            logger.info(f"  Mean surprisal: {train_df['surprisal'].mean():.2f}")

        # Test set summary
        logger.info("Test Set Summary:")
        logger.info(f"  Total words: {len(test_df)}")
        logger.info(f"  Sentences: {test_df['sentence_num'].nunique()}")
        logger.info(f"  Mean reading time: {test_df['reading_time'].mean():.2f} ms")
        logger.info(
            f"  Reading time range: [{test_df['reading_time'].min():.2f}, "
            f"{test_df['reading_time'].max():.2f}] ms"
        )

        if "surprisal" in test_df.columns:
            logger.info(f"  Mean surprisal: {test_df['surprisal'].mean():.2f}")

        # File locations
        train_file = output_dir / "smith2013_train.csv"
        test_file = output_dir / "smith2013_test.csv"
        logger.info(f"Train data saved to: {train_file}")
        logger.info(f"Test data saved to: {test_file}")

    else:
        processed_data = processor.process_full_pipeline(
            output_path=str(output_path),
            create_split=False
        )

        # Summary statistics
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info("Dataset Summary:")
        logger.info(f"  Total words: {len(processed_data)}")
        logger.info(f"  Sentences: {processed_data['sentence_num'].nunique()}")
        logger.info(f"  Mean reading time: {processed_data['reading_time'].mean():.2f} ms")
        logger.info(
            f"  Reading time range: [{processed_data['reading_time'].min():.2f}, "
            f"{processed_data['reading_time'].max():.2f}] ms"
        )

        if "surprisal" in processed_data.columns:
            logger.info(f"  Mean surprisal: {processed_data['surprisal'].mean():.2f}")
            logger.info(f"  Surprisal range: [{processed_data['surprisal'].min():.2f}, {processed_data['surprisal'].max():.2f}]")

        # Spillover features
        logger.info("Spillover Features:")
        logger.info(
            f"  freq_prev1 coverage: {processed_data['freq_prev1'].notna().sum()} / "
            f"{len(processed_data)} ({processed_data['freq_prev1'].notna().sum() / len(processed_data):.1%})"
        )
        logger.info(
            f"  freq_prev2 coverage: {processed_data['freq_prev2'].notna().sum()} / "
            f"{len(processed_data)} ({processed_data['freq_prev2'].notna().sum() / len(processed_data):.1%})"
        )

        if "surprisal_prev1" in processed_data.columns:
            logger.info(
                f"  surprisal_prev1 coverage: {processed_data['surprisal_prev1'].notna().sum()} / "
                f"{len(processed_data)} ({processed_data['surprisal_prev1'].notna().sum() / len(processed_data):.1%})"
            )
            logger.info(
                f"  surprisal_prev2 coverage: {processed_data['surprisal_prev2'].notna().sum()} / "
                f"{len(processed_data)} ({processed_data['surprisal_prev2'].notna().sum() / len(processed_data):.1%})"
            )

        logger.info(f"Processed data saved to: {output_path}")


if __name__ == "__main__":
    main()
