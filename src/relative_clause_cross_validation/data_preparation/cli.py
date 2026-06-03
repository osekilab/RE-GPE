#!/usr/bin/env python3
"""
CLI for Relative Clause cross-validation data preparation

This module provides unified command-line interface for:
- Processing human reading time data
- Creating cross-validation folds
- Filtering folds for critical word overlap
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from human_data_processor import HumanDataProcessor
from cross_validation_splitter import CrossValidationSplitter
from filter_fold_pairs import process_all_folds as filter_folds

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_human_data(args):
    """Process human reading time data from RelativeClauseSet.csv"""
    logger.info("Processing human reading time data...")

    processor = HumanDataProcessor()

    # Determine input file
    if args.input:
        input_path = Path(args.input)
    else:
        # Default location (from project root)
        input_path = Path("data/RelativeClauseSet.csv")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False

    # Process data with RT filtering parameters
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path("src/relative_clause_cross_validation/subject_averages")
    )

    # Get RT filtering parameters (use defaults if not specified)
    min_rt = args.min_rt if hasattr(args, 'min_rt') else 100
    max_rt = args.max_rt if hasattr(args, 'max_rt') else 3000

    processor.process_data(str(input_path), str(output_dir), min_rt=min_rt, max_rt=max_rt)

    logger.info(f"Human data processing completed. Results saved to {output_dir}")
    return True


def create_folds(args):
    """Create 24-fold leave-one-out cross-validation splits"""
    logger.info("Creating 24-fold leave-one-out cross-validation splits...")

    splitter = CrossValidationSplitter()
    splitter.create_folds(
        num_folds=args.num_folds,
        output_dir=Path(args.output_dir) if args.output_dir else Path("src/relative_clause_cross_validation/folds"),
    )

    logger.info("Cross-validation folds created successfully")
    return True


def filter_fold_pairs(args):
    """Filter folds to remove ROI=0 and ROI=-3 word overlaps between train and test"""
    logger.info("Filtering folds to remove critical word overlaps...")

    # Use the imported function
    filter_folds(
        folds_dir=Path(args.folds_dir),
        output_base_dir=Path(args.output_dir),
        max_folds=args.max_folds
    )

    logger.info(f"Filtered folds created in: {Path(args.output_dir) / 'folds_filtered'}")
    return True


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Relative Clause cross-validation data preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process human data
  python cli.py process-human-data --min-rt 100 --max-rt 3000

  # Create 24 folds
  python cli.py create-folds --num-folds 24

  # Filter folds for critical word overlap
  python cli.py filter-folds

  # Full pipeline
  python cli.py process-human-data && \\
  python cli.py create-folds && \\
  python cli.py filter-folds
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process human data command
    process_parser = subparsers.add_parser(
        "process-human-data",
        help="Process human reading time data from RelativeClauseSet.csv"
    )
    process_parser.add_argument(
        "--input",
        type=str,
        help="Path to RelativeClauseSet.csv (default: data/RelativeClauseSet.csv)"
    )
    process_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: src/relative_clause_cross_validation/subject_averages)"
    )
    process_parser.add_argument(
        "--min-rt",
        type=float,
        default=100,
        help="Minimum acceptable RT in ms (default: 100)"
    )
    process_parser.add_argument(
        "--max-rt",
        type=float,
        default=3000,
        help="Maximum acceptable RT in ms (default: 3000)"
    )

    # Create folds command
    folds_parser = subparsers.add_parser(
        "create-folds",
        help="Create 24-fold leave-one-out cross-validation splits"
    )
    folds_parser.add_argument(
        "--num-folds",
        type=int,
        default=24,
        help="Number of folds to create (default: 24)"
    )
    folds_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: src/relative_clause_cross_validation/folds)"
    )

    # Filter folds command
    filter_parser = subparsers.add_parser(
        "filter-folds",
        help="Filter folds to remove critical word overlaps"
    )
    filter_parser.add_argument(
        "--folds-dir",
        type=str,
        default="src/relative_clause_cross_validation/folds",
        help="Directory containing folds (default: src/relative_clause_cross_validation/folds)"
    )
    filter_parser.add_argument(
        "--output-dir",
        type=str,
        default="src/relative_clause_cross_validation",
        help="Base output directory (default: src/relative_clause_cross_validation)"
    )
    filter_parser.add_argument(
        "--max-folds",
        type=int,
        help="Maximum number of folds to process (for testing)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    if args.command == "process-human-data":
        success = process_human_data(args)
    elif args.command == "create-folds":
        success = create_folds(args)
    elif args.command == "filter-folds":
        success = filter_fold_pairs(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
