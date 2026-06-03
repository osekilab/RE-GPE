#!/usr/bin/env python3
"""
CLI for garden-path cross-validation data preparation

This module provides unified command-line interface for:
- Processing human reading time data
- Creating cross-validation folds
- Data enhancement operations
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from cross_validation_splitter import CrossValidationSplitter
from human_data_processor import HumanDataProcessor
from fillers_processor import FillersProcessor
from filter_fold_pairs import process_all_folds as filter_folds

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_human_data(args):
    """Process human reading time data from ClassicGardenPathSet.csv"""
    logger.info("Processing human reading time data...")

    processor = HumanDataProcessor()

    # Determine input file
    if args.input:
        input_path = Path(args.input)
    else:
        # Default location (from project root)
        input_path = Path("data/ClassicGardenPathSet.csv")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False

    # Process data with RT filtering parameters
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path("src/garden_path_cross_validation/subject_averages")
    )

    # Get RT filtering parameters (use defaults if not specified)
    min_rt = args.min_rt if hasattr(args, 'min_rt') else 100
    max_rt = args.max_rt if hasattr(args, 'max_rt') else 3000

    processor.process_data(str(input_path), str(output_dir), min_rt=min_rt, max_rt=max_rt)

    logger.info(f"Human data processing completed. Results saved to {output_dir}")
    return True


def create_folds(args):
    """Create unified cross-validation folds containing all constructions"""
    logger.info("Creating unified cross-validation folds for all constructions...")

    splitter = CrossValidationSplitter()
    splitter.create_folds(
        num_folds=args.num_folds,
        output_dir=Path(args.output_dir) if args.output_dir else Path("src/garden_path_cross_validation/folds"),
    )

    logger.info("Cross-validation folds created successfully")
    return True


def process_fillers(args):
    """Process filler sentences for regression model fitting"""
    logger.info("Processing filler sentences data...")

    processor = FillersProcessor()

    # Determine input file
    if args.input:
        input_path = Path(args.input)
    else:
        # Default location (from project root)
        input_path = Path("data/Fillers.csv")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False

    # Process data
    output_path = (
        args.output if args.output
        else "src/garden_path_cross_validation/folds/fillers_processed.csv"
    )

    # Get RT filtering parameters (use defaults if not specified)
    min_rt = args.min_rt if hasattr(args, 'min_rt') else 100
    max_rt = args.max_rt if hasattr(args, 'max_rt') else 3000

    processor.process(str(input_path), str(output_path), min_rt=min_rt, max_rt=max_rt)

    logger.info(f"Filler data processing completed. Results saved to {output_path}")
    return True


def filter_fold_pairs(args):
    """Filter folds to remove ROI=0 word overlaps between train and test"""
    logger.info("Filtering folds to remove ROI=0 word overlaps...")

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
        description="Garden-path cross-validation data preparation tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (from project root):
  # Process human reading time data
  python src/garden_path_cross_validation/data_preparation/cli.py process-human-data

  # Create unified folds for all constructions
  python src/garden_path_cross_validation/data_preparation/cli.py create-folds --num-folds 24

  # Filter folds to remove ROI=0 word overlaps
  python src/garden_path_cross_validation/data_preparation/cli.py filter-folds

  # Process filler sentences for regression
  python src/garden_path_cross_validation/data_preparation/cli.py process-fillers
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process human data command
    human_parser = subparsers.add_parser(
        "process-human-data", help="Process human reading time data"
    )
    human_parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to ClassicGardenPathSet.csv (default: data/ClassicGardenPathSet.csv)",
    )
    human_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for processed data (default: src/garden_path_cross_validation/subject_averages)",
    )
    human_parser.add_argument(
        "--min-rt",
        type=float,
        default=100,
        help="Minimum RT threshold in ms (default: 100)",
    )
    human_parser.add_argument(
        "--max-rt",
        type=float,
        default=3000,
        help="Maximum RT threshold in ms (default: 3000)",
    )

    # Create folds command
    folds_parser = subparsers.add_parser(
        "create-folds", help="Create cross-validation folds"
    )
    folds_parser.add_argument(
        "--num-folds",
        type=int,
        default=24,
        help="Number of cross-validation folds (default: 24)",
    )
    folds_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for folds (default: src/garden_path_cross_validation/folds)",
    )

    # Process fillers command
    fillers_parser = subparsers.add_parser(
        "process-fillers",
        help="Process filler sentences for regression model fitting"
    )
    fillers_parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to Fillers.csv (default: data/Fillers.csv)",
    )
    fillers_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path for processed data (default: src/garden_path_cross_validation/folds/fillers_processed.csv)",
    )
    fillers_parser.add_argument(
        "--min-rt",
        type=float,
        default=100,
        help="Minimum RT threshold in ms (default: 100)",
    )
    fillers_parser.add_argument(
        "--max-rt",
        type=float,
        default=3000,
        help="Maximum RT threshold in ms (default: 3000)",
    )

    # Filter folds command
    filter_parser = subparsers.add_parser(
        "filter-folds",
        help="Filter folds to remove ROI=0 word overlaps between train and test"
    )
    filter_parser.add_argument(
        "--folds-dir",
        type=str,
        default="src/garden_path_cross_validation/folds",
        help="Directory containing original folds",
    )
    filter_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="src/garden_path_cross_validation",
        help="Base output directory for filtered folds",
    )
    filter_parser.add_argument(
        "--max-folds",
        type=int,
        default=24,
        help="Maximum number of folds to process (default: 24)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "process-human-data":
            success = process_human_data(args)
        elif args.command == "create-folds":
            success = create_folds(args)
        elif args.command == "process-fillers":
            success = process_fillers(args)
        elif args.command == "filter-folds":
            success = filter_fold_pairs(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Error executing {args.command}: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
