#!/usr/bin/env python3
"""
Train models on all filtered folds.

This script trains a model for each fold in the filtered folds directory,
using the garden_path_template.yaml as the base configuration.
"""

import yaml
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
import argparse
import time
from datetime import datetime


def get_available_folds(folds_dir: Path) -> List[int]:
    """Get list of available fold indices from the folds directory.

    Args:
        folds_dir: Directory containing the fold subdirectories

    Returns:
        List of available fold indices
    """
    available_folds = []

    # Check which fold directories exist with both train and test files
    for fold_dir in sorted(folds_dir.glob("fold_*")):
        if fold_dir.is_dir():
            fold_idx = int(fold_dir.name.split("_")[1])
            train_path = fold_dir / "garden_path_train.csv"
            test_path = fold_dir / "garden_path_test.csv"

            if train_path.exists() and test_path.exists():
                available_folds.append(fold_idx)

    return sorted(available_folds)


def create_fold_config(base_config_path: Path, fold_idx: int, config_output_dir: Path, folds_dir: Path, model_output_base_dir: Path) -> Path:
    """
    Create a configuration file for a specific fold.

    Args:
        base_config_path: Path to the template configuration
        fold_idx: Fold index
        config_output_dir: Directory to save the fold-specific config
        folds_dir: Directory containing the fold subdirectories
        model_output_base_dir: Base directory for model outputs

    Returns:
        Path to the created configuration file
    """
    try:
        from ruamel.yaml import YAML
        yaml_loader = YAML()
        yaml_loader.preserve_quotes = True
        yaml_loader.width = 4096  # Prevent line wrapping

        # Load base configuration with ruamel to preserve formatting
        with open(base_config_path, 'r') as f:
            config = yaml_loader.load(f)
    except ImportError:
        # Fallback to standard yaml if ruamel is not available
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        yaml_loader = None

    # Update paths for this fold
    fold_path = folds_dir / f"fold_{fold_idx}"

    # Update training data path
    config['training_kwargs']['data_path'] = str(fold_path / "garden_path_train.csv")

    # Update evaluation data path
    config['training_kwargs']['eval_data_path'] = str(fold_path / "garden_path_test.csv")

    # Update output directory to include fold index
    config['training_kwargs']['output_dir'] = str(model_output_base_dir / f"garden_path_fold_{fold_idx}")

    # Update experiment name to include fold
    config['experiment_name'] = f"garden_path_roi_exclusion_fold_{fold_idx}"

    # Save fold-specific configuration
    fold_config_path = config_output_dir / f"garden_path_fold_{fold_idx}.yaml"

    if yaml_loader:
        # Use ruamel to preserve formatting
        with open(fold_config_path, 'w') as f:
            yaml_loader.dump(config, f)
    else:
        # Fallback to standard yaml
        with open(fold_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False,
                     allow_unicode=True, width=float("inf"))

    return fold_config_path


def run_training(config_path: Path, fold_idx: int, dry_run: bool = False, device: str = None) -> bool:
    """
    Run training for a specific fold.

    Args:
        config_path: Path to the configuration file
        fold_idx: Fold index (for logging)
        dry_run: If True, only print the command without executing
        device: Device to use (e.g., 'cuda:0', 'cuda:1', 'mps', 'cpu')

    Returns:
        True if training succeeded, False otherwise
    """
    cmd = [
        "python", "run.py",
        "--config", str(config_path)
    ]

    # Add device argument if specified
    if device is not None:
        cmd.extend(["--device", device])

    # Add environment variables for MPS support
    env = {
        **subprocess.os.environ,
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "TOKENIZERS_PARALLELISM": "false"
    }

    print(f"\n{'='*80}")
    print(f"Training fold {fold_idx}")
    print(f"Config: {config_path}")
    print(f"Command: PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false {' '.join(cmd)}")
    print(f"{'='*80}\n")

    if dry_run:
        print("[DRY RUN] Would execute the above command")
        return True

    try:
        # Run the training command
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        print(f"\n✓ Fold {fold_idx} training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Fold {fold_idx} training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Training interrupted by user")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Train models on all filtered folds"
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/garden_path_template.yaml"),
        help="Base configuration file to use as template"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/fold_configs"),
        help="Directory to save fold-specific configurations"
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs='+',
        default=None,
        help="Specific fold indices to train (default: all valid folds)"
    )
    parser.add_argument(
        "--start-fold",
        type=int,
        default=None,
        help="Start training from this fold index"
    )
    parser.add_argument(
        "--end-fold",
        type=int,
        default=None,
        help="Stop training at this fold index (inclusive)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be executed without actually running"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run training sequentially (default: sequential)"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue training other folds even if one fails"
    )
    parser.add_argument(
        "--folds-dir",
        type=Path,
        default=Path("../../src/garden_path_cross_validation/folds_filtered"),
        help="Directory containing the filtered fold subdirectories"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base directory for model outputs (default: output/)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (e.g., 'cuda:0', 'cuda:1', 'mps', 'cpu'). If not specified, uses config or auto-detect"
    )

    args = parser.parse_args()

    # Get available folds
    if args.folds:
        # Use specified folds
        available_folds = args.folds
        print(f"Using specified folds: {available_folds}")
    else:
        # Get all available folds from the folds directory
        available_folds = get_available_folds(args.folds_dir)
        print(f"Found {len(available_folds)} available folds: {available_folds}")

    # Apply fold range filters if specified
    if args.start_fold is not None:
        available_folds = [f for f in available_folds if f >= args.start_fold]
    if args.end_fold is not None:
        available_folds = [f for f in available_folds if f <= args.end_fold]

    if not available_folds:
        print("No folds to train after applying filters")
        return

    print(f"\nWill train {len(available_folds)} folds: {available_folds}")

    # Create config directory if it doesn't exist
    args.config_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    successful_folds = []
    failed_folds = []

    # Create configurations and run training for each fold
    start_time = datetime.now()

    for i, fold_idx in enumerate(available_folds, 1):
        print(f"\n{'#'*80}")
        print(f"Processing fold {fold_idx} ({i}/{len(available_folds)})")
        print(f"{'#'*80}")

        try:
            # Create fold-specific configuration
            fold_config_path = create_fold_config(
                args.base_config,
                fold_idx,
                args.config_dir,
                args.folds_dir,
                args.output_dir
            )
            print(f"Created configuration: {fold_config_path}")

            # Run training
            success = run_training(fold_config_path, fold_idx, args.dry_run, device=args.device)

            if success:
                successful_folds.append(fold_idx)
            else:
                failed_folds.append(fold_idx)
                if not args.continue_on_error:
                    print(f"\nStopping due to failure in fold {fold_idx}")
                    break

            # Add a small delay between folds to avoid overwhelming the system
            if i < len(available_folds) and not args.dry_run:
                print(f"\nWaiting 5 seconds before next fold...")
                time.sleep(5)

        except KeyboardInterrupt:
            print(f"\n\nTraining interrupted by user at fold {fold_idx}")
            failed_folds.extend(available_folds[i:])  # Mark remaining as failed
            break

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {duration}")
    print(f"Successful folds ({len(successful_folds)}): {successful_folds}")
    print(f"Failed folds ({len(failed_folds)}): {failed_folds}")

    if not args.dry_run:
        # Save summary to file
        summary_file = Path("training_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Training Summary - {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total folds: {len(available_folds)}\n")
            f.write(f"Successful: {len(successful_folds)} - {successful_folds}\n")
            f.write(f"Failed: {len(failed_folds)} - {failed_folds}\n")
            f.write(f"Duration: {duration}\n")
        print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
