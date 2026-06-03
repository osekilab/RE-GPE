#!/usr/bin/env python3
"""
Batch evaluation script for all fold models.
Runs evaluate_fold.py on all available folds and aggregates the results.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


def find_fold_models(output_dir: Path) -> List[Tuple[int, Path]]:
    """Find all fold model directories."""
    fold_models = []

    for model_dir in sorted(output_dir.glob("garden_path_fold_*")):
        if model_dir.is_dir():
            match = re.search(r'fold_(\d+)', model_dir.name)
            if match:
                fold_num = int(match.group(1))
                checkpoint = model_dir / "final_checkpoint"
                if checkpoint.exists() and (checkpoint / "model.safetensors").exists():
                    fold_models.append((fold_num, checkpoint))

    return sorted(fold_models)


def evaluate_fold(
    fold_num: int,
    checkpoint_path: Path,
    config_template: Path,
    folds_dir: Path,
    eval_filler: bool,
    eval_ucl: bool,
    eval_smith2013: bool,
    eval_natural_stories: bool,
    device: str,
    use_wt_decoding: bool = True
) -> Dict:
    """Run evaluation for a single fold."""

    # Check if already evaluated
    output_file = checkpoint_path / f"fold_{fold_num}_eval.json"
    # if output_file.exists():
    #     print(f"  Loading existing results for fold {fold_num}")
    #     with open(output_file, "r") as f:
    #         return json.load(f)

    print(f"  Evaluating fold {fold_num}...")

    cmd = [
        "uv", "run", "python", "evaluate_fold.py",
        "--checkpoint", str(checkpoint_path),
        "--fold-num", str(fold_num),
        "--config-template", str(config_template),
        "--folds-dir", str(folds_dir),
        "--device", device
    ]

    if eval_filler:
        cmd.append("--eval-filler")

    # Add individual dataset flags
    if eval_ucl:
        cmd.append("--eval-ucl")
    if eval_smith2013:
        cmd.append("--eval-smith2013")
    if eval_natural_stories:
        cmd.append("--eval-natural-stories")

    # Add WT decoding flag
    if not use_wt_decoding:
        cmd.append("--no-wt-decoding")

    env = {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "TOKENIZERS_PARALLELISM": "false"
    }

    try:
        # Run without capturing output for real-time progress visibility
        result = subprocess.run(
            cmd,
            env={**subprocess.os.environ, **env},
            cwd=Path(__file__).parent
        )

        if result.returncode != 0:
            print(f"  Error evaluating fold {fold_num}")
            return None

        # Load and return results
        if output_file.exists():
            with open(output_file, "r") as f:
                return json.load(f)
        else:
            print(f"  No output file created for fold {fold_num}")
            return None

    except Exception as e:
        print(f"  Failed to evaluate fold {fold_num}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate all fold models")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="output",
        help="Directory containing fold models",
    )
    parser.add_argument(
        "--config-template",
        type=str,
        default="configs/garden_path_template.yaml",
        help="Config template path",
    )
    parser.add_argument(
        "--folds-dir",
        type=str,
        default="../../src/garden_path_cross_validation/folds_filtered",
        help="Directory containing fold data",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="all_folds_evaluation_summary.json",
        help="Output file for aggregated results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )
    parser.add_argument(
        "--eval-filler",
        action="store_true",
        help="Also evaluate filler data",
    )

    # Individual dataset arguments
    parser.add_argument(
        "--eval-ucl",
        action="store_true",
        help="Evaluate on UCL self-paced reading corpus",
    )
    parser.add_argument(
        "--eval-smith2013",
        action="store_true",
        help="Evaluate on Smith 2013 self-paced reading corpus",
    )
    parser.add_argument(
        "--eval-natural-stories",
        action="store_true",
        help="Evaluate on Natural Stories corpus",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip folds that already have evaluation results",
    )
    parser.add_argument(
        "--no-wt-decoding",
        dest="use_wt_decoding",
        action="store_false",
        help="Disable WT decoding (use standard surprisal computation)",
    )
    parser.set_defaults(use_wt_decoding=True)
    args = parser.parse_args()

    # Find all fold models
    models_dir = Path(args.models_dir)
    fold_models = find_fold_models(models_dir)

    if not fold_models:
        print(f"No fold models found in {models_dir}")
        return

    print(f"Found {len(fold_models)} fold models")

    # Evaluate each fold
    all_results = {}

    for fold_num, checkpoint_path in tqdm(fold_models, desc="Evaluating folds"):
        if args.skip_existing:
            output_file = checkpoint_path / f"fold_{fold_num}_eval.json"
            if output_file.exists():
                print(f"  Skipping fold {fold_num} (already evaluated)")
                with open(output_file, "r") as f:
                    all_results[fold_num] = json.load(f)
                continue

        result = evaluate_fold(
            fold_num,
            checkpoint_path,
            Path(args.config_template),
            Path(args.folds_dir),
            args.eval_filler,
            args.eval_ucl,
            args.eval_smith2013,
            args.eval_natural_stories,
            args.device,
            args.use_wt_decoding
        )

        if result:
            all_results[fold_num] = result

    # Aggregate results
    print(f"\nAggregating results from {len(all_results)} folds...")

    summary = {
        "n_folds": len(all_results),
        "folds": sorted(all_results.keys()),
        "garden_path": {}
    }

    # Aggregate garden-path results
    for construction in ["MVRR", "NPS", "NPZ"]:
        summary["garden_path"][construction] = {}
        for roi in ["roi0", "roi1", "roi2"]:
            actual_values = []
            predicted_values = []

            for fold_num, result in all_results.items():
                if "garden_path_differences" in result:
                    if construction in result["garden_path_differences"]:
                        if roi in result["garden_path_differences"][construction]:
                            data = result["garden_path_differences"][construction][roi]
                            actual_values.append(data["actual"])
                            predicted_values.append(data["predicted"])

            if actual_values:
                import numpy as np
                summary["garden_path"][construction][roi] = {
                    "actual_mean": float(np.mean(actual_values)),
                    "actual_std": float(np.std(actual_values)),
                    "predicted_mean": float(np.mean(predicted_values)),
                    "predicted_std": float(np.std(predicted_values)),
                    "n_folds": len(actual_values),
                    "actual_values": actual_values,
                    "predicted_values": predicted_values
                }

    # Aggregate filler results
    if args.eval_filler:
        filler_values = []
        filler_coefficients = {}

        for fold_num, result in all_results.items():
            if "filler" in result:
                if "delta_llh" in result["filler"] and result["filler"]["delta_llh"] is not None:
                    filler_values.append(result["filler"]["delta_llh"])

                    # Collect coefficients
                    if "coefficients" in result["filler"]:
                        for coef_name, coef_value in result["filler"]["coefficients"].items():
                            if coef_name not in filler_coefficients:
                                filler_coefficients[coef_name] = []
                            filler_coefficients[coef_name].append(coef_value)

        if filler_values:
            import numpy as np
            summary["filler"] = {
                "delta_llh_mean": float(np.mean(filler_values)),
                "delta_llh_std": float(np.std(filler_values)),
                "delta_llh_sem": float(np.std(filler_values) / np.sqrt(len(filler_values))),
                "delta_llh_min": float(np.min(filler_values)),
                "delta_llh_max": float(np.max(filler_values)),
                "n_folds": len(filler_values),
                "delta_llh_values": filler_values
            }

            # Add mean coefficients
            if filler_coefficients:
                summary["filler"]["coefficients_mean"] = {}
                for coef_name, coef_values in filler_coefficients.items():
                    summary["filler"]["coefficients_mean"][coef_name] = float(np.mean(coef_values))


    # Aggregate individual dataset results
    individual_dataset_names = set()
    for fold_num, result in all_results.items():
        if "individual" in result:
            individual_dataset_names.update(result["individual"].keys())

    for dataset_name in individual_dataset_names:
        dataset_values = []
        dataset_coefficients = {}

        for fold_num, result in all_results.items():
            if "individual" in result and dataset_name in result["individual"]:
                if "delta_llh" in result["individual"][dataset_name] and \
                   result["individual"][dataset_name]["delta_llh"] is not None:
                    dataset_values.append(result["individual"][dataset_name]["delta_llh"])

                    # Collect coefficients
                    if "coefficients" in result["individual"][dataset_name]:
                        for coef_name, coef_value in result["individual"][dataset_name]["coefficients"].items():
                            if coef_name not in dataset_coefficients:
                                dataset_coefficients[coef_name] = []
                            dataset_coefficients[coef_name].append(coef_value)

        if dataset_values:
            import numpy as np
            individual_key = f"individual_{dataset_name}"
            summary[individual_key] = {
                "delta_llh_mean": float(np.mean(dataset_values)),
                "delta_llh_std": float(np.std(dataset_values)),
                "delta_llh_sem": float(np.std(dataset_values) / np.sqrt(len(dataset_values))),
                "delta_llh_min": float(np.min(dataset_values)),
                "delta_llh_max": float(np.max(dataset_values)),
                "n_folds": len(dataset_values),
                "delta_llh_values": dataset_values
            }

            # Add mean coefficients
            if dataset_coefficients:
                summary[individual_key]["coefficients_mean"] = {}
                for coef_name, coef_values in dataset_coefficients.items():
                    summary[individual_key]["coefficients_mean"][coef_name] = float(np.mean(coef_values))

    # Save summary
    summary_file = Path(args.summary_output)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_file}")

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    print(f"\nEvaluated {summary['n_folds']} folds")

    if summary["garden_path"]:
        print("\nGarden-Path Effects (mean ± std across folds):")
        for construction in ["MVRR", "NPS", "NPZ"]:
            if construction in summary["garden_path"]:
                print(f"\n{construction}:")
                for roi in ["roi0", "roi1", "roi2"]:
                    if roi in summary["garden_path"][construction]:
                        data = summary["garden_path"][construction][roi]
                        print(f"  {roi}:")
                        print(f"    Actual:    {data['actual_mean']:6.1f} ± {data['actual_std']:5.1f} ms")
                        print(f"    Predicted: {data['predicted_mean']:6.1f} ± {data['predicted_std']:5.1f} ms")

    if summary.get("filler"):
        filler = summary["filler"]
        print(f"\nFiller Data Delta Log-Likelihood:")
        print(f"  Mean: {filler['delta_llh_mean']:.4f} ± {filler['delta_llh_std']:.4f}")
        print(f"  SEM:  {filler['delta_llh_sem']:.4f}")
        print(f"  Range: [{filler['delta_llh_min']:.4f}, {filler['delta_llh_max']:.4f}]")
        print(f"  N folds: {filler['n_folds']}")
        if "coefficients_mean" in filler and "beta_s" in filler["coefficients_mean"]:
            print(f"  Mean surprisal coef: {filler['coefficients_mean']['beta_s']:.4f}")


    # Print individual dataset results
    for key in summary.keys():
        if key.startswith("individual_"):
            corpus_name = key.replace("individual_", "")
            corpus_data = summary[key]
            print(f"\n{corpus_name.upper()} (Individual) Delta Log-Likelihood:")
            print(f"  Mean: {corpus_data['delta_llh_mean']:.4f} ± {corpus_data['delta_llh_std']:.4f}")
            print(f"  SEM:  {corpus_data['delta_llh_sem']:.4f}")
            print(f"  Range: [{corpus_data['delta_llh_min']:.4f}, {corpus_data['delta_llh_max']:.4f}]")
            print(f"  N folds: {corpus_data['n_folds']}")
            if "coefficients_mean" in corpus_data and "beta_s" in corpus_data["coefficients_mean"]:
                print(f"  Mean surprisal coef: {corpus_data['coefficients_mean']['beta_s']:.4f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
