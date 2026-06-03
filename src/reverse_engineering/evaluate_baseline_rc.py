#!/usr/bin/env python3
"""
Evaluate untrained baseline models for Relative Clause comparison.
Mirrors evaluate_fold_rc.py but loads models directly from HuggingFace instead of checkpoints.
"""
import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data.load_data import load_data
from evaluate_fold_rc import (
    compute_training_data_coefficients,
    evaluate_garden_path
)
from models.filler_regression import compute_filler_coefficients
# evaluate_spr_dataset not needed - using corpus_evaluation instead
from models.corpus_evaluation import evaluate_corpus
from models.utils.utils import set_global_seed


def evaluate_single_fold(args):
    """Evaluate a single fold with untrained baseline model."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Load config
    with open(args.config_template, "r") as f:
        config = yaml.safe_load(f)

    # Update paths for this fold (RC version)
    folds_dir = Path(args.folds_dir)
    test_path = folds_dir / f"fold_{args.fold_num}" / "rc_test.csv"
    train_path = folds_dir / f"fold_{args.fold_num}" / "rc_train.csv"

    # Update the correct config keys
    config["training_kwargs"]["data_path"] = str(train_path)  # For training data
    config["training_kwargs"]["eval_data_path"] = str(test_path)  # For evaluation data

    logger.info(f"Using test data: {test_path}")
    logger.info(f"Using train data: {train_path}")

    # Set seed
    set_global_seed(config.get("seed", 42))

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")
    logger.info(f"WT decoding: {'enabled' if args.use_wt_decoding else 'disabled'}")

    # Add WT decoding setting to config
    config["use_wt_decoding"] = args.use_wt_decoding

    # Load UNTRAINED model from HuggingFace (key difference from evaluate_fold_rc.py)
    model_name = config["model"]
    logger.info(f"Loading UNTRAINED {model_name} from HuggingFace...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded: {model_name} (untrained baseline)")

    # Load data
    train_loader, eval_loader, mean_rt_train, std_rt_train = load_data(config, tokenizer)
    logger.info(f"Data loaded. Mean RT: {mean_rt_train}, Std RT: {std_rt_train}")

    results = {
        "fold": args.fold_num,
        "model_type": "baseline",
        "model_name": model_name,
    }

    # Get configuration parameters
    training_kwargs = config.get("training_kwargs", {})
    use_frequency_spillover = training_kwargs.get("use_frequency_spillover", False)
    use_surprisal_spillover = training_kwargs.get("use_surprisal_spillover", False)
    use_length_spillover = training_kwargs.get("use_length_spillover", False)
    use_training_data_coefficients_eval = training_kwargs.get("use_training_data_coefficients_eval", False)
    filler_data_path = training_kwargs.get("filler_data_path",
                                  "../../src/garden_path_cross_validation/folds/fillers_processed.csv")

    # Get masking parameters (defaults to True for evaluation consistency)
    mask_zero_reading_times = training_kwargs.get("mask_zero_reading_times", True)
    mask_zero_freqs = training_kwargs.get("mask_zero_freqs", True)
    mask_last_region = training_kwargs.get("mask_last_region", True)

    # Compute evaluation coefficients
    if use_training_data_coefficients_eval:
        logger.info("Using training data coefficients for evaluation (as configured)")
        eval_coefficients = compute_training_data_coefficients(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            device=device,
            config=config,
            logger=logger,
            use_wt_decoding=args.use_wt_decoding,
        )
    else:
        logger.info("Computing filler regression coefficients...")
        eval_coefficients = compute_filler_coefficients(
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            use_frequency_spillover=use_frequency_spillover,
            use_surprisal_spillover=use_surprisal_spillover,
            use_length_spillover=use_length_spillover,
            use_word_position=True,
            mask_zero_reading_times=mask_zero_reading_times,
            mask_zero_freqs=mask_zero_freqs,
            mask_last_region=mask_last_region,
            use_wt_decoding=args.use_wt_decoding,
            logger=logger,
            step=0,
            data_path=filler_data_path,
        )

    if eval_coefficients:
        logger.info("Successfully computed evaluation coefficients")
        coeffs_dict = {}
        for key in eval_coefficients.keys():
            if eval_coefficients[key] is not None:
                value = (eval_coefficients[key].item()
                        if eval_coefficients[key].numel() == 1
                        else eval_coefficients[key].mean().item())
                coeffs_dict[key] = value
                logger.info(f"  {key}: {value:.4f}")
        results["eval_coefficients"] = coeffs_dict

    # Evaluate RC (using same function as garden-path but with RC data)
    logger.info("\nEvaluating Relative Clause sentences...")
    rc_results = evaluate_garden_path(
        model=model,
        tokenizer=tokenizer,
        eval_loader=eval_loader,
        device=device,
        config=config,
        logger=logger,
        eval_coefficients=eval_coefficients,
    )
    results["garden_path"] = rc_results
    logger.info(f"RC Test Loss: {rc_results['eval_loss']:.4f}")

    # Extract RC differences for easy access
    if "monitoring_data" in rc_results:
        gp_diffs = {}
        # Get phenomena from config (should be ["RelativeClause"] for RC)
        phenomena = training_kwargs.get("selected_phenomena", ["RelativeClause"])
        for construction in phenomena:
            gp_diffs[construction] = {}
            for roi in [0, 1, 2]:
                key_actual = f"gp_{construction}_roi{roi}_diff_actual"
                key_predicted = f"gp_{construction}_roi{roi}_diff_predicted"
                if key_actual in rc_results["monitoring_data"]:
                    gp_diffs[construction][f"roi{roi}"] = {
                        "actual": rc_results["monitoring_data"][key_actual],
                        "predicted": rc_results["monitoring_data"][key_predicted],
                        "n_pairs": rc_results["monitoring_data"].get(
                            f"gp_{construction}_roi{roi}_n_pairs", 0
                        )
                    }
        results["garden_path_differences"] = gp_diffs

    # Note: SPR datasets evaluation is now controlled solely by command-line flags
    # (--eval-natural-stories, --eval-ucl, --eval-smith2013)
    # Config's evaluate_during_training flag is ignored to avoid duplicate evaluation

    # Evaluate filler data if requested
    if args.eval_filler:
        logger.info("\nEvaluating Filler Data...")

        # Use default path from config
        training_kwargs = config.get("training_kwargs", {})
        filler_path = training_kwargs.get(
            "filler_data_path",
            "../../src/garden_path_cross_validation/folds/fillers_processed.csv"
        )

        filler_results = evaluate_corpus(
            model=model,
            tokenizer=tokenizer,
            corpus_name="filler",
            corpus_path=filler_path,
            config=config,
            device=device,
            use_wt_decoding=args.use_wt_decoding,
            logger=logger,
        )

        results["filler"] = filler_results

        # Log summary
        if "delta_llh" in filler_results and filler_results["delta_llh"] is not None:
            logger.info(
                f"Filler Data Delta Log-Likelihood: "
                f"{filler_results['delta_llh']:.6f}"
            )

    # Evaluate individual datasets
    individual_datasets = []

    # Get SPR datasets configuration from config file
    training_kwargs = config.get("training_kwargs", {})
    spr_datasets_config = training_kwargs.get("spr_datasets", [])

    # Create a mapping from dataset name to config
    spr_config_map = {ds["name"]: ds for ds in spr_datasets_config}

    # Check for UCL dataset
    if args.eval_ucl:
        dataset_config = spr_config_map.get("ucl_selfpaced", {
            "name": "ucl_selfpaced",
            "path": "../../data/ucl_selfpaced_processed.csv",
            "max_length": 40
        })
        individual_datasets.append({
            "name": dataset_config["name"],
            "path": dataset_config["path"],
            "max_length": dataset_config.get("max_length", 40)
        })

    # Check for Smith 2013 dataset
    if args.eval_smith2013:
        dataset_config = spr_config_map.get("smith2013", {
            "name": "smith2013",
            "path": "../../data/smith2013_processed.csv",
            "max_length": 65
        })
        individual_datasets.append({
            "name": dataset_config["name"],
            "path": dataset_config["path"],
            "max_length": dataset_config.get("max_length", 65)
        })

    # Check for Natural Stories dataset
    if args.eval_natural_stories:
        dataset_config = spr_config_map.get("natural_stories", {
            "name": "natural_stories",
            "path": "../../data/natural_stories_processed.csv",
            "max_length": 50
        })
        individual_datasets.append({
            "name": dataset_config["name"],
            "path": dataset_config["path"],
            "max_length": dataset_config.get("max_length", 50)
        })

    # Evaluate each individual dataset
    if individual_datasets:
        results["individual"] = {}

        for dataset in individual_datasets:
            logger.info(f"\nEvaluating {dataset['name'].upper()} dataset...")

            # Update max_length in config
            dataset_config = config.copy()
            dataset_config["training_kwargs"] = config.get("training_kwargs", {}).copy()
            dataset_config["training_kwargs"][f"{dataset['name']}_max_length"] = dataset["max_length"]

            dataset_results = evaluate_corpus(
                model=model,
                tokenizer=tokenizer,
                corpus_name=dataset["name"],
                corpus_path=dataset["path"],
                config=dataset_config,
                device=device,
                use_wt_decoding=args.use_wt_decoding,
                logger=logger,
            )

            results["individual"][dataset["name"]] = dataset_results

            # Log summary
            if "delta_llh" in dataset_results and dataset_results["delta_llh"] is not None:
                logger.info(
                    f"{dataset['name'].upper()} Delta Log-Likelihood: "
                    f"{dataset_results['delta_llh']:.6f}"
                )

    # Save results
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = Path(args.output_dir) / f"baseline_fold_{args.fold_num}_eval.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"BASELINE Fold {args.fold_num} Evaluation Summary")
    print(f"{'='*60}")

    if "garden_path_differences" in results:
        print("\nRelative Clause Effects (Amb - Unamb):")
        # Get phenomena from config (should be ["RelativeClause"] for RC)
        phenomena_to_print = config["training_kwargs"].get(
            "selected_phenomena", ["RelativeClause"]
        )
        for construction in phenomena_to_print:
            if construction in results["garden_path_differences"]:
                print(f"\n{construction}:")
                for roi_key in ["roi0", "roi1", "roi2"]:
                    if roi_key in results["garden_path_differences"][construction]:
                        data = results["garden_path_differences"][construction][roi_key]
                        print(
                            f"  {roi_key}: actual={data['actual']:.1f}ms, "
                            f"predicted={data['predicted']:.1f}ms"
                        )

    if "individual" in results:
        print("\nIndividual Datasets:")
        for dataset_name, dataset_data in results["individual"].items():
            if "delta_llh" in dataset_data and dataset_data["delta_llh"] is not None:
                print(
                    f"  {dataset_name.upper()}: Delta LLH = "
                    f"{dataset_data['delta_llh']:.6f} "
                    f"(n={dataset_data.get('n_words', 0)} words)"
                )

    if "filler" in results:
        print("\nFiller Data:")
        if "delta_llh" in results["filler"] and results["filler"]["delta_llh"] is not None:
            print(f"  Delta Log-Likelihood: {results['filler']['delta_llh']:.6f}")
            print(f"  N words: {results['filler'].get('n_words', 0)}")
            if "coefficients" in results["filler"]:
                print(
                    f"  Surprisal coefficient: "
                    f"{results['filler']['coefficients'].get('beta_s', 0):.4f}"
                )

    print(f"{'='*60}\n")

    return results


def find_valid_folds(folds_dir: Path) -> List[int]:
    """Find all valid fold directories."""
    valid_folds = []

    for fold_dir in sorted(folds_dir.glob("fold_*")):
        if fold_dir.is_dir():
            # Extract fold number
            try:
                fold_num = int(fold_dir.name.split("_")[1])
                # Check if required files exist (RC version)
                train_path = fold_dir / "rc_train.csv"
                test_path = fold_dir / "rc_test.csv"
                if train_path.exists() and test_path.exists():
                    valid_folds.append(fold_num)
            except (IndexError, ValueError):
                continue

    return sorted(valid_folds)


def evaluate_all_folds(args):
    """Evaluate all folds with baseline models.

    Note: For baseline models, only RC data varies across folds.
    Other datasets (SPR) produce identical results for all folds since the
    model is randomly initialized each time.
    """

    # Find all valid folds
    folds_dir = Path(args.folds_dir)
    valid_folds = find_valid_folds(folds_dir)

    if not valid_folds:
        print(f"No valid folds found in {folds_dir}")
        return

    # Check which ones match trained model folds if summary file exists
    if args.trained_summary:
        trained_summary_path = Path(args.trained_summary)
        if trained_summary_path.exists():
            with open(trained_summary_path, "r") as f:
                trained_data = json.load(f)
                trained_folds = trained_data.get("folds", [])
                # Only evaluate folds that have trained models
                valid_folds = [f for f in valid_folds if f in trained_folds]
                print(f"Limiting to {len(valid_folds)} folds that have trained models")

    print(f"Found {len(valid_folds)} valid folds to evaluate")

    # Determine if we need to evaluate non-RC datasets
    # These only need to be evaluated once since baseline model is always the same
    evaluate_other_datasets = (
        args.eval_filler or
        args.eval_ucl or args.eval_smith2013 or args.eval_natural_stories
    )

    if evaluate_other_datasets:
        print("\nNote: Non-RC datasets will be evaluated only once")
        print("(baseline model produces identical results for all folds)")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each fold
    all_results = {}
    shared_other_results = None  # Store results from non-RC datasets

    for idx, fold_num in enumerate(tqdm(valid_folds, desc="Evaluating baseline")):
        # Check if already evaluated
        if args.skip_existing:
            output_file = output_dir / f"baseline_fold_{fold_num}_eval.json"
            if output_file.exists():
                print(f"\n  Skipping fold {fold_num} (already evaluated)")
                with open(output_file, "r") as f:
                    all_results[fold_num] = json.load(f)
                continue

        print(f"\n  Evaluating fold {fold_num}...")

        # For non-RC datasets, only evaluate on first fold
        # and reuse results for subsequent folds
        evaluate_others_this_fold = evaluate_other_datasets and (
            idx == 0 or shared_other_results is None
        )

        # Build arguments for single fold evaluation
        single_args = argparse.Namespace(
            fold_num=fold_num,
            config_template=args.config_template,
            folds_dir=args.folds_dir,
            output_dir=args.output_dir,
            output_file=None,
            device=args.device,
            # Only evaluate other datasets on first fold
            eval_filler=args.eval_filler and evaluate_others_this_fold,
            eval_ucl=args.eval_ucl and evaluate_others_this_fold,
            eval_smith2013=args.eval_smith2013 and evaluate_others_this_fold,
            eval_natural_stories=args.eval_natural_stories and evaluate_others_this_fold,
            use_wt_decoding=args.use_wt_decoding
        )

        try:
            result = evaluate_single_fold(single_args)
            if result:
                # Store shared results from first evaluation
                if evaluate_others_this_fold and shared_other_results is None:
                    shared_other_results = {}
                    if "filler" in result:
                        shared_other_results["filler"] = result["filler"]
                    if "individual" in result:
                        shared_other_results["individual"] = result["individual"]
                    print("  Stored non-RC results for reuse")

                # For subsequent folds, add the shared results
                if not evaluate_others_this_fold and shared_other_results:
                    for key, value in shared_other_results.items():
                        result[key] = value

                all_results[fold_num] = result
        except Exception as e:
            print(f"  Failed to evaluate fold {fold_num}: {e}")
            continue

    # Aggregate results
    print(f"\nAggregating results from {len(all_results)} folds...")

    summary = {
        "model_type": "baseline",
        "n_folds": len(all_results),
        "folds": sorted(all_results.keys()),
        "relative_clause": {}  # Changed from "garden_path"
    }

    # Aggregate RC results
    # Get phenomena from first result to know what to aggregate
    training_kwargs = {}
    if all_results:
        first_result = list(all_results.values())[0]
        # We need to reload config to get phenomena list
        with open(args.config_template, "r") as f:
            config = yaml.safe_load(f)
            training_kwargs = config.get("training_kwargs", {})

    phenomena = training_kwargs.get("selected_phenomena", ["RelativeClause"])

    for construction in phenomena:
        summary["relative_clause"][construction] = {}
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
                summary["relative_clause"][construction][roi] = {
                    "actual_mean": float(np.mean(actual_values)),
                    "actual_std": float(np.std(actual_values)),
                    "predicted_mean": float(np.mean(predicted_values)),
                    "predicted_std": float(np.std(predicted_values)),
                    "n_folds": len(actual_values),
                    "actual_values": actual_values,
                    "predicted_values": predicted_values
                }

    # Aggregate SPR dataset results (check if any SPR results exist)
    # Collect all SPR dataset names from results
    spr_dataset_names = set()
    for fold_num, result in all_results.items():
        for key in result.keys():
            # SPR dataset results are stored with dataset name as key
            if key in ["ucl_selfpaced", "smith2013"] or "_delta_llh" in str(result.get(key, {})):
                spr_dataset_names.add(key)

    # Aggregate results for each SPR dataset
    for dataset_name in spr_dataset_names:
        delta_llh_values = []

        for fold_num, result in all_results.items():
            if dataset_name in result:
                # Handle both key formats
                delta_llh = result[dataset_name].get(f"{dataset_name}_delta_llh",
                           result[dataset_name].get("delta_log_likelihood", 0))
                if delta_llh:
                    delta_llh_values.append(delta_llh)

        if delta_llh_values:
            summary[dataset_name] = {
                "delta_llh_mean": float(np.mean(delta_llh_values)),
                "delta_llh_std": float(np.std(delta_llh_values)),
                "delta_llh_sem": float(np.std(delta_llh_values) / np.sqrt(len(delta_llh_values))),
                "delta_llh_min": float(np.min(delta_llh_values)),
                "delta_llh_max": float(np.max(delta_llh_values)),
                "n_folds": len(delta_llh_values),
                "delta_llh_values": delta_llh_values
            }

    # Aggregate individual dataset results
    individual_dataset_names = set()
    for fold_num, result in all_results.items():
        if "individual" in result:
            individual_dataset_names.update(result["individual"].keys())

    for dataset_name in individual_dataset_names:
        dataset_values = []
        dataset_n_words = []
        dataset_coefficients = {}

        for fold_num, result in all_results.items():
            if "individual" in result and dataset_name in result["individual"]:
                dataset_data = result["individual"][dataset_name]
                if "delta_llh" in dataset_data and dataset_data["delta_llh"] is not None:
                    dataset_values.append(dataset_data["delta_llh"])
                    dataset_n_words.append(dataset_data.get("n_words", 0))

                    # Collect coefficients
                    if "coefficients" in dataset_data:
                        for coef_name, coef_value in dataset_data["coefficients"].items():
                            if coef_name not in dataset_coefficients:
                                dataset_coefficients[coef_name] = []
                            dataset_coefficients[coef_name].append(coef_value)

        if dataset_values:
            individual_key = f"individual_{dataset_name}"
            summary[individual_key] = {
                "delta_llh_mean": float(np.mean(dataset_values)),
                "delta_llh_std": float(np.std(dataset_values)),
                "delta_llh_sem": float(np.std(dataset_values) / np.sqrt(len(dataset_values))),
                "delta_llh_min": float(np.min(dataset_values)),
                "delta_llh_max": float(np.max(dataset_values)),
                "n_folds": len(dataset_values),
                "delta_llh_values": dataset_values,
                "n_words": int(np.mean(dataset_n_words)) if dataset_n_words else 0,
            }

            # Add mean coefficients
            if dataset_coefficients:
                summary[individual_key]["coefficients_mean"] = {}
                for coef_name, coef_values in dataset_coefficients.items():
                    summary[individual_key]["coefficients_mean"][coef_name] = float(np.mean(coef_values))

    # Aggregate filler results
    if args.eval_filler:
        filler_values = []
        filler_n_words = []
        filler_coefficients = {}

        for fold_num, result in all_results.items():
            if "filler" in result:
                filler_data = result["filler"]
                if "delta_llh" in filler_data and filler_data["delta_llh"] is not None:
                    filler_values.append(filler_data["delta_llh"])
                    filler_n_words.append(filler_data.get("n_words", 0))

                    # Collect coefficients
                    if "coefficients" in filler_data:
                        for coef_name, coef_value in filler_data["coefficients"].items():
                            if coef_name not in filler_coefficients:
                                filler_coefficients[coef_name] = []
                            filler_coefficients[coef_name].append(coef_value)

        if filler_values:
            summary["filler"] = {
                "delta_llh_mean": float(np.mean(filler_values)),
                "delta_llh_std": float(np.std(filler_values)),
                "delta_llh_sem": float(np.std(filler_values) / np.sqrt(len(filler_values))),
                "delta_llh_min": float(np.min(filler_values)),
                "delta_llh_max": float(np.max(filler_values)),
                "n_folds": len(filler_values),
                "delta_llh_values": filler_values,
                "n_words": int(np.mean(filler_n_words)) if filler_n_words else 0,
            }

            # Add mean coefficients
            if filler_coefficients:
                summary["filler"]["coefficients_mean"] = {}
                for coef_name, coef_values in filler_coefficients.items():
                    summary["filler"]["coefficients_mean"][coef_name] = float(np.mean(coef_values))

    # Save summary
    summary_file = output_dir / args.summary_output
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_file}")

    # Print summary
    print("\n" + "="*60)
    print("BASELINE EVALUATION SUMMARY (RELATIVE CLAUSE)")
    print("="*60)

    print(f"\nEvaluated {summary['n_folds']} folds")

    if summary["relative_clause"]:
        print("\nRelative Clause Effects (mean ± std across folds):")
        for construction in phenomena:
            if construction in summary["relative_clause"]:
                print(f"\n{construction}:")
                for roi in ["roi0", "roi1", "roi2"]:
                    if roi in summary["relative_clause"][construction]:
                        data = summary["relative_clause"][construction][roi]
                        print(f"  {roi}:")
                        print(f"    Human:    {data['actual_mean']:6.1f} ± {data['actual_std']:5.1f} ms")
                        print(f"    Baseline: {data['predicted_mean']:6.1f} ± {data['predicted_std']:5.1f} ms")

    # Print individual dataset results
    for key in summary.keys():
        if key.startswith("individual_"):
            corpus_name = key.replace("individual_", "")
            corpus_data = summary[key]
            print(f"\n{corpus_name.upper()} (Individual) Delta Log-Likelihood:")
            print(f"  Mean: {corpus_data['delta_llh_mean']:.6f} ± {corpus_data['delta_llh_std']:.6f}")
            print(f"  SEM:  {corpus_data['delta_llh_sem']:.6f}")
            print(f"  Range: [{corpus_data['delta_llh_min']:.6f}, {corpus_data['delta_llh_max']:.6f}]")
            print(f"  N words: {corpus_data['n_words']}")
            print(f"  N folds: {corpus_data['n_folds']}")
            if "coefficients_mean" in corpus_data and "beta_s" in corpus_data["coefficients_mean"]:
                print(f"  Mean surprisal coef: {corpus_data['coefficients_mean']['beta_s']:.4f}")

    if "filler" in summary:
        filler = summary["filler"]
        print(f"\n  Filler Data ({filler['n_folds']} folds):")
        print(f"    Mean: {filler['delta_llh_mean']:.6f} ± {filler['delta_llh_std']:.6f}")
        print(f"    SEM:  {filler['delta_llh_sem']:.6f}")
        print(f"    Range: [{filler['delta_llh_min']:.6f}, {filler['delta_llh_max']:.6f}]")
        print(f"    N words: {filler['n_words']}")
        if "coefficients_mean" in filler and "beta_s" in filler["coefficients_mean"]:
            print(f"    Mean surprisal coef: {filler['coefficients_mean']['beta_s']:.4f}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models for Relative Clause")
    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")

    # Single fold evaluation
    single_parser = subparsers.add_parser("single", help="Evaluate a single fold")
    single_parser.add_argument(
        "--fold-num",
        type=int,
        required=True,
        help="Fold number to evaluate",
    )
    single_parser.add_argument(
        "--config-template",
        type=str,
        default="configs/relative_clause_template.yaml",
        help="Path to config template",
    )
    single_parser.add_argument(
        "--folds-dir",
        type=str,
        default="../../src/relative_clause_cross_validation/folds_filtered",
        help="Directory containing fold data",
    )
    single_parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_evaluation_rc",
        help="Output directory for results",
    )
    single_parser.add_argument(
        "--output-file",
        type=str,
        help="Output JSON file for results (overrides output-dir)",
    )
    single_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )
    single_parser.add_argument(
        "--eval-filler",
        action="store_true",
        help="Also evaluate on filler data",
    )

    # Individual dataset evaluation arguments
    single_parser.add_argument(
        "--eval-ucl",
        action="store_true",
        help="Evaluate on UCL self-paced reading corpus",
    )
    single_parser.add_argument(
        "--eval-smith2013",
        action="store_true",
        help="Evaluate on Smith 2013 self-paced reading corpus",
    )
    single_parser.add_argument(
        "--eval-natural-stories",
        action="store_true",
        help="Evaluate on Natural Stories corpus",
    )
    single_parser.add_argument(
        "--no-wt-decoding",
        dest="use_wt_decoding",
        action="store_false",
        help="Disable WT decoding (use standard surprisal computation)",
    )
    single_parser.set_defaults(use_wt_decoding=True)

    # All folds evaluation
    all_parser = subparsers.add_parser("all", help="Evaluate all folds")
    all_parser.add_argument(
        "--config-template",
        type=str,
        default="configs/relative_clause_template.yaml",
        help="Path to config template",
    )
    all_parser.add_argument(
        "--folds-dir",
        type=str,
        default="../../src/relative_clause_cross_validation/folds_filtered",
        help="Directory containing fold data",
    )
    all_parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_evaluation_rc",
        help="Output directory for results",
    )
    all_parser.add_argument(
        "--summary-output",
        type=str,
        default="baseline_rc_summary.json",
        help="Output file name for aggregated results",
    )
    all_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )
    all_parser.add_argument(
        "--eval-filler",
        action="store_true",
        help="Also evaluate on filler data",
    )

    # Individual dataset evaluation arguments
    all_parser.add_argument(
        "--eval-ucl",
        action="store_true",
        help="Evaluate on UCL self-paced reading corpus",
    )
    all_parser.add_argument(
        "--eval-smith2013",
        action="store_true",
        help="Evaluate on Smith 2013 self-paced reading corpus",
    )
    all_parser.add_argument(
        "--eval-natural-stories",
        action="store_true",
        help="Evaluate on Natural Stories corpus",
    )
    all_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip folds that already have evaluation results",
    )
    all_parser.add_argument(
        "--trained-summary",
        type=str,
        help="Path to trained model summary to limit evaluation to matching folds",
    )
    all_parser.add_argument(
        "--no-wt-decoding",
        dest="use_wt_decoding",
        action="store_false",
        help="Disable WT decoding (use standard surprisal computation)",
    )
    all_parser.set_defaults(use_wt_decoding=True)

    # Parse arguments
    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    # Execute based on mode
    if args.mode == "single":
        evaluate_single_fold(args)
    elif args.mode == "all":
        evaluate_all_folds(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
