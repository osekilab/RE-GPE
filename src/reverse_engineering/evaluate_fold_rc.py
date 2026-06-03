#!/usr/bin/env python3
"""
Relative Clause evaluation script.
Evaluates RC fold models on Relative Clause test data and SPR datasets.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from tqdm import tqdm

from data.load_data import load_data
from models.filler_regression import compute_filler_coefficients
from models.loss import compute_loss
from models.loss_utils import solve_beta
# evaluate_spr_dataset not needed - using corpus_evaluation instead
from models.corpus_evaluation import evaluate_corpus
from models.surprisal import compute_surprisal, compute_surprisal_with_wt_decoding
from models.utils.utils import load_model_and_tokenizer, set_global_seed


def evaluate_garden_path(
    model,
    tokenizer,
    eval_loader,
    device,
    config,
    logger,
    eval_coefficients,
) -> Dict:
    """Evaluate garden-path exactly as in trainer.py."""
    eval_losses = []
    all_monitoring_data = []

    # Get configuration parameters
    training_kwargs = config.get("training_kwargs", {})
    loss_type = training_kwargs.get("loss_type", "mse")
    reduction = training_kwargs.get("reduction", "mean")
    use_frequency_spillover = training_kwargs.get("use_frequency_spillover", False)
    use_surprisal_spillover = training_kwargs.get("use_surprisal_spillover", False)
    use_length_spillover = training_kwargs.get("use_length_spillover", False)
    selected_phenomena = training_kwargs.get("selected_phenomena", ["RelativeClause"])
    # Use eval_phenomena if specified, otherwise fall back to selected_phenomena
    eval_phenomena = training_kwargs.get("eval_phenomena", selected_phenomena)

    logger.info("Evaluating Relative Clause with eval coefficients...")

    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Eval RC", leave=False):
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            # Use pre-computed evaluation coefficients
            loss_results = compute_loss(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                loss_type=loss_type,
                reduction=reduction,
                mask_zero_reading_times=True,
                mask_zero_freqs=True,
                kl_weight=0.0,  # No KL during eval
                model_ref=None,
                # NO ROI exclusion during evaluation
                exclude_roi_from_regression=False,
                # Use pre-computed evaluation coefficients
                use_global_coefficients=eval_coefficients is not None,
                global_coefficients=eval_coefficients,
                # Spillover features
                use_frequency_spillover=use_frequency_spillover,
                use_surprisal_spillover=use_surprisal_spillover,
                use_length_spillover=use_length_spillover,
            )

            # Unpack results
            eval_loss, _, _, _, _, _, monitoring_data = loss_results
            eval_losses.append(eval_loss.item())

            # Collect monitoring data
            if monitoring_data:
                all_monitoring_data.append(monitoring_data)

    # Aggregate monitoring data across batches
    aggregated_monitoring = {}
    if all_monitoring_data and eval_phenomena:
        for construction in eval_phenomena:
            for roi in [0, 1, 2]:
                key_actual = f"gp_{construction}_roi{roi}_diff_actual"
                key_predicted = f"gp_{construction}_roi{roi}_diff_predicted"
                key_n_pairs = f"gp_{construction}_roi{roi}_n_pairs"

                actual_diffs = []
                predicted_diffs = []
                n_pairs_total = 0

                for data in all_monitoring_data:
                    if key_actual in data:
                        actual_diffs.append(data[key_actual])
                        predicted_diffs.append(data[key_predicted])
                        n_pairs_total += data.get(key_n_pairs, 0)

                if actual_diffs:
                    aggregated_monitoring[key_actual] = np.mean(actual_diffs)
                    aggregated_monitoring[key_predicted] = np.mean(predicted_diffs)
                    aggregated_monitoring[key_n_pairs] = n_pairs_total

                    logger.info(
                        f"  {construction} ROI{roi}: actual={aggregated_monitoring[key_actual]:.1f}ms, "
                        f"predicted={aggregated_monitoring[key_predicted]:.1f}ms (n={n_pairs_total})"
                    )

    return {
        "eval_loss": np.mean(eval_losses),
        "eval_loss_std": np.std(eval_losses),
        "monitoring_data": aggregated_monitoring
    }


def compute_training_data_coefficients(
    model,
    tokenizer,
    train_loader,
    device,
    config,
    logger,
    use_wt_decoding: bool = False,
) -> Optional[Dict[str, torch.Tensor]]:
    """Compute regression coefficients using all training data (excluding ROI)."""
    try:
        logger.info("Computing regression coefficients from training data...")

        training_kwargs = config.get("training_kwargs", {})
        use_frequency_spillover = training_kwargs.get("use_frequency_spillover", False)
        use_surprisal_spillover = training_kwargs.get("use_surprisal_spillover", False)
        use_length_spillover = training_kwargs.get("use_length_spillover", False)
        mask_zero_freqs = training_kwargs.get("mask_zero_freqs", True)
        mask_last_region = training_kwargs.get("mask_last_region", True)
        exclude_roi_from_regression = training_kwargs.get("exclude_roi_from_regression", True)
        roi_exclusion_values = training_kwargs.get("roi_exclusion_values", [0, 1, 2])

        # Collect features and targets from all training batches
        all_surprisals = []
        all_features_dict = {}
        all_targets = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Computing coefficients", leave=False):
                batch = {k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                # Get word_ids for aggregation
                word_ids = batch.get("word_ids", None)
                if word_ids is None:
                    continue

                # Compute surprisal at subword level (with WT decoding if enabled)
                if use_wt_decoding:
                    surprisal_subword, _, _ = compute_surprisal_with_wt_decoding(batch, model, tokenizer, word_ids)
                else:
                    surprisal_subword, _, _ = compute_surprisal(batch, model, tokenizer, word_ids)

                # Aggregate surprisal to word level
                mask_word_ids = word_ids != -1
                masked_surprisal_subword = surprisal_subword * mask_word_ids
                masked_word_ids = word_ids * mask_word_ids

                surprisal_word = torch.zeros_like(surprisal_subword)
                surprisal_word.scatter_add_(1, masked_word_ids, masked_surprisal_subword)

                # Get features and masks
                reading_times = batch.get("reading_times", torch.zeros_like(batch["input_ids"]))
                valid_mask = (word_ids != -1) & (reading_times != -1) & (reading_times > 0)

                # Apply zero frequency mask if present
                if mask_zero_freqs and "zero_freq_mask" in batch:
                    zero_freq_mask = batch["zero_freq_mask"]
                    valid_mask = valid_mask & (zero_freq_mask != 0)
                    if use_frequency_spillover:
                        if "zero_freq_mask_prev1" in batch:
                            valid_mask = valid_mask & (batch["zero_freq_mask_prev1"] != 0)
                        if "zero_freq_mask_prev2" in batch:
                            valid_mask = valid_mask & (batch["zero_freq_mask_prev2"] != 0)

                # Apply ROI exclusion if present
                if "roi" in batch and exclude_roi_from_regression:
                    roi_mask = torch.zeros_like(valid_mask)
                    for roi_val in roi_exclusion_values:
                        roi_mask = roi_mask | (batch["roi"] == roi_val)
                    valid_mask = valid_mask & ~roi_mask

                # Process each sentence in the batch
                for idx in range(batch["input_ids"].shape[0]):
                    sentence_mask = valid_mask[idx].clone()

                    # Mask last region if requested
                    if mask_last_region:
                        valid_positions = torch.where(sentence_mask)[0]
                        if len(valid_positions) > 0:
                            last_position = valid_positions[-1]
                            sentence_mask[last_position] = False

                    if sentence_mask.any():
                        # Collect surprisal
                        all_surprisals.append(surprisal_word[idx][sentence_mask])
                        all_targets.append(reading_times[idx][sentence_mask])

                        # Collect other features
                        if "word_lengths" in batch:
                            all_features_dict.setdefault("length", []).append(
                                batch["word_lengths"][idx][sentence_mask]
                            )
                        if "log_unigram_frequencies" in batch:
                            all_features_dict.setdefault("freq", []).append(
                                batch["log_unigram_frequencies"][idx][sentence_mask]
                            )
                        if "word_positions" in batch:
                            all_features_dict.setdefault("position", []).append(
                                batch["word_positions"][idx][sentence_mask]
                            )

                        # Frequency spillover features
                        if use_frequency_spillover:
                            if "log_unigram_frequencies_prev1" in batch:
                                all_features_dict.setdefault("freq_prev1", []).append(
                                    batch["log_unigram_frequencies_prev1"][idx][sentence_mask]
                                )
                            if "log_unigram_frequencies_prev2" in batch:
                                all_features_dict.setdefault("freq_prev2", []).append(
                                    batch["log_unigram_frequencies_prev2"][idx][sentence_mask]
                                )

                        # Surprisal spillover
                        if use_surprisal_spillover:
                            surprisal_prev1 = torch.zeros_like(surprisal_word[idx])
                            surprisal_prev2 = torch.zeros_like(surprisal_word[idx])
                            surprisal_prev1[1:] = surprisal_word[idx][:-1]
                            surprisal_prev2[2:] = surprisal_word[idx][:-2]
                            all_features_dict.setdefault("surprisal_prev1", []).append(
                                surprisal_prev1[sentence_mask]
                            )
                            all_features_dict.setdefault("surprisal_prev2", []).append(
                                surprisal_prev2[sentence_mask]
                            )

                        # Length spillover features
                        if use_length_spillover:
                            if "word_lengths_prev1" in batch:
                                all_features_dict.setdefault("length_prev1", []).append(
                                    batch["word_lengths_prev1"][idx][sentence_mask]
                                )
                            if "word_lengths_prev2" in batch:
                                all_features_dict.setdefault("length_prev2", []).append(
                                    batch["word_lengths_prev2"][idx][sentence_mask]
                                )

        # Concatenate all collected data
        if all_surprisals:
            all_surprisals = torch.cat(all_surprisals)
            all_targets = torch.cat(all_targets)

            # Build design matrix
            features_list = [all_surprisals]
            feature_names = ["beta_s"]

            # Add other features
            if "length" in all_features_dict:
                features_list.append(torch.cat(all_features_dict["length"]))
                feature_names.append("beta_l")
            if "freq" in all_features_dict:
                features_list.append(torch.cat(all_features_dict["freq"]))
                feature_names.append("beta_f")
            if "position" in all_features_dict:
                features_list.append(torch.cat(all_features_dict["position"]))
                feature_names.append("beta_pos")

            # Add spillover features
            if "freq_prev1" in all_features_dict:
                features_list.append(torch.cat(all_features_dict["freq_prev1"]))
                feature_names.append("beta_f_prev1")
            if "freq_prev2" in all_features_dict:
                features_list.append(torch.cat(all_features_dict["freq_prev2"]))
                feature_names.append("beta_f_prev2")

            if "surprisal_prev1" in all_features_dict:
                features_list.append(torch.cat(all_features_dict["surprisal_prev1"]))
                feature_names.append("beta_s_prev1")
            if "surprisal_prev2" in all_features_dict:
                features_list.append(torch.cat(all_features_dict["surprisal_prev2"]))
                feature_names.append("beta_s_prev2")

            if "length_prev1" in all_features_dict:
                features_list.append(torch.cat(all_features_dict["length_prev1"]))
                feature_names.append("beta_l_prev1")
            if "length_prev2" in all_features_dict:
                features_list.append(torch.cat(all_features_dict["length_prev2"]))
                feature_names.append("beta_l_prev2")

            # Stack features and add intercept
            X = torch.stack(features_list, dim=1)
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim=1)
            feature_names = ["beta_0"] + feature_names

            # Solve for coefficients
            betas = solve_beta(X, all_targets)

            # Create coefficient dictionary
            coefficients = {}
            for i, name in enumerate(feature_names):
                coefficients[name] = betas[i].unsqueeze(0)

            logger.info("Successfully computed training data coefficients")
            for key in coefficients.keys():
                if coefficients[key] is not None:
                    value = coefficients[key].item() if coefficients[key].numel() == 1 else coefficients[key].mean().item()
                    logger.info(f"  {key}: {value:.4f}")

            return coefficients
        else:
            logger.warning("No valid training data for coefficient computation")
            return None

    except Exception as e:
        logger.warning(f"Training data coefficient computation failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(description="Fold evaluation matching trainer.py exactly")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--fold-num",
        type=int,
        required=True,
        help="Fold number being evaluated",
    )
    parser.add_argument(
        "--config-template",
        type=str,
        default="configs/relative_clause_template.yaml",
        help="Path to RC config template",
    )
    parser.add_argument(
        "--folds-dir",
        type=str,
        default="../../src/relative_clause_cross_validation/folds_filtered",
        help="Directory containing RC fold data",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output JSON file for results",
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
        help="Evaluate on filler data with delta log-likelihood",
    )

    # Individual dataset evaluation arguments
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
        "--no-wt-decoding",
        dest="use_wt_decoding",
        action="store_false",
        help="Disable WT decoding (use standard surprisal computation)",
    )
    parser.set_defaults(use_wt_decoding=True)
    args = parser.parse_args()

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

    # Update the correct config keys that load_garden_path.py expects
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

    # Load model and tokenizer
    model_name = config["model"]
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load checkpoint weights
    checkpoint_path = Path(args.checkpoint)
    checkpoint_file = checkpoint_path / "model.safetensors"
    if checkpoint_file.exists():
        logger.info(f"Loading model weights from {checkpoint_file}")
        state_dict = load_file(str(checkpoint_file))

        # Handle GPT2's tied weights
        if "lm_head.weight" not in state_dict and "transformer.wte.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"No model checkpoint found at {checkpoint_file}")

    logger.info(f"Model loaded: {model_name}")

    # Load data
    train_loader, eval_loader, mean_rt_train, std_rt_train = load_data(config, tokenizer)
    logger.info(f"Data loaded. Mean RT: {mean_rt_train}, Std RT: {std_rt_train}")

    results = {
        "fold": args.fold_num,
        "checkpoint": str(checkpoint_path),
    }

    # Get configuration parameters
    training_kwargs = config.get("training_kwargs", {})
    use_frequency_spillover = training_kwargs.get("use_frequency_spillover", False)
    use_surprisal_spillover = training_kwargs.get("use_surprisal_spillover", False)
    use_length_spillover = training_kwargs.get("use_length_spillover", False)
    use_training_data_coefficients_eval = training_kwargs.get("use_training_data_coefficients_eval", False)
    filler_data_path = config.get("filler_data_path",
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

    # Evaluate garden-path
    logger.info("\nEvaluating RC test sentences...")
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

    # Extract RC/garden-path differences for easy access
    if "monitoring_data" in rc_results:
        gp_diffs = {}
        # Get phenomena from config (RelativeClause for RC, or MVRR/NPS/NPZ for garden-path)
        phenomena = config["training_kwargs"].get("selected_phenomena", ["RelativeClause"])
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

    # Handle individual dataset evaluations
    individual_datasets = []

    # Get SPR datasets configuration from config file
    training_kwargs = config.get("training_kwargs", {})
    spr_datasets_config = training_kwargs.get("spr_datasets", [])

    # Create a mapping from dataset name to config
    spr_config_map = {ds["name"]: ds for ds in spr_datasets_config}

    # Check for individual SPR datasets
    if args.eval_ucl:
        dataset_config = spr_config_map.get("ucl_selfpaced", {
            "name": "ucl_selfpaced",
            "path": "../../data/ucl_selfpaced_processed.csv",
            "max_length": 40
        })
        individual_datasets.append({
            "name": dataset_config["name"],
            "path": dataset_config["path"],
            "max_length": dataset_config.get("max_length", 40),
            "type": "spr"
        })

    if args.eval_smith2013:
        dataset_config = spr_config_map.get("smith2013", {
            "name": "smith2013",
            "path": "../../data/smith2013_processed.csv",
            "max_length": 65
        })
        individual_datasets.append({
            "name": dataset_config["name"],
            "path": dataset_config["path"],
            "max_length": dataset_config.get("max_length", 65),
            "type": "spr"
        })

    if args.eval_natural_stories:
        dataset_config = spr_config_map.get("natural_stories", {
            "name": "natural_stories",
            "path": "../../data/natural_stories_processed.csv",
            "max_length": 50
        })
        individual_datasets.append({
            "name": dataset_config["name"],
            "path": dataset_config["path"],
            "max_length": dataset_config.get("max_length", 50),
            "type": "spr"
        })

    # Evaluate individual datasets
    if individual_datasets:
        logger.info("\nEvaluating Individual Datasets...")

        if "individual" not in results:
            results["individual"] = {}

        for dataset in individual_datasets:
            dataset_name = dataset["name"]
            dataset_path = dataset["path"]
            dataset_max_length = dataset.get("max_length", 50)
            dataset_type = dataset.get("type", "corpus")

            logger.info(f"\nEvaluating {dataset_name.upper()} ({dataset_type})...")

            # Update config with dataset-specific max_length
            temp_config = config.copy()
            temp_config["training_kwargs"] = config.get("training_kwargs", {}).copy()
            temp_config["training_kwargs"][f"{dataset_name}_max_length"] = dataset_max_length
            # Ensure mask_last_region is set (matching evaluate_fold.py behavior)
            temp_config["training_kwargs"]["mask_last_region"] = mask_last_region

            # Evaluate the corpus
            corpus_results = evaluate_corpus(
                model=model,
                tokenizer=tokenizer,
                corpus_name=dataset_name,
                corpus_path=dataset_path,
                config=temp_config,
                device=device,
                use_wt_decoding=args.use_wt_decoding,
                logger=logger,
            )

            results["individual"][dataset_name] = corpus_results

            # Log summary
            if "delta_llh" in corpus_results and corpus_results["delta_llh"] is not None:
                logger.info(
                    f"{dataset_name.upper()} Delta Log-Likelihood: "
                    f"{corpus_results['delta_llh']:.6f}"
                )

    # Save results
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = checkpoint_path / f"fold_{args.fold_num}_eval.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Fold {args.fold_num} Evaluation Summary")
    print(f"{'='*60}")

    if "garden_path_differences" in results:
        print("\nRelative Clause Effects (Amb - Unamb):")
        # Get phenomena from config (should be ["RelativeClause"] for RC experiments)
        phenomena_to_print = config["training_kwargs"].get("selected_phenomena", ["RelativeClause"])
        for construction in phenomena_to_print:
            if construction in results["garden_path_differences"]:
                print(f"\n{construction}:")
                for roi_key in ["roi0", "roi1", "roi2"]:
                    if roi_key in results["garden_path_differences"][construction]:
                        data = results["garden_path_differences"][construction][roi_key]
                        print(f"  {roi_key}: actual={data['actual']:.1f}ms, "
                              f"predicted={data['predicted']:.1f}ms")

    if "filler" in results:
        print(f"\nFiller Data:")
        if "delta_llh" in results["filler"] and results["filler"]["delta_llh"] is not None:
            print(f"  Delta LLH = {results['filler']['delta_llh']:.6f}")

    if "individual" in results:
        print(f"\nIndividual Dataset Evaluations:")
        for corpus_name, corpus_results in results["individual"].items():
            if "delta_llh" in corpus_results and corpus_results["delta_llh"] is not None:
                print(f"  {corpus_name.upper()}: Delta LLH = {corpus_results['delta_llh']:.6f}")
            elif "error" in corpus_results:
                print(f"  {corpus_name.upper()}: {corpus_results['error']}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
