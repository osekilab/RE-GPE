"""
Simplified trainer with minimal implementation.

Core features only:
- Training with ROI exclusion from regression (always on)
"""

import os
from logging import Logger
from typing import Optional, Dict, Tuple
import json

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from models.loss import compute_loss
from models.utils.utils import get_optimizer, get_scheduler


class Trainer:
    """Simplified trainer with minimal, fixed configuration."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        config: dict,
        logger: Logger,
        device: torch.device,
        model_ref: Optional[PreTrainedModel] = None,
    ):
        """Initialize simplified trainer."""
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.logger = logger
        self.device = device
        self.model_ref = model_ref

        # Extract essential config
        self.config = config
        self.training_kwargs = config.get("training_kwargs", {})

        # ROI exclusion settings (configurable, with defaults)
        self.roi_exclusion_values = self.training_kwargs.get("roi_exclusion_values", [0, 1, 2])
        self.exclude_roi_from_regression = self.training_kwargs.get("exclude_roi_from_regression", True)

        # Training settings
        self.learning_rate = self.training_kwargs.get("learning_rate", 1e-4)
        self.eval_steps = self.training_kwargs.get("eval_steps", 100)
        self.total_steps = self.eval_steps * self.training_kwargs.get("total_steps_multiplier", 10)
        self.accumulation_steps = self.training_kwargs.get("accumulation_steps", 1)
        self.output_dir = self.training_kwargs.get("output_dir", "output")

        # Loss settings
        self.loss_type = self.training_kwargs.get("loss_type", "mse")
        self.reduction = self.training_kwargs.get("reduction", "mean")
        self.kl_weight = self.training_kwargs.get("kl_weight", 0)
        self.kl_variant = self.training_kwargs.get("kl_variant", "full")

        # KL masking settings
        self.kl_exclude_roi = self.training_kwargs.get("kl_exclude_roi", False)
        self.kl_mask_zero_reading_times = self.training_kwargs.get("kl_mask_zero_reading_times", False)
        self.kl_mask_zero_freqs = self.training_kwargs.get("kl_mask_zero_freqs", False)
        self.kl_mask_last_region = self.training_kwargs.get("kl_mask_last_region", False)

        # Masking settings
        self.mask_zero_reading_times = self.training_kwargs.get("mask_zero_reading_times", True)
        self.mask_zero_freqs = self.training_kwargs.get("mask_zero_freqs", True)
        self.mask_last_region = self.training_kwargs.get("mask_last_region", True)

        # Spillover features
        self.use_frequency_spillover = self.training_kwargs.get("use_frequency_spillover", True)
        self.use_surprisal_spillover = self.training_kwargs.get("use_surprisal_spillover", True)
        self.use_length_spillover = self.training_kwargs.get("use_length_spillover", False)

        # Garden-path phenomena to train
        self.selected_phenomena = self.training_kwargs.get("selected_phenomena", ["MVRR", "NPS", "NPZ"])

        # Checkpoint saving - simplified: just save every eval
        self.save_checkpoints = self.training_kwargs.get("save_checkpoints", True)
        self.max_checkpoints = self.training_kwargs.get("max_checkpoints", 3)  # Keep only last N checkpoints

        # Initialize optimizer and scheduler
        self.optimizer = get_optimizer(model, config)
        self.scheduler = get_scheduler(self.optimizer, config, self.total_steps)

        # Filler data path for regression
        self.filler_data_path = self.training_kwargs.get(
            "filler_data_path",
            "src/garden_path_cross_validation/folds/fillers_processed.csv"
        )

        # Training with filler coefficients option
        self.use_filler_coefficients_training = self.training_kwargs.get(
            "use_filler_coefficients_training", False
        )

        # Use only ROI positions for loss (focus on critical regions)
        self.use_roi_only_loss = self.training_kwargs.get(
            "use_roi_only_loss", False
        )
        self.roi_values_for_loss = self.training_kwargs.get(
            "roi_values_for_loss", [0, 1, 2]  # Default: critical and spillover
        )
        self.invert_roi_loss_mask = self.training_kwargs.get(
            "invert_roi_loss_mask", False  # If True, use all ROIs EXCEPT roi_values_for_loss
        )

        # Tracking
        self.step = 0
        self.train_losses = []
        self.saved_checkpoint_steps = []  # Track saved checkpoint steps for cleanup
        self.filler_coefficients = None  # Will store pre-computed filler coefficients
        self.filler_data_path = self.training_kwargs.get("filler_data_path")

        # Baseline (untrained model) coefficient regularization settings
        self.use_baseline_coefficient_regularization = self.training_kwargs.get("use_baseline_coefficient_regularization", False)
        self.baseline_coef_regularization_weight = self.training_kwargs.get("baseline_coef_regularization_weight", 1.0)
        self.baseline_coefficients = None  # Will store pre-computed baseline coefficients

    def train(self) -> None:
        """Simplified training loop."""
        self.logger.info("=" * 80)
        self.logger.info("Starting Training")
        self.logger.info(f"Total steps: {self.total_steps}")
        self.logger.info(f"Eval every: {self.eval_steps} steps")
        self.logger.info(f"ROI exclusion: {self.roi_exclusion_values}")
        if self.use_filler_coefficients_training:
            self.logger.info("Training mode: Using filler coefficients (computed each step)")
        else:
            self.logger.info("Training mode: Using batch coefficients with ROI exclusion")
        if self.use_roi_only_loss:
            if self.invert_roi_loss_mask:
                self.logger.info(f"ROI-only loss: Using all EXCEPT ROI positions {self.roi_values_for_loss} for loss")
            else:
                self.logger.info(f"ROI-only loss: Using only ROI positions {self.roi_values_for_loss} for loss")
        if self.use_baseline_coefficient_regularization:
            self.logger.info(f"Baseline coefficient regularization: L2 weight = {self.baseline_coef_regularization_weight}")
        if self.kl_weight > 0:
            self.logger.info(f"KL divergence: weight={self.kl_weight}, variant={self.kl_variant}")
            kl_masks = []
            if self.kl_exclude_roi:
                kl_masks.append(f"exclude_roi({self.roi_exclusion_values})")
            if self.kl_mask_zero_reading_times:
                kl_masks.append("mask_zero_RT")
            if self.kl_mask_zero_freqs:
                kl_masks.append("mask_OOV")
            if self.kl_mask_last_region:
                kl_masks.append("mask_last_region")
            if kl_masks:
                self.logger.info(f"KL masking: {', '.join(kl_masks)}")
            else:
                self.logger.info("KL masking: padding only (default)")
        self.logger.info("=" * 80)

        # Compute baseline coefficients if needed (before any training)
        if self.use_baseline_coefficient_regularization:
            self.logger.info("Computing baseline (untrained model) coefficients from training data (ROI excluded)...")
            self.baseline_coefficients = self._compute_training_data_coefficients()
            if self.baseline_coefficients:
                self.logger.info("Baseline coefficients computed successfully:")
                for key in sorted(self.baseline_coefficients.keys()):
                    if self.baseline_coefficients[key] is not None:
                        value = (self.baseline_coefficients[key].item()
                                if self.baseline_coefficients[key].numel() == 1
                                else self.baseline_coefficients[key].mean().item())
                        self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.warning("Failed to compute baseline coefficients")

        # Training loop
        self.model.train()
        pbar = tqdm(total=self.total_steps, desc="Training")

        while self.step < self.total_steps:
            for batch in self.train_loader:
                if self.step >= self.total_steps:
                    break

                # Compute filler coefficients if using filler-based training
                if self.use_filler_coefficients_training:
                    self.filler_coefficients = self._compute_filler_coefficients()

                # Training step
                loss, monitoring_data, coefficients = self._training_step(batch)
                self.train_losses.append(loss.item())

                # Log progress
                if self.step % 50 == 0:
                    avg_loss = np.mean(self.train_losses[-50:]) if len(self.train_losses) >= 50 else np.mean(self.train_losses)
                    self.logger.info(f"Step {self.step}: loss={avg_loss:.4f}")

                    # Log regression coefficients
                    if coefficients:
                        self.logger.info("Regression coefficients:")
                        # Check which keys are available
                        coef_keys = list(coefficients.keys())
                        for key in coef_keys:
                            if coefficients[key] is not None:
                                value = coefficients[key].item() if coefficients[key].numel() == 1 else coefficients[key].mean().item()
                                self.logger.info(f"  {key}: {value:.4f}")

                    # Log baseline coefficient regularization loss if available
                    if monitoring_data and "baseline_coef_reg_loss" in monitoring_data:
                        baseline_coef_reg = monitoring_data["baseline_coef_reg_loss"]
                        if baseline_coef_reg > 0:
                            self.logger.info(f"Coefficient Regularization Loss (baseline): {baseline_coef_reg:.4f}")

                    # Log garden-path monitoring data if available
                    if monitoring_data and len(monitoring_data) > 0:
                        gp_keys = [k for k in monitoring_data.keys() if k.startswith("gp_")]
                        if gp_keys:
                            self.logger.info("Training - Garden-path differences (Amb - Unamb):")
                            # Use selected_phenomena for training monitoring (what's actually being trained)
                            for construction in self.selected_phenomena:
                                for roi in [0, 1, 2]:
                                    key_actual = f"gp_{construction}_roi{roi}_diff_actual"
                                    key_predicted = f"gp_{construction}_roi{roi}_diff_predicted"
                                    if key_actual in monitoring_data:
                                        actual_diff = monitoring_data[key_actual]
                                        pred_diff = monitoring_data[key_predicted]
                                        n_pairs = monitoring_data.get(f"gp_{construction}_roi{roi}_n_pairs", 0)

                                        # Get surprisal values if available
                                        surp_amb = monitoring_data.get(f"gp_{construction}_roi{roi}_surprisal_amb", None)
                                        surp_unamb = monitoring_data.get(f"gp_{construction}_roi{roi}_surprisal_unamb", None)
                                        surp_diff = monitoring_data.get(f"gp_{construction}_roi{roi}_surprisal_diff", None)

                                        if surp_diff is not None:
                                            self.logger.info(
                                                f"  {construction} ROI{roi}: "
                                                f"actual={actual_diff:.1f}ms, "
                                                f"predicted={pred_diff:.1f}ms | "
                                                f"surp_diff={surp_diff:.2f} bits "
                                                f"(Amb={surp_amb:.2f}, Unamb={surp_unamb:.2f}) (n={n_pairs})"
                                            )
                                        else:
                                            self.logger.info(
                                                f"  {construction} ROI{roi}: actual={actual_diff:.1f}ms, "
                                                f"predicted={pred_diff:.1f}ms (n={n_pairs})"
                                            )

                # Save checkpoint
                if self.step > 0 and self.step % self.eval_steps == 0:
                    if self.save_checkpoints:
                        self._save_checkpoint({})

                pbar.update(1)
                self.step += 1

        pbar.close()

        # Save final checkpoint
        self._save_final_checkpoint({})

        self.logger.info("Training completed!")

    def _training_step(self, batch: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step with ROI exclusion or filler coefficients."""
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()}

        # Zero gradients
        if self.step % self.accumulation_steps == 0:
            self.optimizer.zero_grad()

        # Determine whether to use filler coefficients or ROI exclusion
        if self.use_filler_coefficients_training and self.filler_coefficients is not None:
            # Use filler coefficients (like evaluation)
            loss_results = compute_loss(
                model=self.model,
                tokenizer=self.tokenizer,
                batch=batch,
                loss_type=self.loss_type,
                reduction=self.reduction,
                mask_zero_reading_times=self.mask_zero_reading_times,
                mask_zero_freqs=self.mask_zero_freqs,
                mask_last_region=self.mask_last_region,
                kl_weight=self.kl_weight,
                kl_variant=self.kl_variant,
                model_ref=self.model_ref,
                # KL masking
                kl_exclude_roi=self.kl_exclude_roi,
                kl_mask_zero_reading_times=self.kl_mask_zero_reading_times,
                kl_mask_zero_freqs=self.kl_mask_zero_freqs,
                kl_mask_last_region=self.kl_mask_last_region,
                # NO ROI exclusion when using filler coefficients
                exclude_roi_from_regression=False,
                # Use pre-computed filler coefficients
                use_global_coefficients=True,
                global_coefficients=self.filler_coefficients,
                # Spillover features
                use_frequency_spillover=self.use_frequency_spillover,
                use_surprisal_spillover=self.use_surprisal_spillover,
                use_length_spillover=self.use_length_spillover,
                # ROI-only loss
                use_roi_only_loss=self.use_roi_only_loss,
                roi_values_for_loss=self.roi_values_for_loss,
                invert_roi_loss_mask=self.invert_roi_loss_mask,
                # Baseline coefficient regularization
                use_baseline_coefficient_regularization=self.use_baseline_coefficient_regularization,
                baseline_coef_regularization_weight=self.baseline_coef_regularization_weight,
                baseline_coefficients=self.baseline_coefficients,
            )
        else:
            # Default: Use batch coefficients with ROI exclusion
            loss_results = compute_loss(
                model=self.model,
                tokenizer=self.tokenizer,
                batch=batch,
                loss_type=self.loss_type,
                reduction=self.reduction,
                mask_zero_reading_times=self.mask_zero_reading_times,
                mask_zero_freqs=self.mask_zero_freqs,
                mask_last_region=self.mask_last_region,
                kl_weight=self.kl_weight,
                kl_variant=self.kl_variant,
                model_ref=self.model_ref,
                # KL masking
                kl_exclude_roi=self.kl_exclude_roi,
                kl_mask_zero_reading_times=self.kl_mask_zero_reading_times,
                kl_mask_zero_freqs=self.kl_mask_zero_freqs,
                kl_mask_last_region=self.kl_mask_last_region,
                exclude_roi_from_regression=self.exclude_roi_from_regression,
                roi_exclusion_values=self.roi_exclusion_values,
                # Spillover features
                use_frequency_spillover=self.use_frequency_spillover,
                use_surprisal_spillover=self.use_surprisal_spillover,
                use_length_spillover=self.use_length_spillover,
                # ROI-only loss
                use_roi_only_loss=self.use_roi_only_loss,
                roi_values_for_loss=self.roi_values_for_loss,
                invert_roi_loss_mask=self.invert_roi_loss_mask,
                # Baseline coefficient regularization
                use_baseline_coefficient_regularization=self.use_baseline_coefficient_regularization,
                baseline_coef_regularization_weight=self.baseline_coef_regularization_weight,
                baseline_coefficients=self.baseline_coefficients,
            )

        # Unpack results
        train_loss, coefficients, _, _, _, _, monitoring_data = loss_results

        # Use training loss as total loss
        total_loss = train_loss

        # Scale for accumulation
        total_loss = total_loss / self.accumulation_steps

        # Backward
        total_loss.backward()

        # Optimizer step
        if (self.step + 1) % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

        # Return unscaled loss for logging
        return total_loss * self.accumulation_steps, monitoring_data, coefficients

    def _compute_filler_coefficients(self) -> Optional[Dict[str, torch.Tensor]]:
        """Compute regression coefficients using filler data."""
        try:
            from models.filler_regression import compute_filler_coefficients

            self.logger.info("Computing filler regression coefficients...")

            coefficients = compute_filler_coefficients(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                config=self.config,
                use_frequency_spillover=self.use_frequency_spillover,
                use_surprisal_spillover=self.use_surprisal_spillover,
                use_length_spillover=self.use_length_spillover,
                use_word_position=True,
                mask_zero_reading_times=self.mask_zero_reading_times,
                mask_zero_freqs=self.mask_zero_freqs,
                mask_last_region=self.mask_last_region,
                logger=self.logger,
                step=self.step,
                data_path=self.filler_data_path,
            )

            if coefficients:
                self.logger.info("Successfully computed filler coefficients")
                return coefficients
            else:
                self.logger.warning("Failed to compute filler coefficients")
                return None

        except Exception as e:
            self.logger.warning(f"Filler coefficient computation failed: {e}")
            return None

    def _compute_training_data_coefficients(self) -> Optional[Dict[str, torch.Tensor]]:
        """Compute regression coefficients from training data (excluding ROI) for baseline regularization."""
        try:
            from models.loss_utils import solve_beta
            from models.surprisal import compute_surprisal
            import torch

            self.logger.info("Computing baseline coefficients from training data (ROI excluded)...")

            # Collect features and targets from all training batches
            all_surprisals = []
            all_features_dict = {}
            all_targets = []

            self.model.eval()
            with torch.no_grad():
                for batch in self.train_loader:
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                            for k, v in batch.items()}

                    word_ids = batch.get("word_ids", None)
                    if word_ids is None:
                        continue

                    # Compute surprisal
                    surprisal_subword, _, _ = compute_surprisal(batch, self.model, self.tokenizer, word_ids)

                    # Aggregate to word level
                    mask_word_ids = word_ids != -1
                    masked_surprisal_subword = surprisal_subword * mask_word_ids
                    masked_word_ids = word_ids * mask_word_ids

                    surprisal_word = torch.zeros_like(surprisal_subword)
                    surprisal_word.scatter_add_(1, masked_word_ids, masked_surprisal_subword)

                    # Create valid mask
                    reading_times = batch.get("reading_times", torch.zeros_like(batch["input_ids"]))
                    valid_mask = (word_ids != -1) & (reading_times != -1) & (reading_times > 0)

                    # Apply masking
                    if self.mask_zero_freqs and "zero_freq_mask" in batch:
                        valid_mask = valid_mask & (batch["zero_freq_mask"] != 0)
                        if self.use_frequency_spillover:
                            if "zero_freq_mask_prev1" in batch:
                                valid_mask = valid_mask & (batch["zero_freq_mask_prev1"] != 0)
                            if "zero_freq_mask_prev2" in batch:
                                valid_mask = valid_mask & (batch["zero_freq_mask_prev2"] != 0)

                    # Exclude ROI positions
                    if "roi" in batch and self.exclude_roi_from_regression:
                        roi_mask = torch.zeros_like(valid_mask)
                        for roi_val in self.roi_exclusion_values:
                            roi_mask = roi_mask | (batch["roi"] == roi_val)
                        valid_mask = valid_mask & ~roi_mask

                    # Collect features
                    for idx in range(batch["input_ids"].shape[0]):
                        sentence_mask = valid_mask[idx].clone()

                        if self.mask_last_region:
                            valid_positions = torch.where(sentence_mask)[0]
                            if len(valid_positions) > 0:
                                sentence_mask[valid_positions[-1]] = False

                        if sentence_mask.any():
                            all_surprisals.append(surprisal_word[idx][sentence_mask])
                            all_targets.append(reading_times[idx][sentence_mask])

                            if "word_lengths" in batch:
                                all_features_dict.setdefault("length", []).append(batch["word_lengths"][idx][sentence_mask])
                            if "log_unigram_frequencies" in batch:
                                all_features_dict.setdefault("freq", []).append(batch["log_unigram_frequencies"][idx][sentence_mask])
                            if "word_positions" in batch:
                                all_features_dict.setdefault("position", []).append(batch["word_positions"][idx][sentence_mask])

                            if self.use_frequency_spillover:
                                if "log_unigram_frequencies_prev1" in batch:
                                    all_features_dict.setdefault("freq_prev1", []).append(batch["log_unigram_frequencies_prev1"][idx][sentence_mask])
                                if "log_unigram_frequencies_prev2" in batch:
                                    all_features_dict.setdefault("freq_prev2", []).append(batch["log_unigram_frequencies_prev2"][idx][sentence_mask])

                            if self.use_surprisal_spillover:
                                surprisal_prev1 = torch.zeros_like(surprisal_word[idx])
                                surprisal_prev2 = torch.zeros_like(surprisal_word[idx])
                                surprisal_prev1[1:] = surprisal_word[idx][:-1]
                                surprisal_prev2[2:] = surprisal_word[idx][:-2]
                                all_features_dict.setdefault("surprisal_prev1", []).append(surprisal_prev1[sentence_mask])
                                all_features_dict.setdefault("surprisal_prev2", []).append(surprisal_prev2[sentence_mask])

                            if self.use_length_spillover:
                                if "word_lengths_prev1" in batch:
                                    all_features_dict.setdefault("length_prev1", []).append(batch["word_lengths_prev1"][idx][sentence_mask])
                                if "word_lengths_prev2" in batch:
                                    all_features_dict.setdefault("length_prev2", []).append(batch["word_lengths_prev2"][idx][sentence_mask])

            # Build coefficient dictionary
            if all_surprisals:
                all_surprisals = torch.cat(all_surprisals)
                all_targets = torch.cat(all_targets)

                features_list = [all_surprisals]
                feature_names = ["beta_s"]

                if "length" in all_features_dict:
                    features_list.append(torch.cat(all_features_dict["length"]))
                    feature_names.append("beta_l")
                if "freq" in all_features_dict:
                    features_list.append(torch.cat(all_features_dict["freq"]))
                    feature_names.append("beta_f")
                if "position" in all_features_dict:
                    features_list.append(torch.cat(all_features_dict["position"]))
                    feature_names.append("beta_pos")
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

                X = torch.stack(features_list, dim=1)
                X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim=1)
                feature_names = ["beta_0"] + feature_names

                betas = solve_beta(X, all_targets)

                coefficients = {name: betas[i].unsqueeze(0) for i, name in enumerate(feature_names)}

                self.model.train()
                return coefficients
            else:
                self.model.train()
                return None

        except Exception as e:
            self.logger.warning(f"Baseline coefficient computation failed: {e}")
            self.model.train()
            return None

    def _save_checkpoint(self, metrics: Dict[str, float]) -> None:
        """Save checkpoint at evaluation step."""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step_{self.step}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        torch.save({
            'step': self.step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, os.path.join(checkpoint_path, 'training_state.pt'))

        # Save metrics
        with open(os.path.join(checkpoint_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        self.logger.info(f"Saved checkpoint at step {self.step}")

        # Track this checkpoint
        self.saved_checkpoint_steps.append(self.step)

        # Remove old checkpoints if we exceed the limit
        if len(self.saved_checkpoint_steps) > self.max_checkpoints:
            old_step = self.saved_checkpoint_steps.pop(0)
            old_path = os.path.join(self.output_dir, f"checkpoint_step_{old_step}")
            if os.path.exists(old_path):
                import shutil
                shutil.rmtree(old_path)
                self.logger.info(f"Removed old checkpoint at step {old_step} (keeping last {self.max_checkpoints})")

    def _save_final_checkpoint(self, metrics: Dict[str, float]) -> None:
        """Save final checkpoint."""
        final_path = os.path.join(self.output_dir, "final_checkpoint")
        os.makedirs(final_path, exist_ok=True)

        # Save model
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)

        # Save final metrics
        with open(os.path.join(final_path, 'final_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        self.logger.info(f"Saved final checkpoint to {final_path}")
