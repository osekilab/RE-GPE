"""Utility functions for loss computation."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


def solve_beta(
    X: torch.Tensor, y: torch.Tensor, regularization: float = 1e-5
) -> torch.Tensor:
    """Solve for optimal coeffs beta^* using Cholesky decomposition."""
    # Compute X^T * X and add regularization
    XTX = torch.matmul(X.T, X) + regularization * torch.eye(
        X.shape[1], dtype=X.dtype, device=X.device
    )
    XTy = torch.matmul(X.T, y)
    L = torch.linalg.cholesky(XTX)
    return torch.cholesky_solve(XTy.unsqueeze(1), L).squeeze()


def compute_kl_full(target_lps: torch.Tensor, ref_lps: torch.Tensor) -> torch.Tensor:
    """Compute full KL divergence term."""
    # Terms are reversed in pytorch kl_div()
    return torch.nn.functional.kl_div(
        target_lps,
        ref_lps,
        reduction="none",
        log_target=True,
    ).sum(-1)


def build_design_matrix(
    surprisal: torch.Tensor,
    word_lengths: torch.Tensor,
    word_frequencies: torch.Tensor,
    word_positions: Optional[torch.Tensor] = None,
    word_frequencies_prev1: Optional[torch.Tensor] = None,
    word_frequencies_prev2: Optional[torch.Tensor] = None,
    surprisal_prev1: Optional[torch.Tensor] = None,
    surprisal_prev2: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """Build design matrix from features."""
    ones = torch.ones_like(surprisal)
    features = [surprisal, word_lengths, word_frequencies]
    coeff_names = ["beta_s", "beta_l", "beta_f"]

    # Add optional features
    if word_positions is not None:
        features.append(word_positions)
        coeff_names.append("beta_pos")

    if word_frequencies_prev1 is not None:
        features.append(word_frequencies_prev1)
        coeff_names.append("beta_f_prev1")

    if word_frequencies_prev2 is not None:
        features.append(word_frequencies_prev2)
        coeff_names.append("beta_f_prev2")

    if surprisal_prev1 is not None:
        features.append(surprisal_prev1)
        coeff_names.append("beta_s_prev1")

    if surprisal_prev2 is not None:
        features.append(surprisal_prev2)
        coeff_names.append("beta_s_prev2")

    # Add intercept
    features.append(ones)
    coeff_names.append("beta_0")

    # Stack features
    X = torch.stack(features, dim=1)

    return X, coeff_names


def apply_roi_weighting(
    loss: torch.Tensor,
    roi_values: torch.Tensor,
    roi_weights: Dict[int, float],
    default_weight: float = 1.0
) -> torch.Tensor:
    """Apply ROI-based loss weighting."""
    # Create weight tensor
    loss_weights = torch.ones_like(loss) * default_weight

    # Apply specific ROI weights
    for roi_value, weight in roi_weights.items():
        if roi_value != "others":
            roi_mask = roi_values == roi_value
            loss_weights[roi_mask] = weight

    # Apply weights to loss
    weighted_loss = loss * loss_weights

    return weighted_loss


def create_exclusion_mask(
    roi_values: torch.Tensor,
    roi_exclusion_values: List[int]
) -> torch.Tensor:
    """Create a mask for excluding specific ROI values."""
    if not roi_exclusion_values:
        return None

    exclude_mask = torch.zeros_like(roi_values, dtype=torch.bool)
    for roi_val in roi_exclusion_values:
        exclude_mask |= (roi_values == roi_val)

    return exclude_mask


def extract_metadata_for_monitoring(
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, Optional[List]]:
    """Extract metadata from batch for monitoring."""
    monitoring_data = {}

    # Extract ambiguity labels
    if "ambiguity" in batch:
        ambiguity_tensor = batch["ambiguity"]
        if hasattr(ambiguity_tensor, "to"):
            ambiguity_tensor = ambiguity_tensor.to(device)

        # Convert to list of strings
        ambiguity_labels = []
        for amb in ambiguity_tensor.cpu().numpy():
            if isinstance(amb, (bytes, np.bytes_)):
                ambiguity_labels.append(amb.decode('utf-8'))
            else:
                ambiguity_labels.append(str(amb))
        monitoring_data["ambiguity_labels"] = ambiguity_labels
    else:
        monitoring_data["ambiguity_labels"] = None

    # Extract construction labels
    if "construction" in batch:
        construction_tensor = batch["construction"]
        if hasattr(construction_tensor, "to"):
            construction_tensor = construction_tensor.to(device)

        construction_labels = []
        for const in construction_tensor.cpu().numpy():
            if isinstance(const, (bytes, np.bytes_)):
                construction_labels.append(const.decode('utf-8'))
            else:
                construction_labels.append(str(const))
        monitoring_data["construction_labels"] = construction_labels
    else:
        monitoring_data["construction_labels"] = None

    # Extract pair IDs
    if "pair_id" in batch:
        pair_id_tensor = batch["pair_id"]
        if hasattr(pair_id_tensor, "to"):
            pair_id_tensor = pair_id_tensor.to(device)

        pair_id_labels = []
        for pid in pair_id_tensor.cpu().numpy():
            if isinstance(pid, (bytes, np.bytes_)):
                pair_id_labels.append(pid.decode('utf-8'))
            else:
                pair_id_labels.append(str(pid))
        monitoring_data["pair_id_labels"] = pair_id_labels
    else:
        monitoring_data["pair_id_labels"] = None

    return monitoring_data


def validate_batch_data(
    batch: Dict[str, torch.Tensor],
    required_fields: List[str],
    logger: Optional[logging.Logger] = None
) -> bool:
    """Validate that batch contains required fields."""
    missing_fields = []
    for field in required_fields:
        if field not in batch:
            missing_fields.append(field)

    if missing_fields:
        if logger:
            logger.warning(f"Batch missing required fields: {missing_fields}")
        return False

    return True


def compute_spillover_features(
    surprisal: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute lag-1 and lag-2 surprisal spillover features."""
    # Initialize spillover tensors
    surprisal_prev1 = torch.zeros_like(surprisal)
    surprisal_prev2 = torch.zeros_like(surprisal)

    # Reshape for processing
    surprisal_reshaped = surprisal.view(batch_size, sequence_length)

    # Compute spillover
    surprisal_prev1[:, 1:] = surprisal_reshaped[:, :-1]
    surprisal_prev2[:, 2:] = surprisal_reshaped[:, :-2]

    # Flatten back
    surprisal_prev1 = surprisal_prev1.view(-1)
    surprisal_prev2 = surprisal_prev2.view(-1)

    return surprisal_prev1, surprisal_prev2
