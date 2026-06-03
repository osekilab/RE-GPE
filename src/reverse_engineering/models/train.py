"""
Simplified training module with minimal implementation.

Use this for clean, manageable training with fixed settings:
- ROI exclusion during training (always on)
- Filler regression for garden-path evaluation
- Natural Stories evaluation with own coefficients
"""

from logging import Logger
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from models.trainer import Trainer


def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_data_loader: DataLoader,
    eval_data_loader: DataLoader,
    config: dict,
    logger: Logger,
    device: torch.device,
    model_ref: Optional[PreTrainedModel] = None,
) -> None:
    """
    Simplified training with minimal, fixed configuration.

    Fixed behaviors:
    - Training: Always excludes ROI [0, 1, 2] from regression
    - Evaluation: Uses filler coefficients for garden-path
    - Evaluation: Natural Stories computes own coefficients

    Args:
        model: The language model to train
        tokenizer: The tokenizer
        train_data_loader: DataLoader for training data
        eval_data_loader: DataLoader for evaluation data
        config: Configuration dictionary
        logger: Logger for logging
        device: Device to run the model on
        model_ref: Reference model for KL divergence (optional)
        mean_rt_train: Not used in simplified version
        std_rt_train: Not used in simplified version
    """
    training_kwargs = config.get("training_kwargs", {})
    roi_exclusion = training_kwargs.get("exclude_roi_from_regression", True)
    roi_values = training_kwargs.get("roi_exclusion_values", [0, 1, 2])

    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info(f"  - ROI exclusion: {'Enabled' if roi_exclusion else 'Disabled'}")
    if roi_exclusion:
        logger.info(f"  - ROI positions excluded: {roi_values}")
    logger.info("=" * 80)

    # Initialize simplified trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_data_loader,
        eval_loader=eval_data_loader,
        config=config,
        logger=logger,
        device=device,
        model_ref=model_ref,
    )

    # Run training
    trainer.train()


__all__ = ["train"]
