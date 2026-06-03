"""
Garden-Path Cross-Validation Fine-Tuning Experiment

This module implements a 23-train/1-test cross-validation methodology for
garden-path effect evaluation using fine-tuned language models.

Key Components:
- Subject-averaged reading time calculation from SAP Classic GP dataset
- Cross-validation splitting (23 items for training, 1 for evaluation)
- Garden-path effect fine-tuning of language models
- Statistical evaluation of garden-path effects across folds

Methodology:
1. Calculate subject-averaged reading times for each word in 24 garden-path items
2. For each fold: use 23 items to fine-tune model, evaluate on 1 held-out item
3. Compare fine-tuned vs baseline models on garden-path effect detection
4. Aggregate results across all 24 folds for robust statistical evaluation
"""
