"""
Data preparation module for garden-path cross-validation

This module handles:
- Human reading time data processing
- Cross-validation fold creation
- Data enhancement
"""

from .cross_validation_splitter import CrossValidationSplitter
from .human_data_processor import HumanDataProcessor

__all__ = ["CrossValidationSplitter", "HumanDataProcessor"]
