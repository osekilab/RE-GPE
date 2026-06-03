"""
Unified SPR (Self-Paced Reading) dataset loader for training and evaluation.
Supports all SPR datasets including Natural Stories, UCL selfpaced, and Smith2013.
"""

import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from data.tokenize_texts import tokenize_texts
from data.tokenize_garden_path import tokenize_garden_path_texts

logger = logging.getLogger(__name__)


class SPRDataset(Dataset):
    """
    Unified dataset class for all SPR corpora.
    Supports spillover features for datasets that have them.
    """

    def __init__(self, data: dict[str, np.ndarray]):
        self.data = data

    def __len__(self):
        return self.data["input_ids"].shape[0]

    def __getitem__(self, idx):
        result = {
            "input_ids": self.data["input_ids"][idx, :],
            "attention_mask": self.data["attention_mask"][idx, :],
            "reading_times": self.data["reading_times"][idx, :],
            "word_ids": self.data["word_ids"][idx, :],
            "word_lengths": self.data["word_lengths"][idx, :],
            "log_unigram_frequencies": self.data["log_unigram_frequencies"][idx, :],
            "zero_freq_mask": self.data["zero_freq_mask"][idx, :],
        }

        # Add word positions if available
        if "word_positions" in self.data:
            result["word_positions"] = self.data["word_positions"][idx, :]

        # Add frequency spillover features if available
        if "log_unigram_frequencies_prev1" in self.data:
            result["log_unigram_frequencies_prev1"] = self.data["log_unigram_frequencies_prev1"][idx, :]
        if "zero_freq_mask_prev1" in self.data:
            result["zero_freq_mask_prev1"] = self.data["zero_freq_mask_prev1"][idx, :]
        if "log_unigram_frequencies_prev2" in self.data:
            result["log_unigram_frequencies_prev2"] = self.data["log_unigram_frequencies_prev2"][idx, :]
        if "zero_freq_mask_prev2" in self.data:
            result["zero_freq_mask_prev2"] = self.data["zero_freq_mask_prev2"][idx, :]

        # Add word length spillover features if available
        if "word_lengths_prev1" in self.data:
            result["word_lengths_prev1"] = self.data["word_lengths_prev1"][idx, :]
        if "word_lengths_prev2" in self.data:
            result["word_lengths_prev2"] = self.data["word_lengths_prev2"][idx, :]

        return result


def process_spr_corpus(
    tokenizer: PreTrainedTokenizer,
    config: dict,
    spr_dataset_name: str,
    evaluate: bool = False
) -> dict[str, np.ndarray]:
    """
    Process SPR corpus data for training or evaluation.

    Supports all SPR datasets with automatic detection of available features.
    Natural Stories and other datasets with spillover features use the advanced
    tokenizer, while simpler datasets use basic tokenization.

    Args:
        tokenizer: The tokenizer to use
        config: Configuration dictionary
        spr_dataset_name: Name of the SPR dataset (e.g., 'natural_stories', 'smith2013', 'ucl_selfpaced')
        evaluate: Whether this is for evaluation

    Returns:
        Tokenized and processed data dictionary
    """
    training_kwargs = config.get("training_kwargs", {})

    # Get dataset configuration from spr_datasets list
    dataset_config = None
    spr_datasets_config = training_kwargs.get("spr_datasets", [])

    for ds in spr_datasets_config:
        if ds["name"] == spr_dataset_name:
            dataset_config = ds
            break

    # Check for filler dataset FIRST (before eval_data_path fallback)
    if dataset_config is None and spr_dataset_name == "filler":
        # Special handling for filler data using filler_data_path
        file_path = training_kwargs.get("filler_data_path", None)
        if file_path is None:
            raise ValueError(
                "filler dataset requested but filler_data_path not found "
                "in config. Please add filler_data_path to training_kwargs."
            )
        max_length = training_kwargs.get("filler_max_length", 40)
        logger.info(
            f"Loading filler data from filler_data_path: {file_path} "
            f"(max_length={max_length})"
        )
    elif dataset_config is None and "eval_data_path" in training_kwargs:
        # Fallback: use eval_data_path for other datasets not in spr_datasets
        file_path = training_kwargs["eval_data_path"]

        # Determine max_length based on dataset name
        if spr_dataset_name == "filler":
            max_length = training_kwargs.get("filler_max_length", 40)
        else:
            # For other datasets not in spr_datasets, try common keys
            max_length = training_kwargs.get(
                f"{spr_dataset_name}_max_length",
                training_kwargs.get("garden_path_max_length", 40)
            )
        logger.info(
            f"Using eval_data_path for '{spr_dataset_name}': {file_path} "
            f"(max_length={max_length})"
        )
    elif dataset_config is not None:
        # Get file path and max_length from spr_datasets config
        file_path = dataset_config["path"]

        if "max_length" not in dataset_config:
            raise ValueError(
                f"max_length not specified for SPR dataset '{spr_dataset_name}'. "
                f"Please add max_length to the dataset configuration in your config file."
            )

        max_length = dataset_config["max_length"]
    else:
        raise ValueError(
            f"No configuration found for SPR dataset '{spr_dataset_name}' in spr_datasets "
            f"and no eval_data_path provided. Please add it to your config file under "
            f"training_kwargs.spr_datasets or ensure eval_data_path is set."
        )

    # Load dataset
    logger.info(f"Loading {spr_dataset_name} data from: {file_path}")
    logger.info(f"Using max_length={max_length} for {spr_dataset_name}")

    spr_data = pd.read_csv(file_path)

    # Handle filler data's different column names
    if spr_dataset_name == "filler":
        if "item" in spr_data.columns and "sentence_num" not in spr_data.columns:
            spr_data["sentence_num"] = spr_data["item"]
        if "word_position" in spr_data.columns and "position" not in spr_data.columns:
            spr_data["position"] = spr_data["word_position"]

    # URL decode words (for datasets that need it, e.g., Natural Stories)
    if "word" in spr_data.columns and spr_data["word"].dtype == object:
        spr_data["word"] = spr_data["word"].str.replace("%2C", ",")

    # Get target column name
    target = training_kwargs.get("train_dataset_target", "reading_time")
    if evaluate:
        target = training_kwargs.get("eval_dataset_target", target)

    # Group by sentence and extract texts and reading times
    texts = spr_data.groupby("sentence_num")["word"].apply(list).tolist()
    reading_times = spr_data.groupby("sentence_num")[target].apply(list).tolist()

    logger.info(f"Loaded {len(texts)} sentences from {spr_dataset_name}")
    if texts and texts[0]:
        logger.info(f"First sentence: {' '.join(texts[0][:10])}...")

    # Check if dataset has spillover features (Natural Stories typically has these)
    has_spillover_features = all(
        col in spr_data.columns
        for col in ["freq_prev1", "length_prev1"]
    )

    if has_spillover_features:
        # Use advanced tokenizer with spillover features
        logger.info(f"{spr_dataset_name} has spillover features, using advanced tokenizer")

        # Extract pre-computed features
        word_lengths = None
        log_frequencies = None
        log_frequencies_prev1 = None
        log_frequencies_prev2 = None
        word_lengths_prev1 = None
        word_lengths_prev2 = None
        word_positions = None

        if "length" in spr_data.columns:
            word_lengths = spr_data.groupby("sentence_num")["length"].apply(list).tolist()

        if "freq" in spr_data.columns:
            log_frequencies = spr_data.groupby("sentence_num")["freq"].apply(list).tolist()

        if "freq_prev1" in spr_data.columns:
            log_frequencies_prev1 = spr_data.groupby("sentence_num")["freq_prev1"].apply(list).tolist()

        if "freq_prev2" in spr_data.columns:
            log_frequencies_prev2 = spr_data.groupby("sentence_num")["freq_prev2"].apply(list).tolist()

        if "length_prev1" in spr_data.columns:
            word_lengths_prev1 = spr_data.groupby("sentence_num")["length_prev1"].apply(list).tolist()

        if "length_prev2" in spr_data.columns:
            word_lengths_prev2 = spr_data.groupby("sentence_num")["length_prev2"].apply(list).tolist()

        if "position" in spr_data.columns:
            word_positions = spr_data.groupby("sentence_num")["position"].apply(list).tolist()

        return tokenize_garden_path_texts(
            texts,
            reading_times,
            tokenizer,
            max_length=max_length,
            word_lengths=word_lengths,
            log_frequencies=log_frequencies,
            log_frequencies_prev1=log_frequencies_prev1,
            log_frequencies_prev2=log_frequencies_prev2,
            word_lengths_prev1=word_lengths_prev1,
            word_lengths_prev2=word_lengths_prev2,
            word_positions=word_positions,
            # SPR datasets don't have garden-path specific fields
            roi_data=None,
            pair_id_data=None,
            construction_data=None,
            ambiguity_data=None,
        )
    else:
        # Use simple tokenizer for datasets without spillover features
        logger.info(f"{spr_dataset_name} using basic tokenizer")
        return tokenize_texts(
            texts,
            reading_times,
            tokenizer,
            max_length=max_length,
        )
