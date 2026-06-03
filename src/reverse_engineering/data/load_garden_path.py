import urllib.parse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .tokenize_garden_path import tokenize_garden_path_texts


class GardenPathDataset(Dataset):
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
            "word_positions": self.data["word_positions"][idx, :],
            # Spillover features
            "log_unigram_frequencies_prev1": self.data["log_unigram_frequencies_prev1"][
                idx, :
            ],
            "zero_freq_mask_prev1": self.data["zero_freq_mask_prev1"][idx, :],
            "log_unigram_frequencies_prev2": self.data["log_unigram_frequencies_prev2"][
                idx, :
            ],
            "zero_freq_mask_prev2": self.data["zero_freq_mask_prev2"][idx, :],
        }

        # Add word length spillover features if available
        if "word_lengths_prev1" in self.data:
            result["word_lengths_prev1"] = self.data["word_lengths_prev1"][idx, :]
        if "word_lengths_prev2" in self.data:
            result["word_lengths_prev2"] = self.data["word_lengths_prev2"][idx, :]
        # Add ROI data if available
        if "roi" in self.data:
            result["roi"] = torch.from_numpy(self.data["roi"][idx, :]).long()

        # Add garden-path metadata - always available for garden-path data
        if "pair_id" in self.data:
            result["pair_id"] = self.data["pair_id"][idx, :]
        if "construction" in self.data:
            result["construction"] = self.data["construction"][idx, :]
        if "ambiguity" in self.data:
            result["ambiguity"] = self.data["ambiguity"][idx, :]

        return result


def process_garden_path_corpus(
    tokenizer: PreTrainedTokenizer, config: dict, evaluate: bool = False
) -> dict[str, np.ndarray]:
    """
    Process garden-path corpus data for reverse-engineering training.

    Args:
        tokenizer: HuggingFace tokenizer
        config: Training configuration
        evaluate: Whether this is evaluation data

    Returns:
        Processed data dictionary for GardenPathDataset
    """
    if evaluate:
        file_path = config["training_kwargs"]["eval_data_path"]
        target = config["training_kwargs"]["eval_dataset_target"]
    else:
        file_path = config["training_kwargs"]["data_path"]
        target = config["training_kwargs"]["train_dataset_target"]

    max_length = config["training_kwargs"]["garden_path_max_length"]
    garden_path_data = pd.read_csv(file_path)

    # URL decode words (fix %2C -> , etc.) - Required for garden-path data
    garden_path_data["word"] = garden_path_data["word"].str.replace(
        "%2C", ","
    )

    # Filter by phenomena based on mode (training vs evaluation)
    if evaluate:
        # For evaluation: check for eval_phenomena first, then eval_all_phenomena flag
        eval_phenomena = config["training_kwargs"].get("eval_phenomena", None)
        eval_all_phenomena = config["training_kwargs"].get("eval_all_phenomena", False)

        if eval_phenomena:
            # Use eval_phenomena if specified (for targeted generalization testing)
            if isinstance(eval_phenomena, str):
                eval_phenomena = [eval_phenomena]
            eval_phenomena = [p.upper() for p in eval_phenomena]
            garden_path_data = garden_path_data[
                garden_path_data["construction"].str.upper().isin(eval_phenomena)
            ]
            print(f"Filtered evaluation data to eval_phenomena {eval_phenomena}: {len(garden_path_data)} words")
        elif eval_all_phenomena:
            # Load all phenomena for multi-phenomenon evaluation
            print(f"Loading all phenomena for evaluation: {len(garden_path_data)} words")
            available_phenomena = garden_path_data["construction"].unique()
            print(f"Available phenomena in evaluation data: {available_phenomena}")
        else:
            # Default: use selected_phenomena for evaluation (backward compatibility)
            selected_phenomena = config["training_kwargs"].get("selected_phenomena", None)
            if selected_phenomena:
                if isinstance(selected_phenomena, str):
                    selected_phenomena = [selected_phenomena]
                selected_phenomena = [p.upper() for p in selected_phenomena]
                garden_path_data = garden_path_data[
                    garden_path_data["construction"].str.upper().isin(selected_phenomena)
                ]
                print(f"Filtered evaluation data to selected_phenomena {selected_phenomena}: {len(garden_path_data)} words")
    else:
        # For training: always use selected_phenomena
        selected_phenomena = config["training_kwargs"].get("selected_phenomena", None)
        if selected_phenomena:
            if isinstance(selected_phenomena, str):
                selected_phenomena = [selected_phenomena]
            # Convert to uppercase for consistent matching
            selected_phenomena = [p.upper() for p in selected_phenomena]
            garden_path_data = garden_path_data[
                garden_path_data["construction"].str.upper().isin(selected_phenomena)
            ]
            print(f"Filtered training data to selected_phenomena {selected_phenomena}: {len(garden_path_data)} words")

    # Group data by sentence
    texts = (
        garden_path_data.groupby("sentence_num")["word"].apply(list).tolist()
    )
    reading_times = (
        garden_path_data.groupby("sentence_num")[target].apply(list).tolist()
    )

    # Extract pre-computed features from garden-path data
    word_lengths = None
    if "length" in garden_path_data.columns:
        word_lengths = (
            garden_path_data.groupby("sentence_num")["length"].apply(list).tolist()
        )

    log_frequencies = None
    if "freq" in garden_path_data.columns:
        log_frequencies = (
            garden_path_data.groupby("sentence_num")["freq"].apply(list).tolist()
        )

    log_frequencies_prev1 = None
    if "freq_prev1" in garden_path_data.columns:
        log_frequencies_prev1 = (
            garden_path_data.groupby("sentence_num")["freq_prev1"].apply(list).tolist()
        )

    log_frequencies_prev2 = None
    if "freq_prev2" in garden_path_data.columns:
        log_frequencies_prev2 = (
            garden_path_data.groupby("sentence_num")["freq_prev2"].apply(list).tolist()
        )

    word_lengths_prev1 = None
    if "length_prev1" in garden_path_data.columns:
        word_lengths_prev1 = (
            garden_path_data.groupby("sentence_num")["length_prev1"].apply(list).tolist()
        )

    word_lengths_prev2 = None
    if "length_prev2" in garden_path_data.columns:
        word_lengths_prev2 = (
            garden_path_data.groupby("sentence_num")["length_prev2"].apply(list).tolist()
        )

    word_positions = None
    if "position" in garden_path_data.columns:
        word_positions = (
            garden_path_data.groupby("sentence_num")["position"].apply(list).tolist()
        )

    # Check if ROI data is available in the CSV
    roi_data = None
    if "roi" in garden_path_data.columns:
        roi_data = (
            garden_path_data.groupby("sentence_num")["roi"].apply(list).tolist()
        )

    # Garden-path metadata - always available
    pair_id_data = None
    construction_data = None
    ambiguity_data = None

    if "item" in garden_path_data.columns:
        pair_id_data = (
            garden_path_data.groupby("sentence_num")["item"]
            .first()  # Use first since they're all the same within a sentence
            .tolist()
        )
    if "construction" in garden_path_data.columns:
        construction_data = (
            garden_path_data.groupby("sentence_num")["construction"]
            .first()  # Use first since they're all the same within a sentence
            .tolist()
        )
    if "ambiguity" in garden_path_data.columns:
        ambiguity_data = (
            garden_path_data.groupby("sentence_num")["ambiguity"]
            .first()  # Use first since they're all the same within a sentence
            .tolist()
        )

    return tokenize_garden_path_texts(
        texts,
        reading_times,
        tokenizer,
        max_length=max_length,
        # Pre-computed features
        word_lengths=word_lengths,
        log_frequencies=log_frequencies,
        log_frequencies_prev1=log_frequencies_prev1,
        log_frequencies_prev2=log_frequencies_prev2,
        word_lengths_prev1=word_lengths_prev1,
        word_lengths_prev2=word_lengths_prev2,
        word_positions=word_positions,
        # ROI and metadata
        roi_data=roi_data,
        pair_id_data=pair_id_data,
        construction_data=construction_data,
        ambiguity_data=ambiguity_data,
    )

