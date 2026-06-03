import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from data.data_utils import (
    get_mean_std_reading_times_dataloader,
    log_dataloader,
    save_data_split_logs,
)
from data.load_spr import SPRDataset, process_spr_corpus
from data.load_garden_path import (
    GardenPathDataset,
    process_garden_path_corpus,
)
from data.batch_samplers import GardenPathBatchSampler


def custom_collate_fn(batch):
    """Custom collate function to handle object arrays (strings) in batch."""
    import torch
    from torch.utils.data.dataloader import default_collate

    # Process each item in the batch
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]

        # Check if this is string/object data
        if len(values) > 0 and isinstance(values[0], np.ndarray) and values[0].dtype == object:
            # Keep as list of numpy arrays for string data
            collated[key] = values
        else:
            # Use default collation for numeric data
            try:
                collated[key] = default_collate(values)
            except:
                # Fallback to keeping as list
                collated[key] = values

    return collated


def load_data(config: dict, tokenizer: PreTrainedTokenizer) -> tuple[DataLoader, DataLoader, float, float]:
    """
    Legacy function for loading data - maintains compatibility with run.py.

    Args:
        config: Configuration dictionary
        tokenizer: The tokenizer to use

    Returns:
        Tuple of (train_loader, eval_loader, mean_rt_train, std_rt_train)
    """
    # Determine dataset type
    train_dataset = config["training_kwargs"].get("train_dataset", "natural_stories")
    eval_dataset = config["training_kwargs"].get("eval_dataset", train_dataset)

    # Create train loader
    train_loader = get_dataloader(
        tokenizer=tokenizer,
        dataset_name=train_dataset,
        config=config,
        num_workers=0,
        evaluate=False,
    )

    # Create eval loader
    eval_loader = get_dataloader(
        tokenizer=tokenizer,
        dataset_name=eval_dataset,
        config=config,
        num_workers=0,
        evaluate=True,
    )

    # Calculate mean and std of reading times for training data
    mean_rt_train, std_rt_train = get_mean_std_reading_times_dataloader(train_loader)

    # Log data statistics
    log_dir = config["training_kwargs"].get("output_dir", None)
    if log_dir:
        log_dataloaders(train_loader, eval_loader, config, tokenizer, log_dir)

    return train_loader, eval_loader, mean_rt_train, std_rt_train


def get_dataloader(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    config: dict,
    num_workers: int = 0,
    evaluate: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for the specified dataset.

    Args:
        tokenizer: The tokenizer to use
        dataset_name: Name of the dataset ('natural_stories', 'garden_path', etc.)
        config: Configuration dictionary
        num_workers: Number of workers for DataLoader
        evaluate: Whether to load evaluation data

    Returns:
        DataLoader instance
    """
    dataset_name = dataset_name.lower()

    # Get batch size from config
    if evaluate:
        batch_size = config["training_kwargs"]["eval_batch_size"]
    else:
        batch_size = config["training_kwargs"]["train_batch_size"]

    # Process the appropriate dataset
    if dataset_name == "garden_path":
        data = process_garden_path_corpus(tokenizer, config, evaluate)
        dataset = GardenPathDataset(data)

        # Check if we should use paired batch sampler
        use_paired_sampler = config["training_kwargs"].get("use_paired_batch_sampler", False)

        if use_paired_sampler and not evaluate:
            # Use GardenPathBatchSampler for training
            selected_phenomena = config["training_kwargs"].get("selected_phenomena", None)
            if selected_phenomena and isinstance(selected_phenomena, str):
                selected_phenomena = [selected_phenomena]

            batch_sampler = GardenPathBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                selected_phenomena=selected_phenomena,
                shuffle=True,
                drop_last=False,
                epoch=0,
            )

            # Log batch sampler info
            batch_info = batch_sampler.get_batch_info()
            print(f"Garden-path batch sampler initialized:")
            print(f"  Strategy: {batch_info['strategy']}")
            print(f"  Batch size: {batch_info['batch_size']}")
            print(f"  Selected phenomena: {batch_info['selected_phenomena']}")
            print(f"  Estimated batches: {batch_info['estimated_batches']}")

            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=custom_collate_fn,
            )
        else:
            # Use standard DataLoader for evaluation or when paired sampling is disabled
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=not evaluate,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=custom_collate_fn,
            )

    else:
        # All other datasets are SPR datasets (natural_stories, smith2013, ucl_selfpaced, etc.)
        data = process_spr_corpus(tokenizer, config, dataset_name, evaluate)
        dataset = SPRDataset(data)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not evaluate,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )


def log_dataloaders(
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: dict,
    tokenizer: PreTrainedTokenizer,
    log_dir: str,
):
    """Log dataloader statistics and save data split information."""
    # Get dataset name from config
    dataset_name = config["training_kwargs"].get("train_dataset", "unknown")

    # Log training dataloader
    if train_dataloader is not None:
        print("\nTraining DataLoader Statistics:")
        mean_rt, std_rt = get_mean_std_reading_times_dataloader(train_dataloader)
        print(f"  Mean RT: {mean_rt:.2f}, Std RT: {std_rt:.2f}")
        print(f"  Number of batches: {len(train_dataloader)}")
        batch_size = config['training_kwargs']['train_batch_size']
        print(f"  Batch size: {batch_size}")

        # Log first batch for debugging
        for batch in train_dataloader:
            print(f"  First batch shape: {batch['input_ids'].shape}")
            if "construction" in batch:
                constructions = set()
                for row in batch["construction"]:
                    for c in row:
                        if str(c) not in ["PAD", ""]:
                            constructions.add(str(c))
                print(f"  Constructions in first batch: {constructions}")
            break

    # Log evaluation dataloader
    if eval_dataloader is not None:
        print("\nEvaluation DataLoader Statistics:")
        mean_rt, std_rt = get_mean_std_reading_times_dataloader(eval_dataloader)
        print(f"  Mean RT: {mean_rt:.2f}, Std RT: {std_rt:.2f}")
        print(f"  Number of batches: {len(eval_dataloader)}")

    # Save data split logs
    if log_dir and train_dataloader is not None:
        try:
            save_data_split_logs(
                train_dataloader,
                eval_dataloader,
                log_dir,
                tokenizer,
                config
            )
            print(f"\nData split logs saved to {log_dir}")
        except Exception as e:
            print(f"Warning: Could not save data split logs: {e}")
