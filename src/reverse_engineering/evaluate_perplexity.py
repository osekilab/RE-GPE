#!/usr/bin/env python3
"""Evaluate subword-level perplexity on SPR corpora.

Matches the same model input pipeline as delta log-likelihood evaluation:
- Sentences tokenized independently (not concatenated)
- BOS token prepended
- Padding to max_length
- Cross-entropy computed over valid (non-padding) subword tokens

Usage:
    # Single model
    uv run python evaluate_perplexity.py \
        --model-path output/gp_all/garden_path_fold_0/final_checkpoint

    # Baseline GPT-2
    uv run python evaluate_perplexity.py --model-path gpt2

    # All folds + baseline
    uv run python evaluate_perplexity.py --all-folds \
        --folds-dir output/gp_all \
        --output-path perplexity_evaluation/results.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Naturalistic corpora for perplexity (uniform max_length=50).
CORPUS_CONFIGS = {
    "natural_stories": {
        "path": "../../data/natural_stories_processed.csv",
        "max_length": 50,
        "display_name": "Natural Stories",
    },
    "smith2013": {
        "path": "../../data/smith2013_processed.csv",
        "max_length": 50,
        "display_name": "Brown (Smith 2013)",
    },
    "ucl_selfpaced": {
        "path": "../../data/ucl_selfpaced_processed.csv",
        "max_length": 50,
        "display_name": "UCL",
    },
}


class SentenceDataset(Dataset):
    """Simple dataset of tokenized sentences for perplexity computation."""

    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.attention_mask[idx], dtype=torch.long
            ),
        }


def load_and_tokenize_corpus(
    corpus_path: str,
    tokenizer,
    max_length: int,
) -> SentenceDataset:
    """Load corpus CSV and tokenize sentences independently.

    Matches the pipeline in corpus_evaluation.py / load_spr.py:
    - Group words by sentence_num to reconstruct sentences
    - Tokenize each sentence independently
    - Pad to max_length
    """
    df = pd.read_csv(corpus_path)

    # Handle column naming
    if "item" in df.columns and "sentence_num" not in df.columns:
        df["sentence_num"] = df["item"]

    # Reconstruct sentences from word columns
    sentences = []
    for _, group in df.groupby("sentence_num", sort=True):
        words = group["word"].tolist()
        sentence = " ".join(str(w) for w in words)
        sentences.append(sentence)

    # Tokenize all sentences with padding (matching the existing pipeline)
    tokenized = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="np",
    )

    return SentenceDataset(tokenized["input_ids"], tokenized["attention_mask"])


def compute_perplexity(
    model,
    tokenizer,
    dataset: SentenceDataset,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Compute subword-level perplexity.

    Matches the surprisal computation in surprisal.py:compute_surprisal():
    - BOS token prepended to each sentence
    - Labels masked with -100 for padding positions
    - Cross-entropy loss averaged over valid tokens
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Prepend BOS token (matching surprisal.py)
            bos = torch.full(
                (input_ids.size(0), 1),
                tokenizer.bos_token_id,
                dtype=torch.long,
                device=device,
            )
            input_ids_with_bos = torch.cat([bos, input_ids], dim=1)
            attention_mask_with_bos = torch.cat(
                [torch.ones_like(bos), attention_mask], dim=1
            )

            # Create labels: mask padding with -100 (matching surprisal.py)
            labels = input_ids_with_bos.clone()
            labels[attention_mask_with_bos == 0] = -100

            # Forward pass
            outputs = model(
                input_ids=input_ids_with_bos,
                attention_mask=attention_mask_with_bos,
                labels=labels,
            )

            # Count valid tokens (non-padding, excluding BOS position)
            # The model's loss is already averaged over valid tokens,
            # but we need per-token loss for correct aggregation across batches.
            # Recompute from logits for precise control.
            logits = outputs.logits  # (batch, seq_len+1, vocab)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Per-token cross-entropy (no reduction)
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Mask out -100 positions
            valid_mask = shift_labels.view(-1) != -100
            valid_loss = per_token_loss[valid_mask].sum().item()
            valid_count = valid_mask.sum().item()

            total_loss += valid_loss
            total_tokens += valid_count

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)

    return {
        "perplexity": perplexity,
        "mean_cross_entropy": mean_loss,
        "total_tokens": total_tokens,
    }


def evaluate_model_on_corpora(
    model_path: str,
    device: torch.device,
    corpora: Dict[str, dict],
    batch_size: int = 64,
) -> Dict[str, Dict]:
    """Evaluate a single model on all corpora."""
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    for corpus_name, corpus_config in corpora.items():
        corpus_path = Path(corpus_config["path"])
        if not corpus_path.exists():
            print(f"  Warning: {corpus_path} not found, skipping")
            continue

        print(f"  Evaluating on {corpus_config['display_name']}...")
        dataset = load_and_tokenize_corpus(
            str(corpus_path),
            tokenizer,
            max_length=corpus_config["max_length"],
        )

        result = compute_perplexity(
            model, tokenizer, dataset, device, batch_size=batch_size
        )
        results[corpus_name] = result
        print(
            f"    PPL: {result['perplexity']:.2f} "
            f"(CE: {result['mean_cross_entropy']:.4f}, "
            f"tokens: {result['total_tokens']})"
        )

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate subword-level perplexity on SPR corpora"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint or HF model name (e.g., gpt2)",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Evaluate all folds + baseline",
    )
    parser.add_argument(
        "--folds-dir",
        type=str,
        default="output/gp_all",
        help="Directory containing <fold-pattern>/final_checkpoint",
    )
    parser.add_argument(
        "--fold-pattern",
        type=str,
        default="garden_path_fold_*",
        help="Glob pattern for fold directories (e.g., 'relative_clause_fold_*')",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="gpt2",
        help="HF model id for the untrained baseline (e.g., gpt2, gpt2-medium, gpt2-large)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation (e.g., when reusing an existing baseline JSON)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="perplexity_evaluation/results.json",
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=list(CORPUS_CONFIGS.keys()),
        default=list(CORPUS_CONFIGS.keys()),
        help="Corpora to evaluate on",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Filter corpora
    corpora = {k: v for k, v in CORPUS_CONFIGS.items() if k in args.corpora}

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.all_folds:
        all_results = {}

        # Baseline
        if args.skip_baseline:
            print("\n=== Skipping baseline evaluation (--skip-baseline) ===")
            all_results["baseline"] = {}
        else:
            print(f"\n=== Evaluating baseline ({args.baseline_model}) ===")
            all_results["baseline"] = evaluate_model_on_corpora(
                args.baseline_model, device, corpora, batch_size=args.batch_size
            )

        # All folds
        folds_dir = Path(args.folds_dir)
        fold_dirs = sorted(
            folds_dir.glob(f"{args.fold_pattern}/final_checkpoint"),
            key=lambda p: int(p.parent.name.split("_")[-1]),
        )

        fold_results = {}
        for fold_path in fold_dirs:
            fold_num = int(fold_path.parent.name.split("_")[-1])
            print(f"\n=== Evaluating fold {fold_num} ===")
            fold_results[fold_num] = evaluate_model_on_corpora(
                str(fold_path), device, corpora, batch_size=args.batch_size
            )

        all_results["folds"] = fold_results

        # Compute summary statistics
        summary = {}
        for corpus_name in corpora:
            ppls = [
                fold_results[f][corpus_name]["perplexity"]
                for f in fold_results
                if corpus_name in fold_results[f]
            ]
            if ppls:
                summary[corpus_name] = {
                    "baseline_ppl": all_results["baseline"].get(
                        corpus_name, {}
                    ).get("perplexity"),
                    "trained_ppl_mean": float(np.mean(ppls)),
                    "trained_ppl_std": float(np.std(ppls)),
                    "trained_ppl_sem": float(
                        np.std(ppls) / np.sqrt(len(ppls))
                    ),
                    "n_folds": len(ppls),
                }
        all_results["summary"] = summary

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for corpus_name, stats in summary.items():
            display_name = corpora[corpus_name]["display_name"]
            print(f"\n{display_name}:")
            base = stats['baseline_ppl']
            base_str = f"{base:.2f}" if base is not None else "N/A (skipped)"
            print(f"  Baseline PPL:     {base_str}")
            print(
                f"  Trained PPL:      {stats['trained_ppl_mean']:.2f} "
                f"± {stats['trained_ppl_sem']:.2f} "
                f"(n={stats['n_folds']})"
            )

        # Save results
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    elif args.model_path:
        results = evaluate_model_on_corpora(
            args.model_path, device, corpora, batch_size=args.batch_size
        )

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    else:
        parser.error("Specify --model-path or --all-folds")


if __name__ == "__main__":
    main()
