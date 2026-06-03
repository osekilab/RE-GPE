#!/usr/bin/env python3
"""Visualization script for the paper figures.

Produces the six figures reported in the paper from the per-fold evaluation
summaries (garden-path and relative-clause effects, delta log-likelihood, and
the cross-construction transfer matrices), across GPT-2 small/medium/large.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


def load_summary_data(summary_path: Path) -> Dict:
    """Load summary JSON file."""
    with open(summary_path, "r") as f:
        return json.load(f)


def exclude_folds_from_summary(summary: Dict, excluded_fold_ids: List[int]) -> Dict:
    """Return a summary dict with per-fold statistics recomputed after dropping
    the given fold ids.

    Works on the output of ``evaluate_all_folds_rc.py`` / ``evaluate_all_folds.py``
    which stores per-fold lists under
      - ``relative_clause[cons][roi].{actual_values, predicted_values}``
      - ``garden_path[phenom][roi].{actual_values, predicted_values}``
      - ``individual_<corpus>.delta_llh_values``
    Mean / std / sem / n_folds are recomputed; entries lacking per-fold lists
    are left untouched.
    """
    if not excluded_fold_ids:
        return summary

    import copy
    new = copy.deepcopy(summary)
    exclude = set(excluded_fold_ids)

    folds_order = summary.get("folds", list(range(summary.get("n_folds", 0))))
    keep_mask = [fid not in exclude for fid in folds_order]
    kept_fold_ids = [fid for fid, keep in zip(folds_order, keep_mask) if keep]

    new["folds"] = kept_fold_ids
    new["n_folds"] = len(kept_fold_ids)

    def _recompute(section: Dict, value_keys_to_stat: Dict[str, str]):
        """Recompute mean/std/sem for each (value_key, stat_prefix) pair."""
        for vkey, prefix in value_keys_to_stat.items():
            if vkey not in section:
                continue
            vals = [v for v, keep in zip(section[vkey], keep_mask) if keep]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            section[vkey] = vals
            section[f"{prefix}_mean"] = float(arr.mean())
            section[f"{prefix}_std"] = float(arr.std(ddof=0))
            if arr.size > 1:
                section[f"{prefix}_sem"] = float(
                    arr.std(ddof=1) / np.sqrt(arr.size)
                )
            else:
                section[f"{prefix}_sem"] = 0.0
            section["n_folds"] = int(arr.size)

    # Garden-path and relative_clause: actual_values / predicted_values
    for top in ("garden_path", "relative_clause"):
        if top not in new:
            continue
        for cons in new[top]:
            for roi in new[top][cons]:
                _recompute(new[top][cons][roi], {
                    "actual_values": "actual",
                    "predicted_values": "predicted",
                })

    # Per-corpus delta_llh_values
    for key in list(new.keys()):
        if not key.startswith("individual_"):
            continue
        _recompute(new[key], {"delta_llh_values": "delta_llh"})
        # delta_llh_min/max if present
        sect = new[key]
        if "delta_llh_values" in sect and sect["delta_llh_values"]:
            sect["delta_llh_min"] = float(min(sect["delta_llh_values"]))
            sect["delta_llh_max"] = float(max(sect["delta_llh_values"]))

    # Filler block (if present)
    if "filler" in new:
        _recompute(new["filler"], {"delta_llh_values": "delta_llh"})
        if "delta_llh_values" in new["filler"] and new["filler"]["delta_llh_values"]:
            new["filler"]["delta_llh_min"] = float(min(new["filler"]["delta_llh_values"]))
            new["filler"]["delta_llh_max"] = float(max(new["filler"]["delta_llh_values"]))

    return new


def plot_garden_path_effects_combined(
    data_dict: Dict[str, Dict],
    baseline_dict: Dict[str, Dict],
    output_dir: Path,
    figsize: Tuple[float, float] = (10.0, 4.5)  # Wider horizontal layout
) -> None:
    """Plot garden-path effects comparison for all model sizes.

    Layout: phenomena (cols) x model sizes (rows)
    Each cell: ROIs color-coded, with Human/Before/After for each ROI

    Args:
        data_dict: Dict mapping model size ("small", "medium", "large") to trained data
        baseline_dict: Dict mapping model size to baseline data
        output_dir: Output directory
        figsize: Figure size
    """
    # Get phenomena and ROIs from first available data
    first_data = next(iter(data_dict.values()))
    gp_data = first_data["garden_path"]
    phenomena = list(gp_data.keys())
    rois = list(gp_data[phenomena[0]].keys())

    model_sizes = ["small", "medium", "large"]
    model_labels = ["Small", "Medium", "Large"]

    # Colors for Human/Before/After (colorblind-friendly: Okabe-Ito palette)
    condition_colors = {
        "human": "#999999",      # Gray
        "before": "#E69F00",     # Yellow/Orange
        "after": "#0072B2",      # Blue
    }

    roi_labels = {
        "roi0": "ROI 0",
        "roi1": "ROI 1",
        "roi2": "ROI 2",
    }

    # Pattern for Human/Before/After
    # Human: solid, Before: light/hatched, After: darker

    fig, axes = plt.subplots(
        len(model_sizes), len(phenomena),
        figsize=figsize,
        sharex=False,
        sharey=True
    )

    # Ensure axes is 2D
    if len(model_sizes) == 1:
        axes = axes.reshape(1, -1)
    if len(phenomena) == 1:
        axes = axes.reshape(-1, 1)

    bar_width = 0.08
    roi_gap = 0.12  # Gap between ROI groups
    condition_gap = 0.02  # Small gap between Human/Before/After

    for row, (size, size_label) in enumerate(zip(model_sizes, model_labels)):
        if size not in data_dict or size not in baseline_dict:
            continue

        for col, phenomenon in enumerate(phenomena):
            ax = axes[row, col]

            x_tick_positions = []
            x_tick_labels = []

            for roi_idx, roi in enumerate(rois):
                # Get data
                trained_data = data_dict[size]["garden_path"][phenomenon][roi]
                baseline_data_roi = baseline_dict[size]["garden_path"][phenomenon][roi]

                actual_mean = trained_data["actual_mean"]
                actual_std = trained_data["actual_std"]
                n_folds = trained_data["n_folds"]
                baseline_mean = baseline_data_roi["predicted_mean"]
                baseline_std = baseline_data_roi["predicted_std"]
                trained_mean = trained_data["predicted_mean"]
                trained_std = trained_data["predicted_std"]

                # Calculate x positions for this ROI group
                base_x = roi_idx * (3 * bar_width + condition_gap * 2 + roi_gap)
                x_human = base_x
                x_before = base_x + bar_width + condition_gap
                x_after = base_x + 2 * bar_width + 2 * condition_gap

                # Human bar (blue)
                ax.bar(
                    x_human, actual_mean, bar_width,
                    yerr=actual_std / np.sqrt(n_folds),
                    capsize=1.5,
                    color=condition_colors["human"],
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5
                )

                # Before fine-tuning bar (gray)
                ax.bar(
                    x_before, baseline_mean, bar_width,
                    yerr=baseline_std / np.sqrt(n_folds),
                    capsize=1.5,
                    color=condition_colors["before"],
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5
                )

                # After fine-tuning bar (purple)
                ax.bar(
                    x_after, trained_mean, bar_width,
                    yerr=trained_std / np.sqrt(n_folds),
                    capsize=1.5,
                    color=condition_colors["after"],
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5
                )

                # Track tick position (center of ROI group)
                x_tick_positions.append((x_human + x_after) / 2)
                x_tick_labels.append(roi_labels[roi])

            # Add horizontal line at 0
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

            # Set x-axis ticks
            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels(x_tick_labels, fontsize=10)

            # Title for top row (phenomenon names)
            if row == 0:
                ax.set_title(phenomenon, fontsize=12, fontweight="bold")

            # Y-label for leftmost column (model size bold, unit normal)
            if col == 0:
                ax.set_ylabel(f"$\\bf{{{size_label}}}$\nΔRT [ms]", fontsize=11)

            # Grid
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_axisbelow(True)

            # Y-axis ticks every 50 ms so 50 / 150 are shown
            ax.yaxis.set_major_locator(MultipleLocator(50))

            # Adjust tick label size
            ax.tick_params(axis='y', labelsize=10)

    # Add legend at top
    # Create custom legend for Human/Before/After with colors
    from matplotlib.patches import Patch as LegendPatch
    legend_elements = [
        LegendPatch(facecolor=condition_colors["human"], alpha=0.8, edgecolor="black", label="Human"),
        LegendPatch(facecolor=condition_colors["before"], alpha=0.8, edgecolor="black", label="Pre-fine-tuning"),
        LegendPatch(facecolor=condition_colors["after"], alpha=0.8, edgecolor="black", label="Post-fine-tuning"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=11
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Save figure
    output_path = output_dir / "garden_path_effects_combined.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved garden-path effects plot to {output_path}")

    plt.close()


def plot_delta_llh_combined(
    data_dict: Dict[str, Dict],
    baseline_dict: Dict[str, Dict],
    output_dir: Path,
    figsize: Tuple[float, float] = (8.0, 3.5)  # Wider horizontal layout
) -> None:
    """Plot delta log-likelihood comparison for all model sizes.

    Layout: corpora (columns) with model sizes grouped within each, dataset names on top.

    Args:
        data_dict: Dict mapping model size to trained data
        baseline_dict: Dict mapping model size to baseline data
        output_dir: Output directory
        figsize: Figure size
    """
    # Corpus mapping for display names
    corpus_map = {
        "natural_stories": "Natural Stories",
        "smith2013": "Brown",
        "ucl_selfpaced": "UCL"
    }

    # Get available corpora from data
    first_data = next(iter(data_dict.values()))
    available_corpora = []
    for key in first_data.keys():
        if key.startswith("individual_"):
            corpus_name = key.replace("individual_", "")
            if corpus_name in corpus_map:
                available_corpora.append(corpus_name)

    # Sort by desired order
    corpus_order = ["natural_stories", "smith2013", "ucl_selfpaced"]
    corpora = [c for c in corpus_order if c in available_corpora]

    if not corpora:
        print("No corpus data found")
        return

    model_sizes = ["small", "medium", "large"]
    model_labels = ["S", "M", "L"]  # Short labels for compact display

    # Colors - colorblind-friendly (Okabe-Ito palette)
    colors = {
        "baseline": "#E69F00",   # Yellow/Orange for baseline (Before fine-tuning)
        "trained": "#0072B2",    # Blue for trained (After fine-tuning)
    }

    fig, axes = plt.subplots(1, len(corpora), figsize=figsize, sharey=True)

    if len(corpora) == 1:
        axes = [axes]

    bar_width = 0.10
    group_gap = 0.08

    for col, corpus in enumerate(corpora):
        ax = axes[col]
        corpus_key = f"individual_{corpus}"

        x_tick_positions = []
        x_tick_labels = []

        for j, (size, label) in enumerate(zip(model_sizes, model_labels)):
            if size not in data_dict or size not in baseline_dict:
                continue

            # Get data
            trained_corpus = data_dict[size].get(corpus_key, {})
            baseline_corpus = baseline_dict[size].get(corpus_key, {})

            # Extract delta_llh
            trained_mean = trained_corpus.get("delta_llh_mean", trained_corpus.get("delta_llh", 0))
            trained_sem = trained_corpus.get("delta_llh_sem", 0)
            baseline_mean = baseline_corpus.get("delta_llh_mean", baseline_corpus.get("delta_llh", 0))
            baseline_sem = baseline_corpus.get("delta_llh_sem", 0)

            # Calculate positions
            base_x = j * (2 * bar_width + group_gap)
            x_baseline = base_x
            x_trained = base_x + bar_width

            # Baseline bar (no error bars: baseline has no per-fold variance)
            ax.bar(
                x_baseline, baseline_mean, bar_width,
                color=colors["baseline"],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5
            )

            # Trained bar
            ax.bar(
                x_trained, trained_mean, bar_width,
                yerr=trained_sem,
                capsize=2,
                color=colors["trained"],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
                error_kw={"elinewidth": 2.5, "capthick": 2.5, "ecolor": "black"},
            )

            # Track tick position (center of baseline/trained pair)
            x_tick_positions.append((x_baseline + x_trained) / 2)
            x_tick_labels.append(label)

        # Set x-axis ticks (model size labels - bold)
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, fontsize=16, fontweight="bold")

        # Dataset name as title (on top)
        ax.set_title(corpus_map[corpus], fontsize=16, fontweight="bold")

        # Add horizontal line at 0
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=0.5)

        # Grid
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

        # Y-axis label only for leftmost
        if col == 0:
            ax.set_ylabel("Δllh", fontsize=16)

        # Adjust tick label size
        ax.yaxis.set_major_locator(MultipleLocator(0.005))
        ax.tick_params(axis='y', labelsize=15)

    # Legend at top
    legend_elements = [
        Patch(facecolor=colors["baseline"], alpha=0.8, edgecolor="black", label="Pre-fine-tuning"),
        Patch(facecolor=colors["trained"], alpha=0.8, edgecolor="black", label="Post-fine-tuning"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        fontsize=16
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.78)

    # Save figure
    output_path = output_dir / "delta_llh_combined.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved delta log-likelihood plot to {output_path}")

    plt.close()


def plot_transfer_matrix_single_size(
    transfer_data_dict: Dict[str, Dict[str, Dict]],
    baseline_dict: Dict[str, Dict],
    output_dir: Path,
    model_size: str = "small",
    roi: str = "roi1",
    figsize: Tuple[float, float] = (8.0, 5.5)
) -> None:
    """Plot transfer learning results for a single model size (garden-path only).

    Layout: 3 rows (trained on) x 3 cols (evaluated on: MVRR, NPS, NPZ)
    Each cell: Human/Before/After bars
    Style matches garden_path_effects_combined.png

    Args:
        transfer_data_dict: Dict mapping training phenomenon to data_dict
        baseline_dict: Dict mapping model size to baseline data
        output_dir: Output directory
        model_size: Model size to use ("small", "medium", "large")
        roi: ROI to display ("roi0", "roi1", "roi2")
        figsize: Figure size
    """
    training_phenomena = ["MVRR", "NPS", "NPZ"]
    eval_phenomena = ["MVRR", "NPS", "NPZ"]

    model_label = {"small": "Small", "medium": "Medium", "large": "Large"}[model_size]

    # Colors for Human/Before/After (colorblind-friendly: Okabe-Ito palette)
    condition_colors = {
        "human": "#999999",      # Gray
        "before": "#E69F00",     # Yellow/Orange
        "after": "#0072B2",      # Blue
    }

    # Create figure
    fig, axes = plt.subplots(
        len(training_phenomena), len(eval_phenomena),
        figsize=figsize,
        sharey=True
    )

    # Bar layout parameters (slightly narrower for cleaner look)
    bar_width = 0.22
    bar_gap = 0.03

    for row, train_phenom in enumerate(training_phenomena):
        if train_phenom not in transfer_data_dict:
            continue

        train_data = transfer_data_dict[train_phenom]
        if model_size not in train_data:
            continue

        for col, eval_phenom in enumerate(eval_phenomena):
            ax = axes[row, col]

            # Get data
            trained_data_gp = train_data[model_size]["garden_path"][eval_phenom][roi]
            baseline_data_roi = baseline_dict[model_size]["garden_path"][eval_phenom][roi]

            actual_mean = trained_data_gp["actual_mean"]
            actual_std = trained_data_gp["actual_std"]
            n_folds = trained_data_gp["n_folds"]
            baseline_mean = baseline_data_roi["predicted_mean"]
            baseline_std = baseline_data_roi["predicted_std"]
            trained_mean = trained_data_gp["predicted_mean"]
            trained_std = trained_data_gp["predicted_std"]

            # Human/Before/After bars with small gaps
            x_positions = [0, bar_width + bar_gap, 2 * (bar_width + bar_gap)]

            ax.bar(
                x_positions[0], actual_mean, bar_width,
                yerr=actual_std / np.sqrt(n_folds),
                capsize=2,
                color=condition_colors["human"],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5
            )
            ax.bar(
                x_positions[1], baseline_mean, bar_width,
                yerr=baseline_std / np.sqrt(n_folds),
                capsize=2,
                color=condition_colors["before"],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5
            )
            ax.bar(
                x_positions[2], trained_mean, bar_width,
                yerr=trained_std / np.sqrt(n_folds),
                capsize=2,
                color=condition_colors["after"],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5
            )

            ax.set_xticks([])

            # Title for top row
            if row == 0:
                ax.set_title(f"Eval: {eval_phenom}", fontsize=17, fontweight="bold")

            # Highlight diagonal (trained = evaluated) - consistent color with all_sizes
            if train_phenom == eval_phenom:
                ax.set_facecolor("#E8F8E8")

            # Add horizontal line at 0
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

            # Y-label for leftmost column: "Train: X" bold, "ΔRT [ms]" normal
            if col == 0:
                ax.set_ylabel(f"$\\bf{{Train: {train_phenom}}}$\nΔRT [ms]", fontsize=16)

            # Grid
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.tick_params(axis='y', labelsize=15)

    # Add legend at top
    from matplotlib.patches import Patch as LegendPatch
    legend_elements = [
        LegendPatch(facecolor=condition_colors["human"], alpha=0.8, edgecolor="black", label="Human"),
        LegendPatch(facecolor=condition_colors["before"], alpha=0.8, edgecolor="black", label="Pre-fine-tuning"),
        LegendPatch(facecolor=condition_colors["after"], alpha=0.8, edgecolor="black", label="Post-fine-tuning"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        fontsize=16
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)

    # Save figure
    output_path = output_dir / f"transfer_matrix_{model_size}_{roi}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved transfer matrix ({model_size}) to {output_path}")

    plt.close()


def plot_transfer_matrix_all_sizes(
    transfer_data_dict: Dict[str, Dict[str, Dict]],
    baseline_dict: Dict[str, Dict],
    output_dir: Path,
    roi: str = "roi1",
    figsize: Tuple[float, float] = (18.0, 17.0),
    include_delta_llh: bool = True,
    include_horizontal_separator: bool = True,
    include_vertical_separator: bool = True,
    output_suffix: str = "",
    model_sizes: Optional[List[str]] = None,
    legend_y: float = 1.005,
    subplots_top: float = 0.94,
) -> None:
    """Plot transfer learning results for all model sizes (appendix figure).

    Layout: 9 rows (3 models x 3 train) x {3 or 4} cols
    Style matches garden_path_effects_combined.png and delta_llh_combined.png

    Args:
        transfer_data_dict: Dict mapping training phenomenon to data_dict
        baseline_dict: Dict mapping model size to baseline data
        output_dir: Output directory
        roi: ROI to display
        figsize: Figure size
        include_delta_llh: Include the rightmost Δllh column
        include_separators: Draw the horizontal lines between model-size groups
            and the vertical line before the Δllh column
        output_suffix: Extra string appended to the output filename (before .png)
    """
    training_phenomena = ["MVRR", "NPS", "NPZ"]
    eval_phenomena = ["MVRR", "NPS", "NPZ"]
    if model_sizes is None:
        model_sizes = ["small", "medium", "large"]
    model_labels = {"small": "Small", "medium": "Medium", "large": "Large"}
    datasets = ["natural_stories", "smith2013", "ucl_selfpaced"]
    dataset_labels = {"natural_stories": "Nat.", "smith2013": "Brn.", "ucl_selfpaced": "UCL"}

    # Colors matching garden_path_effects_combined.png
    condition_colors = {
        "human": "#999999",
        "before": "#E69F00",
        "after": "#0072B2",
    }

    n_rows = len(model_sizes) * len(training_phenomena)
    n_cols = len(eval_phenomena) + (1 if include_delta_llh else 0)

    # Use GridSpec with different column widths
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=figsize)
    width_ratios = [1, 1, 1, 1.4] if include_delta_llh else [1, 1, 1]
    gs = GridSpec(n_rows, n_cols, figure=fig, width_ratios=width_ratios)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)])

    # Bar parameters matching garden_path_effects_combined style
    bar_width = 0.08
    condition_gap = 0.02

    for model_idx, model_size in enumerate(model_sizes):
        for row, train_phenom in enumerate(training_phenomena):
            if train_phenom not in transfer_data_dict:
                continue

            train_data = transfer_data_dict[train_phenom]
            if model_size not in train_data:
                continue

            actual_row = model_idx * len(training_phenomena) + row

            # --- Garden-path columns (0, 1, 2) ---
            for col, eval_phenom in enumerate(eval_phenomena):
                ax = axes[actual_row, col]

                trained_data_gp = train_data[model_size]["garden_path"][eval_phenom][roi]
                baseline_data_roi = baseline_dict[model_size]["garden_path"][eval_phenom][roi]

                actual_mean = trained_data_gp["actual_mean"]
                actual_std = trained_data_gp["actual_std"]
                n_folds = trained_data_gp["n_folds"]
                baseline_mean = baseline_data_roi["predicted_mean"]
                baseline_std = baseline_data_roi["predicted_std"]
                trained_mean = trained_data_gp["predicted_mean"]
                trained_std = trained_data_gp["predicted_std"]

                # Bar positions (matching garden_path_effects_combined)
                base_x = 0
                x_human = base_x
                x_before = base_x + bar_width + condition_gap
                x_after = base_x + 2 * bar_width + 2 * condition_gap

                ax.bar(x_human, actual_mean, bar_width,
                       yerr=actual_std / np.sqrt(n_folds), capsize=1.5,
                       color=condition_colors["human"], alpha=0.8,
                       edgecolor="black", linewidth=0.5)
                ax.bar(x_before, baseline_mean, bar_width,
                       yerr=baseline_std / np.sqrt(n_folds), capsize=1.5,
                       color=condition_colors["before"], alpha=0.8,
                       edgecolor="black", linewidth=0.5)
                ax.bar(x_after, trained_mean, bar_width,
                       yerr=trained_std / np.sqrt(n_folds), capsize=1.5,
                       color=condition_colors["after"], alpha=0.8,
                       edgecolor="black", linewidth=0.5)

                ax.set_xticks([])
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

                # Highlight diagonal (trained on same phenomenon)
                if train_phenom == eval_phenom:
                    ax.set_facecolor("#E8F8E8")

                # Column headers (Eval:) for top row only
                if actual_row == 0:
                    ax.set_title(f"Eval: {eval_phenom}", fontsize=16, fontweight="bold")

                # Y-label: "Train: X" bold + "ΔRT [ms]", model size on NPS row (middle)
                if col == 0:
                    if row == 1:  # NPS row (middle) - show model size here
                        ax.set_ylabel(f"$\\bf{{{model_labels[model_size]}}}$\n$\\bf{{Train: {train_phenom}}}$\nΔRT [ms]", fontsize=16)
                    else:
                        ax.set_ylabel(f"$\\bf{{Train: {train_phenom}}}$\nΔRT [ms]", fontsize=16)

                ax.grid(True, alpha=0.3, axis="y")
                ax.set_axisbelow(True)
                ax.yaxis.set_major_locator(MultipleLocator(50))
                ax.tick_params(axis='y', labelsize=15)

            # --- Delta log-lik column (3 datasets with x-tick labels) ---
            if include_delta_llh:
                col = len(eval_phenomena)
                ax = axes[actual_row, col]

                # Bar parameters for delta log-lik (wider bars like delta_llh_combined.png)
                dll_bar_width = 0.10
                dll_condition_gap = 0.02
                dll_group_gap = 0.08

                x_tick_positions = []
                x_tick_labels = []

                for ds_idx, ds in enumerate(datasets):
                    key = f"individual_{ds}"
                    baseline_corpus = baseline_dict[model_size].get(key, {})
                    trained_corpus = train_data[model_size].get(key, {})

                    baseline_val = baseline_corpus.get("delta_llh_mean", baseline_corpus.get("delta_llh", 0))
                    baseline_sem = baseline_corpus.get("delta_llh_sem", 0)
                    trained_val = trained_corpus.get("delta_llh_mean", trained_corpus.get("delta_llh", 0))
                    trained_sem = trained_corpus.get("delta_llh_sem", 0)

                    base_x = ds_idx * (2 * dll_bar_width + dll_condition_gap + dll_group_gap)
                    x_before = base_x
                    x_after = base_x + dll_bar_width + dll_condition_gap

                    ax.bar(x_before, baseline_val, dll_bar_width,
                           color=condition_colors["before"], alpha=0.8,
                           edgecolor="black", linewidth=0.5)
                    ax.bar(x_after, trained_val, dll_bar_width,
                           yerr=trained_sem, capsize=2,
                           color=condition_colors["after"], alpha=0.8,
                           edgecolor="black", linewidth=0.5,
                           error_kw={"elinewidth": 2.5, "capthick": 2.5, "ecolor": "black"})

                    # Track tick position and label
                    x_tick_positions.append((x_before + x_after) / 2)
                    x_tick_labels.append(dataset_labels[ds])

                # Set x-axis ticks (dataset labels - bold like delta_llh_combined.png)
                ax.set_xticks(x_tick_positions)
                ax.set_xticklabels(x_tick_labels, fontsize=16, fontweight="bold")

                # Column header for top row (using Greek delta like delta_llh_combined.png)
                if actual_row == 0:
                    ax.set_title("Δllh", fontsize=16, fontweight="bold")

                ax.grid(True, alpha=0.3, axis="y")
                ax.set_axisbelow(True)
                ax.yaxis.set_major_locator(MultipleLocator(0.01))
                ax.tick_params(axis='y', labelsize=15)

    # Synchronize y-axis for GP columns and delta-llh columns separately
    # GP columns
    gp_ylims = []
    for r in range(n_rows):
        for c in range(len(eval_phenomena)):
            gp_ylims.extend(axes[r, c].get_ylim())
    gp_ymin, gp_ymax = min(gp_ylims), max(gp_ylims)
    for r in range(n_rows):
        for c in range(len(eval_phenomena)):
            axes[r, c].set_ylim(gp_ymin, gp_ymax)

    # Delta log-lik column
    if include_delta_llh:
        dll_ylims = []
        for r in range(n_rows):
            dll_ylims.extend(axes[r, len(eval_phenomena)].get_ylim())
        dll_ymin, dll_ymax = min(dll_ylims), max(dll_ylims)
        for r in range(n_rows):
            axes[r, len(eval_phenomena)].set_ylim(dll_ymin, dll_ymax)

    # Add legend at top (matching garden_path_effects_combined style)
    legend_elements = [
        Patch(facecolor=condition_colors["human"], alpha=0.8, edgecolor="black", label="Human"),
        Patch(facecolor=condition_colors["before"], alpha=0.8, edgecolor="black", label="Pre-fine-tuning"),
        Patch(facecolor=condition_colors["after"], alpha=0.8, edgecolor="black", label="Post-fine-tuning"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=3,
        fontsize=16
    )

    plt.subplots_adjust(top=subplots_top, bottom=0.03, left=0.14, right=0.98, hspace=0.65, wspace=0.30)

    # Need layout to compute separator line positions
    if include_horizontal_separator or (include_vertical_separator and include_delta_llh):
        fig.canvas.draw()

    # Horizontal separator lines between model size groups
    if include_horizontal_separator:
        for model_idx in range(1, len(model_sizes)):
            # Draw line between model groups
            separator_row = model_idx * len(training_phenomena)
            ax_above = axes[separator_row - 1, 0]
            ax_below = axes[separator_row, 0]
            bbox_above = ax_above.get_position()
            bbox_below = ax_below.get_position()
            line_y = (bbox_above.y0 + bbox_below.y1) / 2
            line = plt.Line2D(
                [0.08, 0.98], [line_y, line_y],
                transform=fig.transFigure,
                color='#666666', linewidth=2.0, linestyle='-', alpha=0.8
            )
            fig.add_artist(line)

    # Vertical separator line between GP columns and Delta Log-Lik column
    if include_vertical_separator and include_delta_llh:
        ax_npz = axes[0, 2]  # NPZ column (last GP column)
        ax_dll = axes[0, 3]  # Delta Log-Lik column
        bbox_npz = ax_npz.get_position()
        bbox_dll = ax_dll.get_position()
        # Pulled left from midpoint to avoid overlapping Δllh y-tick labels
        line_x = bbox_npz.x1 + 0.15 * (bbox_dll.x0 - bbox_npz.x1)
        bbox_top = axes[0, 0].get_position()
        bbox_bottom = axes[-1, 0].get_position()
        line = plt.Line2D(
            [line_x, line_x], [bbox_bottom.y0, bbox_top.y1],
            transform=fig.transFigure,
            color='#666666', linewidth=2.0, linestyle='-', alpha=0.8
        )
        fig.add_artist(line)

    # Save
    output_path = output_dir / f"transfer_matrix_all_sizes_{roi}{output_suffix}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved transfer matrix (all sizes) to {output_path}")
    plt.close()


def load_transfer_data(base_dir: Path, phenomenon: str) -> Tuple[Dict[str, Dict], bool]:
    """Load data for transfer learning experiments (trained on one phenomenon).

    Args:
        base_dir: Base directory containing JSON files
        phenomenon: The phenomenon to load (mvrr, nps, npz)

    Returns:
        Tuple of (data_dict, success)
    """
    data_dict = {}
    size_suffixes = {
        "small": "",
        "medium": "_md",
        "large": "_lg",
    }

    for size, suffix in size_suffixes.items():
        filename = f"all_folds_evaluation_summary_{phenomenon}{suffix}.json"
        filepath = base_dir / filename
        if filepath.exists():
            print(f"  Loading {size} model from {filepath}")
            data_dict[size] = load_summary_data(filepath)
        else:
            print(f"  Warning: {filepath} not found")

    return data_dict, bool(data_dict)


def plot_rc_effects_combined(
    data_dict: Dict[str, Dict],
    baseline_dict: Dict[str, Dict],
    output_dir: Path,
    figsize: Tuple[float, float] = (4.0, 4.5)
) -> None:
    """Plot relative clause effects comparison for all model sizes.

    Layout: model sizes (rows) x ROIs (grouped bars within single column)

    Args:
        data_dict: Dict mapping model size to trained data
        baseline_dict: Dict mapping model size to baseline data
        output_dir: Output directory
        figsize: Figure size
    """
    model_sizes = ["small", "medium", "large"]
    model_labels = ["Small", "Medium", "Large"]
    rois = ["roi0", "roi1", "roi2"]

    # Colors for Human/Before/After (colorblind-friendly: Okabe-Ito palette)
    condition_colors = {
        "human": "#999999",      # Gray
        "before": "#E69F00",     # Yellow/Orange
        "after": "#0072B2",      # Blue
    }

    roi_labels = {
        "roi0": "ROI 0",
        "roi1": "ROI 1",
        "roi2": "ROI 2",
    }

    fig, axes = plt.subplots(
        len(model_sizes), 1,
        figsize=figsize,
        sharex=False,  # Show ROI labels on all rows
        sharey=True
    )

    bar_width = 0.08
    roi_gap = 0.12
    condition_gap = 0.02

    for row, (size, size_label) in enumerate(zip(model_sizes, model_labels)):
        if size not in data_dict or size not in baseline_dict:
            continue

        ax = axes[row]
        x_tick_positions = []
        x_tick_labels = []

        for roi_idx, roi in enumerate(rois):
            # Get data
            trained_data = data_dict[size]["relative_clause"]["RelativeClause"][roi]
            baseline_data_roi = baseline_dict[size]["relative_clause"]["RelativeClause"][roi]

            actual_mean = trained_data["actual_mean"]
            actual_std = trained_data["actual_std"]
            n_folds = trained_data["n_folds"]
            baseline_mean = baseline_data_roi["predicted_mean"]
            baseline_std = baseline_data_roi["predicted_std"]
            trained_mean = trained_data["predicted_mean"]
            trained_std = trained_data["predicted_std"]

            # Calculate x positions for this ROI group
            base_x = roi_idx * (3 * bar_width + condition_gap * 2 + roi_gap)
            x_human = base_x
            x_before = base_x + bar_width + condition_gap
            x_after = base_x + 2 * bar_width + 2 * condition_gap

            # Human bar
            ax.bar(
                x_human, actual_mean, bar_width,
                yerr=actual_std / np.sqrt(n_folds),
                capsize=1.5,
                color=condition_colors["human"],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5
            )

            # Before fine-tuning bar
            ax.bar(
                x_before, baseline_mean, bar_width,
                yerr=baseline_std / np.sqrt(n_folds),
                capsize=1.5,
                color=condition_colors["before"],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5
            )

            # After fine-tuning bar
            ax.bar(
                x_after, trained_mean, bar_width,
                yerr=trained_std / np.sqrt(n_folds),
                capsize=1.5,
                color=condition_colors["after"],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5
            )

            x_tick_positions.append((x_human + x_after) / 2)
            x_tick_labels.append(roi_labels[roi])

        # Add horizontal line at 0
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

        # Set x-axis ticks
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, fontsize=10)

        # Y-label (model size)
        ax.set_ylabel(f"$\\bf{{{size_label}}}$\nΔRT [ms]", fontsize=11)

        # Grid
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.tick_params(axis='y', labelsize=10)

    # Title
    axes[0].set_title("Relative Clause", fontsize=12, fontweight="bold")

    # Add legend at top
    from matplotlib.patches import Patch as LegendPatch
    legend_elements = [
        LegendPatch(facecolor=condition_colors["human"], alpha=0.8, edgecolor="black", label="Human"),
        LegendPatch(facecolor=condition_colors["before"], alpha=0.8, edgecolor="black", label="Pre-fine-tuning"),
        LegendPatch(facecolor=condition_colors["after"], alpha=0.8, edgecolor="black", label="Post-fine-tuning"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=10
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Save figure
    output_path = output_dir / "rc_effects_combined.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved RC effects plot to {output_path}")

    plt.close()


def plot_rc_delta_llh_combined(
    data_dict: Dict[str, Dict],
    baseline_dict: Dict[str, Dict],
    output_dir: Path,
    figsize: Tuple[float, float] = (8.0, 3.5)
) -> None:
    """Plot delta log-likelihood comparison for RC models.

    Same layout as plot_delta_llh_combined.

    Args:
        data_dict: Dict mapping model size to trained data
        baseline_dict: Dict mapping model size to baseline data
        output_dir: Output directory
        figsize: Figure size
    """
    # Corpus mapping for display names
    corpus_map = {
        "natural_stories": "Natural Stories",
        "smith2013": "Brown",
        "ucl_selfpaced": "UCL"
    }

    # Get available corpora from data
    first_data = next(iter(data_dict.values()))
    available_corpora = []
    for key in first_data.keys():
        if key.startswith("individual_"):
            corpus_name = key.replace("individual_", "")
            if corpus_name in corpus_map:
                available_corpora.append(corpus_name)

    # Sort by desired order
    corpus_order = ["natural_stories", "smith2013", "ucl_selfpaced"]
    corpora = [c for c in corpus_order if c in available_corpora]

    if not corpora:
        print("No corpus data found for RC")
        return

    model_sizes = ["small", "medium", "large"]
    model_labels = ["S", "M", "L"]

    # Colors
    colors = {
        "baseline": "#E69F00",
        "trained": "#0072B2",
    }

    fig, axes = plt.subplots(1, len(corpora), figsize=figsize, sharey=True)

    if len(corpora) == 1:
        axes = [axes]

    bar_width = 0.10
    group_gap = 0.08

    for col, corpus in enumerate(corpora):
        ax = axes[col]
        corpus_key = f"individual_{corpus}"

        x_tick_positions = []
        x_tick_labels = []

        for j, (size, label) in enumerate(zip(model_sizes, model_labels)):
            if size not in data_dict or size not in baseline_dict:
                continue

            trained_corpus = data_dict[size].get(corpus_key, {})
            baseline_corpus = baseline_dict[size].get(corpus_key, {})

            trained_mean = trained_corpus.get("delta_llh_mean", trained_corpus.get("delta_llh", 0))
            trained_sem = trained_corpus.get("delta_llh_sem", 0)
            baseline_mean = baseline_corpus.get("delta_llh_mean", baseline_corpus.get("delta_llh", 0))
            baseline_sem = baseline_corpus.get("delta_llh_sem", 0)

            base_x = j * (2 * bar_width + group_gap)
            x_baseline = base_x
            x_trained = base_x + bar_width

            ax.bar(
                x_baseline, baseline_mean, bar_width,
                color=colors["baseline"],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5
            )

            ax.bar(
                x_trained, trained_mean, bar_width,
                yerr=trained_sem,
                capsize=2,
                color=colors["trained"],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
                error_kw={"elinewidth": 2.5, "capthick": 2.5, "ecolor": "black"},
            )

            x_tick_positions.append((x_baseline + x_trained) / 2)
            x_tick_labels.append(label)

        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, fontsize=16, fontweight="bold")
        ax.set_title(corpus_map[corpus], fontsize=16, fontweight="bold")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

        if col == 0:
            ax.set_ylabel("Δllh", fontsize=16)

        ax.yaxis.set_major_locator(MultipleLocator(0.005))
        ax.tick_params(axis='y', labelsize=15)

    # Legend
    legend_elements = [
        Patch(facecolor=colors["baseline"], alpha=0.8, edgecolor="black", label="Pre-fine-tuning"),
        Patch(facecolor=colors["trained"], alpha=0.8, edgecolor="black", label="Post-fine-tuning"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        fontsize=16
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.78)

    output_path = output_dir / "rc_delta_llh_combined.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved RC delta log-likelihood plot to {output_path}")

    plt.close()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize evaluation results for ARR"
    )
    parser.add_argument(
        "--summary-small",
        type=Path,
        default=Path("all_folds_evaluation_summary.json"),
        help="Path to small model summary JSON"
    )
    parser.add_argument(
        "--summary-medium",
        type=Path,
        default=Path("all_folds_evaluation_summary_md.json"),
        help="Path to medium model summary JSON"
    )
    parser.add_argument(
        "--summary-large",
        type=Path,
        default=Path("all_folds_evaluation_summary_lg.json"),
        help="Path to large model summary JSON"
    )
    parser.add_argument(
        "--baseline-small",
        type=Path,
        default=Path("baseline_evaluation/baseline_summary.json"),
        help="Path to small baseline summary JSON"
    )
    parser.add_argument(
        "--baseline-medium",
        type=Path,
        default=Path("baseline_evaluation_md/baseline_summary.json"),
        help="Path to medium baseline summary JSON"
    )
    parser.add_argument(
        "--baseline-large",
        type=Path,
        default=Path("baseline_evaluation_lg/baseline_summary.json"),
        help="Path to large baseline summary JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for plots"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory containing all JSON files"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data for each model size
    data_dict = {}
    baseline_dict = {}

    size_files = [
        ("small", args.summary_small, args.baseline_small),
        ("medium", args.summary_medium, args.baseline_medium),
        ("large", args.summary_large, args.baseline_large),
    ]

    for size, summary_path, baseline_path in size_files:
        if summary_path.exists():
            print(f"Loading {size} model summary from {summary_path}")
            data_dict[size] = load_summary_data(summary_path)
        else:
            print(f"Warning: {summary_path} not found")

        if baseline_path.exists():
            print(f"Loading {size} baseline from {baseline_path}")
            baseline_dict[size] = load_summary_data(baseline_path)
        else:
            print(f"Warning: {baseline_path} not found")

    if not data_dict:
        print("Error: No data loaded")
        return

    # Create the paper figures only.
    print("\nFigure: garden_path_effects_combined ...")
    plot_garden_path_effects_combined(data_dict, baseline_dict, args.output_dir)

    print("\nFigure: delta_llh_combined ...")
    plot_delta_llh_combined(data_dict, baseline_dict, args.output_dir)

    # Load cross-construction transfer data (used by the transfer matrices).
    transfer_phenomena = ["mvrr", "nps", "npz"]
    transfer_data_dict = {}
    for phenomenon in transfer_phenomena:
        print(f"\n--- Loading {phenomenon.upper()} transfer data ---")
        transfer_data, success = load_transfer_data(args.base_dir, phenomenon)
        if success:
            transfer_data_dict[phenomenon.upper()] = transfer_data
        else:
            print(f"Skipping {phenomenon.upper()} transfer (no data)")

    if transfer_data_dict:
        # Figure: transfer_matrix_small_roi1
        print("\nFigure: transfer_matrix_small_roi1 ...")
        plot_transfer_matrix_single_size(
            transfer_data_dict,
            baseline_dict,
            args.output_dir,
            model_size="small",
            roi="roi1",
        )

        # Figure: transfer_matrix_all_sizes_roi1_md_lg_clean (appendix)
        print("\nFigure: transfer_matrix_all_sizes_roi1_md_lg_clean ...")
        plot_transfer_matrix_all_sizes(
            transfer_data_dict,
            baseline_dict,
            args.output_dir,
            roi="roi1",
            figsize=(14.0, 12.0),
            include_delta_llh=False,
            include_horizontal_separator=True,
            include_vertical_separator=False,
            output_suffix="_md_lg_clean",
            model_sizes=["medium", "large"],
            legend_y=1.02,
            subplots_top=0.93,
        )

    # --- RC (Relative Clause) plots ---
    print("\n" + "=" * 50)
    print("Creating RC (Relative Clause) plots...")
    print("=" * 50)

    # Load RC data
    rc_data_dict = {}
    rc_baseline_dict = {}

    rc_size_files = [
        ("small", Path("all_folds_evaluation_summary_rc.json"),
         Path("baseline_evaluation_rc/baseline_summary.json")),
        ("medium", Path("all_folds_evaluation_summary_rc_md.json"),
         Path("baseline_evaluation_rc_md/baseline_summary.json")),
        ("large", Path("all_folds_evaluation_summary_rc_lg.json"),
         Path("baseline_evaluation_rc_lg/baseline_summary.json")),
    ]

    # Folds whose fine-tuning diverged for the Large RC model; excluded from
    # every Large-RC statistic for consistency with the appendix PPL/BLiMP
    # table.
    RC_LG_EXCLUDED = [20, 22]

    for size, summary_path, baseline_path in rc_size_files:
        if summary_path.exists():
            print(f"Loading RC {size} model summary from {summary_path}")
            data = load_summary_data(summary_path)
            if size == "large":
                print(f"  Dropping RC Large failed folds {RC_LG_EXCLUDED} from aggregates")
                data = exclude_folds_from_summary(data, RC_LG_EXCLUDED)
            rc_data_dict[size] = data
        else:
            print(f"Warning: {summary_path} not found")

        if baseline_path.exists():
            print(f"Loading RC {size} baseline from {baseline_path}")
            rc_baseline_dict[size] = load_summary_data(baseline_path)
        else:
            print(f"Warning: {baseline_path} not found")

    if rc_data_dict and rc_baseline_dict:
        print("\nCreating RC effects plot...")
        plot_rc_effects_combined(rc_data_dict, rc_baseline_dict, args.output_dir)

        print("\nCreating RC delta log-likelihood plot...")
        plot_rc_delta_llh_combined(rc_data_dict, rc_baseline_dict, args.output_dir)
    else:
        print("Skipping RC plots (no data)")

    print("\nDone!")


if __name__ == "__main__":
    main()
