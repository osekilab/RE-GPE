#!/usr/bin/env python3
"""Generate the appendix LaTeX table summarising perplexity + BLiMP results
for pre-fine-tuning, GP-fine-tuned, and RC-fine-tuned checkpoints.

Reads:
  perplexity_evaluation/results{,_md,_lg,_rc,_rc_md,_rc_lg}.json
  blimp_evaluation{,_md,_lg,_rc,_rc_md,_rc_lg}/summary.json

Writes the table to stdout and to ``figures/ppl_blimp_appendix.tex``.

Notes:
  - RC Large folds 20 and 22 failed to converge; they are excluded from
    every RC-Large statistic (perplexity, BLiMP) in this table.
  - Pre-fine-tuning values are shared between GP and RC (same baseline ckpt).
"""

import json
from pathlib import Path
import numpy as np


SIZES = [
    ("Small",
     "perplexity_evaluation/results.json",
     "perplexity_evaluation/results_rc.json",
     "blimp_evaluation",
     "blimp_evaluation_rc"),
    ("Medium",
     "perplexity_evaluation/results_md.json",
     "perplexity_evaluation/results_rc_md.json",
     "blimp_evaluation_md",
     "blimp_evaluation_rc_md"),
    ("Large",
     "perplexity_evaluation/results_lg.json",
     "perplexity_evaluation/results_rc_lg.json",
     "blimp_evaluation_lg",
     "blimp_evaluation_rc_lg"),
]
CORPORA = ["natural_stories", "smith2013", "ucl_selfpaced"]
VOCAB_SIZE = 50257  # GPT-2 BPE vocabulary
# RC Large: folds 20, 22 failed to converge (training divergence).
RC_LG_EXCLUDED_FOLDS = {20, 22}


def ppl_stats(ppl_file, exclude_folds=None):
    """Return {corpus: (baseline_ppl, trained_mean, trained_sem, n)} for a size."""
    exclude_folds = set(exclude_folds or [])
    d = json.loads(Path(ppl_file).read_text())
    out = {}
    for c in CORPORA:
        baseline = d.get("baseline", {}).get(c, {}).get("perplexity")
        vals = [
            d["folds"][fn][c]["perplexity"]
            for fn in d["folds"]
            if int(fn) not in exclude_folds and c in d["folds"][fn]
        ]
        arr = np.array(vals, dtype=float)
        mean = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        out[c] = (baseline, mean, sem, len(arr))
    return out


def blimp_stats(blimp_dir, exclude_folds=None):
    """Return (baseline_acc, trained_mean, trained_sem, n)."""
    exclude_folds = set(exclude_folds or [])
    summary = json.loads((Path(blimp_dir) / "summary.json").read_text())
    baseline_entry = summary.get("baseline") or {}
    baseline = baseline_entry.get("blimp")
    vals = [
        summary["folds"][fn]["blimp"]
        for fn in summary["folds"]
        if int(fn) not in exclude_folds and "blimp" in summary["folds"][fn]
    ]
    arr = np.array(vals, dtype=float)
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return baseline, mean, sem, len(arr)


def fmt_pre_ppl(v):
    return f"{v:.2f}"


def fmt_pre_blimp(v):
    return f"{v:.3f}"


def fmt_post_ppl(mean, sem):
    return f"{mean:.2f} \\pm {sem:.2f}"


def fmt_post_blimp(mean, sem):
    return f"{mean:.3f} \\pm {sem:.3f}"


def main():
    lines = [
        r"% Requires: \usepackage{siunitx,multirow,booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\sisetup{separate-uncertainty=true, table-align-uncertainty=true}",
        r"\begin{tabular}{@{}ll",
        r"  S[table-format=5.2(2.2)]",
        r"  S[table-format=5.2(2.2)]",
        r"  S[table-format=5.2(2.2)]",
        r"  @{\hspace{1.5em}}",
        r"  S[table-format=1.3(1.3)]@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Perplexity ($\downarrow$)} "
        r"& {Accuracy ($\uparrow$)} \\",
        r"\cmidrule(lr){3-5} \cmidrule(l){6-6}",
        r"Size & Condition & {Natural Stories} & {Brown} & {UCL} & {BLiMP} \\",
        r"\midrule",
    ]

    for idx, (size, gp_ppl_f, rc_ppl_f, gp_blimp_dir, rc_blimp_dir) in enumerate(SIZES):
        rc_exclude = RC_LG_EXCLUDED_FOLDS if size == "Large" else set()

        gp_ppl = ppl_stats(gp_ppl_f)
        rc_ppl = ppl_stats(rc_ppl_f, exclude_folds=rc_exclude)
        gp_blimp = blimp_stats(gp_blimp_dir)
        rc_blimp = blimp_stats(rc_blimp_dir, exclude_folds=rc_exclude)

        base_ns = gp_ppl["natural_stories"][0]
        base_sm = gp_ppl["smith2013"][0]
        base_uc = gp_ppl["ucl_selfpaced"][0]
        base_blimp = gp_blimp[0]

        lines.append(fr"\multirow{{3}}{{*}}{{{size}}}")
        lines.append(
            "  & Pre        & "
            f"{fmt_pre_ppl(base_ns)} & {fmt_pre_ppl(base_sm)} & "
            f"{fmt_pre_ppl(base_uc)} & {fmt_pre_blimp(base_blimp)} \\\\"
        )
        lines.append(
            "  & Post (GP)  & "
            f"{fmt_post_ppl(gp_ppl['natural_stories'][1], gp_ppl['natural_stories'][2])} & "
            f"{fmt_post_ppl(gp_ppl['smith2013'][1], gp_ppl['smith2013'][2])} & "
            f"{fmt_post_ppl(gp_ppl['ucl_selfpaced'][1], gp_ppl['ucl_selfpaced'][2])} & "
            f"{fmt_post_blimp(gp_blimp[1], gp_blimp[2])} \\\\"
        )
        lines.append(
            "  & Post (RC)  & "
            f"{fmt_post_ppl(rc_ppl['natural_stories'][1], rc_ppl['natural_stories'][2])} & "
            f"{fmt_post_ppl(rc_ppl['smith2013'][1], rc_ppl['smith2013'][2])} & "
            f"{fmt_post_ppl(rc_ppl['ucl_selfpaced'][1], rc_ppl['ucl_selfpaced'][2])} & "
            f"{fmt_post_blimp(rc_blimp[1], rc_blimp[2])} \\\\"
        )
        if idx != len(SIZES) - 1:
            lines.append(r"\addlinespace[2pt]")

    lines.extend([
        r"\midrule",
        r"\multicolumn{2}{@{}l}{Uniform baseline}",
        fr"  & \multicolumn{{1}}{{c}}{{{VOCAB_SIZE}}}",
        fr"  & \multicolumn{{1}}{{c}}{{{VOCAB_SIZE}}}",
        fr"  & \multicolumn{{1}}{{c}}{{{VOCAB_SIZE}}}",
        r"  & \multicolumn{1}{c}{0.500} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Perplexity (lower is better) on three naturalistic corpora "
        r"and BLiMP overall accuracy (higher is better) for pre- and "
        r"post-fine-tuned LMs (GP: garden-path effects, RC: SRC/ORC "
        r"asymmetry).  Values for post-fine-tuned LMs are means~$\pm$~standard "
        r"errors across folds. The uniform baseline assumes a uniform "
        r"distribution over the vocabulary.}",
        r"\label{tab:ppl_blimp_appendix}",
        r"\end{table*}",
    ])
    text = "\n".join(lines) + "\n"

    print(text)

    out = Path("figures/ppl_blimp_appendix.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
