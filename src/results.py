"""Reading results/runs.jsonl and turning it into tables.

One line of JSON per run goes in; a tidy frame, a mean-plus-or-minus-std
summary and the deliverable Markdown table come out.

Two conventions kept throughout:

**Aggregate over seeds, never over folds.** Three seeds of the same
(arm, target, fraction) are repeats of one measurement and get averaged with a
spread. Different targets are different questions -- Cora accuracy and
Elliptic AUC-PR are not commensurable and are never pooled into one number.

**Arm D collapses by taking the best source.** Arm D asks "would a single
source have done just as well?", so the fair comparison against arm A is the
*best* single source, not the average of them. Averaging would flatter arm A
by dragging in whichever source happened to be useless. The per-source detail
stays available for notebook 07.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import config

ARM_LABELS = {
    "A_transfer": "A transfer (3 sources)",
    "B_random": "B random-init",
    "C_scratch": "C from-scratch",
    "D_expert": "D single-source expert",
}

#: Order arms appear in tables and legends.
ARM_ORDER = ["B_random", "D_expert", "A_transfer", "C_scratch"]


def load_frame(path: Path = config.RUNS_JSONL) -> "Any":
    """Every run as one tidy row."""
    import pandas as pd

    rows = []
    for record in config.load_runs(path):
        cfg, met = record["config"], record["metrics"]
        rows.append({
            "run_id": record["run_id"],
            "config_hash": record["config_hash"],
            "git_commit": record.get("git_commit"),
            "arm": cfg["arm"],
            "target": cfg["target_domain"],
            "sources": tuple(cfg["source_domains"]),
            "source_label": ", ".join(cfg["source_domains"]) or "-",
            "seed": cfg["seed"],
            "fraction": cfg["label_fraction"],
            "metric": met["primary_metric"],
            "score": met["primary_value"],
            "converged": met.get("converged", False),
            "train_nodes": met.get("train_nodes"),
            "seconds": met.get("total_seconds"),
        })
    # Build with an explicit column list so an empty results file still yields
    # a frame with the right shape. Otherwise every downstream groupby fails
    # with a bare KeyError, which is a confusing way to be told "no runs yet"
    # -- and that is exactly the state the notebook is in before the sweep.
    columns = ["run_id", "config_hash", "git_commit", "arm", "target", "sources",
               "source_label", "seed", "fraction", "metric", "score",
               "converged", "train_nodes", "seconds"]
    df = pd.DataFrame(rows, columns=columns)
    df["arm_label"] = df["arm"].map(ARM_LABELS)
    return df


def coverage(df, expected: int = 360) -> dict[str, Any]:
    """Is the matrix complete, and did everything converge?

    Worth checking before reading any result: a table built from a partial
    sweep looks exactly like a table built from a complete one.
    """
    return {
        "rows": len(df),
        "expected": expected,
        "complete": len(df) >= expected,
        "unique_runs": df["run_id"].nunique() if len(df) else 0,
        "duplicated_run_ids": int(len(df) - df["run_id"].nunique()) if len(df) else 0,
        "not_converged": int((~df["converged"]).sum()) if len(df) else 0,
        "targets": sorted(df["target"].unique()) if len(df) else [],
        "seeds": sorted(df["seed"].unique()) if len(df) else [],
        "fractions": sorted(df["fraction"].unique()) if len(df) else [],
        "total_gpu_hours": round(df["seconds"].sum() / 3600, 2) if len(df) else 0.0,
    }


def best_expert(df) -> "Any":
    """Collapse arm D to its best single source, per (target, fraction, seed).

    Arm A is compared against the strongest expert, not an average one. If the
    best single source matches arm A, the cross-domain claim fails -- and that
    is the comparison arm D exists to make possible.
    """
    import pandas as pd

    experts = df[df["arm"] == "D_expert"]
    if experts.empty:
        return experts
    idx = experts.groupby(["target", "fraction", "seed"])["score"].idxmax()
    best = experts.loc[idx].copy()
    best["arm"] = "D_expert"
    return best


def collapsed(df) -> "Any":
    """One row per (arm, target, fraction, seed), arm D reduced to its best."""
    import pandas as pd

    others = df[df["arm"] != "D_expert"]
    return pd.concat([others, best_expert(df)], ignore_index=True)


def summarise(df) -> "Any":
    """Mean and std over seeds for every (arm, target, fraction)."""
    import pandas as pd

    flat = collapsed(df)
    g = flat.groupby(["target", "fraction", "arm"], as_index=False).agg(
        metric=("metric", "first"),
        mean=("score", "mean"),
        std=("score", "std"),
        n_seeds=("score", "size"),
        all_converged=("converged", "all"),
    )
    g["std"] = g["std"].fillna(0.0)
    return g


def transfer_gain(df) -> "Any":
    """The two comparisons the project exists to make.

    `gain_vs_random`  = arm A - arm B. Does pretraining help at all?
    `gain_vs_expert`  = arm A - best arm D. Does pretraining on SEVERAL
                        unrelated domains beat pretraining on one?

    The second is the load-bearing one. A positive first number with a
    non-positive second means "pretraining helps", not "cross-domain
    pretraining helps".

    Gains are computed per seed and then averaged, so the reported spread is
    the spread of the difference -- which is what decides whether a gap is
    real. Averaging the arms first and subtracting would discard the pairing
    and overstate the uncertainty.
    """
    import pandas as pd

    flat = collapsed(df)
    wide = flat.pivot_table(index=["target", "fraction", "seed"],
                            columns="arm", values="score")
    out = pd.DataFrame(index=wide.index)
    if "A_transfer" in wide and "B_random" in wide:
        out["gain_vs_random"] = wide["A_transfer"] - wide["B_random"]
    if "A_transfer" in wide and "D_expert" in wide:
        out["gain_vs_expert"] = wide["A_transfer"] - wide["D_expert"]
    if "C_scratch" in wide and "A_transfer" in wide:
        out["gap_to_scratch"] = wide["C_scratch"] - wide["A_transfer"]

    agg = out.groupby(["target", "fraction"]).agg(["mean", "std", "size"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def verdict(gains, column: str = "gain_vs_expert") -> "Any":
    """Label each cell by whether the gap clears its own seed-to-seed spread.

    Deliberately blunt. A difference smaller than the variation between seeds
    is not evidence, however much one might want it to be.
    """
    import pandas as pd

    mean_col, std_col = f"{column}_mean", f"{column}_std"
    rows = []
    for _, r in gains.iterrows():
        mean, std = r[mean_col], r[std_col]
        if np.isnan(mean):
            label = "no data"
        elif abs(mean) <= std:
            label = "indistinguishable"
        elif mean > 0:
            label = "positive"
        else:
            label = "negative"
        rows.append({"target": r["target"], "fraction": r["fraction"],
                     "mean": mean, "std": std, "verdict": label})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The deliverable table
# --------------------------------------------------------------------------

def markdown_table(df, path: Path | None = None) -> str:
    """results/table.md -- 4 domains x 4 arms x 5 label fractions, mean +/- std."""
    summary = summarise(df)
    if summary.empty:
        return "_No runs logged yet._\n"

    lines = [
        "# Results",
        "",
        "Test-set score, mean +/- std over 3 seeds. Each domain uses its own",
        "headline metric, so numbers are comparable **down** a column and",
        "never **across** domains.",
        "",
        "Arm D is collapsed to its best single source per cell -- the fair",
        "comparison for arm A is the strongest expert, not an average one.",
        "",
    ]

    for target in sorted(summary["target"].unique()):
        block = summary[summary["target"] == target]
        metric = block["metric"].iloc[0]
        lines += [f"## {target}  ({metric})", ""]
        fractions = sorted(block["fraction"].unique())
        header = "| arm | " + " | ".join(f"{f:.0%}" for f in fractions) + " |"
        lines += [header, "|" + "---|" * (len(fractions) + 1)]
        for arm in ARM_ORDER:
            row = [ARM_LABELS.get(arm, arm)]
            for frac in fractions:
                cell = block[(block["arm"] == arm) & (block["fraction"] == frac)]
                if cell.empty:
                    row.append("-")
                else:
                    m, s = cell["mean"].iloc[0], cell["std"].iloc[0]
                    flag = "" if cell["all_converged"].iloc[0] else "*"
                    row.append(f"{m:.3f} ± {s:.3f}{flag}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines += ["`*` marks a cell where at least one run did not reach a",
              "validation plateau; its score is a lower bound.", ""]
    text = "\n".join(lines)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return text
