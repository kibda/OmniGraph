"""Aggregation logic, checked on synthetic runs with known answers.

These are the functions that turn 360 JSON lines into the numbers in the
README. A bug here does not crash anything -- it just reports the wrong
finding -- so the arithmetic is pinned against hand-computed values on data
small enough to verify by eye.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, results  # noqa: E402
from src.config import RunConfig  # noqa: E402


@pytest.fixture
def synthetic(tmp_path):
    """A tiny matrix with values chosen so every aggregate is known."""
    path = tmp_path / "runs.jsonl"
    # arm A scores 0.80/0.82/0.84 -> mean 0.82
    # arm B scores 0.70/0.72/0.74 -> mean 0.72  => gain_vs_random = +0.10 exactly
    # arm D photo 0.75.., ppi 0.90.. -> best expert is ppi
    #   => gain_vs_expert = 0.82 - 0.92 = -0.10 exactly
    plan = [
        ("A_transfer", ("photo", "ppi"), [0.80, 0.82, 0.84]),
        ("B_random", (), [0.70, 0.72, 0.74]),
        ("C_scratch", (), [0.90, 0.92, 0.94]),
        ("D_expert", ("photo",), [0.75, 0.77, 0.79]),
        ("D_expert", ("ppi",), [0.90, 0.92, 0.94]),
    ]
    for arm, sources, scores in plan:
        for seed, score in enumerate(scores):
            cfg = RunConfig(arm=arm, target_domain="cora", source_domains=sources,
                            seed=seed, label_fraction=1.0)
            config.log_run(cfg, {"primary_metric": "accuracy",
                                 "primary_value": score,
                                 "converged": True, "train_nodes": 100,
                                 "total_seconds": 1.0}, path=path)
    return path


def test_load_frame_shape(synthetic):
    df = results.load_frame(synthetic)
    assert len(df) == 15
    assert set(df["arm"]) == {"A_transfer", "B_random", "C_scratch", "D_expert"}


def test_best_expert_picks_the_strongest_source(synthetic):
    """Arm A must be compared against the best expert, not an average one."""
    df = results.load_frame(synthetic)
    best = results.best_expert(df)
    assert len(best) == 3, "one row per seed"
    assert set(best["source_label"]) == {"ppi"}, "ppi outscores photo at every seed"


def test_summarise_means_and_spread(synthetic):
    df = results.load_frame(synthetic)
    s = results.summarise(df)
    a = s[(s["arm"] == "A_transfer")].iloc[0]
    assert a["mean"] == pytest.approx(0.82)
    assert a["n_seeds"] == 3
    # arm D is collapsed to its best source, so its mean is ppi's
    d = s[(s["arm"] == "D_expert")].iloc[0]
    assert d["mean"] == pytest.approx(0.92)


def test_transfer_gain_is_paired_per_seed(synthetic):
    """Gains are differences within a seed, then averaged."""
    df = results.load_frame(synthetic)
    g = results.transfer_gain(df).iloc[0]
    assert g["gain_vs_random_mean"] == pytest.approx(0.10)
    assert g["gain_vs_expert_mean"] == pytest.approx(-0.10)
    assert g["gap_to_scratch_mean"] == pytest.approx(0.10)
    # every seed has the same difference, so the spread is exactly zero
    assert g["gain_vs_random_std"] == pytest.approx(0.0, abs=1e-12)


def test_verdict_calls_small_gaps_indistinguishable(synthetic):
    """A difference inside its own seed spread is not evidence."""
    import pandas as pd

    gains = pd.DataFrame([
        {"target": "a", "fraction": 1.0, "g_mean": 0.001, "g_std": 0.05},
        {"target": "b", "fraction": 1.0, "g_mean": 0.20, "g_std": 0.01},
        {"target": "c", "fraction": 1.0, "g_mean": -0.20, "g_std": 0.01},
    ])
    v = results.verdict(gains, column="g")
    assert list(v["verdict"]) == ["indistinguishable", "positive", "negative"]


def test_markdown_table_marks_unconverged(tmp_path):
    """A cell containing a non-converged run must be flagged as a lower bound."""
    path = tmp_path / "runs.jsonl"
    for seed in range(3):
        cfg = RunConfig(arm="B_random", target_domain="cora", source_domains=(),
                        seed=seed, label_fraction=1.0)
        config.log_run(cfg, {"primary_metric": "accuracy", "primary_value": 0.5,
                             "converged": seed != 0, "train_nodes": 10,
                             "total_seconds": 1.0}, path=path)
    text = results.markdown_table(results.load_frame(path))
    assert "0.500 ± 0.000*" in text, "unconverged cell was not flagged"


def test_coverage_detects_incomplete_matrix(synthetic):
    cov = results.coverage(results.load_frame(synthetic), expected=360)
    assert cov["complete"] is False
    assert cov["duplicated_run_ids"] == 0
    assert cov["rows"] == 15


def test_empty_results_do_not_crash(tmp_path):
    df = results.load_frame(tmp_path / "nothing.jsonl")
    assert len(df) == 0
    assert "No runs logged yet" in results.markdown_table(df)
