"""The fast multi-label F1 must agree with scikit-learn exactly.

`src/metrics.py` computes micro- and macro-F1 by hand because scikit-learn's
general implementation dominated the runtime of a PPI probe. A hand-rolled
metric is only worth having if it is provably the same number, so this pins it
against the reference on random data, degenerate edge cases included.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import _multilabel_f1, compute_metrics  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("shape", [(50, 7), (200, 121), (13, 3)])
def test_matches_sklearn(seed, shape):
    rng = np.random.default_rng(seed)
    truth = rng.random(shape) < 0.35
    pred = rng.random(shape) < 0.35

    ours = _multilabel_f1(truth, pred)
    assert ours["micro_f1"] == pytest.approx(
        f1_score(truth, pred, average="micro", zero_division=0), abs=1e-12)
    assert ours["macro_f1"] == pytest.approx(
        f1_score(truth, pred, average="macro", zero_division=0), abs=1e-12)


@pytest.mark.parametrize("case", ["all_true", "all_false", "perfect", "inverted"])
def test_degenerate_cases(case):
    """Edge cases are where a hand-rolled metric usually diverges."""
    rng = np.random.default_rng(0)
    truth = rng.random((40, 6)) < 0.4
    pred = {
        "all_true": np.ones_like(truth),
        "all_false": np.zeros_like(truth),
        "perfect": truth.copy(),
        "inverted": ~truth,
    }[case].astype(bool)

    ours = _multilabel_f1(truth, pred)
    assert ours["micro_f1"] == pytest.approx(
        f1_score(truth, pred, average="micro", zero_division=0), abs=1e-12)
    assert ours["macro_f1"] == pytest.approx(
        f1_score(truth, pred, average="macro", zero_division=0), abs=1e-12)


def test_threshold_at_logit_zero_equals_sigmoid_at_half():
    """compute_metrics thresholds raw logits; that must equal sigmoid >= 0.5."""
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, size=(60, 9))
    truth = (rng.random((60, 9)) < 0.4).astype(float)

    ours = compute_metrics("multilabel", truth, logits)
    via_sigmoid = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int)
    assert ours["micro_f1"] == pytest.approx(
        f1_score(truth, via_sigmoid, average="micro", zero_division=0), abs=1e-12)
