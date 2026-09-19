"""Scoring, with a different headline metric per domain.

One metric does not fit these four domains, and using one anyway would be
actively misleading:

* **Cora, Photo** -- ordinary multi-class. Accuracy is honest here because no
  class dominates badly (worst imbalance 5.9x). Macro-F1 is reported
  alongside, since it weights every class equally and so notices when a model
  quietly abandons the rare ones.

* **PPI** -- multi-label. A node carries ~37 of 121 labels at once, so
  "accuracy" is not even well defined: is a prediction correct when it gets 36
  of 37 right? Micro-F1 pools every (node, label) decision into one
  contingency table, which is the standard answer.

* **Elliptic** -- binary and 9.2x imbalanced. Accuracy is a trap: always
  predicting "licit" scores 90.2% while catching not one illicit transaction.
  AUC-PR only rewards finding the rare positive class, and unlike AUC-ROC it
  does not flatter a model just because the negatives are easy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The single number reported per domain in the headline table.
PRIMARY_METRIC = {
    "cora": "accuracy",
    "photo": "accuracy",
    "ppi": "micro_f1",
    "elliptic": "auc_pr",
}

#: Which metrics are meaningful for each task type.
METRICS_FOR_TASK = {
    "multiclass": ["accuracy", "macro_f1"],
    "binary": ["auc_pr", "macro_f1", "accuracy", "auc_roc"],
    "multilabel": ["micro_f1", "macro_f1"],
}


def _multilabel_f1(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Micro- and macro-F1 for multi-label targets, computed directly.

    scikit-learn's f1_score is general and correspondingly slow: scoring PPI's
    11,389 validation nodes across 121 labels took long enough that a single
    probe run cost 88 seconds, almost all of it in metric computation rather
    than training. Over the full 240-run matrix that is hours of pure
    overhead. The contingency counts are three boolean reductions, so we do
    them here. `tests/test_metrics.py` checks the result against scikit-learn.

    micro: pool every (node, label) decision into one table, then F1 of that.
    macro: F1 per label, then the unweighted mean -- so a rare label counts
    as much as a common one.
    """
    tp = np.logical_and(pred, truth).sum(axis=0).astype(np.float64)
    fp = np.logical_and(pred, ~truth).sum(axis=0).astype(np.float64)
    fn = np.logical_and(~pred, truth).sum(axis=0).astype(np.float64)

    micro_den = 2 * tp.sum() + fp.sum() + fn.sum()
    micro = 2 * tp.sum() / micro_den if micro_den > 0 else 0.0

    per_label_den = 2 * tp + fp + fn
    per_label = np.divide(2 * tp, per_label_den,
                          out=np.zeros_like(tp), where=per_label_den > 0)
    return {"micro_f1": float(micro), "macro_f1": float(per_label.mean())}


def compute_metrics(task: str, y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    """Score predictions. `logits` are raw model outputs, not probabilities.

    Kept separate from the training loop so every arm is scored by identical
    code -- a comparison between arms is only meaningful if nothing but the
    encoder differs.
    """
    from sklearn.metrics import (average_precision_score, f1_score,
                                 roc_auc_score)

    if task == "multilabel":
        # Thresholding the logit at 0 is identical to thresholding the
        # sigmoid at 0.5, and skips computing the sigmoid at all.
        pred = logits >= 0.0
        truth = y_true > 0.5
        return {
            **_multilabel_f1(truth, pred),
            "mean_labels_predicted": float(pred.sum(axis=1).mean()),
        }

    pred = logits.argmax(axis=1)
    out: dict[str, float] = {
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }

    if task == "binary":
        # Softmax over two columns, then take the probability of class 1
        # (illicit) -- the rare class is the one we care about finding.
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        prob_pos = (exp[:, 1] / exp.sum(axis=1))
        if len(np.unique(y_true)) > 1:
            out["auc_pr"] = float(average_precision_score(y_true, prob_pos))
            out["auc_roc"] = float(roc_auc_score(y_true, prob_pos))
            # The floor AUC-PR of a random classifier is the positive rate,
            # so the raw number means nothing without it.
            out["positive_rate"] = float((y_true == 1).mean())
        else:
            out["auc_pr"] = float("nan")
            out["auc_roc"] = float("nan")
            out["positive_rate"] = float((y_true == 1).mean())
    return out


def primary_metric_for(domain_name: str, task: str) -> str:
    """The metric early stopping watches and the results table reports."""
    if domain_name in PRIMARY_METRIC:
        return PRIMARY_METRIC[domain_name]
    return {"multiclass": "accuracy", "binary": "auc_pr",
            "multilabel": "micro_f1"}[task]


def majority_baseline(task: str, y_true: np.ndarray) -> dict[str, float]:
    """What you get for free, without a model.

    Every reported number should be read against this. On Elliptic the
    majority baseline scores 0.902 accuracy, which is why accuracy is not the
    headline there.
    """
    if task == "multilabel":
        # Predict the marginal: every label that fires more than half the time.
        pred = np.tile((y_true.mean(axis=0) >= 0.5).astype(int), (len(y_true), 1))
        from sklearn.metrics import f1_score
        return {
            "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        }

    counts = np.bincount(y_true.astype(int))
    majority = int(counts.argmax())
    pred = np.full_like(y_true, majority)
    from sklearn.metrics import f1_score
    out = {
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }
    if task == "binary":
        # A random scorer's AUC-PR equals the positive rate.
        out["auc_pr"] = float((y_true == 1).mean())
    return out


def format_metrics(metrics: dict[str, Any], primary: str | None = None) -> str:
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            marker = "*" if k == primary else ""
            parts.append(f"{marker}{k}={v:.4f}")
    return "  ".join(parts)
