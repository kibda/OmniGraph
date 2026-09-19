"""Looking at what an encoder learned, without training anything on it.

Used by notebook 03 to answer "did pretraining change the representation, and
did the change help?" -- questions that need a measurement, not just a picture.

A t-SNE plot is suggestive but it is not evidence: t-SNE will happily produce
convincing-looking clusters from pure noise, and its layout depends on the
perplexity and the seed. So every figure here is paired with a number that
does not depend on a projection.
"""

from __future__ import annotations

import numpy as np


def tsne_2d(embeddings: np.ndarray, seed: int = 0, subsample: int | None = 3000,
            perplexity: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Project embeddings to 2D. Returns (coords, indices_used).

    Subsamples for speed -- t-SNE is O(n log n) at best and unreadable above a
    few thousand points anyway. The indices are returned so labels can be
    lined up with the coordinates.
    """
    from sklearn.manifold import TSNE

    n = embeddings.shape[0]
    rng = np.random.default_rng(seed)
    idx = (rng.choice(n, size=subsample, replace=False)
           if subsample and n > subsample else np.arange(n))

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5.0, (len(idx) - 1) / 3)),
        init="pca",
        random_state=seed,
        max_iter=1000,
    )
    return tsne.fit_transform(embeddings[idx]), idx


def knn_accuracy(embeddings: np.ndarray, labels: np.ndarray,
                 train_idx: np.ndarray, test_idx: np.ndarray,
                 k: int = 10) -> float:
    """Accuracy of a k-nearest-neighbour classifier in embedding space.

    A direct measure of whether the embedding puts same-class nodes near each
    other. It trains nothing -- no weights, no gradient -- so it measures the
    geometry of the representation itself rather than what a classifier can
    squeeze out of it. Neighbours are drawn from the training set only.
    """
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(embeddings[train_idx], labels[train_idx])
    return float(clf.score(embeddings[test_idx], labels[test_idx]))


def quick_linear_probe(embeddings: np.ndarray, labels: np.ndarray,
                       train_idx: np.ndarray, test_idx: np.ndarray,
                       seed: int = 0) -> dict[str, float]:
    """A fast scikit-learn linear probe, for sanity-checking only.

    Notebook 04 builds the real probe in PyTorch with early stopping on
    validation, which is what every reported number comes from. This one
    exists so notebook 03 can ask "did pretraining help at all?" without
    waiting for the full apparatus.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    clf = LogisticRegression(max_iter=3000, random_state=seed)
    clf.fit(embeddings[train_idx], labels[train_idx])
    pred = clf.predict(embeddings[test_idx])
    return {
        "accuracy": float((pred == labels[test_idx]).mean()),
        "macro_f1": float(f1_score(labels[test_idx], pred, average="macro")),
    }


def embedding_health(embeddings: np.ndarray) -> dict[str, float]:
    """Cheap diagnostics for a collapsed or degenerate representation.

    `effective_rank` is the one to watch. It is the exponential of the entropy
    of the normalised singular values -- roughly, how many dimensions the
    embedding actually uses. An encoder that has collapsed to a handful of
    directions will show an effective rank far below its output width, and
    no downstream probe can recover what it threw away.
    """
    z = embeddings - embeddings.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(z, compute_uv=False)
    p = sv / max(sv.sum(), 1e-12)
    p = p[p > 0]
    entropy = -(p * np.log(p)).sum()

    return {
        "dims": int(embeddings.shape[1]),
        "effective_rank": float(np.exp(entropy)),
        "mean_norm": float(np.linalg.norm(embeddings, axis=1).mean()),
        "std_overall": float(embeddings.std()),
        "dead_dims": int((embeddings.std(axis=0) < 1e-6).sum()),
    }


def compare_encoders(results: dict[str, dict[str, float]]) -> "Any":  # noqa: F821
    """Tidy the per-encoder measurements into one table."""
    import pandas as pd

    return pd.DataFrame(results).T
