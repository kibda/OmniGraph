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


# --------------------------------------------------------------------------
# Domain similarity (notebook 07)
# --------------------------------------------------------------------------

def degree_distance(domain_a, domain_b, bins: int = 60) -> float:
    """How unlike two domains' degree distributions are.

    Jensen-Shannon distance between the two log-degree histograms, on a shared
    bin grid. Chosen over a raw KL divergence for two reasons: it is symmetric
    (there is no "reference" domain here), and it is finite even when one
    distribution puts zero mass where the other does not -- which happens
    constantly, since Elliptic's maximum degree is 472 and Photo's is 1434.

    0 means identical, 1 means maximally different.
    """
    from scipy.spatial.distance import jensenshannon

    from .describe import degree_array

    a = np.log1p(degree_array(domain_a))
    b = np.log1p(degree_array(domain_b))
    lo, hi = 0.0, float(max(a.max(), b.max()))
    grid = np.linspace(lo, hi, bins + 1)

    pa, _ = np.histogram(a, bins=grid, density=False)
    pb, _ = np.histogram(b, bins=grid, density=False)
    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)
    return float(jensenshannon(pa, pb, base=2))


def spectral_gap(domain, max_nodes: int = 20_000, seed: int = 0) -> float:
    """The normalised Laplacian's spectral gap on the largest connected component.

    Also called the Fiedler value: the second-smallest eigenvalue of the
    normalised Laplacian. It measures how hard the graph is to cut in two.
    Near 0 means the graph falls apart into weakly connected communities;
    larger means it is well mixed. This is a global property that degree
    statistics cannot see -- two graphs can share a degree distribution and
    have entirely different community structure.

    Two details that matter, both learned by getting NaN without them:

    * **Largest connected component only.** A disconnected graph has one zero
      eigenvalue per component, so its Fiedler value is exactly 0 and carries
      no information. Cora and Elliptic are both disconnected, so the gap is
      computed on their largest component -- which is the quantity people mean
      when they quote a spectral gap for a real-world graph.

    * **Shift-invert.** Asking ARPACK for the smallest-magnitude eigenvalues
      directly (`which="SM"`) converges badly and returned NaN on two of four
      domains here. Shifting around sigma=0 turns it into a largest-magnitude
      problem, which ARPACK handles well.

    Large graphs are subsampled first: this is a summary statistic, not worth
    a full eigendecomposition of a 203,769-node Laplacian.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh

    from .features import _to_sparse_adjacency

    g = domain.graphs[0]
    n = g.num_nodes
    edge_index = g.edge_index

    if n > max_nodes:
        import torch
        from torch_geometric.utils import k_hop_subgraph

        rng = torch.Generator().manual_seed(seed)
        seeds = torch.randperm(n, generator=rng)[: max_nodes // 10]
        subset, edge_index, _, _ = k_hop_subgraph(
            seeds, 2, edge_index, relabel_nodes=True, num_nodes=n
        )
        n = int(subset.numel())

    adj = _to_sparse_adjacency(edge_index, n)

    # Restrict to the largest connected component.
    n_comp, labels = connected_components(adj, directed=False)
    if n_comp > 1:
        biggest = np.bincount(labels).argmax()
        keep = labels == biggest
        adj = adj[keep][:, keep]

    deg = np.asarray(adj.sum(axis=1)).ravel()
    if adj.shape[0] < 3 or deg.min() <= 0:
        keep = deg > 0
        if keep.sum() < 3:
            return float("nan")
        adj = adj[keep][:, keep]
        deg = np.asarray(adj.sum(axis=1)).ravel()

    d_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    laplacian = (sp.eye(adj.shape[0]) - d_inv_sqrt @ adj @ d_inv_sqrt).tocsc()

    try:
        vals = eigsh(laplacian, k=2, sigma=0.0, which="LM",
                     return_eigenvectors=False, maxiter=10_000)
    except Exception:  # noqa: BLE001 - ARPACK may still fail on odd graphs
        return float("nan")

    vals = np.sort(np.real(vals))
    # vals[0] is the ~0 eigenvalue of the connected component; vals[1] is the gap.
    return float(vals[1]) if vals.size > 1 else float("nan")


def domain_similarity_table(domains: dict) -> "Any":  # noqa: F821
    """Pairwise degree-distribution distance, plus each domain's spectral gap.

    Notebook 07 asks whether these predict transfer gain. Both are computed
    from graph structure alone -- no labels, no features -- so they are
    available before any transfer is attempted, which is what would make them
    useful if the relationship held.
    """
    import pandas as pd

    names = list(domains)
    dist = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            dist.loc[a, b] = 0.0 if a == b else degree_distance(domains[a], domains[b])

    gaps = pd.Series({n: spectral_gap(d) for n, d in domains.items()},
                     name="spectral_gap")
    return dist, gaps
