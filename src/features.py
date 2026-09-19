"""Make four incompatible feature spaces into one shared input space.

The problem, in one line: Cora has 1433 sparse binary features, Photo has 745,
PPI has 50 dense continuous ones, Elliptic has 165. One shared encoder cannot
consume four different input widths that mean four different things.

The fix has two halves.

**Half one: compress content to a common width.** Per-domain TruncatedSVD to
128 dimensions. SVD is chosen over PCA because it works directly on sparse
matrices without centring them (centring a sparse matrix makes it dense --
for Cora that is a 2708x1433 dense matrix where 98.7% of entries were zero).
The transform is fit on TRAINING NODES ONLY, per domain.

**Half two: add structure the content cannot express.** Five label-free
features describing where a node sits in its graph: log1p(degree), clustering
coefficient, PageRank, k-core number, triangle count. Standardized per domain,
with the scaler also fit on training nodes only.

128 + 5 = 133 dimensions, identical across all four domains.

A distinction worth being precise about, because it decides what counts as
leakage here:

  * The structural features are computed from the FULL graph, including test
    nodes. That is not leakage. They are label-free -- degree and PageRank
    say nothing about what class a node is -- and this is the standard
    transductive setting, where the graph is known in advance and only the
    labels are hidden.
  * The SVD basis and the scaler's mean/std ARE fitted objects. If a test
    node contributed to them, information about the test set would be baked
    into the representation of every node. So those are fit on training rows
    only, and tests/test_no_leakage.py asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .datasets import GraphDomain
from .splits import NodeSplit, graph_offsets

#: Order of the five structural features. Fixed, because downstream code and
#: plots index into it by position.
STRUCTURAL_NAMES: list[str] = [
    "log1p_degree",
    "clustering",
    "pagerank",
    "kcore",
    "triangles",
]


# --------------------------------------------------------------------------
# Symmetrising
# --------------------------------------------------------------------------

def symmetrize_domain(domain: GraphDomain, verbose: bool = True) -> GraphDomain:
    """Return a copy of the domain with undirected edges.

    Only Elliptic needs this. Its edges point one way (bitcoin flows forward
    in time), which leaves 27% of its nodes with no incoming edge -- and a GIN
    layer aggregates over incoming edges, so those nodes would receive no
    messages at all and their embedding would be a function of their own
    features alone.

    It is a real trade: symmetrising discards flow direction, which is
    arguably signal in a fraud graph. We take it because three of the five
    structural features below (clustering, k-core, triangles) are undirected
    notions anyway, and because the other three domains are undirected -- a
    shared encoder should not have to cope with one domain behaving
    structurally unlike the rest.
    """
    import copy

    from torch_geometric.utils import to_undirected

    if domain.is_undirected:
        if verbose:
            print(f"  {domain.name:<9} already undirected, unchanged")
        return domain

    new = copy.copy(domain)
    new.graphs = []
    before = after = 0
    for g in domain.graphs:
        g2 = g.clone()
        before += int(g2.edge_index.size(1))
        g2.edge_index = to_undirected(g2.edge_index, num_nodes=g2.num_nodes)
        after += int(g2.edge_index.size(1))
        new.graphs.append(g2)
    if verbose:
        print(f"  {domain.name:<9} symmetrised: {before:,} -> {after:,} edge_index columns")
    return new


# --------------------------------------------------------------------------
# Structural features
# --------------------------------------------------------------------------

def _to_sparse_adjacency(edge_index, num_nodes: int):
    """Binary symmetric adjacency as a scipy CSR matrix, no self-loops.

    Self-loops are dropped because a self-loop would inflate the degree and
    corrupt the clustering coefficient, which is defined on simple graphs.
    """
    import scipy.sparse as sp

    ei = edge_index.cpu().numpy()
    src, dst = ei[0], ei[1]
    keep = src != dst
    src, dst = src[keep], dst[keep]

    a = sp.coo_matrix(
        (np.ones(src.shape[0], dtype=np.float32), (src, dst)),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    # Symmetrise and re-binarise: (A + A^T) has 2 where both directions exist.
    a = a.maximum(a.T)
    a.data[:] = 1.0
    a.eliminate_zeros()
    return a


def _pagerank(adj, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """PageRank by power iteration on the sparse adjacency.

    Interpretation: the probability that a random surfer, who follows a
    random edge 85% of the time and teleports to a uniformly random node the
    other 15%, is sitting on this node in the long run. It is a measure of
    importance that accounts for *whose* neighbour you are, not just how many
    you have.

    Dangling nodes (degree 0) would leak probability mass, so their share is
    redistributed uniformly each iteration.
    """
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    deg = np.asarray(adj.sum(axis=1)).ravel()
    dangling = deg == 0
    inv_deg = np.where(dangling, 0.0, 1.0 / np.maximum(deg, 1e-12))

    # Row-normalise: column j of M gets mass from each neighbour i as 1/deg_i.
    transition = adj.T.multiply(inv_deg).tocsr()

    rank = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        dangling_mass = damping * rank[dangling].sum() / n
        new_rank = damping * (transition @ rank) + teleport + dangling_mass
        if np.abs(new_rank - rank).sum() < tol:
            rank = new_rank
            break
        rank = new_rank
    return rank


def _triangles_and_clustering(adj) -> tuple[np.ndarray, np.ndarray]:
    """Triangles through each node, and its local clustering coefficient.

    Triangles: for node i, the number of pairs of its neighbours that are
    themselves connected. Computed as 0.5 * sum_j A_ij * (A^2)_ij -- the
    elementwise product restricts A^2 to actual edges, which keeps the
    intermediate sparse instead of materialising a dense N x N matrix.

    Clustering coefficient: what fraction of the pairs of my neighbours are
    connected to each other, i.e. 2T / (d(d-1)). It answers "do my friends
    know each other?" and is 0 for a node with fewer than two neighbours.
    A citation network and a co-purchase network differ sharply here.
    """
    a2 = adj @ adj
    tri = np.asarray(adj.multiply(a2).sum(axis=1)).ravel() / 2.0

    deg = np.asarray(adj.sum(axis=1)).ravel()
    pairs = deg * (deg - 1.0) / 2.0
    clustering = np.divide(tri, pairs, out=np.zeros_like(tri), where=pairs > 0)
    return tri, clustering


def _core_numbers(adj) -> np.ndarray:
    """k-core number of every node.

    Definition, by the peeling procedure that computes it: repeatedly delete
    every node whose current degree is below k. A node's core number is the
    largest k for which it is still standing when that process settles.

    It measures something different from degree. A hub with 500 neighbours
    that are all leaves has huge degree but a core number of 1, because once
    the leaves are peeled away it has nothing left holding it up. A node in a
    dense mutually-connected cluster has a high core number even with modest
    degree. Notebook 07 uses exactly this kind of distinction when asking
    whether structurally similar domains transfer better.

    networkx implements the peeling in optimised form, which matters because
    Elliptic has 203,769 nodes.
    """
    import networkx as nx

    g = nx.from_scipy_sparse_array(adj)
    g.remove_edges_from(nx.selfloop_edges(g))
    core = nx.core_number(g)
    return np.array([core.get(i, 0) for i in range(adj.shape[0])], dtype=np.float64)


def structural_features_for_graph(edge_index, num_nodes: int) -> np.ndarray:
    """The five structural features for one graph, shape (num_nodes, 5)."""
    adj = _to_sparse_adjacency(edge_index, num_nodes)
    deg = np.asarray(adj.sum(axis=1)).ravel()

    tri, clustering = _triangles_and_clustering(adj)
    pagerank = _pagerank(adj)
    kcore = _core_numbers(adj)

    return np.column_stack([
        np.log1p(deg),   # log1p because degree is heavy-tailed: Photo has a
                         # node with 1434 neighbours and a median of 22.
        clustering,
        pagerank,
        kcore,
        tri,
    ]).astype(np.float32)


def structural_features_for_domain(domain: GraphDomain, verbose: bool = True) -> np.ndarray:
    """Structural features for every pooled node, shape (total_nodes, 5).

    Computed per graph and then stacked: PPI's 24 graphs are separate
    organisms and a node's clustering coefficient must not be computed across
    a graph boundary that does not exist.
    """
    import time

    parts = []
    t0 = time.perf_counter()
    for g in domain.graphs:
        parts.append(structural_features_for_graph(g.edge_index, g.num_nodes))
    out = np.concatenate(parts, axis=0)
    if verbose:
        print(f"  {domain.name:<9} structural features {out.shape} "
              f"in {time.perf_counter() - t0:.1f}s")
    return out


# --------------------------------------------------------------------------
# The fitted transform
# --------------------------------------------------------------------------

@dataclass
class UnifiedFeatureTransform:
    """Per-domain transform producing the shared 133-dim input space.

    Holds three fitted objects, all fit on training rows only:
      * `svd`          -- the SVD basis for this domain's raw features
      * `struct_scaler`-- mean/std of the five structural features

    `fitted_on` records exactly which node indices were used, which is what
    the leakage test inspects.
    """

    domain: str
    svd_dim: int = 128
    svd: Any = None
    struct_scaler: Any = None
    n_components_used: int = 0
    padded_dims: int = 0
    raw_dim: int = 0
    fitted_on: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    explained_variance: float = 0.0

    # -- fit --------------------------------------------------------------

    def fit(self, domain: GraphDomain, train_idx: np.ndarray,
            structural: np.ndarray, seed: int = 0, verbose: bool = True) -> "UnifiedFeatureTransform":
        """Fit the SVD basis and the structural scaler on training nodes only."""
        import torch
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import StandardScaler

        x = torch.cat([g.x for g in domain.graphs], dim=0).cpu().numpy()
        self.raw_dim = int(x.shape[1])
        self.fitted_on = np.asarray(train_idx, dtype=np.int64).copy()

        # TruncatedSVD needs n_components < n_features. PPI has only 50 raw
        # features, so it cannot yield 128 components -- we take as many as
        # exist and zero-pad the rest, which keeps the width identical across
        # domains without inventing information.
        k = min(self.svd_dim, self.raw_dim - 1)
        self.n_components_used = k
        self.padded_dims = self.svd_dim - k

        self.svd = TruncatedSVD(n_components=k, random_state=seed, algorithm="randomized")
        self.svd.fit(x[train_idx])
        self.explained_variance = float(self.svd.explained_variance_ratio_.sum())

        self.struct_scaler = StandardScaler()
        self.struct_scaler.fit(structural[train_idx])

        if verbose:
            pad = f", zero-padding {self.padded_dims}" if self.padded_dims else ""
            print(f"  {self.domain:<9} SVD {self.raw_dim} -> {k} dims{pad}  "
                  f"| explains {self.explained_variance:.1%} of variance  "
                  f"| fit on {len(train_idx):,} train nodes")
        return self

    # -- transform --------------------------------------------------------

    def transform(self, domain: GraphDomain, structural: np.ndarray) -> np.ndarray:
        """Apply the fitted transform to every node. Shape (N, svd_dim + 5)."""
        import torch

        if self.svd is None or self.struct_scaler is None:
            raise RuntimeError("transform() called before fit()")

        x = torch.cat([g.x for g in domain.graphs], dim=0).cpu().numpy()
        reduced = self.svd.transform(x)

        if self.padded_dims:
            pad = np.zeros((reduced.shape[0], self.padded_dims), dtype=reduced.dtype)
            reduced = np.hstack([reduced, pad])

        struct = self.struct_scaler.transform(structural)
        return np.hstack([reduced, struct]).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self.svd_dim + len(STRUCTURAL_NAMES)


def build_unified_features(
    domain: GraphDomain,
    split: NodeSplit,
    seed: int = 0,
    svd_dim: int = 128,
    structural: np.ndarray | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, UnifiedFeatureTransform, np.ndarray]:
    """End to end for one domain: structural features, fit, transform.

    Returns (unified_features, fitted_transform, raw_structural_features).
    The raw structural features are handed back so notebooks can plot them
    before standardization.
    """
    if structural is None:
        structural = structural_features_for_domain(domain, verbose=verbose)

    transform = UnifiedFeatureTransform(domain=domain.name, svd_dim=svd_dim)
    transform.fit(domain, split.train_idx, structural, seed=seed, verbose=verbose)
    unified = transform.transform(domain, structural)
    return unified, transform, structural
