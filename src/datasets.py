"""Load the four domains behind one uniform interface.

The four graphs are genuinely different objects: one is a single citation
network, one is a single co-purchase network, one is 24 separate protein
graphs, and one is a single transaction graph where most nodes have no label
at all. Downstream code should not have to special-case them, so every
domain is wrapped in a `GraphDomain` that always exposes a *list* of graphs
plus honest metadata about what the labels mean.

Nothing here fits a transform, splits anything, or touches labels for
learning. This module only loads and describes. Feature unification is
notebook 02 (src/features.py); splitting is notebook 04.

If a download fails we raise `DatasetDownloadError` and stop. We never
silently substitute a different dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import config


class DatasetDownloadError(RuntimeError):
    """Raised when a dataset cannot be fetched or read.

    Deliberately fatal. Substituting a different dataset would silently
    change what the experiment measures.
    """


# --------------------------------------------------------------------------
# The container every domain is wrapped in
# --------------------------------------------------------------------------

@dataclass
class GraphDomain:
    """One domain, normalised to a common shape.

    `graphs` is always a list, even when the domain is a single graph. That
    is the one concession that lets PPI (24 graphs) and Cora (1 graph) flow
    through the same code path.
    """

    name: str                       # short key used in configs: cora, photo, ...
    display_name: str               # for tables and plot titles
    task: str                       # multiclass | multilabel | binary
    graphs: list                    # list[torch_geometric.data.Data]
    num_raw_features: int
    num_classes: int                # classes, or number of labels if multilabel
    primary_metric: str             # the headline metric for this domain
    description: str                # what the nodes and edges actually are
    label_semantics: dict[int, str] = field(default_factory=dict)
    has_unlabeled: bool = False     # True when some nodes carry no usable label
    unlabeled_value: int | None = None   # the y value that means "no label"
    source_split: list[str] = field(default_factory=list)  # per-graph origin tag

    # -- shape ------------------------------------------------------------

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)

    @property
    def total_nodes(self) -> int:
        return int(sum(g.num_nodes for g in self.graphs))

    @property
    def total_directed_edges(self) -> int:
        """Columns in edge_index, summed.

        PyG stores an undirected edge as *two* columns (u->v and v->u), so
        this is roughly twice the number of edges you would draw on paper.
        `total_undirected_edges` is the number most papers quote.
        """
        return int(sum(g.edge_index.size(1) for g in self.graphs))

    @property
    def total_undirected_edges(self) -> int:
        import torch_geometric.utils as pyg_utils

        total = 0
        for g in self.graphs:
            if pyg_utils.is_undirected(g.edge_index, num_nodes=g.num_nodes):
                total += g.edge_index.size(1) // 2
            else:
                total += g.edge_index.size(1)
        return int(total)

    @property
    def is_undirected(self) -> bool:
        import torch_geometric.utils as pyg_utils

        return all(
            pyg_utils.is_undirected(g.edge_index, num_nodes=g.num_nodes)
            for g in self.graphs
        )

    @property
    def avg_degree(self) -> float:
        """Mean number of neighbours per node, counting each direction once."""
        return self.total_directed_edges / max(self.total_nodes, 1)

    def __repr__(self) -> str:
        return (
            f"GraphDomain({self.name}, {self.num_graphs} graph(s), "
            f"{self.total_nodes:,} nodes, {self.total_undirected_edges:,} edges, "
            f"{self.num_raw_features} feats, task={self.task})"
        )


# --------------------------------------------------------------------------
# Per-domain loaders
# --------------------------------------------------------------------------

def _root_for(name: str, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else config.DATA_ROOT
    p = base / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_cora(root: Path | None = None) -> GraphDomain:
    from torch_geometric.datasets import Planetoid

    ds = Planetoid(root=str(_root_for("Planetoid", root)), name="Cora")
    data = ds[0]
    return GraphDomain(
        name="cora",
        display_name="Cora (citation)",
        task="multiclass",
        graphs=[data],
        num_raw_features=ds.num_features,
        num_classes=ds.num_classes,
        primary_metric="accuracy",
        description=(
            "A citation network of machine-learning papers. A NODE is one paper. "
            "An EDGE means one paper cites the other (stored undirected here). "
            "The 1433 raw FEATURES are a bag-of-words vector: entry i is 1 if the "
            "paper's abstract contains vocabulary word i, else 0 -- so the feature "
            "space is sparse and binary. The LABEL is the paper's subfield, one of "
            "7 (e.g. Reinforcement_Learning, Neural_Networks). The task is: given "
            "the words in a paper and who it cites, predict its subfield."
        ),
        source_split=["full"],
    )


def _load_photo(root: Path | None = None) -> GraphDomain:
    from torch_geometric.datasets import Amazon

    ds = Amazon(root=str(_root_for("Amazon", root)), name="Photo")
    data = ds[0]
    return GraphDomain(
        name="photo",
        display_name="Amazon Photo (co-purchase)",
        task="multiclass",
        graphs=[data],
        num_raw_features=ds.num_features,
        num_classes=ds.num_classes,
        primary_metric="accuracy",
        description=(
            "A segment of the Amazon co-purchase graph for photography products. "
            "A NODE is one product. An EDGE means the two products were frequently "
            "bought together. The 745 raw FEATURES are a bag-of-words encoding of "
            "the product's customer reviews. The LABEL is the product category, one "
            "of 8. The task is: given review text and what a product is co-bought "
            "with, predict its category. Note there is no official train/test split "
            "-- we make our own in notebook 04."
        ),
        source_split=["full"],
    )


def _load_ppi(root: Path | None = None) -> GraphDomain:
    from torch_geometric.datasets import PPI

    base = str(_root_for("PPI", root))
    graphs, origin = [], []
    for split in ("train", "val", "test"):
        ds = PPI(root=base, split=split)
        for g in ds:
            graphs.append(g)
            origin.append(split)
    ref = PPI(root=base, split="train")
    return GraphDomain(
        name="ppi",
        display_name="PPI (protein interaction)",
        task="multilabel",
        graphs=graphs,
        num_raw_features=ref.num_features,
        num_classes=ref[0].y.size(1),
        primary_metric="micro_f1",
        description=(
            "Protein-protein interaction networks, one graph per human tissue. A "
            "NODE is a protein. An EDGE means the two proteins physically interact. "
            "The 50 raw FEATURES are positional gene sets, motif gene sets and "
            "immunological signatures. The LABELS are 121 gene-ontology terms and "
            "this is MULTI-LABEL: a protein can carry many functions at once, so y "
            "is a 0/1 vector of length 121 per node, not a single class index. "
            "This is the only domain made of many separate graphs (24 of them), "
            "which is why GraphDomain always holds a list."
        ),
        source_split=origin,
    )


def _load_elliptic(root: Path | None = None) -> GraphDomain:
    from torch_geometric.datasets import EllipticBitcoinDataset

    ds = EllipticBitcoinDataset(root=str(_root_for("Elliptic", root)))
    data = ds[0]
    return GraphDomain(
        name="elliptic",
        display_name="Elliptic (bitcoin transactions)",
        task="binary",
        graphs=[data],
        num_raw_features=ds.num_features,
        num_classes=2,
        primary_metric="auc_pr",
        description=(
            "A graph of Bitcoin transactions. A NODE is one transaction. An EDGE "
            "means bitcoin flowed from one transaction to the next. The 165 raw "
            "FEATURES are 94 local features (timestep, fees, volume, in/out degree) "
            "plus 72 aggregated features from the node's one-hop neighbours. The "
            "LABEL is licit vs illicit -- illicit meaning traced to scams, ransomware "
            "or dark markets. Two things make this domain hard and interesting: only "
            "about a quarter of nodes are labelled at all, and among those the "
            "illicit class is roughly 10x rarer than the licit one. That is why the "
            "headline metric here is AUC-PR and not accuracy."
        ),
        label_semantics={0: "licit", 1: "illicit", 2: "unknown (no label)"},
        has_unlabeled=True,
        unlabeled_value=2,
        source_split=["full"],
    )


#: Registry. Order is the order tables and plots use.
LOADERS: dict[str, Callable[..., GraphDomain]] = {
    "cora": _load_cora,
    "photo": _load_photo,
    "ppi": _load_ppi,
    "elliptic": _load_elliptic,
}

DOMAIN_ORDER: list[str] = ["cora", "photo", "ppi", "elliptic"]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def load_domain(name: str, root: Path | None = None, verbose: bool = True) -> GraphDomain:
    """Load one domain by key. Raises DatasetDownloadError on any failure.

    The first call downloads and preprocesses; later calls read the cached
    `processed/` folder and are fast.
    """
    key = name.lower()
    if key not in LOADERS:
        raise KeyError(f"Unknown domain {name!r}. Known: {list(LOADERS)}")

    if verbose:
        print(f"  loading {key} ...", end=" ", flush=True)
    try:
        domain = LOADERS[key](root)
    except Exception as exc:  # noqa: BLE001 - any failure must stop the run
        raise DatasetDownloadError(
            f"\n\nFAILED to load domain {key!r}.\n"
            f"  reason: {type(exc).__name__}: {exc}\n"
            f"  data root: {root or config.DATA_ROOT}\n\n"
            "STOPPING rather than substituting a different dataset. "
            "Check the network connection, or delete the partially written "
            f"folder under the data root and retry."
        ) from exc
    if verbose:
        print(f"ok  ({domain.total_nodes:,} nodes)")
    return domain


def load_all_domains(
    root: Path | None = None,
    names: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, GraphDomain]:
    """Load every domain into a dict keyed by short name."""
    names = names or DOMAIN_ORDER
    if verbose:
        print(f"Loading {len(names)} domains into {root or config.DATA_ROOT}")
    out = {n: load_domain(n, root=root, verbose=verbose) for n in names}
    if verbose:
        print("All domains loaded.\n")
    return out


def verify_label_semantics(domain: GraphDomain) -> dict[str, Any]:
    """Check what the label tensor actually contains, instead of assuming.

    Dataset conventions drift between library versions. Rather than trust a
    docstring about which integer means "illicit", we look at the tensor and
    report what is there. Notebook 01 prints this.
    """
    import torch

    y = domain.graphs[0].y
    info: dict[str, Any] = {
        "dtype": str(y.dtype),
        "shape": tuple(y.shape),
        "ndim": y.ndim,
    }
    if y.ndim == 1:
        vals, counts = torch.unique(y, return_counts=True)
        info["unique_values"] = vals.tolist()
        info["value_counts"] = {int(v): int(c) for v, c in zip(vals, counts)}
        info["declared_meaning"] = {
            int(k): v for k, v in domain.label_semantics.items()
        } or "(single-class-index labels, meanings are dataset-defined)"
    else:
        info["labels_per_node"] = int(y.size(1))
        info["positives_per_node_mean"] = float(y.sum(dim=1).float().mean())
        info["note"] = "multi-label: y is a 0/1 matrix, one column per label"
    return info
