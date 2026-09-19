"""Train / validation / test splits over nodes.

Notebook 02 needs these before it can fit anything, because every fitted
transform in this project is fit on training nodes only. Notebook 04 builds
on them for the label-fraction sweep.

Two conventions used throughout:

**Pooled indexing.** A domain may hold several graphs (PPI holds 24). We
concatenate them in order and number nodes 0..N-1 across the whole domain, so
a split is one flat array of indices regardless of how many graphs there are.
`graph_offsets` maps back when needed.

**Scorable nodes.** Only nodes that carry a usable label can be in a split.
For Elliptic that excludes the 157,205 nodes labelled "unknown". Those nodes
stay in the graph -- they carry features and edges and they help message
passing -- they simply never appear in train, validation or test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .datasets import GraphDomain


@dataclass
class NodeSplit:
    """One train/val/test partition of a domain's scorable nodes.

    Indices are pooled node ids (see module docstring). They are disjoint by
    construction, and their union is the set of scorable nodes -- never the
    set of all nodes, when some carry no label.
    """

    domain: str
    seed: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    total_nodes: int
    scorable_nodes: int
    stratified: bool

    def __post_init__(self) -> None:
        # These invariants are the whole point of the class; check them once,
        # here, rather than trusting every caller.
        sets = [set(self.train_idx.tolist()), set(self.val_idx.tolist()),
                set(self.test_idx.tolist())]
        assert not (sets[0] & sets[1]), "train and val overlap"
        assert not (sets[0] & sets[2]), "train and test overlap"
        assert not (sets[1] & sets[2]), "val and test overlap"
        assert sum(len(s) for s in sets) == self.scorable_nodes, (
            "splits do not cover exactly the scorable nodes"
        )

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train_idx),
            "val": len(self.val_idx),
            "test": len(self.test_idx),
        }

    def mask(self, which: str, num_nodes: int | None = None) -> np.ndarray:
        """Boolean mask over all pooled nodes for one split."""
        n = num_nodes or self.total_nodes
        m = np.zeros(n, dtype=bool)
        m[getattr(self, f"{which}_idx")] = True
        return m

    def __repr__(self) -> str:
        s = self.sizes
        return (
            f"NodeSplit({self.domain}, seed={self.seed}, "
            f"train={s['train']:,} val={s['val']:,} test={s['test']:,}, "
            f"scorable={self.scorable_nodes:,}/{self.total_nodes:,})"
        )


# --------------------------------------------------------------------------
# Pooling helpers
# --------------------------------------------------------------------------

def graph_offsets(domain: GraphDomain) -> np.ndarray:
    """Where each graph starts in the pooled node numbering.

    For a domain with graphs of size [1767, 1377, ...] this returns
    [0, 1767, 3144, ...] plus a final entry equal to the total node count.
    """
    sizes = [g.num_nodes for g in domain.graphs]
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


def pooled_labels(domain: GraphDomain) -> np.ndarray:
    """Labels for every pooled node.

    Shape (N,) for single-label domains, (N, L) for multi-label.
    """
    import torch

    y = torch.cat([g.y for g in domain.graphs], dim=0)
    return y.cpu().numpy()


def scorable_mask(domain: GraphDomain) -> np.ndarray:
    """Which pooled nodes carry a usable label.

    Everything except the sentinel "unknown" value, where a domain has one.
    """
    y = pooled_labels(domain)
    if domain.has_unlabeled and domain.unlabeled_value is not None:
        return y != domain.unlabeled_value
    return np.ones(y.shape[0], dtype=bool)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def make_node_split(
    domain: GraphDomain,
    seed: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> NodeSplit:
    """Split a domain's scorable nodes into train / val / test.

    Stratified by class where that is meaningful -- for a domain as skewed as
    Elliptic (9.2x), a plain random split can leave the rare class badly
    represented in validation, which would make early stopping erratic.

    PPI is multi-label: a node carries ~37 of 121 labels at once, so there is
    no single class to stratify on. It gets a plain random split.

    Note this splits *nodes*, including for PPI, whose 24 graphs are pooled.
    That is a deliberate choice and notebook 04 revisits it -- an inductive
    split by whole graph tests something subtly different.
    """
    from sklearn.model_selection import train_test_split

    assert 0 < train_frac < 1 and 0 < val_frac < 1
    assert train_frac + val_frac < 1, "no test fraction would be left"

    y = pooled_labels(domain)
    ok = scorable_mask(domain)
    candidates = np.flatnonzero(ok)

    # Stratify only when there is one label per node.
    stratify_on: np.ndarray | None = None
    if domain.task in ("multiclass", "binary") and y.ndim == 1:
        stratify_on = y[candidates]

    test_frac = 1.0 - train_frac - val_frac
    train_idx, rest_idx = train_test_split(
        candidates,
        train_size=train_frac,
        random_state=seed,
        shuffle=True,
        stratify=stratify_on,
    )
    # Re-derive the stratification labels for the remaining pool.
    rest_strat = y[rest_idx] if stratify_on is not None else None
    val_share = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=val_share,
        random_state=seed,
        shuffle=True,
        stratify=rest_strat,
    )

    return NodeSplit(
        domain=domain.name,
        seed=seed,
        train_idx=np.sort(train_idx),
        val_idx=np.sort(val_idx),
        test_idx=np.sort(test_idx),
        total_nodes=int(y.shape[0]),
        scorable_nodes=int(ok.sum()),
        stratified=stratify_on is not None,
    )


def split_summary(domain: GraphDomain, split: NodeSplit) -> dict[str, Any]:
    """Numbers worth printing: sizes, and whether stratification held."""
    y = pooled_labels(domain)
    out: dict[str, Any] = {
        "domain": domain.name,
        "seed": split.seed,
        "total_nodes": split.total_nodes,
        "scorable_nodes": split.scorable_nodes,
        "excluded_unlabelled": split.total_nodes - split.scorable_nodes,
        "stratified": split.stratified,
        **split.sizes,
    }
    if y.ndim == 1:
        # Class balance per split -- should match across splits if stratified.
        for which in ("train", "val", "test"):
            idx = getattr(split, f"{which}_idx")
            vals, counts = np.unique(y[idx], return_counts=True)
            out[f"{which}_class_frac"] = {
                int(v): round(float(c / counts.sum()), 4) for v, c in zip(vals, counts)
            }
    else:
        for which in ("train", "val", "test"):
            idx = getattr(split, f"{which}_idx")
            out[f"{which}_labels_per_node"] = round(float(y[idx].sum(axis=1).mean()), 2)
    return out


def print_split_summary(domain: GraphDomain, split: NodeSplit) -> None:
    s = split_summary(domain, split)
    print(f"\n--- {domain.name} (seed {s['seed']}) ---")
    print(f"  total nodes      : {s['total_nodes']:,}")
    if s["excluded_unlabelled"]:
        print(f"  excluded (no label): {s['excluded_unlabelled']:,} "
              f"-- still in the graph, never scored")
    print(f"  scorable nodes   : {s['scorable_nodes']:,}")
    print(f"  train/val/test   : {s['train']:,} / {s['val']:,} / {s['test']:,}")
    print(f"  stratified       : {s['stratified']}")
    if "train_class_frac" in s:
        print("  class balance holds across splits:")
        classes = sorted(s["train_class_frac"])
        header = "     " + "".join(f"{('c' + str(c)):>9}" for c in classes)
        print(header)
        for which in ("train", "val", "test"):
            row = s[f"{which}_class_frac"]
            print(f"  {which:<5}" + "".join(f"{row.get(c, 0.0):>9.3f}" for c in classes))
    else:
        print(f"  labels/node      : train {s['train_labels_per_node']}, "
              f"val {s['val_labels_per_node']}, test {s['test_labels_per_node']}")


def subsample_train(
    split: NodeSplit,
    y: np.ndarray,
    fraction: float,
    seed: int,
    task: str,
    min_per_class: int = 1,
) -> np.ndarray:
    """Take a `fraction` of the training nodes, for the label-efficiency sweep.

    Only the TRAINING set shrinks. Validation and test stay at full size in
    every run, so a 1% point and a 100% point are scored on exactly the same
    nodes and the curve measures label efficiency rather than evaluation noise.

    Single-label domains are subsampled per class, so a 1% draw cannot
    accidentally omit a class entirely -- on Elliptic, where the illicit class
    is 9.8% of labelled nodes, a naive 1% random draw would sometimes contain
    no positives at all and the probe would have nothing to learn.

    PPI is multi-label: a node carries ~37 of 121 labels, so there is no single
    class to stratify on and it gets a uniform random draw.
    """
    if fraction >= 1.0:
        return split.train_idx

    rng = np.random.default_rng(seed)
    train_idx = split.train_idx

    if task == "multilabel" or y.ndim > 1:
        k = max(min_per_class, int(round(len(train_idx) * fraction)))
        return np.sort(rng.choice(train_idx, size=min(k, len(train_idx)), replace=False))

    labels = y[train_idx]
    chosen = []
    for cls in np.unique(labels):
        pool = train_idx[labels == cls]
        k = max(min_per_class, int(round(len(pool) * fraction)))
        k = min(k, len(pool))
        chosen.append(rng.choice(pool, size=k, replace=False))
    return np.sort(np.concatenate(chosen))


def label_fraction_report(split: NodeSplit, y: np.ndarray, task: str,
                          fractions: list[float], seed: int = 0) -> "Any":
    """How many labelled nodes each point of the sweep actually gets."""
    import pandas as pd

    rows = []
    for frac in fractions:
        idx = subsample_train(split, y, frac, seed, task)
        row = {
            "fraction": frac,
            "train_nodes": len(idx),
            "of_full_train": len(idx) / len(split.train_idx),
        }
        if y.ndim == 1:
            counts = np.bincount(y[idx].astype(int))
            row["smallest_class"] = int(counts[counts > 0].min())
            row["classes_present"] = int((counts > 0).sum())
        else:
            row["labels_never_positive"] = int((y[idx].sum(axis=0) == 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("fraction")
