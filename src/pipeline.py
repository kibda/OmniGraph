"""Preparing a domain for a run, with the caching the full matrix needs.

The 12-run matrix re-prepares the same domains many times, so this module
draws a careful line between what may be cached and what must not be.

**Cacheable** -- anything that does not depend on the split:
  * the loaded, symmetrized `GraphDomain`
  * the five structural features (they read only the graph, never a label)

**Never cacheable** -- anything fitted:
  * the SVD basis, the SVD scaling scalar, the structural scaler

That second list is the whole point. Those are fit on the training split, so
they change with the seed. Caching them across seeds would silently mix one
fold's training nodes into another fold's transform -- exactly the leak that
`tests/test_no_leakage.py` exists to prevent, reintroduced through a
performance optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import datasets, features, splits
from .datasets import GraphDomain
from .splits import NodeSplit

#: domain name -> symmetrized GraphDomain. Split-independent.
_DOMAIN_CACHE: dict[str, GraphDomain] = {}

#: domain name -> (total_nodes, 5) structural features. Split-independent,
#: label-free, and by far the most expensive thing to recompute.
_STRUCTURAL_CACHE: dict[str, np.ndarray] = {}


@dataclass
class PreparedDomain:
    """One domain, ready to encode: features, split, labels, metadata."""

    name: str
    domain: GraphDomain
    split: NodeSplit
    X: np.ndarray                 # (total_nodes, 133)
    y: np.ndarray
    transform: Any
    seed: int

    @property
    def task(self) -> str:
        return self.domain.task

    @property
    def num_classes(self) -> int:
        return self.domain.num_classes

    def __repr__(self) -> str:
        return (f"PreparedDomain({self.name}, seed={self.seed}, "
                f"X{self.X.shape}, train={len(self.split.train_idx):,})")


def get_domain(name: str, verbose: bool = False) -> GraphDomain:
    """Load and symmetrize once, then reuse."""
    if name not in _DOMAIN_CACHE:
        d = datasets.load_domain(name, verbose=verbose)
        _DOMAIN_CACHE[name] = features.symmetrize_domain(d, verbose=verbose)
    return _DOMAIN_CACHE[name]


def get_structural(name: str, verbose: bool = False) -> np.ndarray:
    """Structural features, computed once per domain.

    Safe to cache across seeds and folds: these read the graph only. The
    leakage test asserts they do not change when labels are shuffled.
    """
    if name not in _STRUCTURAL_CACHE:
        _STRUCTURAL_CACHE[name] = features.structural_features_for_domain(
            get_domain(name), verbose=verbose
        )
    return _STRUCTURAL_CACHE[name]


def prepare(name: str, seed: int, svd_scaling: str = "block",
            svd_dim: int = 128, verbose: bool = False) -> PreparedDomain:
    """Everything needed to run an arm on this domain at this seed.

    The split and every fitted transform are rebuilt for the given seed. Only
    the graph and the structural features come from cache.
    """
    domain = get_domain(name, verbose=verbose)
    structural = get_structural(name, verbose=verbose)
    split = splits.make_node_split(domain, seed=seed)
    X, transform, _ = features.build_unified_features(
        domain, split, seed=seed, svd_dim=svd_dim, svd_scaling=svd_scaling,
        structural=structural, verbose=verbose,
    )
    return PreparedDomain(
        name=name, domain=domain, split=split, X=X,
        y=splits.pooled_labels(domain), transform=transform, seed=seed,
    )


def prepare_all(seed: int, names: list[str] | None = None,
                svd_scaling: str = "block", verbose: bool = False
                ) -> dict[str, PreparedDomain]:
    """Prepare every domain at one seed."""
    names = names or datasets.DOMAIN_ORDER
    return {n: prepare(n, seed, svd_scaling=svd_scaling, verbose=verbose)
            for n in names}


def source_domains_for(target: str, all_domains: list[str] | None = None) -> tuple[str, ...]:
    """The three sources for a leave-one-domain-out fold.

    One fold per domain: hold that domain out entirely, pretrain on the other
    three, then probe on the held-out one.
    """
    all_domains = all_domains or datasets.DOMAIN_ORDER
    return tuple(d for d in all_domains if d != target)


def clear_caches() -> None:
    """Drop the caches. Useful when memory matters more than speed."""
    _DOMAIN_CACHE.clear()
    _STRUCTURAL_CACHE.clear()
