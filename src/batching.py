"""Turning a domain into batches the GPU can actually hold.

Cora and Photo fit on a 4 GB card whole. Elliptic does not: a full-graph DGI
step on its 203,769 nodes peaks at 3.73 GB, which leaves no headroom and would
fail the moment anything else touches the GPU.

The usual answer is neighbour sampling (`NeighborLoader`), but that needs
`pyg-lib` or `torch-sparse`, neither of which ships a Windows wheel for this
torch version. So we sample directly, and the specific choice matters:

**We take the exact 2-hop induced subgraph around a set of seed nodes.**

A 2-layer GIN has a receptive field of exactly 2 hops -- a node's embedding is
a function of its 2-hop neighbourhood and nothing beyond it. So the embedding
a seed node gets inside this subgraph is *identical* to the embedding it would
get from the full graph. This is not an approximation, unlike neighbour
sampling, which subsamples each hop and produces embeddings that differ from
the full-graph ones. We pay for exactness with a variable batch size.

Multi-graph domains (PPI's 24 tissues) need none of this: each graph is
already small, and one graph per batch keeps the DGI summary vector
meaningful -- it describes one tissue rather than a blend of eight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from .datasets import GraphDomain
from .splits import graph_offsets

#: A single graph larger than this gets subgraph-sampled rather than used whole.
#: Set from the measured 4 GB ceiling: Photo (7,650 nodes) runs whole at 0.37 GB,
#: Elliptic (203,769) peaks at 3.73 GB.
DEFAULT_MAX_FULL_NODES = 60_000

#: Seeds per sampled batch. 4,096 seeds expand to roughly 50k nodes on Elliptic.
DEFAULT_SEEDS = 4_096


@dataclass
class GraphBatch:
    """One unit of work: a graph (or subgraph) plus its features.

    `node_idx` maps rows back to pooled domain node ids, so embeddings
    computed here can be written into the right slots of a full embedding
    matrix.
    """

    domain: str
    x: torch.Tensor
    edge_index: torch.Tensor
    node_idx: torch.Tensor
    is_full_graph: bool
    graph_id: int = 0

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))

    def to(self, device) -> "GraphBatch":
        return GraphBatch(
            domain=self.domain,
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            node_idx=self.node_idx,
            is_full_graph=self.is_full_graph,
            graph_id=self.graph_id,
        )

    def __repr__(self) -> str:
        kind = "full" if self.is_full_graph else "2-hop sample"
        return (f"GraphBatch({self.domain}, {self.num_nodes:,} nodes, "
                f"{self.num_edges:,} edges, {kind})")


class DomainBatcher:
    """Produces batches for one domain, cycling forever.

    Round-robin multi-domain training pulls exactly one batch per domain per
    step, so this has to be an endless stream rather than an epoch iterator.
    """

    def __init__(
        self,
        domain: GraphDomain,
        x: np.ndarray,
        max_full_nodes: int = DEFAULT_MAX_FULL_NODES,
        seeds: int = DEFAULT_SEEDS,
        num_hops: int = 2,
        seed: int = 0,
    ) -> None:
        self.domain = domain
        self.name = domain.name
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.offsets = graph_offsets(domain)
        self.max_full_nodes = max_full_nodes
        self.seeds = seeds
        self.num_hops = num_hops
        self.generator = torch.Generator().manual_seed(seed)

        self.multi_graph = domain.num_graphs > 1
        self.needs_sampling = (
            not self.multi_graph and domain.graphs[0].num_nodes > max_full_nodes
        )
        self._cursor = 0

    # -- description -------------------------------------------------------

    @property
    def strategy(self) -> str:
        if self.multi_graph:
            return f"one graph per batch ({self.domain.num_graphs} graphs)"
        if self.needs_sampling:
            return f"{self.num_hops}-hop subgraph from {self.seeds:,} seeds"
        return "full graph"

    @property
    def batches_per_epoch(self) -> int:
        """How many batches cover the domain once."""
        if self.multi_graph:
            return self.domain.num_graphs
        if self.needs_sampling:
            return max(1, self.domain.graphs[0].num_nodes // self.seeds)
        return 1

    # -- production --------------------------------------------------------

    def _graph_batch(self, gi: int) -> GraphBatch:
        g = self.domain.graphs[gi]
        lo, hi = int(self.offsets[gi]), int(self.offsets[gi + 1])
        return GraphBatch(
            domain=self.name,
            x=self.x[lo:hi],
            edge_index=g.edge_index,
            node_idx=torch.arange(lo, hi),
            is_full_graph=True,
            graph_id=gi,
        )

    def _sampled_batch(self) -> GraphBatch:
        from torch_geometric.utils import k_hop_subgraph

        g = self.domain.graphs[0]
        n = g.num_nodes
        seed_nodes = torch.randperm(n, generator=self.generator)[: self.seeds]
        subset, sub_ei, _, _ = k_hop_subgraph(
            seed_nodes, self.num_hops, g.edge_index,
            relabel_nodes=True, num_nodes=n,
        )
        return GraphBatch(
            domain=self.name,
            x=self.x[subset],
            edge_index=sub_ei,
            node_idx=subset,
            is_full_graph=False,
        )

    def next_batch(self) -> GraphBatch:
        if self.multi_graph:
            batch = self._graph_batch(self._cursor % self.domain.num_graphs)
            self._cursor += 1
            return batch
        if self.needs_sampling:
            return self._sampled_batch()
        return self._graph_batch(0)

    def __iter__(self) -> Iterator[GraphBatch]:
        while True:
            yield self.next_batch()


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

@torch.no_grad()
def full_embeddings(
    encoder,
    domain: GraphDomain,
    x: np.ndarray,
    device,
    max_full_nodes: int = DEFAULT_MAX_FULL_NODES,
) -> np.ndarray:
    """Embed every node in the domain. Shape (total_nodes, out_dim).

    Inference is far cheaper than training -- no activations are retained for
    a backward pass -- so a graph that needs sampling during training often
    fits whole here. We still chunk by graph for multi-graph domains.

    BatchNorm is put in eval mode, so it uses its running statistics rather
    than the statistics of whatever happens to be in this batch. Without that,
    a node's embedding would depend on which other nodes were embedded
    alongside it, which would make the downstream probe non-deterministic.
    """
    encoder.eval()
    xs = torch.as_tensor(x, dtype=torch.float32)
    offsets = graph_offsets(domain)
    out = np.empty((domain.total_nodes, encoder.out_dim), dtype=np.float32)

    for gi, g in enumerate(domain.graphs):
        lo, hi = int(offsets[gi]), int(offsets[gi + 1])
        xb = xs[lo:hi].to(device)
        ei = g.edge_index.to(device)
        z = encoder(xb, ei)
        out[lo:hi] = z.cpu().numpy()
        del xb, ei, z
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def describe_batching(batchers: dict[str, DomainBatcher]) -> None:
    """Print what each domain will actually do during training."""
    print(f"  {'domain':<10} {'strategy':<38} {'batch nodes':>12} {'batch edges':>12}")
    print("  " + "-" * 76)
    for name, b in batchers.items():
        sample = b.next_batch()
        print(f"  {name:<10} {b.strategy:<38} {sample.num_nodes:>12,} "
              f"{sample.num_edges:>12,}")
