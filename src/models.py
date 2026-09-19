"""The encoder, and the self-supervised objective that trains it.

Two pieces:

**GINEncoder** -- the shared encoder. Graph Isomorphism Network, 2 layers,
hidden 256, output 128. One instance is shared across every domain; that
sharing is the whole experiment.

**DeepGraphInfomax** -- the pretraining objective. Self-supervised: it never
reads a label. It learns by asking the encoder to tell a real node embedding
apart from a fake one, where the fake comes from a corrupted copy of the same
graph. To win that game the encoder has to produce embeddings that carry
genuine information about the graph.

The discriminator is per-domain while the encoder is shared. That split is
deliberate: the discriminator's job is domain-specific bookkeeping (what does
a "typical" node look like *here*), and forcing one discriminator to serve
four unlike domains would push domain-identity information into the encoder,
which is exactly what we do not want it to specialise in.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------

class GINEncoder(nn.Module):
    """2-layer Graph Isomorphism Network.

    What one GIN layer does, in words: a node's new representation is an MLP
    applied to (its own features, scaled by 1+eps) plus (the SUM of its
    neighbours' features).

    The choice of SUM is the point of GIN, and it is why this architecture was
    picked over GCN or GraphSAGE. Mean and max aggregation are provably unable
    to distinguish certain graph structures -- a node with three identical
    neighbours looks the same as a node with one, under mean. Sum keeps that
    difference, which makes GIN as powerful as the Weisfeiler-Lehman test at
    telling graphs apart. For a project whose whole question is whether
    *structural* knowledge transfers across domains, an encoder that can
    actually see structure matters.

    Two layers means each node sees its 2-hop neighbourhood. Deeper GNNs tend
    to over-smooth: after many rounds of averaging with neighbours, every node
    converges to the same vector and the representation stops discriminating.
    """

    def __init__(
        self,
        in_dim: int = 133,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim, self.hidden_dim, self.out_dim = in_dim, hidden_dim, out_dim
        self.num_layers, self.dropout = num_layers, dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dims[i + 1]),
            )
            # train_eps=True lets the model learn how much weight to give a
            # node's own features versus its neighbours', rather than fixing it.
            self.convs.append(GINConv(mlp, train_eps=True))
            # Normalisation between layers only -- never on the output. See
            # the note in forward() for why that distinction is critical.
            if i < num_layers - 1:
                self.norms.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                if self.dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        # The final layer gets neither normalisation nor activation, and both
        # omissions matter.
        #
        # No ReLU: the embedding should be free to take negative values, and a
        # ReLU here would discard half the representational space.
        #
        # No BatchNorm, for a subtler and more damaging reason. BatchNorm
        # forces each output dimension to mean ~0 ACROSS NODES. The DGI
        # summary vector is sigmoid(mean over nodes), so a normalised output
        # makes that summary a constant ~0.5 vector no matter what the graph
        # contains -- identical for the real graph and the corrupted one. The
        # discriminator then has nothing to discriminate on, the loss parks at
        # ln(2) = 0.693, and accuracy sits at chance while everything still
        # looks like it is training. This was observed, not theorised.
        return x

    @torch.no_grad()
    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Embeddings with gradients off and BatchNorm in eval mode."""
        was_training = self.training
        self.eval()
        out = self.forward(x, edge_index)
        if was_training:
            self.train()
        return out

    def reset_parameters(self) -> None:
        for module in self.modules():
            if hasattr(module, "reset_parameters") and module is not self:
                module.reset_parameters()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------
# Deep Graph Infomax
# --------------------------------------------------------------------------

def corrupt_features(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Build the negative example: same graph, shuffled node features.

    This is DGI's corruption function. The edges are left completely
    untouched; only the rows of the feature matrix are permuted, so node 5
    now wears node 900's features while keeping node 5's neighbourhood.

    Why that is the right corruption: it breaks the correspondence between a
    node's own content and its structural context, while preserving the
    graph's degree distribution and global statistics exactly. So the encoder
    cannot win the discrimination game by learning something trivial about
    graph size or density -- the only way to tell real from fake is to
    actually model how content and structure fit together.
    """
    perm = torch.randperm(x.size(0), device=x.device, generator=generator)
    return x[perm]


class DGIDiscriminator(nn.Module):
    """Bilinear scorer: how well does this node embedding match the summary?

    score(h, s) = h^T W s

    One of these per domain. It is a small matrix (128 x 128) compared with
    the shared encoder, which is the point -- domain-specific bookkeeping
    stays here rather than leaking into the representation.
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        """Returns one logit per node."""
        return torch.matmul(h, torch.matmul(self.weight, summary))


@dataclass
class DGIOutput:
    loss: torch.Tensor
    pos_logit_mean: float
    neg_logit_mean: float
    accuracy: float


class DeepGraphInfomax(nn.Module):
    """Shared encoder plus one discriminator head per domain.

    The training signal, step by step:

      1. Encode the real graph          -> H       (one vector per node)
      2. Summarise it                   -> s = sigmoid(mean(H))
      3. Encode a corrupted copy        -> H~
      4. Score every real node against the summary, and every fake one too
      5. Push real scores up, fake scores down (binary cross-entropy)

    The summary s is a single vector describing the whole graph. Asking "does
    this node belong to this graph?" forces each node embedding to carry
    information about global structure, not just its immediate neighbourhood
    -- which is what makes the learned representation useful for a downstream
    task it was never told about.

    No labels are read anywhere in this process.
    """

    def __init__(self, encoder: GINEncoder, domains: list[str]) -> None:
        super().__init__()
        self.encoder = encoder
        self.discriminators = nn.ModuleDict(
            {name: DGIDiscriminator(encoder.out_dim) for name in domains}
        )

    @staticmethod
    def summarise(h: torch.Tensor) -> torch.Tensor:
        """Readout: the graph-level summary vector."""
        return torch.sigmoid(h.mean(dim=0))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        domain: str,
        generator: torch.Generator | None = None,
    ) -> DGIOutput:
        h_real = self.encoder(x, edge_index)
        summary = self.summarise(h_real)

        x_fake = corrupt_features(x, generator=generator)
        h_fake = self.encoder(x_fake, edge_index)

        disc = self.discriminators[domain]
        pos_logits = disc(h_real, summary)
        neg_logits = disc(h_fake, summary)

        loss = (
            F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        ) / 2.0

        with torch.no_grad():
            correct = (pos_logits > 0).sum() + (neg_logits <= 0).sum()
            accuracy = float(correct) / (pos_logits.numel() + neg_logits.numel())
            pos_mean = float(pos_logits.mean())
            neg_mean = float(neg_logits.mean())

        return DGIOutput(
            loss=loss,
            pos_logit_mean=pos_mean,
            neg_logit_mean=neg_mean,
            accuracy=accuracy,
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def encoder_parameters(self) -> int:
        return self.encoder.num_parameters


# --------------------------------------------------------------------------
# Probes (used from notebook 04 onward, defined here to keep models together)
# --------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """A single linear layer on top of frozen embeddings.

    Deliberately the weakest possible classifier. If a linear probe can read
    a property off the embeddings, that property is *linearly available* in
    the representation -- which is a statement about the encoder, not about
    the probe. A deeper probe could learn the task itself and would tell us
    nothing about what pretraining achieved.
    """

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()


def build_encoder(config=None, **overrides) -> GINEncoder:
    """Construct the encoder from a RunConfig, so every arm builds it identically."""
    if config is not None:
        kwargs = dict(
            in_dim=config.svd_dim + config.n_structural,
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            num_layers=config.num_layers,
        )
    else:
        kwargs = dict(in_dim=133, hidden_dim=256, out_dim=128, num_layers=2)
    kwargs.update(overrides)
    return GINEncoder(**kwargs)
