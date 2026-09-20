"""A second pretraining objective: masked feature reconstruction (GraphMAE).

**Why a second objective exists at all.** The DGI result in notebook 06 is
negative, but it cannot distinguish two very different explanations:

    (a) self-supervised pretraining does not transfer across these domains
    (b) DGI specifically is a weak objective here

Evidence for (b) is uncomfortably strong: DGI solves its pretext task by step
21 of 300, reaching 100% discriminator accuracy. Once every real node scores
above every fake one, the cross-entropy gradient is almost zero and the
remaining steps teach nothing. A critic can fairly say the homework was too
easy, so the student never learned.

This objective is designed to be hard to saturate:

    hide the features of half the nodes, then reconstruct them from their
    neighbours

There is no point at which that is "solved" -- the reconstruction is always
imperfect and always supplies gradient. It is also **generative** where DGI is
**contrastive**, so it tests a genuinely different hypothesis class rather
than a variation on the same one.

Everything downstream is unchanged: the same 133-dim features, the same
frozen-encoder probe, the same metrics, the same splits, the same seeds. Only
the pretraining objective differs, which is what makes the comparison mean
something.

Following Hou et al. (2022), *GraphMAE: Self-Supervised Masked Graph
Autoencoders*, with two choices that paper argues for and that matter here:

* **A learnable mask token**, not zeros. Zeroing a feature vector is not a
  neutral act in our 133-dim space -- zero is a meaningful value there (it is
  what PPI's 79 padded dimensions hold), so zeros would be confusable with
  real data. A learned token is unambiguous.
* **Scaled cosine error**, not mean squared error. MSE is dominated by
  whichever feature dimensions happen to have the largest magnitude, which
  after our per-domain SVD scaling is the leading components. Cosine error
  weighs direction rather than magnitude, and the exponent sharpens the
  gradient toward examples still being reconstructed badly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

from .models import GINEncoder, frozen_batchnorm_stats


@dataclass
class MAEOutput:
    """Mirrors DGIOutput so the shared training loop needs no special-casing."""

    loss: torch.Tensor
    accuracy: float        # mean cosine similarity on masked nodes, in [0, 1]
    masked_nodes: int
    mask_rate: float


class FeatureDecoder(nn.Module):
    """Reconstructs masked node features. One per domain.

    A single GIN layer rather than an MLP, because reconstruction should be
    allowed to use the graph: the whole premise is that a node's features are
    predictable from its neighbourhood. An MLP decoder could only invert the
    encoder pointwise and would make the task trivially about compression
    rather than about structure.

    Kept deliberately shallow. A powerful decoder lets the encoder off the
    hook -- if the decoder can reconstruct from a poor embedding, the encoder
    never has to produce a good one.
    """

    def __init__(self, in_dim: int = 128, hidden_dim: int = 256,
                 out_dim: int = 133) -> None:
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.conv = GINConv(mlp, train_eps=True)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(z, edge_index)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if hasattr(m, "reset_parameters") and m is not self:
                m.reset_parameters()


def scaled_cosine_error(pred: torch.Tensor, target: torch.Tensor,
                        gamma: float = 2.0) -> torch.Tensor:
    """(1 - cosine similarity) ^ gamma, averaged.

    `gamma > 1` sharpens the loss: nodes already reconstructed well contribute
    almost nothing, so the gradient concentrates on the ones still wrong. With
    gamma = 1 this is plain cosine error and training plateaus earlier.
    """
    cos = F.cosine_similarity(pred, target, dim=-1)
    return (1.0 - cos).pow(gamma).mean()


class GraphMAE(nn.Module):
    """Shared encoder, per-domain decoder, masked feature reconstruction.

    The per-domain decoder mirrors DGI's per-domain discriminator and exists
    for the same reason: reconstructing Cora's bag-of-words is a different job
    from reconstructing Elliptic's engineered transaction statistics, even
    though both arrive as 133 numbers. Forcing one decoder to do both would
    push domain-identity information into the shared encoder, which is exactly
    what we do not want it to specialise in.

    No labels are read anywhere in this file.
    """

    def __init__(self, encoder: GINEncoder, domains: list[str],
                 mask_rate: float = 0.5, gamma: float = 2.0,
                 replace_rate: float = 0.0) -> None:
        super().__init__()
        self.encoder = encoder
        self.mask_rate = mask_rate
        self.gamma = gamma
        self.replace_rate = replace_rate

        # A learned stand-in for "this node's features are hidden".
        self.mask_token = nn.Parameter(torch.zeros(1, encoder.in_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Bridges encoder output back to decoder input. GraphMAE calls this
        # re-masking: the encoder's embedding of a masked node is itself
        # blanked before decoding, so the decoder cannot simply read the
        # answer out of that node's own embedding and must use neighbours.
        self.encoder_to_decoder = nn.Linear(encoder.out_dim, encoder.out_dim,
                                            bias=False)
        self.decoders = nn.ModuleDict({
            name: FeatureDecoder(encoder.out_dim, encoder.hidden_dim,
                                 encoder.in_dim)
            for name in domains
        })

    def _choose_mask(self, num_nodes: int, device,
                     generator: torch.Generator | None = None) -> torch.Tensor:
        k = max(1, int(self.mask_rate * num_nodes))
        perm = torch.randperm(num_nodes, device=device, generator=generator)
        return perm[:k]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, domain: str,
                generator: torch.Generator | None = None) -> MAEOutput:
        num_nodes = x.size(0)
        masked_idx = self._choose_mask(num_nodes, x.device, generator)

        corrupted = x.clone()
        corrupted[masked_idx] = self.mask_token.to(x.dtype)

        z = self.encoder(corrupted, edge_index)

        # Re-mask: blank the masked nodes' own embeddings before decoding.
        z = self.encoder_to_decoder(z)
        z = z.clone()
        z[masked_idx] = 0.0

        # BatchNorm statistics must come from real graphs only -- inference
        # never sees a masked one. Same reasoning as the DGI corrupted pass.
        with frozen_batchnorm_stats(self.decoders[domain]):
            reconstruction = self.decoders[domain](z, edge_index)

        pred = reconstruction[masked_idx]
        target = x[masked_idx]
        loss = scaled_cosine_error(pred, target, self.gamma)

        with torch.no_grad():
            cos = F.cosine_similarity(pred, target, dim=-1).mean()
            accuracy = float((cos + 1.0) / 2.0)   # map [-1,1] to [0,1]

        return MAEOutput(loss=loss, accuracy=accuracy,
                         masked_nodes=int(masked_idx.numel()),
                         mask_rate=self.mask_rate)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
