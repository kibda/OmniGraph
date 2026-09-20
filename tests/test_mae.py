"""The second pretraining objective, and the schema change that enabled it.

Two things are guarded here:

1. **MAE behaves like a pretraining objective** -- it masks what it claims to
   mask, reconstructs from neighbours rather than from the node itself, and
   produces a loss that actually decreases.
2. **Adding it did not invalidate the 360 runs already logged.** Introducing
   `pretrain_objective` changed every config hash, which would have made
   run_all.py silently re-run seven hours of finished work. That is the kind
   of regression that costs a day and leaves no trace, so it is pinned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, models, pretrain  # noqa: E402
from src.config import RunConfig  # noqa: E402
from src.mae import GraphMAE, scaled_cosine_error  # noqa: E402


def tiny_graph(n: int = 60, dim: int = 133, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    src = torch.randint(0, n, (2, 300), generator=g)
    edge_index = torch.cat([src, src.flip(0)], dim=1)   # make it undirected
    return x, edge_index


# --------------------------------------------------------------------------
# Hash stability -- the expensive regression
# --------------------------------------------------------------------------

def test_default_objective_does_not_change_the_hash():
    """A config at the default objective must hash as it did before the field
    existed, or every completed run looks unfinished."""
    cfg = RunConfig(arm="B_random", target_domain="cora", seed=0,
                    label_fraction=0.01)
    payload = {k: v for k, v in cfg.to_dict().items() if k != "notes"}
    assert payload["pretrain_objective"] == "dgi"
    # the field is present in the config record but excluded from the hash
    assert "pretrain_objective" in cfg.to_dict()
    assert cfg.run_id == "B_random__cora__lf0.01__s0__cd9e88f562ab"


def test_non_default_objective_does_change_the_hash():
    a = RunConfig(arm="A_transfer", target_domain="cora",
                  source_domains=("photo",), seed=0)
    b = RunConfig(arm="A_transfer", target_domain="cora",
                  source_domains=("photo",), seed=0, pretrain_objective="mae")
    assert a.hash != b.hash
    assert "mae" in b.run_id and "mae" not in a.run_id


def test_logged_runs_still_resolve_to_their_ids():
    """Every row already in results/runs.jsonl must still match its config."""
    rows = config.load_runs()
    if not rows:
        pytest.skip("no results logged yet")
    for row in rows:
        cfg = RunConfig(**{k: (tuple(v) if k == "source_domains" else v)
                           for k, v in row["config"].items()})
        assert cfg.run_id == row["run_id"], (
            f"{row['run_id']} no longer reproduces its own id -- resumability "
            "is broken and run_all.py would redo finished work"
        )


# --------------------------------------------------------------------------
# The objective itself
# --------------------------------------------------------------------------

def test_scaled_cosine_error_is_zero_for_perfect_reconstruction():
    x = torch.randn(20, 8)
    assert scaled_cosine_error(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_scaled_cosine_error_ignores_magnitude():
    """Cosine error measures direction, which is why it was chosen over MSE."""
    x = torch.randn(20, 8)
    assert scaled_cosine_error(x * 7.5, x).item() == pytest.approx(0.0, abs=1e-5)


def test_mask_rate_is_respected():
    x, edge_index = tiny_graph()
    enc = models.build_encoder()
    mae = GraphMAE(enc, ["d"], mask_rate=0.5)
    out = mae(x, edge_index, "d")
    assert out.masked_nodes == pytest.approx(x.size(0) * 0.5, abs=1)


def test_masked_nodes_cannot_see_their_own_features():
    """The mask token must actually replace the features.

    If a masked node kept its own features, the task would be trivial and the
    encoder would learn nothing -- the same failure mode as DGI saturating.
    """
    x, edge_index = tiny_graph()
    enc = models.build_encoder()
    mae = GraphMAE(enc, ["d"], mask_rate=1.0)     # mask everything

    captured = {}
    original_forward = enc.forward

    def spy(inp, ei):
        captured["input"] = inp.detach().clone()
        return original_forward(inp, ei)

    enc.forward = spy
    mae(x, edge_index, "d")
    enc.forward = original_forward

    seen = captured["input"]
    assert not torch.allclose(seen, x), "features reached the encoder unmasked"
    # with mask_rate=1.0 every row should be the same mask token
    assert torch.allclose(seen[0], seen[1], atol=1e-6)


def test_loss_decreases_with_training():
    """The objective must be learnable at all."""
    torch.manual_seed(0)
    x, edge_index = tiny_graph()
    enc = models.build_encoder()
    mae = GraphMAE(enc, ["d"])
    opt = torch.optim.Adam(mae.parameters(), lr=1e-2)

    first = None
    for step in range(40):
        opt.zero_grad()
        out = mae(x, edge_index, "d")
        out.loss.backward()
        opt.step()
        if step == 0:
            first = float(out.loss.detach())
    assert float(out.loss.detach()) < first, "MAE loss did not decrease"


def test_build_objective_dispatches():
    enc = models.build_encoder()
    assert isinstance(pretrain.build_objective(enc, ["d"], "dgi"),
                      models.DeepGraphInfomax)
    assert isinstance(pretrain.build_objective(enc, ["d"], "mae"), GraphMAE)
    with pytest.raises(ValueError):
        pretrain.build_objective(enc, ["d"], "nonsense")


def test_per_domain_decoders_exist():
    """One decoder per domain, mirroring DGI's per-domain discriminators."""
    enc = models.build_encoder()
    mae = GraphMAE(enc, ["cora", "photo", "ppi"])
    assert set(mae.decoders.keys()) == {"cora", "photo", "ppi"}
    # the encoder is shared, the decoders are not
    assert mae.encoder is enc
