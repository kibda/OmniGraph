"""The run-matrix definition, and the safety of its one optimisation.

Two things are checked here:

1. The matrix is what the protocol says it is -- leave-one-domain-out, every
   arm present, arm D running once per individual source.
2. The embedding cache cannot change a result. It exists to remove four fifths
   of the pretraining work in the full matrix; an optimisation that quietly
   altered a number would be far worse than no optimisation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import arms  # noqa: E402
from src.config import RunConfig  # noqa: E402


# --------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------

def test_matrix_is_leave_one_domain_out():
    """Every fold pretrains on exactly the domains it does not probe."""
    for cfg in arms.matrix_configs():
        if cfg.arm in ("B_random", "C_scratch"):
            assert cfg.source_domains == (), f"{cfg.arm} must have no sources"
        else:
            assert cfg.target_domain not in cfg.source_domains, (
                f"{cfg.run_id}: target leaked into the pretraining sources"
            )


def test_matrix_shape():
    cfgs = arms.matrix_configs()
    targets, seeds, fractions = 4, 3, 5
    # per (target, seed, fraction): A + B + C + one D per source (3)
    assert len(cfgs) == targets * seeds * fractions * 6 == 360

    by_arm = {}
    for c in cfgs:
        by_arm[c.arm] = by_arm.get(c.arm, 0) + 1
    assert by_arm["A_transfer"] == 60
    assert by_arm["B_random"] == 60
    assert by_arm["C_scratch"] == 60
    assert by_arm["D_expert"] == 180, "arm D runs once per individual source"


def test_arm_a_uses_all_three_sources():
    for cfg in arms.matrix_configs():
        if cfg.arm == "A_transfer":
            assert len(cfg.source_domains) == 3
        if cfg.arm == "D_expert":
            assert len(cfg.source_domains) == 1


def test_run_ids_are_unique():
    """Two runs sharing an id would silently overwrite each other on resume."""
    cfgs = arms.matrix_configs()
    ids = [c.run_id for c in cfgs]
    assert len(set(ids)) == len(ids)


def test_every_source_appears_as_an_expert():
    """Each fold must test each of its three sources individually."""
    from src.pipeline import source_domains_for

    cfgs = arms.matrix_configs()
    for target in ["cora", "photo", "ppi", "elliptic"]:
        experts = {c.source_domains[0] for c in cfgs
                   if c.arm == "D_expert" and c.target_domain == target}
        assert experts == set(source_domains_for(target))


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------

def test_cache_key_ignores_label_fraction():
    """Arms A/B/D build their encoder without labels, so fraction is irrelevant."""
    base = dict(arm="A_transfer", target_domain="cora",
                source_domains=("photo", "ppi", "elliptic"), seed=0)
    k1 = arms.embedding_cache_key(RunConfig(**base, label_fraction=0.01))
    k2 = arms.embedding_cache_key(RunConfig(**base, label_fraction=1.0))
    assert k1 == k2


@pytest.mark.parametrize("field,value", [
    ("seed", 1),
    ("target_domain", "photo"),
    ("arm", "D_expert"),
    ("hidden_dim", 64),
    ("svd_scaling", "none"),
])
def test_cache_key_separates_anything_that_matters(field, value):
    """Anything that changes the encoder must produce a different key."""
    base = RunConfig(arm="A_transfer", target_domain="cora",
                     source_domains=("photo", "ppi", "elliptic"), seed=0)
    from dataclasses import replace
    other = replace(base, **{field: value})
    assert arms.embedding_cache_key(base) != arms.embedding_cache_key(other)


def test_cache_key_separates_different_experts():
    """Arm D on photo and arm D on ppi must not share cached embeddings."""
    common = dict(arm="D_expert", target_domain="cora", seed=0)
    k_photo = arms.embedding_cache_key(RunConfig(**common, source_domains=("photo",)))
    k_ppi = arms.embedding_cache_key(RunConfig(**common, source_domains=("ppi",)))
    assert k_photo != k_ppi


def test_budget_smoke_is_smaller_everywhere():
    full, smoke = arms.Budget(), arms.Budget.smoke()
    assert smoke.pretrain_steps < full.pretrain_steps
    assert smoke.probe_epochs < full.probe_epochs
    assert smoke.scratch_epochs < full.scratch_epochs
