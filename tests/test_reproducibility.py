"""The same config and seed must give the same numbers, always.

This is the guarantee results/runs.jsonl rests on: a row records a config hash
and a seed, and that pair has to be enough to re-derive the numbers. It is also
what makes scripts/run_all.py safe to resume -- a run skipped because it is
already logged must be the run that would have been produced.

It is easy to lose by accident. Two ways it was actually lost during
development, both caught here:

  * `run_arm` did not reseed, so encoder weights came from whatever global RNG
    state the previous run left behind.
  * CUDA scatter-add (GIN's aggregation) uses atomics whose summation order
    varies between runs, and float addition is not associative.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import arms, config, pipeline  # noqa: E402
from src.config import RunConfig  # noqa: E402
from src.seeding import set_seed  # noqa: E402


def test_cublas_workspace_is_configured_at_import():
    """The env var must be set before any CUDA context exists."""
    import os
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_set_seed_is_reproducible_for_plain_tensors():
    import torch

    set_seed(11)
    a = torch.randn(64, 64)
    set_seed(11)
    b = torch.randn(64, 64)
    assert torch.equal(a, b)


@pytest.fixture(scope="module")
def prepared():
    return pipeline.prepare_all(seed=0, names=["cora", "photo"])


def test_prepared_features_are_reproducible():
    """Same domain, same seed -> identical 133-dim features."""
    import numpy as np

    a = pipeline.prepare("cora", seed=0)
    b = pipeline.prepare("cora", seed=0)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.split.train_idx, b.split.train_idx)


def test_different_seeds_give_different_splits():
    """Otherwise the three seeds would not be measuring anything."""
    import numpy as np

    a = pipeline.prepare("cora", seed=0)
    b = pipeline.prepare("cora", seed=1)
    assert not np.array_equal(a.split.train_idx, b.split.train_idx)


@pytest.mark.slow
def test_arm_run_is_reproducible_despite_interleaving(prepared):
    """The headline guarantee, under adversarial conditions.

    Another arm-run is interleaved between repeats specifically to disturb the
    global RNG state. If `run_arm` did not reseed, this is where it would show.
    """
    device = config.get_device()
    budget = arms.Budget.smoke()

    target = RunConfig(arm="D_expert", target_domain="cora",
                       source_domains=("photo",), seed=0, label_fraction=1.0)
    disturber = replace(target, arm="B_random", source_domains=())

    values = []
    for _ in range(2):
        values.append(arms.run_arm(target, prepared, device, budget,
                                   verbose=False).primary_value)
        arms.run_arm(disturber, prepared, device, budget, verbose=False)

    assert values[0] == values[1], (
        f"same config gave {values[0]} then {values[1]} -- runs are not "
        "reproducible, so runs.jsonl cannot be trusted or resumed"
    )


@pytest.mark.slow
def test_embedding_cache_does_not_change_results(prepared):
    """The cache is an optimisation; it must be invisible in the numbers."""
    device = config.get_device()
    budget = arms.Budget.smoke()
    cfg = RunConfig(arm="D_expert", target_domain="cora",
                    source_domains=("photo",), seed=0, label_fraction=1.0)

    uncached = arms.run_arm(cfg, prepared, device, budget, verbose=False)

    cache: dict = {}
    first = arms.run_arm(cfg, prepared, device, budget, verbose=False,
                         embedding_cache=cache)
    # Same encoder, different label fraction -> must hit the cache.
    other_fraction = replace(cfg, label_fraction=0.5)
    assert arms.embedding_cache_key(other_fraction) in cache

    second = arms.run_arm(cfg, prepared, device, budget, verbose=False,
                          embedding_cache=cache)
    assert second.pretrain_summary.get("reused_from_cache") is True
    assert first.primary_value == second.primary_value == uncached.primary_value
