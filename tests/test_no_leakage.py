"""Assert that no test-split node contributes to any fitted transform.

This is the test the whole feature-unification step has to pass. If a test
node's features helped choose the SVD basis, or helped compute the mean and
standard deviation used to standardize, then information about the test set is
baked into every node's representation and every downstream number is
optimistic by an unknown amount.

Three levels of check, weakest to strongest:

  1. **Bookkeeping** -- the indices we recorded as "fitted on" are disjoint
     from validation and test.
  2. **Behavioural** -- refit with the test rows corrupted to garbage. If the
     fit never saw them, the fitted parameters must come out bit-identical.
     This catches leaks that bookkeeping would miss, e.g. a transform that
     quietly fits on everything and only reports the train indices.
  3. **Scope** -- structural features are allowed to see the whole graph
     (they are label-free, and this is a transductive setting), but they must
     not see labels. Asserted by construction.

Run with:  .venv\\Scripts\\python.exe -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import datasets, features, splits  # noqa: E402

#: Cora and Photo are small enough to fit and refit quickly. The leakage
#: property is structural -- it does not depend on which domain we use -- so
#: testing on the small ones keeps the suite fast enough to actually run.
TEST_DOMAINS = ["cora", "photo"]
SEED = 0


@pytest.fixture(scope="module")
def loaded():
    """Load once, reuse across tests."""
    out = {}
    for name in TEST_DOMAINS:
        domain = datasets.load_domain(name, verbose=False)
        domain = features.symmetrize_domain(domain, verbose=False)
        split = splits.make_node_split(domain, seed=SEED)
        structural = features.structural_features_for_domain(domain, verbose=False)
        out[name] = (domain, split, structural)
    return out


# --------------------------------------------------------------------------
# 1. Bookkeeping
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", TEST_DOMAINS)
def test_splits_are_disjoint(loaded, name):
    """Train, validation and test must not share a single node."""
    _, split, _ = loaded[name]
    train, val, test = set(split.train_idx), set(split.val_idx), set(split.test_idx)
    assert not (train & val), f"{name}: train and val overlap"
    assert not (train & test), f"{name}: train and test overlap"
    assert not (val & test), f"{name}: val and test overlap"


@pytest.mark.parametrize("name", TEST_DOMAINS)
def test_transform_records_only_training_nodes(loaded, name):
    """The indices the transform reports fitting on exclude val and test."""
    domain, split, structural = loaded[name]
    transform = features.UnifiedFeatureTransform(domain=name)
    transform.fit(domain, split.train_idx, structural, seed=SEED, verbose=False)

    fitted = set(transform.fitted_on.tolist())
    assert fitted == set(split.train_idx.tolist()), "fitted_on is not exactly the train set"
    assert not (fitted & set(split.test_idx.tolist())), "a TEST node was fitted on"
    assert not (fitted & set(split.val_idx.tolist())), "a VAL node was fitted on"


# --------------------------------------------------------------------------
# 2. Behavioural -- the check that actually catches a real leak
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", TEST_DOMAINS)
def test_svd_basis_unchanged_when_test_rows_are_corrupted(loaded, name):
    """Corrupt every test row, refit, and require identical SVD components.

    If the SVD had seen the test rows, replacing them with large random noise
    would move the basis. Bit-identical components are proof it did not.
    """
    import torch

    domain, split, structural = loaded[name]

    clean = features.UnifiedFeatureTransform(domain=name)
    clean.fit(domain, split.train_idx, structural, seed=SEED, verbose=False)

    # Build a domain whose test rows are garbage.
    corrupted = _corrupt_rows(domain, split.test_idx, seed=SEED)

    dirty = features.UnifiedFeatureTransform(domain=name)
    dirty.fit(corrupted, split.train_idx, structural, seed=SEED, verbose=False)

    np.testing.assert_array_equal(
        clean.svd.components_, dirty.svd.components_,
        err_msg=f"{name}: SVD basis moved when TEST rows changed -- the fit saw them",
    )
    np.testing.assert_array_equal(
        clean.svd.singular_values_, dirty.svd.singular_values_,
        err_msg=f"{name}: SVD singular values moved when TEST rows changed",
    )


@pytest.mark.parametrize("name", TEST_DOMAINS)
def test_scaler_stats_unchanged_when_test_rows_are_corrupted(loaded, name):
    """Same corruption test for the structural-feature scaler."""
    domain, split, structural = loaded[name]

    clean = features.UnifiedFeatureTransform(domain=name)
    clean.fit(domain, split.train_idx, structural, seed=SEED, verbose=False)

    dirty_structural = structural.copy()
    rng = np.random.default_rng(SEED)
    dirty_structural[split.test_idx] = rng.normal(500.0, 50.0, size=(len(split.test_idx), structural.shape[1]))

    dirty = features.UnifiedFeatureTransform(domain=name)
    dirty.fit(domain, split.train_idx, dirty_structural, seed=SEED, verbose=False)

    np.testing.assert_array_equal(
        clean.struct_scaler.mean_, dirty.struct_scaler.mean_,
        err_msg=f"{name}: scaler mean moved when TEST rows changed -- the fit saw them",
    )
    np.testing.assert_array_equal(
        clean.struct_scaler.scale_, dirty.struct_scaler.scale_,
        err_msg=f"{name}: scaler scale moved when TEST rows changed",
    )


@pytest.mark.parametrize("name", TEST_DOMAINS)
def test_val_rows_also_excluded_from_fit(loaded, name):
    """Validation must be as protected as test.

    Early stopping reads validation, so a transform fitted on validation
    rows would make the stopping decision optimistic too.
    """
    domain, split, structural = loaded[name]

    clean = features.UnifiedFeatureTransform(domain=name)
    clean.fit(domain, split.train_idx, structural, seed=SEED, verbose=False)

    corrupted = _corrupt_rows(domain, split.val_idx, seed=SEED)
    dirty = features.UnifiedFeatureTransform(domain=name)
    dirty.fit(corrupted, split.train_idx, structural, seed=SEED, verbose=False)

    np.testing.assert_array_equal(
        clean.svd.components_, dirty.svd.components_,
        err_msg=f"{name}: SVD basis moved when VAL rows changed -- the fit saw them",
    )


# --------------------------------------------------------------------------
# 3. The test that proves the test works
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", TEST_DOMAINS)
def test_corruption_would_be_detected(loaded, name):
    """A deliberately leaky fit MUST fail the corruption check.

    Without this, the tests above could pass for the wrong reason -- e.g. if
    the corruption were too mild to move anything. Here we fit on ALL nodes
    on purpose and assert the basis does move.
    """
    domain, split, structural = loaded[name]
    all_idx = np.arange(domain.total_nodes)

    leaky_clean = features.UnifiedFeatureTransform(domain=name)
    leaky_clean.fit(domain, all_idx, structural, seed=SEED, verbose=False)

    corrupted = _corrupt_rows(domain, split.test_idx, seed=SEED)
    leaky_dirty = features.UnifiedFeatureTransform(domain=name)
    leaky_dirty.fit(corrupted, all_idx, structural, seed=SEED, verbose=False)

    assert not np.array_equal(
        leaky_clean.svd.components_, leaky_dirty.svd.components_
    ), (
        f"{name}: corrupting test rows did NOT move a fit that saw them. "
        "The corruption is too weak, so the leakage tests prove nothing."
    )


# --------------------------------------------------------------------------
# 4. Scope: structural features are label-free
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", TEST_DOMAINS)
def test_structural_features_ignore_labels(loaded, name):
    """Shuffling every label must not change a single structural feature.

    Structural features may look at the whole graph -- that is the
    transductive setting and is not leakage -- but they must never look at y.
    """
    import torch

    domain, _, structural = loaded[name]

    shuffled = _shuffle_labels(domain, seed=SEED)
    structural_after = features.structural_features_for_domain(shuffled, verbose=False)

    np.testing.assert_array_equal(
        structural, structural_after,
        err_msg=f"{name}: structural features changed when labels were shuffled",
    )


@pytest.mark.parametrize("name", TEST_DOMAINS)
def test_unified_output_shape_and_finiteness(loaded, name):
    """133 dims for every domain, and no NaN or inf anywhere."""
    domain, split, structural = loaded[name]
    unified, transform, _ = features.build_unified_features(
        domain, split, seed=SEED, structural=structural, verbose=False
    )
    assert unified.shape == (domain.total_nodes, 133), (
        f"{name}: expected (N, 133), got {unified.shape}"
    )
    assert np.isfinite(unified).all(), f"{name}: non-finite values in unified features"
    assert transform.output_dim == 133


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _corrupt_rows(domain, idx: np.ndarray, seed: int):
    """Copy of the domain with the given pooled rows replaced by loud noise."""
    import copy

    import torch

    rng = np.random.default_rng(seed)
    new = copy.copy(domain)
    offsets = splits.graph_offsets(domain)
    new.graphs = [g.clone() for g in domain.graphs]

    for gi, g in enumerate(new.graphs):
        lo, hi = offsets[gi], offsets[gi + 1]
        local = idx[(idx >= lo) & (idx < hi)] - lo
        if local.size:
            noise = rng.normal(1000.0, 100.0, size=(local.size, g.x.size(1)))
            g.x[torch.as_tensor(local, dtype=torch.long)] = torch.as_tensor(
                noise, dtype=g.x.dtype
            )
    return new


def _shuffle_labels(domain, seed: int):
    """Copy of the domain with labels randomly permuted within each graph."""
    import copy

    import torch

    rng = np.random.default_rng(seed)
    new = copy.copy(domain)
    new.graphs = [g.clone() for g in domain.graphs]
    for g in new.graphs:
        perm = torch.as_tensor(rng.permutation(g.num_nodes), dtype=torch.long)
        g.y = g.y[perm]
    return new
