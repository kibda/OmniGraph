"""The four arms, behind one function.

    A_transfer   pretrain DGI on the 3 source domains -> freeze -> probe target
    B_random     untrained encoder                    -> freeze -> probe target
    C_scratch    encoder + head trained on target labels (the upper bound)
    D_expert     pretrain DGI on ONE source           -> freeze -> probe target

Arm D is not optional. Without it, "arm A beats arm B" only establishes that
*some* pretraining helps -- it cannot distinguish that from the specific claim
this project is testing, which is that pretraining on *several unrelated*
domains helps. If a single-source expert does just as well, the cross-domain
story collapses.

Everything an arm does after the encoder exists is shared code: the same
splits, the same label-fraction subsample, the same probe, the same metrics.
Only the encoder differs, which is what makes the comparison a statement about
encoders.

**BatchNorm on the target.** Arms A and D pretrain on source domains, so their
BatchNorm running statistics describe those sources, not the target. We
recalibrate them on the target's features before embedding it. This is
label-free (forward passes only) and it is applied to arms A, B and D alike.

Without it the comparison is confounded exactly as it was in notebook 03: arm
B's encoder is calibrated on the target while arm A's is not, so arm A would be
penalised for a normalisation mismatch rather than for anything about its
representation. `calibrate_target_bn=False` reproduces the unadapted variant,
and notebook 05 reports both so the size of the effect is visible rather than
assumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import batching, metrics, models, pretrain, probe, splits
from .config import RunConfig
from .pipeline import PreparedDomain, source_domains_for
from .seeding import set_seed

ARMS = ["A_transfer", "B_random", "C_scratch", "D_expert"]


@dataclass
class Budget:
    """Training budgets, so a smoke test and a real run share one code path."""

    pretrain_steps: int = 300
    pretrain_patience: int = 30
    pretrain_lr: float = 1e-3
    probe_epochs: int = 3000
    probe_patience: int = 200
    probe_lr: float = 0.5
    # Arm C is the upper bound, so it must actually reach one. At 300/50 it
    # was still climbing on PPI and Elliptic in notebook 04, which makes the
    # reported ceiling a lower bound and understates every arm's headroom.
    # Arm C is only 60 of the 360 runs, so the extra budget is cheap.
    scratch_epochs: int = 600
    scratch_patience: int = 80
    scratch_lr: float = 1e-3

    @classmethod
    def smoke(cls) -> "Budget":
        """Deliberately too small to produce a real result.

        The point of a smoke test is to exercise every code path cheaply and
        catch shape errors, device errors and obvious nonsense. Numbers from
        this budget are NOT results and notebook 05 says so explicitly.
        """
        return cls(
            pretrain_steps=30, pretrain_patience=10,
            probe_epochs=300, probe_patience=50,
            scratch_epochs=30, scratch_patience=10,
        )

    def describe(self) -> str:
        return (f"pretrain {self.pretrain_steps} steps | probe {self.probe_epochs} "
                f"epochs | scratch {self.scratch_epochs} epochs")


@dataclass
class ArmOutcome:
    """One completed arm-run."""

    config: RunConfig
    test_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    primary_metric: str = ""
    primary_value: float = float("nan")
    converged: bool = False
    train_nodes: int = 0
    pretrain_seconds: float = 0.0
    probe_seconds: float = 0.0
    total_seconds: float = 0.0
    pretrain_summary: dict[str, Any] = field(default_factory=dict)

    def as_metrics_dict(self) -> dict[str, Any]:
        """The payload written to results/runs.jsonl."""
        return {
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "test": self.test_metrics,
            "val": self.val_metrics,
            "converged": self.converged,
            "train_nodes": self.train_nodes,
            "pretrain_seconds": round(self.pretrain_seconds, 2),
            "probe_seconds": round(self.probe_seconds, 2),
            "total_seconds": round(self.total_seconds, 2),
            "pretrain": self.pretrain_summary,
        }


# --------------------------------------------------------------------------
# Encoder construction, per arm
# --------------------------------------------------------------------------

def _pretrained_encoder(config: RunConfig, prepared: dict[str, PreparedDomain],
                        device, budget: Budget, verbose: bool):
    """Pretrain on the configured source domains. Arms A and D.

    Arm A passes three sources and gets round-robin training; arm D passes one
    and gets the same loop with a single batcher. Identical code, so the only
    difference between the arms is how many domains are in the mix.
    """
    sources = list(config.source_domains)
    encoder = models.build_encoder(config)
    src_domains = {s: prepared[s].domain for s in sources}
    src_features = {s: prepared[s].X for s in sources}

    if len(sources) == 1:
        name = sources[0]
        return pretrain.pretrain_single_domain(
            encoder, src_domains[name], src_features[name], device,
            steps=budget.pretrain_steps, lr=budget.pretrain_lr,
            patience=budget.pretrain_patience, seed=config.seed, verbose=verbose,
            objective=config.pretrain_objective,
        )
    return pretrain.pretrain_multi_domain(
        encoder, src_domains, src_features, device,
        steps=budget.pretrain_steps, lr=budget.pretrain_lr,
        patience=budget.pretrain_patience, seed=config.seed, verbose=verbose,
        objective=config.pretrain_objective,
    )


def _embed_target(encoder, target: PreparedDomain, device,
                  calibrate_target_bn: bool, seed: int) -> np.ndarray:
    """Embed the target with a frozen encoder.

    See the module docstring for why BatchNorm is recalibrated here. It uses
    forward passes only -- no gradients, no labels.
    """
    if calibrate_target_bn:
        batching.calibrate_batchnorm(encoder, target.domain, target.X, device,
                                     passes=3, seed=seed)
    encoder.eval()
    return batching.full_embeddings(encoder, target.domain, target.X, device)


# --------------------------------------------------------------------------
# The single entry point
# --------------------------------------------------------------------------

def embedding_cache_key(config: RunConfig) -> tuple:
    """What determines a frozen arm's embeddings.

    Deliberately omits `label_fraction`. Arms A, B and D never see a label
    while their encoder is built, so the embeddings for 1% and for 100% of the
    labels are bit-identical -- only the probe that reads them differs. The
    full matrix runs five fractions per fold, so reusing them removes four
    fifths of the pretraining work.

    Arm C is excluded from caching entirely: it trains on the target's labels,
    so its encoder genuinely differs at every fraction.
    """
    return (config.arm, config.target_domain, tuple(config.source_domains),
            config.seed, config.svd_scaling, config.hidden_dim,
            config.out_dim, config.num_layers, config.pretrain_objective)


def run_arm(
    config: RunConfig,
    prepared: dict[str, PreparedDomain],
    device,
    budget: Budget | None = None,
    calibrate_target_bn: bool = True,
    verbose: bool = True,
    embedding_cache: dict | None = None,
) -> ArmOutcome:
    """Run one arm end to end and return its scored outcome.

    `prepared` must hold every domain the config references, already prepared
    at `config.seed`.

    Pass an `embedding_cache` dict to reuse frozen-arm embeddings across label
    fractions. The caller owns it, so it can be cleared between folds to keep
    memory bounded.
    """
    budget = budget or Budget()
    t0 = time.perf_counter()

    # Reseed HERE, at the start of every arm-run.
    #
    # Without this, encoder weights come from whatever global RNG state the
    # previous run happened to leave behind, so the same config run twice
    # gives different numbers -- observed as 0.7675 then 0.7860 for identical
    # inputs. The downstream helpers seed their own generators, but the
    # encoder is constructed before any of them is reached.
    #
    # This is what makes results/runs.jsonl reproducible: the same config
    # hash and seed must yield the same numbers regardless of what ran first,
    # which is also what lets a crashed sweep resume without re-deriving
    # everything.
    set_seed(config.seed)

    target = prepared[config.target_domain]
    primary = metrics.primary_metric_for(target.name, target.task)

    # The label-fraction subsample applies to the TRAINING set only; val and
    # test stay at full size so every point of the sweep is scored alike.
    train_idx = splits.subsample_train(
        target.split, target.y, config.label_fraction, config.seed, target.task
    )

    outcome = ArmOutcome(config=config, primary_metric=primary,
                         train_nodes=len(train_idx))

    if config.arm == "C_scratch":
        if verbose:
            print(f"  arm C: end-to-end on {target.name} "
                  f"({len(train_idx):,} labelled train nodes)")
        result = probe.train_end_to_end(
            models.build_encoder(config), target.domain, target.X, target.y,
            train_idx, target.split.val_idx, target.split.test_idx,
            target.task, target.num_classes, target.name, device,
            epochs=budget.scratch_epochs, lr=budget.scratch_lr,
            patience=budget.scratch_patience, seed=config.seed, verbose=False,
        )
        outcome.probe_seconds = result.seconds
    else:
        key = embedding_cache_key(config)
        cached = embedding_cache.get(key) if embedding_cache is not None else None

        if cached is not None:
            embeddings, outcome.pretrain_summary = cached
            outcome.pretrain_summary = dict(outcome.pretrain_summary,
                                            reused_from_cache=True)
            if verbose:
                print(f"  {config.arm}: reusing cached embeddings "
                      f"(label fraction does not change them)")
        elif config.arm == "B_random":
            if verbose:
                print(f"  arm B: untrained encoder, BatchNorm calibrated on {target.name}")
            encoder = pretrain.random_init_encoder(
                target.domain, target.X, device, seed=config.seed, config=config
            )
            embeddings = batching.full_embeddings(encoder, target.domain,
                                                  target.X, device)
        else:
            t_pre = time.perf_counter()
            encoder, history = _pretrained_encoder(config, prepared, device,
                                                   budget, verbose)
            outcome.pretrain_seconds = time.perf_counter() - t_pre
            outcome.pretrain_summary = {
                "objective": config.pretrain_objective,
            "sources": list(config.source_domains),
                "steps_run": len(history.steps),
                "best_loss": history.best_loss,
                "final_disc_accuracy": history.accuracy[-1] if history.accuracy else None,
                "stopped_early": history.stopped_early,
            }
            embeddings = _embed_target(encoder, target, device,
                                       calibrate_target_bn, config.seed)

        if embedding_cache is not None and cached is None:
            embedding_cache[key] = (embeddings, dict(outcome.pretrain_summary))

        result = probe.train_linear_probe(
            embeddings, target.y, train_idx,
            target.split.val_idx, target.split.test_idx,
            target.task, target.num_classes, target.name, device,
            epochs=budget.probe_epochs, lr=budget.probe_lr,
            weight_decay=config.probe_weight_decay,
            patience=budget.probe_patience, seed=config.seed, verbose=False,
            standardize=config.probe_standardize,
            standardize_idx=target.split.train_idx,   # label-free, full split
        )
        outcome.probe_seconds = result.seconds

    outcome.test_metrics = result.test_metrics
    outcome.val_metrics = result.val_metrics
    outcome.primary_value = result.test_metrics.get(primary, float("nan"))
    outcome.converged = result.stopped_early
    outcome.total_seconds = time.perf_counter() - t0

    if verbose:
        flag = "" if outcome.converged else "   [DID NOT CONVERGE]"
        print(f"    -> test {primary} = {outcome.primary_value:.4f}"
              f"  ({outcome.total_seconds:.1f}s){flag}")
    return outcome


# --------------------------------------------------------------------------
# Building the run matrix
# --------------------------------------------------------------------------

def matrix_configs(
    targets: list[str] | None = None,
    seeds: list[int] | None = None,
    fractions: list[float] | None = None,
    include_experts: bool = True,
    base: RunConfig | None = None,
) -> list[RunConfig]:
    """Every RunConfig in the protocol.

    Leave-one-domain-out: one fold per target domain. For each fold, each seed
    and each label fraction, we run arm A (all three sources), arm B, arm C,
    and one arm D per individual source -- three of them, since "which single
    source" is itself the question arm D answers.
    """
    from . import datasets

    targets = targets or datasets.DOMAIN_ORDER
    seeds = seeds or [0, 1, 2]
    fractions = fractions or [0.01, 0.05, 0.10, 0.50, 1.00]
    template = base or RunConfig()

    configs: list[RunConfig] = []
    for target in targets:
        sources = source_domains_for(target)
        for seed in seeds:
            for frac in fractions:
                common = dict(target_domain=target, seed=seed, label_fraction=frac)
                configs.append(_derive(template, arm="A_transfer",
                                       source_domains=sources, **common))
                configs.append(_derive(template, arm="B_random",
                                       source_domains=(), **common))
                configs.append(_derive(template, arm="C_scratch",
                                       source_domains=(), **common))
                if include_experts:
                    for src in sources:
                        configs.append(_derive(template, arm="D_expert",
                                               source_domains=(src,), **common))
    return configs


def _derive(template: RunConfig, **overrides) -> RunConfig:
    from dataclasses import replace

    return replace(template, **overrides)


def describe_matrix(configs: list[RunConfig]) -> "Any":
    """Count the runs by arm and target, so the scale is visible up front."""
    import pandas as pd

    df = pd.DataFrame([{"arm": c.arm, "target": c.target_domain,
                        "seed": c.seed, "fraction": c.label_fraction}
                       for c in configs])
    return df.pivot_table(index="arm", columns="target", values="seed",
                          aggfunc="count", fill_value=0)
