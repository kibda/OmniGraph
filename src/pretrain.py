"""Self-supervised pretraining with Deep Graph Infomax.

Two entry points:

* `pretrain_single_domain` -- the arm D "expert", and what notebook 03 runs.
* `pretrain_multi_domain`  -- the arm A encoder. Round-robin: one batch from
  each domain per step, one shared encoder, a separate discriminator head per
  domain.

Round-robin rather than concatenating the domains into one pool, because the
domains differ in size by three orders of magnitude. Sampling proportionally
would make Elliptic supply ~75% of gradient steps and Cora ~1%, and the
"shared" encoder would be an Elliptic encoder with a rounding error of Cora in
it. One batch each per step gives every domain equal say.

No labels are read anywhere in this file.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .batching import DomainBatcher, GraphBatch
from .models import DeepGraphInfomax, GINEncoder


@dataclass
class PretrainHistory:
    """Everything worth plotting or logging from a pretraining run."""

    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    per_domain_loss: dict[str, list[float]] = field(default_factory=dict)
    best_step: int = -1
    best_loss: float = float("inf")
    stopped_early: bool = False
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "per_domain_loss": self.per_domain_loss,
            "best_step": self.best_step,
            "best_loss": self.best_loss,
            "stopped_early": self.stopped_early,
            "seconds": self.seconds,
        }


def build_objective(encoder, domains: list[str], objective: str = "dgi"):
    """Wrap the shared encoder in a pretraining objective.

    Both objectives expose the same interface -- callable with
    (x, edge_index, domain, generator), returning something with `.loss` and
    `.accuracy` -- so the training loop below needs no knowledge of which one
    it is driving. That is what keeps the two experiments comparable: only the
    objective differs, every other line of code is shared.
    """
    if objective == "dgi":
        return DeepGraphInfomax(encoder, domains)
    if objective == "mae":
        from .mae import GraphMAE

        return GraphMAE(encoder, domains)
    raise ValueError(f"unknown pretraining objective {objective!r}")


def _run_batch(model, batch: GraphBatch, device,
               generator: torch.Generator | None = None):
    batch = batch.to(device)
    return model(batch.x, batch.edge_index, batch.domain, generator=generator)


def _pretrain(
    model: DeepGraphInfomax,
    batchers: dict[str, DomainBatcher],
    device,
    steps: int,
    lr: float,
    patience: int,
    seed: int,
    log_every: int,
    verbose: bool,
) -> PretrainHistory:
    """Shared loop. One step = one batch from every domain in `batchers`."""
    model.to(device).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator(device=device).manual_seed(seed)

    history = PretrainHistory()
    history.per_domain_loss = {name: [] for name in batchers}

    best_state = copy.deepcopy(model.state_dict())
    since_improved = 0
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        optimiser.zero_grad(set_to_none=True)

        step_loss = 0.0
        step_acc = 0.0
        for name, batcher in batchers.items():
            out = _run_batch(model, batcher.next_batch(), device, generator)
            # Backward per domain rather than summing first: this keeps only
            # one domain's activation graph alive at a time, which is what
            # lets Elliptic and Cora share a 4 GB card.
            (out.loss / len(batchers)).backward()
            step_loss += float(out.loss.detach()) / len(batchers)
            step_acc += out.accuracy / len(batchers)
            history.per_domain_loss[name].append(float(out.loss.detach()))

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()

        history.steps.append(step)
        history.loss.append(step_loss)
        history.accuracy.append(step_acc)

        # Early stopping on the self-supervised objective itself. There is no
        # validation label to watch here -- DGI never sees one -- so we stop
        # when its own loss stops improving. Labels enter only at the probe.
        if step_loss < history.best_loss - 1e-5:
            history.best_loss = step_loss
            history.best_step = step
            best_state = copy.deepcopy(model.state_dict())
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                history.stopped_early = True
                if verbose:
                    print(f"    early stop at step {step} "
                          f"(no improvement for {patience} steps)")
                break

        if verbose and (step % log_every == 0 or step == 1):
            print(f"    step {step:>4}/{steps}  loss {step_loss:.4f}  "
                  f"disc acc {step_acc:.3f}")

    history.seconds = time.perf_counter() - t0
    # Restore the best encoder, not the last one.
    model.load_state_dict(best_state)
    if verbose:
        print(f"    done in {history.seconds:.1f}s | best loss {history.best_loss:.4f} "
              f"at step {history.best_step}")
    return history


def pretrain_single_domain(
    encoder: GINEncoder,
    domain,
    x: np.ndarray,
    device,
    steps: int = 300,
    lr: float = 1e-3,
    patience: int = 30,
    seed: int = 0,
    log_every: int = 25,
    verbose: bool = True,
    objective: str = "dgi",
    **batcher_kwargs,
) -> tuple[GINEncoder, PretrainHistory]:
    """Pretrain on one domain. Arm D, and what notebook 03 demonstrates."""
    batcher = DomainBatcher(domain, x, seed=seed, **batcher_kwargs)
    model = build_objective(encoder, [domain.name], objective)
    if verbose:
        print(f"  pretraining [{objective}] on {domain.name} | {batcher.strategy}")
    history = _pretrain(model, {domain.name: batcher}, device, steps, lr,
                        patience, seed, log_every, verbose)
    return model.encoder, history


def pretrain_multi_domain(
    encoder: GINEncoder,
    domains: dict,
    features_by_domain: dict[str, np.ndarray],
    device,
    steps: int = 300,
    lr: float = 1e-3,
    patience: int = 30,
    seed: int = 0,
    log_every: int = 25,
    verbose: bool = True,
    objective: str = "dgi",
    **batcher_kwargs,
) -> tuple[GINEncoder, PretrainHistory]:
    """Pretrain one shared encoder on several domains, round-robin. Arm A."""
    batchers = {
        name: DomainBatcher(d, features_by_domain[name], seed=seed, **batcher_kwargs)
        for name, d in domains.items()
    }
    model = build_objective(encoder, list(domains), objective)
    if verbose:
        print(f"  pretraining [{objective}] on {len(domains)} domains round-robin: {list(domains)}")
        for name, b in batchers.items():
            print(f"    {name:<10} {b.strategy}")
    history = _pretrain(model, batchers, device, steps, lr, patience, seed,
                        log_every, verbose)
    return model.encoder, history


def random_init_encoder(domain, x: np.ndarray, device, seed: int = 0,
                        config=None, calibrate: bool = True, **encoder_kwargs):
    """Arm B: an untrained encoder, with BatchNorm statistics calibrated.

    The calibration is not optional cosmetics, it is what makes arm B a valid
    control. BatchNorm in a freshly constructed module holds placeholder
    running statistics (mean 0, variance 1) and in eval mode applies them
    literally -- meaning it performs no normalisation at all. Probing that is
    not "the same encoder without pretraining", it is a different architecture
    with its normalisation layers disabled.

    Measured on Amazon Photo, where GIN's SUM aggregation over an average
    degree of 31 compounds across two layers: the uncalibrated encoder
    produced embeddings with mean norm 1730 and effective rank 5.3 of 128
    dimensions. Calibrated, the same weights give mean norm 7.7 and effective
    rank 16.6.

    Calibration runs forward passes only. No gradients, no labels.
    """
    from .batching import calibrate_batchnorm
    from .models import build_encoder
    from .seeding import set_seed

    set_seed(seed)
    encoder = build_encoder(config, **encoder_kwargs).to(device)
    if calibrate:
        calibrate_batchnorm(encoder, domain, x, device, passes=3, seed=seed)
    encoder.eval()
    return encoder
