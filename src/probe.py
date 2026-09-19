"""The downstream evaluation: a frozen encoder and a linear probe.

**Why a linear probe, and why frozen.** The question is what the *encoder*
learned, not what a classifier can squeeze out of it. A single linear layer
can only read properties that are already linearly available in the
representation, so its score is a statement about the embedding. A deeper
probe could learn the task largely by itself and would blur the very
difference between arms that the experiment exists to measure.

Two training functions live here, and they share every piece of machinery they
can -- same optimiser, same early-stopping rule, same scoring code -- so that a
comparison between arms reflects the encoder and nothing else:

* `train_linear_probe` -- arms A, B and D. Encoder frozen, probe trained.
* `train_end_to_end`   -- arm C. Encoder and head trained together on the
  target's labels. This is the upper bound: what you get if you forget
  transfer entirely and just train on the task.

Early stopping watches **validation** and only validation. Test is read once,
at the end, by `evaluate_split`.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .metrics import compute_metrics, primary_metric_for
from .models import LinearProbe


@dataclass
class ProbeResult:
    """Everything a single arm-run produces."""

    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    best_epoch: int = -1
    best_val_score: float = -np.inf
    epochs_run: int = 0
    stopped_early: bool = False
    seconds: float = 0.0
    train_nodes: int = 0
    history: dict[str, list[float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "best_epoch": self.best_epoch,
            "best_val_score": self.best_val_score,
            "epochs_run": self.epochs_run,
            "stopped_early": self.stopped_early,
            "seconds": self.seconds,
            "train_nodes": self.train_nodes,
        }


def loss_for_task(task: str):
    """Multi-label needs BCE per label; everything else is cross-entropy."""
    if task == "multilabel":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def targets_for_task(y: np.ndarray, task: str, device) -> torch.Tensor:
    if task == "multilabel":
        return torch.as_tensor(y, dtype=torch.float32, device=device)
    return torch.as_tensor(y, dtype=torch.long, device=device)


def output_dim_for(task: str, y: np.ndarray, num_classes: int) -> int:
    if task == "multilabel":
        return int(y.shape[1])
    return int(num_classes)


# --------------------------------------------------------------------------
# Arms A, B, D -- frozen encoder
# --------------------------------------------------------------------------

def standardize_embeddings(embeddings: np.ndarray, fit_idx: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance each embedding dimension, fit on `fit_idx`.

    This is a conditioning fix, not a modelling choice. A linear probe on
    standardized features spans exactly the same function class as one on raw
    features -- the optimum is identical -- so this changes only how quickly
    gradient descent reaches it, and it changes it identically for every arm.

    It matters a great deal in practice. Unstandardized, a PPI probe needed
    1,991 epochs to plateau; standardized it converged in 79. Differently
    conditioned embeddings otherwise converge at different rates, which would
    make an under-trained probe favour whichever arm happened to be easier to
    optimise -- the same class of mistake as the uncalibrated BatchNorm in
    arm B.

    Fit on training nodes only. Note this uses NO labels -- only which nodes
    are in the training split -- so it stays valid at every label fraction.
    """
    mu = embeddings[fit_idx].mean(axis=0)
    sd = embeddings[fit_idx].std(axis=0)
    return (embeddings - mu) / np.maximum(sd, 1e-6)


def train_linear_probe(
    embeddings: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    task: str,
    num_classes: int,
    domain_name: str,
    device,
    epochs: int = 3000,
    lr: float = 0.5,
    weight_decay: float = 0.0,
    patience: int = 200,
    seed: int = 0,
    verbose: bool = False,
    standardize: bool = True,
    standardize_idx: np.ndarray | None = None,
) -> ProbeResult:
    """Train a linear classifier on FROZEN embeddings.

    The embeddings arrive as a plain array and are never differentiated
    through -- the encoder is not merely set to `requires_grad=False`, it is
    not in the graph at all. Freezing is the point of the experiment, so it is
    enforced structurally rather than by remembering to set a flag.

    Defaults (lr 0.5, 3000 epochs, patience 200) were chosen so that all four
    domains actually reach a validation plateau rather than hitting the epoch
    cap while still improving. That distinction is not cosmetic: at the
    original 300-epoch cap, Elliptic's AUC-PR read 0.454 while its converged
    value is 0.663. These are optimiser settings, shared identically by every
    arm, so they cannot favour one arm over another.
    """
    from .seeding import set_seed

    set_seed(seed)
    t0 = time.perf_counter()

    if standardize:
        # Standardize on the FULL training split, not the label-fraction
        # subsample: it needs no labels, and estimating a mean from the 17
        # nodes of a 1% draw would be far noisier than it needs to be.
        fit_idx = standardize_idx if standardize_idx is not None else train_idx
        embeddings = standardize_embeddings(embeddings, fit_idx)

    z = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    targets = targets_for_task(y, task, device)
    out_dim = output_dim_for(task, y, num_classes)
    primary = primary_metric_for(domain_name, task)

    probe = LinearProbe(z.size(1), out_dim).to(device)
    optimiser = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = loss_for_task(task)

    tr = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    va = torch.as_tensor(val_idx, dtype=torch.long, device=device)

    result = ProbeResult(train_nodes=len(train_idx))
    result.history = {"train_loss": [], "val_score": []}
    best_state = copy.deepcopy(probe.state_dict())
    since_improved = 0

    for epoch in range(1, epochs + 1):
        probe.train()
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(probe(z[tr]), targets[tr])
        loss.backward()
        optimiser.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(z[va]).cpu().numpy()
        val_scores = compute_metrics(task, y[val_idx], val_logits)
        score = val_scores.get(primary, float("nan"))

        result.history["train_loss"].append(float(loss.detach()))
        result.history["val_score"].append(float(score))

        # VALIDATION only. Test is not read anywhere in this loop.
        if score > result.best_val_score + 1e-6:
            result.best_val_score = float(score)
            result.best_epoch = epoch
            best_state = copy.deepcopy(probe.state_dict())
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                result.stopped_early = True
                break

        if verbose and epoch % 50 == 0:
            print(f"      epoch {epoch:>4}  loss {float(loss):.4f}  val {primary} {score:.4f}")

    result.epochs_run = epoch
    probe.load_state_dict(best_state)          # restore the best, not the last
    probe.eval()

    result.val_metrics = evaluate_split(probe, z, y, val_idx, task)
    result.test_metrics = evaluate_split(probe, z, y, test_idx, task)
    result.seconds = time.perf_counter() - t0
    return result


@torch.no_grad()
def evaluate_split(probe, z: torch.Tensor, y: np.ndarray,
                   idx: np.ndarray, task: str) -> dict[str, float]:
    """Score one split. The only place test labels are ever touched."""
    probe.eval()
    i = torch.as_tensor(idx, dtype=torch.long, device=z.device)
    logits = probe(z[i]).cpu().numpy()
    return compute_metrics(task, y[idx], logits)


# --------------------------------------------------------------------------
# Arm C -- end to end, the upper bound
# --------------------------------------------------------------------------

def train_end_to_end(
    encoder,
    domain,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    task: str,
    num_classes: int,
    domain_name: str,
    device,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 30,
    seed: int = 0,
    verbose: bool = False,
) -> ProbeResult:
    """Train encoder and head together on the target's own labels. Arm C.

    This is the upper bound the frozen arms are measured against: what the
    same architecture achieves when it is allowed to fit the task directly,
    with no transfer involved. If a frozen arm approaches it, the frozen
    representation is carrying almost everything the task needs.

    Runs on the full graph in one step per epoch, which the 4 GB budget allows
    for Cora, Photo and any single PPI graph. Elliptic uses the same 2-hop
    subgraph batching as pretraining.
    """
    from .batching import DomainBatcher, full_embeddings
    from .seeding import set_seed

    set_seed(seed)
    t0 = time.perf_counter()

    targets = targets_for_task(y, task, device)
    out_dim = output_dim_for(task, y, num_classes)
    primary = primary_metric_for(domain_name, task)

    encoder = encoder.to(device)
    head = LinearProbe(encoder.out_dim, out_dim).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimiser = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = loss_for_task(task)

    batcher = DomainBatcher(domain, x, seed=seed)
    train_mask = np.zeros(domain.total_nodes, dtype=bool)
    train_mask[train_idx] = True

    result = ProbeResult(train_nodes=len(train_idx))
    result.history = {"train_loss": [], "val_score": []}
    best_state = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(head.state_dict()))
    since_improved = 0

    # One EPOCH is one pass over the domain, not one batch. Without this,
    # "300 epochs" means 300 full passes for Cora (1 batch each) but only 12.5
    # passes for PPI (24 graphs each), and the two arms are not comparably
    # trained. Measured: PPI arm C scored 0.497 micro-F1 at 200 batch-steps and
    # 0.774 once it was actually trained.
    #
    # Validating once per epoch rather than once per step also removes most of
    # the cost, since validation embeds the whole domain.
    steps_per_epoch = max(1, batcher.batches_per_epoch)

    for epoch in range(1, epochs + 1):
        encoder.train(); head.train()
        epoch_loss, seen = 0.0, 0

        for _ in range(steps_per_epoch):
            batch = batcher.next_batch().to(device)

            # Which nodes of this batch are labelled training nodes?
            local_train = np.flatnonzero(train_mask[batch.node_idx.numpy()])
            if local_train.size == 0:
                continue
            local = torch.as_tensor(local_train, dtype=torch.long, device=device)
            global_ids = batch.node_idx.numpy()[local_train]

            optimiser.zero_grad(set_to_none=True)
            logits = head(encoder(batch.x, batch.edge_index))[local]
            loss = criterion(logits, targets[torch.as_tensor(global_ids, dtype=torch.long,
                                                             device=device)])
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.detach()); seen += 1

        if seen == 0:
            continue
        loss = torch.tensor(epoch_loss / seen)

        z_all = full_embeddings(encoder, domain, x, device)
        with torch.no_grad():
            zt = torch.as_tensor(z_all, dtype=torch.float32, device=device)
            val_logits = head(zt[torch.as_tensor(val_idx, dtype=torch.long,
                                                 device=device)]).cpu().numpy()
        score = compute_metrics(task, y[val_idx], val_logits).get(primary, float("nan"))

        result.history["train_loss"].append(float(loss.detach()))
        result.history["val_score"].append(float(score))

        if score > result.best_val_score + 1e-6:
            result.best_val_score = float(score)
            result.best_epoch = epoch
            best_state = (copy.deepcopy(encoder.state_dict()),
                          copy.deepcopy(head.state_dict()))
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                result.stopped_early = True
                break

        if verbose and epoch % 25 == 0:
            print(f"      epoch {epoch:>4}  loss {float(loss):.4f}  val {primary} {score:.4f}")

    result.epochs_run = epoch
    encoder.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])

    z_all = full_embeddings(encoder, domain, x, device)
    zt = torch.as_tensor(z_all, dtype=torch.float32, device=device)
    result.val_metrics = evaluate_split(head, zt, y, val_idx, task)
    result.test_metrics = evaluate_split(head, zt, y, test_idx, task)
    result.seconds = time.perf_counter() - t0
    return result
