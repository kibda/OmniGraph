"""Paths, device detection, and run bookkeeping.

Three jobs:
  1. Find a writable dataset root that works on a laptop, on Kaggle, and on
     Colab, without the caller hard-coding any path.
  2. Pick the compute device (CUDA if present, else CPU).
  3. Give every run a stable fingerprint (config hash) so rows in
     results/runs.jsonl can be matched up, deduplicated, and resumed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Project layout
# --------------------------------------------------------------------------

#: Repository root: the directory that contains src/.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
RUNS_JSONL: Path = RESULTS_DIR / "runs.jsonl"


def default_data_root() -> Path:
    """Where datasets get downloaded to.

    Resolution order, first match wins:
      1. $OMNIGRAPH_DATA          -- explicit override, always respected
      2. /kaggle/working/data     -- Kaggle notebooks (the writable layer)
      3. /content/data            -- Google Colab
      4. <project>/data           -- local checkout

    Kaggle's /kaggle/input is read-only, so we never download into it:
    PyTorch Geometric needs to write a processed/ folder next to raw/.
    """
    override = os.environ.get("OMNIGRAPH_DATA")
    if override:
        return Path(override).expanduser().resolve()
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working/data")
    if Path("/content").is_dir():
        return Path("/content/data")
    return PROJECT_ROOT / "data"


DATA_ROOT: Path = default_data_root()


def ensure_dirs() -> None:
    """Create the directories we write to. Safe to call repeatedly."""
    for d in (DATA_ROOT, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------

def get_device(prefer_cuda: bool = True):
    """Return the torch device to run on.

    torch is imported lazily so this module can be imported for its paths
    before torch is installed -- which is exactly the situation in the very
    first setup notebook.
    """
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report() -> dict[str, Any]:
    """Facts about the compute device, for the setup notebook to print."""
    import torch

    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_build": torch.version.cuda,
        "device": str(get_device()),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu_name=props.name,
            gpu_total_memory_gb=round(props.total_memory / 1024**3, 2),
            gpu_capability=f"sm_{props.major}{props.minor}",
            gpu_count=torch.cuda.device_count(),
        )
    return info


# --------------------------------------------------------------------------
# Run configuration and fingerprinting
# --------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Everything that determines what a run computes.

    The hash of this object identifies a run. Two runs with the same hash and
    the same seed should produce the same numbers; that is what makes
    scripts/run_all.py resumable.

    Every field is flat and JSON-serialisable -- no objects, no paths -- so
    the hash is stable across machines.
    """

    # --- what we are measuring -------------------------------------------
    arm: str = "A_transfer"                 # A_transfer|B_random|C_scratch|D_expert
    target_domain: str = "cora"             # the held-out domain we probe on
    source_domains: tuple[str, ...] = ()    # pretraining sources; () for B and C
    label_fraction: float = 1.0             # 1%,5%,10%,50%,100% -> 0.01 .. 1.0
    seed: int = 0

    # --- feature unification ---------------------------------------------
    svd_dim: int = 128
    n_structural: int = 5     # log1p(deg), clustering, pagerank, k-core, triangles
    # svd_dim + n_structural == encoder input dim (133)

    # --- encoder ----------------------------------------------------------
    encoder: str = "gin"
    hidden_dim: int = 256
    out_dim: int = 128
    num_layers: int = 2

    # --- pretraining (DGI) ------------------------------------------------
    pretrain_epochs: int = 300
    pretrain_lr: float = 1e-3
    pretrain_patience: int = 30

    # --- downstream linear probe ------------------------------------------
    probe_epochs: int = 300
    probe_lr: float = 1e-2
    probe_weight_decay: float = 0.0
    probe_patience: int = 30

    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_domains"] = list(self.source_domains)
        return d

    @property
    def hash(self) -> str:
        """12-char fingerprint of the configuration, seed included.

        `notes` is excluded on purpose: it is a human comment, not an input
        to the computation, so editing it must not invalidate a finished run.
        """
        payload = {k: v for k, v in self.to_dict().items() if k != "notes"}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def run_id(self) -> str:
        return (
            f"{self.arm}__{self.target_domain}"
            f"__lf{self.label_fraction:g}__s{self.seed}__{self.hash}"
        )


# --------------------------------------------------------------------------
# results/runs.jsonl
# --------------------------------------------------------------------------

def git_commit() -> str | None:
    """Short commit hash, so a result can be traced back to the code."""
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def log_run(
    config: RunConfig,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
    path: Path = RUNS_JSONL,
) -> dict[str, Any]:
    """Append one result row to results/runs.jsonl and return it.

    One line of JSON per run, append-only. We never rewrite history, so a
    crashed sweep can be restarted without losing what already finished.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": config.run_id,
        "config_hash": config.hash,
        "seed": config.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "config": config.to_dict(),
        "metrics": metrics,
    }
    if extra:
        record["extra"] = extra
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def load_runs(path: Path = RUNS_JSONL) -> list[dict[str, Any]]:
    """Read every result row. Returns [] if nothing has been run yet."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def completed_run_ids(path: Path = RUNS_JSONL) -> set[str]:
    """Run ids already present -- what makes run_all.py resumable."""
    return {r["run_id"] for r in load_runs(path)}
