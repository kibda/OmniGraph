"""One function to make a run reproducible.

Randomness in this project enters from four places, and all four have to be
pinned or the same config will give different numbers on a rerun:

  * Python's `random`      -- used by some sampling helpers
  * NumPy                  -- TruncatedSVD, split shuffling
  * PyTorch (CPU)          -- weight init, dropout, DGI corruption
  * PyTorch (CUDA)         -- the same, on the GPU
"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed every RNG this project touches. Returns the seed, for logging.

    `deterministic=True` also turns off cuDNN's autotuner. The autotuner
    benchmarks several convolution algorithms and picks the fastest, which
    can differ between runs and change results in the last decimal places.
    We trade a little speed for exact reproducibility; that is the right
    trade for an experiment whose whole point is comparing arms.
    """
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # cuDNN flags alone are not enough. GIN aggregates neighbours with a
        # scatter-add, which on CUDA uses atomics whose summation order varies
        # between runs -- floating-point addition is not associative, so the
        # result varies in the last bits. Through pretraining and a probe those
        # differences amplify: the same config scored 0.7823, 0.7731 and 0.7694
        # on three consecutive runs before this was switched on.
        #
        # This costs roughly 20% runtime and is worth it: without it,
        # results/runs.jsonl cannot be reproduced from a config hash and seed,
        # which is the guarantee the whole results table rests on.
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:  # noqa: BLE001 - never fail a run over this
            warnings.warn(
                f"Could not enable deterministic algorithms ({exc}). Runs will "
                "still be seeded, but may not be bit-reproducible.",
                RuntimeWarning, stacklevel=2,
            )

    return seed


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker process.

    Worker processes are forked/spawned after `set_seed` ran, so they need
    their own seeding or every worker would draw the same stream.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
