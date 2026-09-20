"""OmniGraph: cross-domain graph foundation model experiments.

Every piece of experiment logic lives in this package. Notebooks import from
here and never define logic of their own.
"""

import os as _os

# Deterministic cuBLAS. This MUST be set before the CUDA context is created,
# which means before torch is first used -- so it lives here, in the package
# __init__, rather than in seeding.py where it would already be too late.
#
# Without it, cuBLAS picks reduction orders that vary between runs, and GIN's
# scatter-add message passing is non-deterministic on GPU for the same reason.
# Measured on an identical config run three times: 0.7823, 0.7731, 0.7694.
# With this plus torch.use_deterministic_algorithms, all three agree exactly.
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
