"""Environment introspection for notebook 00.

Answers three questions a reader of the repo will ask before trusting any
number in it: what am I running on, are the versions the ones that were
pinned, and can the code actually reach a GPU.
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path
from typing import Any

from . import config

#: Packages the experiments actually depend on, in import order.
REQUIRED = [
    "torch",
    "torch_geometric",
    "numpy",
    "scipy",
    "sklearn",
    "networkx",
    "pandas",
    "matplotlib",
]


def detect_platform() -> str:
    """Which of the three environments we support are we in?"""
    if Path("/kaggle/working").is_dir():
        return "kaggle"
    if Path("/content").is_dir() and "google.colab" in sys.modules:
        return "colab"
    if Path("/content").is_dir():
        return "colab-like"
    return "local"


def package_versions(names: list[str] | None = None) -> dict[str, str]:
    """Import each package and report its version, or why it failed.

    We import rather than read metadata: a package can be present in
    site-packages and still fail to import (a broken CUDA build, a missing
    DLL on Windows). Importing is the check that matters.
    """
    names = names or REQUIRED
    out: dict[str, str] = {}
    for name in names:
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "(no __version__)")
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            out[name] = f"MISSING/BROKEN: {type(exc).__name__}: {exc}"
    return out


def full_report() -> dict[str, Any]:
    """Everything notebook 00 prints, as one dict."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "environment": detect_platform(),
        "project_root": str(config.PROJECT_ROOT),
        "data_root": str(config.DATA_ROOT),
        "data_root_exists": config.DATA_ROOT.exists(),
        "results_jsonl": str(config.RUNS_JSONL),
        "packages": package_versions(),
    }
    try:
        report["device"] = config.device_report()
    except Exception as exc:  # noqa: BLE001
        report["device"] = {"error": f"{type(exc).__name__}: {exc}"}
    return report


def print_report(report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pretty-print the environment report. Returns it too, for asserting on."""
    report = report or full_report()

    def header(title: str) -> None:
        print(f"\n{title}\n{'-' * len(title)}")

    header("Interpreter")
    print(f"  python       : {report['python']}")
    print(f"  executable   : {report['executable']}")
    print(f"  platform     : {report['platform']}")
    print(f"  environment  : {report['environment']}")

    header("Paths")
    print(f"  project root : {report['project_root']}")
    print(f"  data root    : {report['data_root']}  (exists={report['data_root_exists']})")
    print(f"  runs.jsonl   : {report['results_jsonl']}")

    header("Packages")
    for name, ver in report["packages"].items():
        flag = "FAIL" if ver.startswith("MISSING") else "ok  "
        print(f"  [{flag}] {name:<18} {ver}")

    header("Compute device")
    dev = report["device"]
    if "error" in dev:
        print(f"  could not query torch: {dev['error']}")
    else:
        print(f"  torch        : {dev['torch_version']}  (cuda build: {dev['cuda_build']})")
        print(f"  cuda available: {dev['cuda_available']}")
        print(f"  SELECTED     : {dev['device']}")
        if dev["cuda_available"]:
            print(f"  gpu          : {dev['gpu_name']}")
            print(f"  gpu memory   : {dev['gpu_total_memory_gb']} GB")
            print(f"  capability   : {dev['gpu_capability']}")
    return report


def assert_environment_ok(report: dict[str, Any] | None = None) -> None:
    """Fail loudly if anything required is missing.

    Better to stop here, in the setup notebook, than to discover a broken
    install three notebooks later in the middle of a training loop.
    """
    report = report or full_report()
    broken = [n for n, v in report["packages"].items() if v.startswith("MISSING")]
    if broken:
        raise RuntimeError(
            f"These packages are missing or fail to import: {broken}. "
            "Run the install cell above, then restart the kernel."
        )
    print("\nEnvironment OK: every required package imports cleanly.")
