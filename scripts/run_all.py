"""Execute the full run matrix, headless and resumable.

    .venv\\Scripts\\python.exe scripts\\run_all.py                 # everything
    .venv\\Scripts\\python.exe scripts\\run_all.py --smoke         # tiny budget
    .venv\\Scripts\\python.exe scripts\\run_all.py --targets cora  # one fold
    .venv\\Scripts\\python.exe scripts\\run_all.py --dry-run       # just plan

**Resumability.** `results/runs.jsonl` is append-only and every row carries a
`run_id` derived from the config hash and seed. On start-up we read the ids
already present and skip them, so a crashed or interrupted sweep restarts
where it stopped. Because runs are bit-reproducible, a skipped run is exactly
the run that would have been produced.

**Ordering.** Runs are grouped by (target, seed) and then by arm, so that the
embedding cache is useful: arms A, B and D build their encoder without ever
seeing a label, which makes their embeddings identical across the five label
fractions. Grouping this way turns five pretraining runs into one. The cache
is cleared between groups so memory stays bounded.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import arms, config, datasets, pipeline  # noqa: E402
from src.config import RunConfig  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run the OmniGraph experiment matrix.")
    p.add_argument("--targets", nargs="*", default=None,
                   help="target domains to run (default: all four folds)")
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="seeds (default: 0 1 2)")
    p.add_argument("--fractions", nargs="*", type=float, default=None,
                   help="label fractions (default: 0.01 0.05 0.1 0.5 1.0)")
    p.add_argument("--arms", nargs="*", default=None,
                   help="restrict to these arms")
    p.add_argument("--objective", default="dgi", choices=["dgi", "mae"],
                   help="pretraining objective: dgi (contrastive, the original "
                        "runs) or mae (masked feature reconstruction). Results "
                        "are logged with distinct run ids, so the two never "
                        "collide and either can be resumed independently.")
    p.add_argument("--smoke", action="store_true",
                   help="tiny budget -- exercises the code, produces no results")
    p.add_argument("--no-experts", action="store_true",
                   help="skip arm D (NOT recommended: it is the control that "
                        "separates 'cross-domain pretraining helps' from "
                        "'any pretraining helps')")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit without running anything")
    p.add_argument("--out", type=Path, default=config.RUNS_JSONL,
                   help="results jsonl (default: results/runs.jsonl)")
    p.add_argument("--force", action="store_true",
                   help="re-run even if a run_id is already logged")
    return p.parse_args(argv)


def group_key(cfg: RunConfig) -> tuple:
    """Runs sharing this key can share one embedding cache."""
    return (cfg.target_domain, cfg.seed)


def main(argv=None) -> int:
    args = parse_args(argv)
    config.ensure_dirs()

    budget = arms.Budget.smoke() if args.smoke else arms.Budget()
    template = RunConfig(pretrain_objective=args.objective)
    configs = arms.matrix_configs(
        targets=args.targets, seeds=args.seeds, fractions=args.fractions,
        include_experts=not args.no_experts, base=template,
    )
    if args.objective != "dgi":
        # Arm B and arm C never pretrain, so their numbers do not depend on the
        # objective -- but they are still re-run here on purpose. Comparing an
        # MAE arm A against a DGI-era arm B would confound the objective with
        # the machine the baseline was produced on, and these runs are cheap.
        print(f"  objective     : {args.objective}  (arms B and C re-run as "
              f"same-machine baselines)")
    if args.arms:
        configs = [c for c in configs if c.arm in set(args.arms)]

    done = set() if args.force else config.completed_run_ids(args.out)
    todo = [c for c in configs if c.run_id not in done]

    print("=" * 72)
    print("OmniGraph run matrix")
    print("=" * 72)
    print(f"  budget        : {budget.describe()}"
          + ("   [SMOKE -- not results]" if args.smoke else ""))
    print(f"  results file  : {args.out}")
    print(f"  planned runs  : {len(configs)}")
    print(f"  already done  : {len(configs) - len(todo)}")
    print(f"  to run now    : {len(todo)}")
    if args.no_experts:
        print("  WARNING: arm D disabled. 'Transfer works' will not be "
              "distinguishable from 'any pretraining works'.")
    print()

    if not todo:
        print("Nothing to do -- every planned run is already in the results file.")
        return 0

    by_group: dict[tuple, list[RunConfig]] = defaultdict(list)
    for cfg in todo:
        by_group[group_key(cfg)].append(cfg)
    # Cheap arms first inside a group, so a crash still leaves useful rows.
    arm_order = {"B_random": 0, "D_expert": 1, "A_transfer": 2, "C_scratch": 3}
    for group in by_group.values():
        group.sort(key=lambda c: (arm_order.get(c.arm, 9),
                                  c.source_domains, c.label_fraction))

    print(f"  grouped into {len(by_group)} (target, seed) blocks")
    if args.dry_run:
        for (target, seed), group in sorted(by_group.items()):
            print(f"    {target:<9} seed {seed}: {len(group)} runs")
        print("\n--dry-run: nothing executed.")
        return 0

    device = config.get_device()
    print(f"  device        : {device}")
    print()

    started = time.perf_counter()
    completed = failed = 0

    for gi, ((target, seed), group) in enumerate(sorted(by_group.items()), start=1):
        print("-" * 72)
        print(f"[block {gi}/{len(by_group)}]  target={target}  seed={seed}  "
              f"({len(group)} runs)")
        print("-" * 72)

        needed = {target} | {s for c in group for s in c.source_domains}
        prepared = {n: pipeline.prepare(n, seed) for n in sorted(needed)}
        cache: dict = {}

        for cfg in group:
            label = cfg.arm + (f"[{cfg.source_domains[0]}]"
                               if cfg.arm == "D_expert" else "")
            print(f"  {label:<22} lf={cfg.label_fraction:<5g} ", end="", flush=True)
            try:
                outcome = arms.run_arm(cfg, prepared, device, budget,
                                       verbose=False, embedding_cache=cache)
                config.log_run(cfg, outcome.as_metrics_dict(), path=args.out)
                flag = "" if outcome.converged else "  [no plateau]"
                print(f"{outcome.primary_metric}={outcome.primary_value:.4f}"
                      f"  {outcome.total_seconds:5.1f}s{flag}", flush=True)
                completed += 1
            except Exception:  # noqa: BLE001 - one bad run must not kill the sweep
                failed += 1
                print("FAILED", flush=True)
                traceback.print_exc()

        cache.clear()
        prepared.clear()
        elapsed = time.perf_counter() - started
        rate = elapsed / max(completed, 1)
        left = len(todo) - completed - failed
        print(f"  block done. {completed}/{len(todo)} runs, "
              f"{elapsed / 60:.1f} min elapsed, "
              f"~{rate * left / 60:.0f} min remaining")
        print()

    total = time.perf_counter() - started
    print("=" * 72)
    print(f"completed {completed} runs in {total / 60:.1f} min"
          + (f", {failed} FAILED" if failed else ""))
    print(f"results appended to {args.out}")
    if failed:
        print("Re-run the same command to retry only the failures.")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
