"""Show how far the run matrix has got, and what it says so far.

    .venv\\Scripts\\python.exe scripts\\progress.py
    .venv\\Scripts\\python.exe scripts\\progress.py --watch      # refresh every 30s
    .venv\\Scripts\\python.exe scripts\\progress.py --results    # add the score tables

Reads results/runs.jsonl only. It touches no GPU and takes no lock, so it is
safe to run at any time while the sweep is going.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import arms, config, results  # noqa: E402

BAR_WIDTH = 44


def bar(done: int, total: int) -> str:
    filled = int(BAR_WIDTH * done / max(total, 1))
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + "]"


def render(path: Path, show_results: bool) -> None:
    df = results.load_frame(path)
    expected = len(arms.matrix_configs())
    done = len(df)

    print("=" * 72)
    print(f"OmniGraph run matrix     {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 72)
    print(f"  {bar(done, expected)}  {done}/{expected}  ({done / expected:.0%})")

    if done == 0:
        print("\n  No runs logged yet.")
        return

    # Elapsed wall-clock from the first to the most recent row.
    stamps = sorted(r["timestamp"] for r in config.load_runs(path))
    t0 = datetime.fromisoformat(stamps[0])
    t1 = datetime.fromisoformat(stamps[-1])
    elapsed = (t1 - t0).total_seconds()
    rate = elapsed / max(done, 1)
    remaining = rate * (expected - done)

    print(f"  elapsed  {elapsed / 60:6.1f} min"
          f"   |  GPU time logged {df['seconds'].sum() / 3600:.2f} h")
    if done < expected:
        print(f"  ETA      {remaining / 60:6.1f} min at the average rate so far")
        print("           (PPI and Elliptic folds run slower than Cora and Photo,")
        print("            so a mid-sweep estimate drifts)")
    else:
        print("  COMPLETE")

    print("\n  by fold:")
    for target in ["cora", "photo", "ppi", "elliptic"]:
        sub = df[df["target"] == target]
        per_fold = expected // 4
        seeds = sorted(int(x) for x in sub["seed"].unique())
        mark = "done" if len(sub) >= per_fold else "...."
        print(f"    {target:<9} {len(sub):>3}/{per_fold}  {mark}   seeds {seeds}")

    bad = df[~df["converged"]]
    print(f"\n  did not reach a validation plateau: {len(bad)} of {done}")
    if len(bad):
        for (t, a), n in bad.groupby(["target", "arm"]).size().items():
            print(f"    {t:<9} {a:<12} {n}")
        print("    (these cells are marked * in results/table.md -- lower bounds)")

    if not show_results:
        print("\n  --results shows the score tables, --watch refreshes every 30s")
        return

    print()
    for target in sorted(df["target"].unique()):
        sub = df[df["target"] == target]
        summary = results.summarise(sub)
        if summary.empty:
            continue
        metric = summary["metric"].iloc[0]
        fractions = sorted(summary["fraction"].unique())
        print("-" * 72)
        print(f"  {target}  ({metric})   {len(sub)} runs, seeds "
              f"{sorted(int(x) for x in sub['seed'].unique())}")
        print("    " + " " * 24 + "".join(f"{f:>13.0%}" for f in fractions))
        for arm in results.ARM_ORDER:
            row = summary[summary["arm"] == arm].sort_values("fraction")
            if row.empty:
                continue
            cells = "".join(f"{m:>7.3f}±{s:<5.3f}"
                            for m, s in zip(row["mean"], row["std"]))
            print(f"    {results.ARM_LABELS[arm]:<24}{cells}")

        gains = results.transfer_gain(sub)
        if "gain_vs_expert_mean" in gains:
            v = results.verdict(gains, "gain_vs_expert")
            counts = v["verdict"].value_counts().to_dict()
            print(f"    arm A vs BEST single-source expert: {counts}")
    print("-" * 72)
    print("  'indistinguishable' = the gap is smaller than its own spread")
    print("  across seeds, which is not evidence of a difference.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch", action="store_true", help="refresh every 30 seconds")
    p.add_argument("--results", action="store_true", help="also print score tables")
    p.add_argument("--interval", type=int, default=30)
    p.add_argument("--path", type=Path, default=config.RUNS_JSONL)
    args = p.parse_args(argv)

    if not args.watch:
        render(args.path, args.results)
        return 0

    try:
        while True:
            print("\033[2J\033[H", end="")   # clear screen
            render(args.path, args.results)
            print(f"\n  refreshing every {args.interval}s -- Ctrl+C to stop")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped watching (the sweep itself is unaffected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
