"""Rebuild a results jsonl from a run_all.py console log.

    python scripts/recover_from_log.py mae_log.txt results/runs_mae.jsonl --objective mae

Every line run_all.py prints carries the whole result -- arm, source, label
fraction, metric, score, and whether the run reached a validation plateau. So
a saved console log is a complete record, and a lost results file can be
reconstructed from it rather than re-run.

This exists because a Kaggle session ended without committing its output and
113 minutes of finished GPU work was sitting only in a notebook's scrollback.

**What is recovered and what is not.** Scores, convergence flags and per-run
wall times come back exactly. Timestamps, the platform string and the git
commit cannot -- the log does not carry them -- so rows are marked
`"recovered_from_log": true` rather than silently fabricated. Any analysis
that only reads scores is unaffected; anything auditing provenance can see
these rows were reconstructed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, pipeline  # noqa: E402
from src.config import RunConfig  # noqa: E402

BLOCK = re.compile(r"\[block\s+\d+/\d+\]\s+target=(\w+)\s+seed=(\d+)")
RUN = re.compile(
    r"^\s+(?P<arm>[A-Z]_\w+?)(?:\[(?P<source>\w+)\])?\s+"
    r"lf=(?P<frac>[\d.]+)\s+"
    r"(?P<metric>\w+)=(?P<score>[\d.]+)\s+"
    r"(?P<secs>[\d.]+)s(?P<flag>\s+\[no plateau\])?\s*$"
)


def parse(text: str) -> list[dict]:
    target = seed = None
    rows = []
    for line in text.splitlines():
        block = BLOCK.search(line)
        if block:
            target, seed = block.group(1), int(block.group(2))
            continue
        match = RUN.match(line)
        if match and target is not None:
            g = match.groupdict()
            rows.append({
                "arm": g["arm"],
                "source": g["source"],
                "target": target,
                "seed": seed,
                "fraction": float(g["frac"]),
                "metric": g["metric"],
                "score": float(g["score"]),
                "seconds": float(g["secs"]),
                "converged": g["flag"] is None,
            })
    return rows


def rebuild(rows: list[dict], out: Path, objective: str) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    written = 0
    for r in rows:
        if r["arm"] == "D_expert":
            sources = (r["source"],)
        elif r["arm"] == "A_transfer":
            sources = pipeline.source_domains_for(r["target"])
        else:
            sources = ()

        cfg = RunConfig(
            arm=r["arm"], target_domain=r["target"], source_domains=sources,
            label_fraction=r["fraction"], seed=r["seed"],
            pretrain_objective=objective,
        )
        config.log_run(
            cfg,
            {
                "primary_metric": r["metric"],
                "primary_value": r["score"],
                # The log prints only the headline metric, so that is all we
                # can restore. Secondary metrics (macro-F1 and friends) are
                # genuinely gone and are not invented here.
                "test": {r["metric"]: r["score"]},
                "val": {},
                "converged": r["converged"],
                "train_nodes": None,
                "total_seconds": r["seconds"],
                "recovered_from_log": True,
            },
            path=out,
        )
        written += 1
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--objective", default="dgi", choices=["dgi", "mae"])
    args = p.parse_args(argv)

    rows = parse(args.log.read_text(encoding="utf-8"))
    print(f"parsed {len(rows)} runs from {args.log}")

    targets = sorted({r["target"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    fracs = sorted({r["fraction"] for r in rows})
    print(f"  targets   : {targets}")
    print(f"  seeds     : {seeds}")
    print(f"  fractions : {fracs}")
    print(f"  arms      : {sorted({r['arm'] for r in rows})}")
    print(f"  no plateau: {sum(not r['converged'] for r in rows)}")

    written = rebuild(rows, args.out, args.objective)
    print(f"\nwrote {written} rows to {args.out}")

    ids = {r["run_id"] for r in config.load_runs(args.out)}
    if len(ids) != written:
        print(f"WARNING: {written - len(ids)} duplicate run ids")
        return 1
    print("every run id is unique")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
