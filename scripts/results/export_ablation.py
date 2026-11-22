#!/usr/bin/env python3
"""Write ablation_<gpu>.csv for the report's profiled-input heatmap (fig:ablation), one
file per ablation batch: one row per (ablation row, benchmark) cell, ablation rows
in ROWS order and benchmarks in BENCHMARKS order within each, with x and y indices,
GASP's median absolute speedup error in that cell and its change from all-static
(delta, percentage points).

The numbers come from read_ablation.summarize, the same code as make ablation-results.

Example usage:
    scripts/results/export_ablation.py --out ~/grad/gpu/research/report/data
    scripts/results/export_ablation.py --out <dir> <batch_dir> ...
"""

import csv
import sys

from batches import gpu_of, parse_args
from read_ablation import load_batch, summarize

BASE = "all-static"
ROWS = ["crude", "profiled-cache", "profiled-coalescing", "profiled-bank", "profiled-counts", "all-profiled"]
BENCHMARKS = ["bank_conflict", "divergence", "l1_lines", "l2", "mm", "sfu", "transpose", "vecadd"]
GASP = "GASPerf"


def main() -> int:
    args = parse_args(__doc__.splitlines()[0], batches=True)
    out_dir = args.out
    gpus = [gpu_of(batch) for batch in args.batch_dirs]
    cells: dict[tuple[str, str, str, str], dict] = {}
    for gpu, batch in zip(gpus, args.batch_dirs):
        records, missing, row_order = load_batch(batch, None)
        if missing:
            print(f"{gpu}: missing {missing}", file=sys.stderr)
        for s in summarize(records, row_order):
            cells[(s["gpu"], s["benchmark"], s["model"], s["row"])] = s

    def value(gpu: str, benchmark: str, model: str, row: str, key: str) -> float:
        s = cells.get((gpu, benchmark, model, row))
        if s is None or s[key] is None:
            raise SystemExit(f"no {key} for {gpu} {benchmark} {model} {row}")
        return s[key]

    for gpu in gpus:
        path = out_dir / f"ablation_{gpu}.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y", "row", "benchmark", "value", "delta"])
            for y, row in enumerate(ROWS):
                for x, benchmark in enumerate(BENCHMARKS):
                    v = value(gpu, benchmark, GASP, row, "abs_speedup_err_median")
                    base = value(gpu, benchmark, GASP, BASE, "abs_speedup_err_median")
                    w.writerow([x, y, row, benchmark, f"{v:.1f}", f"{v - base:.1f}"])
        print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
