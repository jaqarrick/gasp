#!/usr/bin/env python3
"""Write ablation_<gpu>.csv for the profiled-input heatmap, one file per batch.

Each cell is GASP's median absolute speedup error and its change from all-static,
from read_ablation.summarize (the same as make ablation-results). ablation_summary.csv
has all of that summary: every GPU, model, benchmark (and "all") and row, with runtime
and speedup errors and each one's change from the same model's all-static row.

Example usage:
    scripts/results/export_ablation.py --out /some/directory
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
METRICS = ["abs_err_median", "abs_err_mean", "abs_speedup_err_median", "abs_speedup_err_mean"]
SUMMARY_COLUMNS = (
    ["gpu", "model", "benchmark", "row", "kernels", "speedups"]
    + METRICS
    + [f"{m}_delta" for m in METRICS]
    + ["itilp_capped", "itmlp_capped"]
)


def write_summary(path, cells: dict) -> None:
    """Every summarize() row, with each metric's change from the same model's all-static row."""
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for (gpu, benchmark, model, _), s in cells.items():
            base = cells.get((gpu, benchmark, model, BASE), {})
            row = {k: s[k] for k in SUMMARY_COLUMNS if k in s}
            for m in METRICS:
                if s[m] is not None and base.get(m) is not None:
                    row[f"{m}_delta"] = s[m] - base[m]
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"wrote {path}")


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

    write_summary(out_dir / "ablation_summary.csv", cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
