#!/usr/bin/env python3
"""Write mm_<gpu>.csv: median speedup over `basic` per mm variant, measured and predicted.

Reads the speedup_<gpu>.csv files in --out (from export_speedup_scatter.py).

Example usage:
    scripts/results/export_mm.py --out /to/some/dir
"""

import csv
import statistics

from batches import parse_args

VARIANTS = ["tiled", "uncoalesced", "double", "bounds", "sfu"]


def main() -> None:
    data = parse_args(__doc__.splitlines()[0], batches=False).out
    sources = sorted(data.glob("speedup_*.csv"))
    if not sources:
        raise SystemExit(f"no speedup_<gpu>.csv in {data}; run export_speedup_scatter.py first")
    for source in sources:
        gpu = source.stem.removeprefix("speedup_")
        rows = [r for r in csv.DictReader(source.open()) if r["benchmark"] == "mm"]
        path = data / f"mm_{gpu}.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant", "n", "measured", "gasp", "gpuperf", "gasp_err", "gpuperf_err"])
            for v in VARIANTS:
                rs = [r for r in rows if r["kernel"] == v]
                m = [float(r["measured"]) for r in rs]
                g = [float(r["gasperf"]) for r in rs]
                p = [float(r["gpuperf"]) for r in rs]
                err = lambda pred: statistics.median(abs(a - b) / b * 100 for a, b in zip(pred, m))
                w.writerow([v, len(rs), f"{statistics.median(m):.4g}", f"{statistics.median(g):.4g}",
                            f"{statistics.median(p):.4g}", f"{err(g):.1f}", f"{err(p):.1f}"])
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
