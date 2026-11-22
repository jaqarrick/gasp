#!/usr/bin/env python3
"""Write the report's bar-chart data for the non-mm benchmarks (fig:bars, fig:bank-l1): for each
comparison, the median over its configs of the measured speedup over the config's
reference kernel and of each model's predicted speedup, and each model's median absolute
speedup error.

    bars_<gpu>.csv   vecadd, divergence, SFU, transpose, L2
    bank_<gpu>.csv   bank-conflict degrees (both bank_conflict configs)
    l1_<gpu>.csv     L1 line patterns

Reads every speedup_<gpu>.csv in --out (from export_speedup_scatter.py).

Example usage:
    scripts/results/export_bars.py --out ~/grad/gpu/research/report/data
"""

import csv
import statistics

from batches import parse_args

# file -> [(key, benchmark, kernel, config prefix or None)]
GROUPS = {
    "bars": [
        ("vecaddpoor", "vecadd", "poor", None),
        ("vecaddstride", "vecadd", "stride", None),
        ("divergence", "divergence", "uniform", None),
        ("sfu", "sfu", "sfu_free", None),
        ("transpose", "transpose", "padded", None),
        ("ltwo", "l2", "l2_resident", None),
    ],
    "bank": [(f"d{d}", "bank_conflict", f"degree_{d}", None) for d in (2, 4, 8, 16, 32)],
    "l1": [
        (k.replace("_", ""), "l1_lines", k, None)
        for k in ("s5_l2_misaligned", "s8_l2_stride2", "s8_l8_groups4", "s16_l4_stride4",
                  "s16_l16_groups2", "s32_l8_stride8", "s32_l32_stride32")
    ],
}


def main() -> None:
    data = parse_args(__doc__.splitlines()[0], batches=False).out
    sources = sorted(data.glob("speedup_*.csv"))
    if not sources:
        raise SystemExit(f"no speedup_<gpu>.csv in {data}; run export_speedup_scatter.py first")
    for source in sources:
        gpu = source.stem.removeprefix("speedup_")
        rows = list(csv.DictReader(source.open()))
        for name, items in GROUPS.items():
            path = data / f"{name}_{gpu}.csv"
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["key", "n", "measured", "gasp", "gpuperf", "gasp_err", "gpuperf_err"])
                for key, bench, kernel, _ in items:
                    rs = [r for r in rows if r["benchmark"] == bench and r["kernel"] == kernel]
                    m = [float(r["measured"]) for r in rs]
                    g = [float(r["gasperf"]) for r in rs]
                    p = [float(r["gpuperf"]) for r in rs]
                    err = lambda pred: statistics.median(abs(a - b) / b * 100 for a, b in zip(pred, m))
                    w.writerow([key, len(rs), f"{statistics.median(m):.4g}",
                                f"{statistics.median(g):.4g}", f"{statistics.median(p):.4g}",
                                f"{err(g):.1f}", f"{err(p):.1f}"])
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
