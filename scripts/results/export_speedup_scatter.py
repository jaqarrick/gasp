#!/usr/bin/env python3
"""Write speedup_<gpu>.csv for the report's speedup scatter (fig:scatter): one row per
compared kernel (every kernel but its config's reference), all-static row, with the
measured speedup and both models' predicted speedups. export_mm.py and export_bars.py
read these files, so run this first (make report-data does).

Example usage:
    scripts/results/export_speedup_scatter.py --out ~/grad/gpu/research/report/data
    scripts/results/export_speedup_scatter.py --out <dir> <batch_dir> ...
"""

import csv
import sys

from batches import gpu_of, parse_args
from read_ablation import load_batch


def main() -> int:
    args = parse_args(__doc__.splitlines()[0], batches=True)
    for batch in args.batch_dirs:
        gpu = gpu_of(batch)
        records, missing, _ = load_batch(batch, None)
        if missing:
            print(f"{gpu}: missing {missing}", file=sys.stderr)
        rows: dict[tuple[str, str], dict] = {}
        for r in records:
            if r["row"] != "all-static" or r["is_reference"]:
                continue
            if r.get("predicted_speedup") is None or r.get("measured_speedup") is None:
                continue
            key = (r["config"], r["kernel"])
            row = rows.setdefault(key, {"benchmark": r["benchmark"], "config": r["config"],
                                        "kernel": r["kernel"], "measured": r["measured_speedup"]})
            row[r["model"].lower()] = r["predicted_speedup"]
        path = args.out / f"speedup_{gpu}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["benchmark", "config", "kernel", "measured", "gasperf", "gpuperf"])
            w.writeheader()
            for row in rows.values():
                if row.get("gasperf", 0) > 0 and row.get("gpuperf", 0) > 0 and row["measured"] > 0:
                    w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
        print(f"wrote {path} ({len(rows)} kernels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
