#!/usr/bin/env python3
"""Read every result.json of a batch from scripts/batch_experiment.sh and print one
row per config, kernel and model: the launch, then predicted and measured T_GPU and
speedup with their errors. A dry-run batch has no measurements, so those columns
are blank. Only GASPerf rows by default; --model picks another model, or all.

Example usage:
    scripts/read_batch.py experiments/results/e2e/rtx2080ti/batch_20261113_153012
    scripts/read_batch.py --csv <batch_dir> > batch.csv
    scripts/read_batch.py --model all <batch_dir>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = [
    "config",
    "kernel",
    "block",
    "grid",
    "problem_size",
    "model",
    "t_gpu_pred",
    "t_gpu_meas",
    "t_gpu_err_pct",
    "speedup_pred",
    "speedup_meas",
    "speedup_err_pct",
]


def dims(values: list[int]) -> str:
    return "x".join(str(v) for v in values)


def rows_for(config: str, doc: dict, only_model: str | None) -> list[dict]:
    measured = (doc.get("result") or {}).get("kernels", {})
    rows = []
    for name, k in doc["prediction"]["kernels"].items():
        for model, pred in k["models"].items():
            if only_model is not None and model != only_model:
                continue
            meas = measured.get(name, {}).get("models", {}).get(model)
            row = {
                "config": config,
                "kernel": name,
                "block": dims(k["block_dim"]),
                "grid": dims(k["grid_dim"]),
                "problem_size": k["problem_size"],
                "model": model,
                "t_gpu_pred": pred["t_gpu"],
                "speedup_pred": pred["speedup"],
            }
            if meas:
                row.update(
                    t_gpu_meas=meas["t_gpu"]["measured"],
                    t_gpu_err_pct=meas["t_gpu"]["error_pct"],
                    speedup_meas=meas["speedup"]["measured"],
                    speedup_err_pct=meas["speedup"]["error_pct"],
                )
            rows.append(row)
    return rows


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 100 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def print_table(rows: list[dict]) -> None:
    cells = [COLUMNS] + [[fmt(r.get(c)) for c in COLUMNS] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(COLUMNS))]
    for n, row in enumerate(cells):
        # text left, numbers right
        print(
            "  ".join(
                cell.ljust(w) if i < 6 else cell.rjust(w)
                for i, (cell, w) in enumerate(zip(row, widths))
            ).rstrip()
        )
        if n == 0:
            print("  ".join("-" * w for w in widths))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batch_dir", type=Path, help="a batch_<timestamp> folder")
    parser.add_argument("--csv", action="store_true", help="write CSV to stdout")
    parser.add_argument(
        "--model",
        default="GASPerf",
        help="model to show, or 'all' (default: GASPerf)",
    )
    args = parser.parse_args()

    if not args.batch_dir.is_dir():
        print(f"error: {args.batch_dir} is not a directory", file=sys.stderr)
        return 1

    rows, missing = [], []
    for config_dir in sorted(p for p in args.batch_dir.iterdir() if p.is_dir()):
        result = config_dir / "result.json"
        try:
            doc = json.loads(result.read_text())
        except (OSError, json.JSONDecodeError) as e:
            # a failed config leaves no or a partial result.json
            missing.append(f"{config_dir.name}: {e}")
            continue
        rows.extend(
            rows_for(config_dir.name, doc, None if args.model == "all" else args.model)
        )

    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    else:
        print_table(rows)
    for m in missing:
        print(f"skipped {m}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
