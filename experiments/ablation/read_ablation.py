#!/usr/bin/env python3
"""Read every ablation.json of a batch from scripts/batch_ablation.sh and print the
error decomposition: per benchmark (and all together), model and ablation row, the
mean and median absolute T_GPU error and speedup error, and how many kernels had
ITILP or ITMLP at a cap (where ILP or MLP can't change the prediction). --records
prints every record instead. Only GASPerf by default; --model picks another model,
or all.

The speedup columns leave out each config's reference kernel, whose speedup is 1
by definition. Every row carries the GPU, so the CSVs of several batches (one per
GPU) can be concatenated.

Example usage:
    experiments/ablation/read_ablation.py experiments/results/ablation/rtx2080ti/batch_20261118_101500
    experiments/ablation/read_ablation.py --csv <batch_dir> > ablation.csv
    experiments/ablation/read_ablation.py --records --model all <batch_dir>
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

RECORD_COLUMNS = [
    "gpu",
    "benchmark",
    "config",
    "row",
    "kernel",
    "model",
    "pico",
    "predicted_cycles",
    "measured_cycles",
    "error_pct",
    "predicted_speedup",
    "measured_speedup",
    "speedup_error_pct",
    "itilp_capped",
    "itmlp_capped",
]

SUMMARY_COLUMNS = [
    "gpu",
    "benchmark",
    "model",
    "row",
    "kernels",
    "abs_err_mean",
    "abs_err_median",
    "speedups",
    "abs_speedup_err_mean",
    "abs_speedup_err_median",
    "itilp_capped",
    "itmlp_capped",
]

# columns printed left-aligned; the rest are numbers
TEXT_COLUMNS = {"gpu", "benchmark", "config", "row", "kernel", "model", "pico"}

ALL_BENCHMARKS = "all"


def benchmark_of(config: str) -> str:
    """benchmarks/<benchmark>/configs/<name>.toml -> <benchmark>."""
    parts = Path(config).parts
    if len(parts) >= 3 and parts[-2] == "configs":
        return parts[-3]
    return Path(config).stem


def records_of(config_dir: str, doc: dict, only_model: str | None) -> list[dict]:
    """The records of one ablation.json, with the GPU, benchmark and config added."""
    benchmark = benchmark_of(doc["config"])
    out = []
    for r in doc["records"]:
        if only_model is not None and r["model"] != only_model:
            continue
        out.append(
            {
                **r,
                "gpu": doc["gpu"],
                "benchmark": benchmark,
                "config": config_dir,
                "is_reference": r["kernel"] == doc["reference"],
            }
        )
    return out


def abs_values(records: list[dict], key: str) -> list[float]:
    return [abs(r[key]) for r in records if r.get(key) is not None]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(records: list[dict], row_order: list[str]) -> list[dict]:
    """One summary row per (benchmark, model, ablation row), plus the same over
    all benchmarks. Rows keep the order of the ablation's ROWS."""
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for r in records:
        for benchmark in (r["benchmark"], ALL_BENCHMARKS):
            groups.setdefault((r["gpu"], benchmark, r["model"], r["row"]), []).append(r)

    def order(key: tuple[str, str, str, str]) -> tuple:
        gpu, benchmark, model, row = key
        rank = row_order.index(row) if row in row_order else len(row_order)
        # every benchmark first, then all of them together
        return (gpu, benchmark == ALL_BENCHMARKS, benchmark, model, rank, row)

    summary = []
    for key in sorted(groups, key=order):
        gpu, benchmark, model, row = key
        group = groups[key]
        errors = abs_values(group, "error_pct")
        compared = [r for r in group if not r["is_reference"]]
        speedup_errors = abs_values(compared, "speedup_error_pct")
        summary.append(
            {
                "gpu": gpu,
                "benchmark": benchmark,
                "model": model,
                "row": row,
                "kernels": len(group),
                "abs_err_mean": mean(errors),
                "abs_err_median": median(errors),
                "speedups": len(speedup_errors),
                "abs_speedup_err_mean": mean(speedup_errors),
                "abs_speedup_err_median": median(speedup_errors),
                "itilp_capped": sum(1 for r in group if r.get("itilp_capped")),
                "itmlp_capped": sum(1 for r in group if r.get("itmlp_capped")),
            }
        )
    return summary


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 100 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def print_table(columns: list[str], rows: list[dict]) -> None:
    cells = [columns] + [[fmt(r.get(c)) for c in columns] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    for n, row in enumerate(cells):
        print(
            "  ".join(
                cell.ljust(w) if columns[i] in TEXT_COLUMNS else cell.rjust(w)
                for i, (cell, w) in enumerate(zip(row, widths))
            ).rstrip()
        )
        if n == 0:
            print("  ".join("-" * w for w in widths))


def load_batch(
    batch_dir: Path, only_model: str | None
) -> tuple[list[dict], list[str], list[str]]:
    """The records of every ablation.json in a batch, the configs that left none
    (as "<config>: <error>"), and the ablation rows in the order the batch ran them."""
    records, missing, row_order = [], [], []
    for config_dir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
        path = config_dir / "ablation.json"
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            # a failed config leaves no or a partial ablation.json
            missing.append(f"{config_dir.name}: {e}")
            continue
        for row in doc["rows"]:
            if row not in row_order:
                row_order.append(row)
        records.extend(records_of(config_dir.name, doc, only_model))
    return records, missing, row_order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batch_dir", type=Path, help="a batch_<timestamp> folder")
    parser.add_argument("--csv", action="store_true", help="write CSV to stdout")
    parser.add_argument(
        "--model",
        default="GASPerf",
        help="model to show, or 'all' (default: GASPerf)",
    )
    parser.add_argument(
        "--records",
        action="store_true",
        help="every record instead of the summary",
    )
    args = parser.parse_args()

    if not args.batch_dir.is_dir():
        print(f"error: {args.batch_dir} is not a directory", file=sys.stderr)
        return 1

    records, missing, row_order = load_batch(
        args.batch_dir, None if args.model == "all" else args.model
    )

    if args.records:
        columns, rows = RECORD_COLUMNS, records
    else:
        columns, rows = SUMMARY_COLUMNS, summarize(records, row_order)

    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    else:
        print_table(columns, rows)
    for m in missing:
        print(f"skipped {m}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
