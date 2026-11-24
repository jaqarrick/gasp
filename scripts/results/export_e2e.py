#!/usr/bin/env python3
"""Write the end-to-end result CSVs (e2e_*.csv) for the report from the ablation batches.

Uses the all-static row of each batch, with each run and comparison counted once.
Errors come from ablation.json, terms and inputs from result.json; result_agrees is 0
where the two predictions differ. The files are listed in docs/experiments.md, section 8.

Example usage:
    scripts/results/export_e2e.py --out /path/to/data/dir
    scripts/results/export_e2e.py --out <dir> <batch_dir> ...
"""

import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

from batches import gpu_of, parse_args
from read_ablation import load_batch

TERMS = [
    "t_comp", "t_mem", "t_overlap", "t_exec", "issue_time", "smem_pipe_time",
    "fp64_pipe_time", "sfu_pipe_time", "o_cfdiv", "l1_time", "l2_time", "dram_time",
    "amat", "itilp", "itmlp", "cwp", "mwp",
]

INPUTS = [
    "l1_hit_ratio", "l2_hit_ratio", "l2_hit_ratio_concurrent", "average_trans_warp",
    "bank_conflict_count", "total_inst_count", "mem_inst_count", "sfu_inst_count",
    "fp64_inst_count", "occupancy",
]

KERNEL_COLUMNS = (
    ["benchmark", "config", "kernel", "model", "reference", "pico",
     "t_gpu_pred", "t_gpu_meas", "t_gpu_err_pct",
     "speedup_pred", "speedup_meas", "speedup_err_pct",
     "duplicate_run", "duplicate_speedup", "result_agrees"]
    + TERMS
    + [f"{name}_{kind}" for name in INPUTS for kind in ("static", "measured")]
)

METRIC_COLUMNS = [
    "kernels", "t_gpu_mdape", "t_gpu_mape", "t_gpu_mpe",
    "speedups", "speedup_mdape", "speedup_mape", "speedup_within_25",
    "rank_correct", "rank_pairs",
]

KERNEL_SUMMARY_COLUMNS = [
    "gpu", "model", "benchmark", "kernel", "runs", "t_gpu_err_min", "t_gpu_err_median",
    "t_gpu_err_max", "speedups", "speedup_mdape", "speedup_meas_median", "speedup_pred_median",
]

INPUT_COLUMNS = [
    "gpu", "benchmark", "input", "runs", "exact", "nonzero_measured",
    "mdape", "mape", "mpe", "zero_measured",
]

BANK_COLUMNS = [
    "gpu", "model", "config", "pred_added", "meas_added", "added_ratio",
    "t_gpu_err_min", "t_gpu_err_max", "count_static_d1", "count_meas_d1",
    "count_static_d32", "count_meas_d32", "fallbacks",
]


def fmt(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return "" if value is None else value


def write(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(row.get(k)) for k in columns})
    print(f"wrote {path} ({len(rows)} rows)")


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def runs(records: list[dict]) -> list[dict]:
    return [r for r in records if not r["duplicate_run"] and r.get("error_pct") is not None]


def speedups(records: list[dict]) -> list[dict]:
    return [r for r in records if not r["is_reference"] and not r["duplicate_speedup"]
            and r.get("speedup_error_pct") is not None]


def load_result(batch: Path, config: str) -> dict | None:
    try:
        return json.loads((batch / config / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def kernel_row(r: dict, doc: dict | None) -> dict:
    row = {
        "benchmark": r["benchmark"], "config": r["config"], "kernel": r["kernel"],
        "model": r["model"], "reference": r["is_reference"], "pico": r.get("pico"),
        "t_gpu_pred": r["predicted_cycles"], "t_gpu_meas": r["measured_cycles"],
        "t_gpu_err_pct": r["error_pct"], "speedup_pred": r.get("predicted_speedup"),
        "speedup_meas": r.get("measured_speedup"), "speedup_err_pct": r.get("speedup_error_pct"),
        "duplicate_run": r["duplicate_run"], "duplicate_speedup": r["duplicate_speedup"],
    }
    if doc is None:
        return row
    pred = doc["prediction"]["kernels"].get(r["kernel"], {}).get("models", {}).get(r["model"], {})
    meas = doc.get("result", {}).get("kernels", {}).get(r["kernel"], {})
    if "t_gpu" in pred:
        row["result_agrees"] = (
            abs(pred["t_gpu"] - r["predicted_cycles"]) <= 1e-6 * abs(r["predicted_cycles"])
            and meas.get("measured", {}).get("cycles") == r["measured_cycles"]
        )
    for term in TERMS:
        row[term] = pred.get(term)
    inputs = meas.get("inputs", {})
    for name in INPUTS:
        value = inputs.get(name) or {}
        row[f"{name}_static"] = value.get("predicted")
        row[f"{name}_measured"] = value.get("measured")
    return row


def metrics(records: list[dict]) -> dict:
    """The METRIC_COLUMNS of one model's records."""
    errors = [r["error_pct"] for r in runs(records)]
    absolute = [abs(e) for e in errors]
    speedup = [abs(r["speedup_error_pct"]) for r in speedups(records)]
    by_config: dict[str, list[dict]] = {}
    for r in records:
        by_config.setdefault(r["config"], []).append(r)
    seen: set[frozenset] = set()
    correct = pairs = 0
    for config in sorted(by_config):
        for a, b in itertools.combinations(by_config[config], 2):
            pair = frozenset((a["run_key"], b["run_key"]))
            if a["run_key"] is not None and b["run_key"] is not None:
                if pair in seen:
                    continue
                seen.add(pair)
            pairs += 1
            correct += sign(a["measured_cycles"] - b["measured_cycles"]) == sign(
                a["predicted_cycles"] - b["predicted_cycles"])
    return {
        "kernels": len(errors), "t_gpu_mdape": median(absolute), "t_gpu_mape": mean(absolute),
        "t_gpu_mpe": mean(errors), "speedups": len(speedup), "speedup_mdape": median(speedup),
        "speedup_mape": mean(speedup), "speedup_within_25": sum(e <= 25 for e in speedup),
        "rank_correct": correct, "rank_pairs": pairs,
    }


def kernel_summary(gpu: str, model: str, benchmark: str, kernel: str, records: list[dict]) -> dict:
    errors = [r["error_pct"] for r in runs(records)]
    compared = speedups(records)
    return {
        "gpu": gpu, "model": model, "benchmark": benchmark, "kernel": kernel,
        "runs": len(errors), "t_gpu_err_min": min(errors, default=None),
        "t_gpu_err_median": median(errors), "t_gpu_err_max": max(errors, default=None),
        "speedups": len(compared),
        "speedup_mdape": median([abs(r["speedup_error_pct"]) for r in compared]),
        "speedup_meas_median": median([r["measured_speedup"] for r in compared]),
        "speedup_pred_median": median([r["predicted_speedup"] for r in compared]),
    }


def input_summary(gpu: str, benchmark: str, name: str, rows: list[dict]) -> dict:
    pairs = [(row[f"{name}_static"], row[f"{name}_measured"]) for row in rows]
    pairs = [(s, m) for s, m in pairs if s is not None and m is not None]
    exact = sum(abs(s - m) <= 1e-9 * max(1.0, abs(m)) for s, m in pairs)
    errors = [(s - m) / m * 100 for s, m in pairs if m != 0]
    absolute = [abs(e) for e in errors]
    return {
        "gpu": gpu, "benchmark": benchmark, "input": name, "runs": len(pairs), "exact": exact,
        "nonzero_measured": len(errors), "mdape": median(absolute), "mape": mean(absolute),
        "mpe": mean(errors), "zero_measured": sum(m == 0 and s != 0 for s, m in pairs),
    }


def bank_cost(gpu: str, model: str, config: str, rows: list[dict]) -> dict | None:
    by_kernel = {r["kernel"]: r for r in rows}
    d1, d32 = by_kernel.get("degree_1"), by_kernel.get("degree_32")
    if d1 is None or d32 is None:
        return None
    pred = d32["t_gpu_pred"] - d1["t_gpu_pred"]
    meas = d32["t_gpu_meas"] - d1["t_gpu_meas"]
    errors = [r["t_gpu_err_pct"] for r in rows]
    return {
        "gpu": gpu, "model": model, "config": config, "pred_added": pred, "meas_added": meas,
        "added_ratio": pred / meas if meas else None,
        "t_gpu_err_min": min(errors), "t_gpu_err_max": max(errors),
        "count_static_d1": d1.get("bank_conflict_count_static"),
        "count_meas_d1": d1.get("bank_conflict_count_measured"),
        "count_static_d32": d32.get("bank_conflict_count_static"),
        "count_meas_d32": d32.get("bank_conflict_count_measured"),
        "fallbacks": sum(r["pico"] == "fallback" for r in rows),
    }


def main() -> int:
    args = parse_args(__doc__.splitlines()[0], batches=True)
    summary, benchmarks, kernels, bank, inputs = [], [], [], [], []
    for batch in args.batch_dirs:
        gpu = gpu_of(batch)
        records, missing, _ = load_batch(batch, None)
        if missing:
            print(f"{gpu}: missing {missing}", file=sys.stderr)
        records = [r for r in records if r["row"] == "all-static"]
        docs = {c: load_result(batch, c) for c in {r["config"] for r in records}}
        rows = [kernel_row(r, docs[r["config"]]) for r in records]
        write(args.out / f"e2e_{gpu}.csv", KERNEL_COLUMNS, rows)
        disagree = sum(row.get("result_agrees") is False for row in rows)
        duplicate = sum(r["duplicate_run"] for r in records)
        print(f"  {disagree} rows differ from result.json, {duplicate} duplicate runs")
        distinct = [row for row in rows if row["model"] == "GASPerf" and not row["duplicate_run"]]
        for benchmark in ["all"] + sorted({row["benchmark"] for row in distinct}):
            group = [row for row in distinct if benchmark in ("all", row["benchmark"])]
            for name in INPUTS:
                inputs.append(input_summary(gpu, benchmark, name, group))
        for model in sorted({r["model"] for r in records}):
            mine = [r for r in records if r["model"] == model]
            summary.append({"gpu": gpu, "model": model, **metrics(mine)})
            for benchmark in sorted({r["benchmark"] for r in mine}):
                group = [r for r in mine if r["benchmark"] == benchmark]
                benchmarks.append({"gpu": gpu, "model": model, "benchmark": benchmark, **metrics(group)})
                for kernel in sorted({r["kernel"] for r in group}):
                    kernels.append(kernel_summary(gpu, model, benchmark, kernel,
                                                  [r for r in group if r["kernel"] == kernel]))
            for config in sorted({r["config"] for r in rows if r["benchmark"] == "bank_conflict"}):
                cost = bank_cost(gpu, model, config,
                                 [r for r in rows if r["config"] == config and r["model"] == model])
                if cost is not None:
                    bank.append(cost)
    write(args.out / "e2e_summary.csv", ["gpu", "model"] + METRIC_COLUMNS, summary)
    write(args.out / "e2e_benchmarks.csv", ["gpu", "model", "benchmark"] + METRIC_COLUMNS, benchmarks)
    write(args.out / "e2e_kernels.csv", KERNEL_SUMMARY_COLUMNS, kernels)
    write(args.out / "e2e_bank_cost.csv", BANK_COLUMNS, bank)
    write(args.out / "e2e_inputs.csv", INPUT_COLUMNS, inputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
