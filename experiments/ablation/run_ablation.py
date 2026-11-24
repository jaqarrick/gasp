"""Predict a config once per ablation row and compare each with saved measurements.

Needs no GPU: the measurements come from `gasp --run --save-metrics`.

    make ablation ARGS="--gpu rtx2080ti --config benchmarks/vecadd/configs/vecadd.toml \
        --metrics vecadd.metrics.json --out vecadd.ablation.json"
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from gasp import __version__
from gasp.cli import (
    KernelResult,
    load_hardware,
    load_saved_metrics,
    predict_all_models,
    prepare_kernel,
    reference_times,
)
from gasp.cli.config import ConfigError, load_config
from gasp.cli.utils import GPU_PROFILES
from gasp.compare import measured_speedup, relative_error_pct, speedup
from gasp.logger import Logger
from gasp.overrides import parse_overrides

# row name -> its --override values (handoff, "The ablation")
ROWS: dict[str, list[str]] = {
    "all-static": [],
    "crude": ["coalescing=crude", "bank=crude"],
    "profiled-cache": ["cache=profiled"],
    "profiled-coalescing": ["coalescing=profiled"],
    "profiled-bank": ["bank=profiled"],
    "profiled-counts": ["counts=profiled"],
    "all-profiled": ["all=profiled"],
}

# no ncu counter for these, so all-profiled keeps the static values
NOT_PROFILED = ["ilp", "mlp", "sync_inst_count"]

COLUMNS = [
    "row",
    "kernel",
    "model",
    "overrides",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", help="built-in GPU name")
    parser.add_argument("--profile", help="path to custom hardware profile JSON")
    parser.add_argument("--config", required=True, help="TOML file listing the kernels")
    parser.add_argument(
        "--metrics", required=True, help="metrics file from gasp --save-metrics"
    )
    parser.add_argument(
        "--reference", default=None, help="kernel speedups are relative to"
    )
    parser.add_argument("--out", default=None, help="write the results as JSON")
    parser.add_argument("--verbose", action="store_true")
    return parser


def row_records(
    row: str, results: list[KernelResult], measured: dict, reference: str
) -> list[dict]:
    # speedups are within one model
    t_reference = reference_times(results, reference)
    records = []
    for r in results:
        name = r.spec.name
        p = r.prediction
        t_gpu = p.t_gpu
        predicted = speedup(t_reference[r.model], t_gpu)
        actual = measured_speedup(measured[reference], measured[name]).median
        records.append(
            {
                "row": row,
                "kernel": name,
                "model": r.model,
                "overrides": r.overrides,
                "pico": "fallback" if r.pico_error else "ok",
                "predicted_cycles": t_gpu,
                "measured_cycles": measured[name].cycles,
                "error_pct": relative_error_pct(t_gpu, measured[name].cycles),
                "predicted_speedup": predicted,
                "measured_speedup": actual,
                "speedup_error_pct": relative_error_pct(predicted, actual),
                "itilp_capped": p.itilp_capped,
                "itmlp_capped": p.itmlp_capped,
                # ILP and MLP are never overridden, so the static ones were used
                "ilp": p.static_inputs.ilp if p.static_inputs else None,
                "mlp": p.static_inputs.mlp if p.static_inputs else None,
                "itilp": p.itilp,
                "itilp_max": p.itilp_max,
                "itmlp": p.itmlp,
                "mwp_peak_bw": p.mwp_peak_bw,
            }
        )
    return records


def git_commit() -> str:
    """Short commit of the checkout, with a mark if it has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit} (uncommitted changes)" if dirty else commit


def format_cell(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.gpu is None and args.profile is None:
        parser.error("give --gpu or --profile")

    logger = Logger(args.verbose)
    gpu, arch, hp = load_hardware(args.gpu, args.profile, logger)
    profile = args.profile or str(GPU_PROFILES[gpu])
    try:
        specs = load_config(args.config)
    except ConfigError as e:
        logger.error(str(e))
        sys.exit(1)
    names = [spec.name for spec in specs]
    reference = args.reference or names[0]
    if reference not in names:
        parser.error(f"--reference {reference} is not one of: {', '.join(names)}")

    saved = load_saved_metrics(args.metrics, gpu, arch, specs, logger)
    measured = {name: saved.kernels[name].profile for name in names}

    # the static analysis runs once per kernel, each model once per row
    prepared = [
        prepare_kernel(spec, arch, hp, warn_shared_program=False, logger=logger)
        for spec in specs
    ]
    records = []
    for row, values in ROWS.items():
        choices = parse_overrides(values)
        results = predict_all_models(prepared, choices, saved, logger)
        records += row_records(row, results, measured, reference)

    logger.table(
        f"Ablation: {args.config} on {gpu} (reference: {reference};"
        f" all-profiled keeps static {', '.join(NOT_PROFILED)})",
        COLUMNS,
        [[format_cell(record[c]) for c in COLUMNS] for record in records],
        force=True,
    )
    if args.out is not None:
        doc = {
            "gasp_version": __version__,
            "commit": git_commit(),
            "gpu": gpu,
            "arch": arch,
            "profile": profile,
            "config": args.config,
            "metrics": args.metrics,
            "reference": reference,
            "rows": ROWS,
            "not_profiled": NOT_PROFILED,
            "records": records,
        }
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        logger.success(f"Wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
