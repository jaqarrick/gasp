"""The ablation: predict every kernel of a config once per row of ROWS, each row
a different set of overrides, and compare each with the saved measurements.

    make ablation ARGS="--gpu rtx2080ti --config benchmarks/vecadd/vecadd.toml \\
        --metrics vecadd.metrics.json --out vecadd.ablation.csv"

Needs no GPU: the measurements come from a metrics file written earlier by
`gasp --run --save-metrics`. See README.md here."""

import argparse
import csv
import sys

from gasp.cli import (
    KernelResult,
    load_hardware,
    load_saved_metrics,
    predict_kernel,
    prepare_kernel,
)
from gasp.cli.config import ConfigError, load_config
from gasp.compare import measured_speedup, speedup
from gasp.logger import Logger
from gasp.model_opts import NAMES as MODEL_OPT_NAMES
from gasp.model_opts import ModelOptsError, parse_model_opts
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

COLUMNS = [
    "row",
    "kernel",
    "overrides",
    "model_opts",
    "pico",
    "predicted_cycles",
    "measured_cycles",
    "error_pct",
    "predicted_speedup",
    "measured_speedup",
    "speedup_error_pct",
]


def error_pct(predicted: float, measured: float) -> float | None:
    return None if measured == 0 else (predicted - measured) / measured * 100


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
    parser.add_argument(
        "--model-opts",
        action="append",
        default=[],
        metavar="NAME[,NAME...]",
        help="model options every row is predicted with (repeatable): "
        f"{', '.join(MODEL_OPT_NAMES)}, or all",
    )
    parser.add_argument("--out", default=None, help="write the results as CSV")
    parser.add_argument("--verbose", action="store_true")
    return parser


def row_records(
    row: str, results: list[KernelResult], measured: dict, reference: str
) -> list[dict]:
    t_reference = next(r.prediction.t_gpu for r in results if r.spec.name == reference)
    records = []
    for r in results:
        name = r.spec.name
        t_gpu = r.prediction.t_gpu
        predicted = speedup(t_reference, t_gpu)
        actual = measured_speedup(measured[reference], measured[name]).median
        records.append(
            {
                "row": row,
                "kernel": name,
                "overrides": r.overrides,
                "model_opts": r.model_opts,
                "pico": "fallback" if r.pico_error else "ok",
                "predicted_cycles": t_gpu,
                "measured_cycles": measured[name].cycles,
                "error_pct": error_pct(t_gpu, measured[name].cycles),
                "predicted_speedup": predicted,
                "measured_speedup": actual,
                "speedup_error_pct": error_pct(predicted, actual),
            }
        )
    return records


def format_cell(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.gpu is None and args.profile is None:
        parser.error("give --gpu or --profile")
    try:
        options = parse_model_opts(args.model_opts)
    except ModelOptsError as e:
        parser.error(f"--model-opts: {e}")

    logger = Logger(args.verbose)
    gpu, arch, hp = load_hardware(args.gpu, args.profile, logger)
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

    # the static analysis runs once per kernel, the model once per row
    prepared = [
        prepare_kernel(spec, arch, hp, warn_shared_program=False, logger=logger)
        for spec in specs
    ]
    records = []
    for row, values in ROWS.items():
        choices = parse_overrides(values)
        results = [
            predict_kernel(p, choices, saved, logger, options) for p in prepared
        ]
        records += row_records(row, results, measured, reference)

    logger.table(
        f"Ablation: {args.config} on {gpu} (reference: {reference})",
        COLUMNS,
        [[format_cell(record[c]) for c in COLUMNS] for record in records],
        force=True,
    )
    if args.out is not None:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(records)
        logger.success(f"Wrote {len(records)} rows to {args.out}")


if __name__ == "__main__":
    main()
