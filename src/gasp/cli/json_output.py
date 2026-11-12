"""The --json document: the prediction for every kernel and model, and with a
measurement (--run or --metrics) the measured values beside it."""

import dataclasses
import statistics
from typing import Any

from gasp.compare import (
    input_comparison_values,
    measured_speedup,
    relative_error_pct,
    speedup,
)
from gasp.metrics_file import median_metrics
from gasp.profiler import ProfileOutput

from .results import KernelResult, reference_times, unique_kernels

# PredictionOutput fields from modules before the model
_KERNEL_FIELDS = ("static_inputs", "total_warps", "occupancy", "model")


def _model_json(r: KernelResult, t_reference: float) -> dict[str, Any]:
    values = dataclasses.asdict(r.prediction)
    for field in _KERNEL_FIELDS:
        values.pop(field)
    values["speedup"] = speedup(t_reference, r.prediction.t_gpu)
    return values


def prediction_json(results: list[KernelResult], reference: str) -> dict[str, Any]:
    t_reference = reference_times(results, reference)
    kernels: dict[str, Any] = {}
    for k in unique_kernels(results):
        spec, p = k.spec, k.prediction
        kernels[spec.name] = {
            "file": spec.file,
            "kernel": k.kernel_name,
            "grid_dim": list(spec.grid_dim),
            "block_dim": list(spec.block_dim),
            "problem_size": spec.problem_size,
            "footprint": spec.footprint,
            "reuse": spec.reuse,
            "loop_iterations": spec.loop_iterations,
            "loop_counts": spec.loop_counts,
            "params": spec.params,
            "pico": "fallback" if k.pico_error else "ok",
            "occupancy": p.occupancy,
            "total_warps": p.total_warps,
            "inputs": (
                dataclasses.asdict(p.static_inputs) if p.static_inputs else None
            ),
            "models": {
                r.model: _model_json(r, t_reference[r.model])
                for r in results
                if r.spec.name == spec.name
            },
        }
    return {"kernels": kernels}


def _measured_json(m: ProfileOutput) -> dict[str, Any]:
    return {
        "cycles": m.cycles,
        "cycles_min": m.cycles_min,
        "cycles_max": m.cycles_max,
        "duration_ns": statistics.median(run.duration_ns for run in m.runs),
        "shared_mem_config_bytes": m.shared_mem_config_bytes,
        "runs": [
            {"cycles": run.cycles, "duration_ns": run.duration_ns} for run in m.runs
        ],
        "metrics": median_metrics(m),
    }


def _speedup_json(
    r: KernelResult,
    t_reference: float,
    reference: ProfileOutput,
    m: ProfileOutput,
    is_reference: bool,
) -> dict[str, Any]:
    if is_reference:
        # exactly 1, as report_measured shows it
        predicted = measured = low = high = 1.0
    else:
        predicted = speedup(t_reference, r.prediction.t_gpu)
        s = measured_speedup(reference, m)
        measured, low, high = s.median, s.low, s.high
    return {
        "predicted": predicted,
        "measured": measured,
        "measured_low": low,
        "measured_high": high,
        "error_pct": relative_error_pct(predicted, measured),
    }


def result_json(
    results: list[KernelResult],
    measured: dict[str, ProfileOutput],
    reference: str,
    repeats: int,
) -> dict[str, Any]:
    t_reference = reference_times(results, reference)
    kernels: dict[str, Any] = {}
    for k in unique_kernels(results):
        name = k.spec.name
        m = measured[name]
        models = {}
        for r in results:
            if r.spec.name != name:
                continue
            t_gpu = r.prediction.t_gpu
            models[r.model] = {
                "t_gpu": {
                    "predicted": t_gpu,
                    "measured": m.cycles,
                    "error_pct": relative_error_pct(t_gpu, m.cycles),
                },
                "speedup": _speedup_json(
                    r, t_reference[r.model], measured[reference], m, name == reference
                ),
            }
        kernels[name] = {
            "measured": _measured_json(m),
            # static inputs come before the model, so any one model's will do
            "inputs": input_comparison_values(k.prediction, median_metrics(m)),
            "models": models,
        }
    return {"repeats": repeats, "kernels": kernels}


def build_json(
    gpu: str,
    arch: str,
    reference: str,
    overrides: str,
    results: list[KernelResult],
    measured: dict[str, ProfileOutput] | None,
    repeats: int | None,
) -> dict[str, Any]:
    return {
        "gpu": gpu,
        "arch": arch,
        "reference": reference,
        "overrides": overrides,
        "prediction": prediction_json(results, reference),
        "result": (
            None
            if measured is None or repeats is None
            else result_json(results, measured, reference, repeats)
        ),
    }
