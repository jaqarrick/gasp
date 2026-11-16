import shlex
from dataclasses import dataclass

from gasp.overrides import profiled_values
from gasp.predict import PredictionOutput
from gasp.profiler import WARPS_ACTIVE_PCT_METRIC, ProfileOutput


@dataclass
class MeasuredSpeedup:
    median: float
    low: float
    high: float


def speedup(t_reference: float, t_kernel: float) -> float:
    """Above 1 means the kernel is faster than the reference."""
    return t_reference / t_kernel


def relative_error_pct(predicted: float, measured: float) -> float | None:
    if measured == 0:
        return None
    return (predicted - measured) / measured * 100


def relative_error(predicted: float, measured: float) -> str:
    error = relative_error_pct(predicted, measured)
    return "n/a" if error is None else f"{error:+.1f}%"


def measured_speedup(
    reference: ProfileOutput, kernel: ProfileOutput
) -> MeasuredSpeedup:
    """Speedup of the medians. Runs aren't paired, so the range is the widest possible."""
    return MeasuredSpeedup(
        median=speedup(reference.cycles, kernel.cycles),
        low=speedup(reference.cycles_min, kernel.cycles_max),
        high=speedup(reference.cycles_max, kernel.cycles_min),
    )


def work_mismatches(
    problem_sizes: dict[str, int],
    program_args: dict[str, list[str]],
    params: dict[str, dict[str, int]] | None = None,
) -> list[str]:
    """Warnings for settings that suggest the kernels don't do the same work."""
    warnings = []
    if len(set(problem_sizes.values())) > 1:
        values = ", ".join(f"{name}={v}" for name, v in problem_sizes.items())
        warnings.append(
            f"Kernels may not do the same work, problem sizes differ: {values}"
        )
    if len({tuple(args) for args in program_args.values()}) > 1:
        values = ", ".join(
            f"{name}='{shlex.join(args)}'" for name, args in program_args.items()
        )
        warnings.append(
            f"Kernels may not do the same work, program arguments differ: {values}"
        )
    if params and len({tuple(sorted(p.items())) for p in params.values()}) > 1:
        values = "; ".join(
            f"{name}: {', '.join(f'{k}={v}' for k, v in p.items()) or 'none'}"
            for name, p in params.items()
        )
        warnings.append(
            f"Kernels may not do the same work, kernel params differ: {values}"
        )
    return warnings


def carveout_mismatches(
    predictions: dict[str, PredictionOutput], measured: dict[str, ProfileOutput]
) -> list[str]:
    """Kernels whose measured shared-memory carveout isn't the predicted one,
    so the L1 size the cache model used is wrong for them."""
    warnings = []
    for name, prediction in predictions.items():
        actual = measured[name].shared_mem_config_bytes
        predicted = prediction.shared_mem_carveout_bytes
        if actual is not None and actual != predicted:
            warnings.append(
                f"{name}: the driver picked a {actual:,} B shared-memory "
                f"carveout, the model assumed {predicted:,} B, so its L1 size "
                f"({prediction.l1_cache_size_bytes:,} B) is off"
            )
    return warnings


# (quantity, ModelInputs field, format): static model inputs ncu also measures
_INPUT_QUANTITIES: tuple[tuple[str, str, str], ...] = (
    ("L1 hit ratio", "l1_hit_ratio", ".3f"),
    ("L2 hit ratio (whole kernel, GPUPerf)", "l2_hit_ratio", ".3f"),
    ("Transactions / global access (ncu: loads)", "average_trans_warp", ".2f"),
    ("Bank conflicts / warp", "bank_conflict_count", ".2f"),
    ("Instructions / warp", "total_inst_count", ".1f"),
    ("Global mem instructions / warp", "mem_inst_count", ".1f"),
    ("SFU instructions / warp", "sfu_inst_count", ".1f"),
)


@dataclass
class InputComparison:
    key: str
    quantity: str
    predicted: float
    measured: float
    fmt: str


def _comparisons(
    prediction: PredictionOutput, metrics: dict[str, float]
) -> list[InputComparison]:
    static = prediction.static_inputs
    if static is None:
        return []
    measured = profiled_values(metrics, prediction.total_warps)
    found = []
    for quantity, field, fmt in _INPUT_QUANTITIES:
        if field in measured:
            found.append(
                InputComparison(
                    field, quantity, getattr(static, field), measured[field], fmt
                )
            )
        if field == "l2_hit_ratio" and "l2_hit_ratio" in measured:
            concurrent = static.l2_hit_ratio_concurrent
            if concurrent is not None:
                found.append(
                    InputComparison(
                        "l2_hit_ratio_concurrent",
                        "L2 hit ratio (concurrent, GASPerf)",
                        concurrent,
                        measured["l2_hit_ratio"],
                        fmt,
                    )
                )
    if WARPS_ACTIVE_PCT_METRIC in metrics:
        found.append(
            InputComparison(
                "occupancy",
                "Occupancy (theoretical vs achieved)",
                prediction.occupancy,
                metrics[WARPS_ACTIVE_PCT_METRIC] / 100,
                ".3f",
            )
        )
    return found


def input_comparisons(
    prediction: PredictionOutput, metrics: dict[str, float]
) -> list[list[str]]:
    return [
        [
            c.quantity,
            format(c.predicted, c.fmt),
            format(c.measured, c.fmt),
            format(c.predicted - c.measured, "+" + c.fmt),
            relative_error(c.predicted, c.measured),
        ]
        for c in _comparisons(prediction, metrics)
    ]


def input_comparison_values(
    prediction: PredictionOutput, metrics: dict[str, float]
) -> dict[str, dict[str, float | None]]:
    return {
        c.key: {
            "predicted": c.predicted,
            "measured": c.measured,
            "error_pct": relative_error_pct(c.predicted, c.measured),
        }
        for c in _comparisons(prediction, metrics)
    }
