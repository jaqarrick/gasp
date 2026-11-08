import shlex
from dataclasses import dataclass

from gasp.profiler import ProfileOutput


@dataclass
class MeasuredSpeedup:
    median: float
    low: float
    high: float


def speedup(t_reference: float, t_kernel: float) -> float:
    """Above 1 means the kernel is faster than the reference."""
    return t_reference / t_kernel


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
