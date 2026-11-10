"""The metrics file: the ncu measurements of a --run, saved as JSON so they can
be replayed without a GPU (--metrics) and feed profiled overrides.

Only raw ncu values are stored. Model inputs are worked out from them on load
(gasp.overrides), so changing a conversion doesn't need a new profile."""

import json
import statistics
from dataclasses import asdict, dataclass
from typing import Protocol

from gasp import __version__
from gasp.profiler import KernelMeasurement, ProfileOutput


class MetricsFileError(ValueError):
    pass


@dataclass
class SavedKernel:
    kernel_name: str  # mangled
    block_size: int
    grid_size: int
    profile: ProfileOutput


@dataclass
class SavedMetrics:
    gpu: str
    arch: str
    kernels: dict[str, SavedKernel]  # by spec name


class Launched(Protocol):
    """What check_matches reads from a kernel spec."""

    name: str
    block_size: int
    grid_size: int


def save_metrics(path: str, saved: SavedMetrics) -> None:
    data = {
        "gasp_version": __version__,
        "gpu": saved.gpu,
        "arch": saved.arch,
        "kernels": {
            name: {
                "kernel_name": k.kernel_name,
                "block_size": k.block_size,
                "grid_size": k.grid_size,
                "runs": [asdict(run) for run in k.profile.runs],
            }
            for name, k in saved.kernels.items()
        },
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    except OSError as e:
        raise MetricsFileError(f"cannot write metrics file {path}: {e.strerror}") from e


def load_metrics(path: str) -> SavedMetrics:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise MetricsFileError(f"cannot read metrics file {path}: {e.strerror}") from e
    except json.JSONDecodeError as e:
        raise MetricsFileError(f"invalid JSON in metrics file {path}: {e}") from e

    try:
        kernels = {
            name: SavedKernel(
                kernel_name=k["kernel_name"],
                block_size=k["block_size"],
                grid_size=k["grid_size"],
                profile=ProfileOutput(
                    runs=[
                        KernelMeasurement(
                            duration_ns=run["duration_ns"],
                            cycles=run["cycles"],
                            metrics=dict(run["metrics"]),
                        )
                        for run in k["runs"]
                    ]
                ),
            )
            for name, k in data["kernels"].items()
        }
        saved = SavedMetrics(gpu=data["gpu"], arch=data["arch"], kernels=kernels)
    except (KeyError, TypeError, AttributeError) as e:
        raise MetricsFileError(
            f"metrics file {path} is not in the format --save-metrics writes "
            f"({type(e).__name__}: {e})"
        ) from e

    empty = [name for name, k in saved.kernels.items() if not k.profile.runs]
    if empty:
        raise MetricsFileError(
            f"metrics file {path} has no runs for: {', '.join(empty)}"
        )
    return saved


def check_matches(
    saved: SavedMetrics, gpu: str, arch: str, specs: list[Launched]
) -> None:
    """The file must be a profile of these kernels, on this GPU, with these
    launches: profiled values are per warp of the launch that was measured."""
    if (saved.gpu, saved.arch) != (gpu, arch):
        raise MetricsFileError(
            f"the metrics file was profiled on {saved.gpu} ({saved.arch}), "
            f"this run is for {gpu} ({arch})"
        )
    for spec in specs:
        kernel = saved.kernels.get(spec.name)
        if kernel is None:
            raise MetricsFileError(
                f"the metrics file has no kernel named {spec.name} "
                f"(it has: {', '.join(saved.kernels) or 'none'})"
            )
        launched = (kernel.grid_size, kernel.block_size)
        if launched != (spec.grid_size, spec.block_size):
            raise MetricsFileError(
                f"{spec.name}: the metrics file was profiled with "
                f"<<<{kernel.grid_size}, {kernel.block_size}>>>, this run uses "
                f"<<<{spec.grid_size}, {spec.block_size}>>>"
            )


def median_metrics(profile: ProfileOutput) -> dict[str, float]:
    """One value per metric: the median across runs, for metrics every run has."""
    names = set(profile.runs[0].metrics)
    for run in profile.runs[1:]:
        names &= set(run.metrics)
    return {
        name: statistics.median(run.metrics[name] for run in profile.runs)
        for name in sorted(names)
    }
