from dataclasses import dataclass

from gasp.models import DEFAULT_MODEL
from gasp.predict import PredictionOutput

from .config import KernelSpec


@dataclass
class KernelResult:
    spec: KernelSpec
    kernel_name: str  # mangled
    prediction: PredictionOutput
    pico_error: bool
    overrides: str = "none"
    model: str = DEFAULT_MODEL


def reference_times(results: list[KernelResult], reference: str) -> dict[str, float]:
    """The reference kernel's T_GPU under each model. Only comparing within a model never across"""
    return {r.model: r.prediction.t_gpu for r in results if r.spec.name == reference}


def unique_kernels(results: list[KernelResult]) -> list[KernelResult]:
    """One result per kernel"""
    seen: dict[str, KernelResult] = {}
    for r in results:
        seen.setdefault(r.spec.name, r)
    return list(seen.values())
