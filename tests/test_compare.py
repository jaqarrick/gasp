"""Tests for gasp.compare — speedups and work-mismatch warnings."""

import pytest

from gasp.compare import (
    carveout_mismatches,
    measured_speedup,
    speedup,
    work_mismatches,
)
from gasp.predict import PredictionOutput
from gasp.profiler import SHARED_CONFIG_METRIC, KernelMeasurement, ProfileOutput


def profile(*cycles: float) -> ProfileOutput:
    return ProfileOutput(
        runs=[KernelMeasurement(duration_ns=1.0, cycles=c, metrics={}) for c in cycles]
    )


def test_speedup_above_one_means_faster_than_reference():
    assert speedup(200.0, 100.0) == 2.0
    assert speedup(100.0, 200.0) == 0.5


def test_measured_speedup_uses_medians():
    s = measured_speedup(profile(100, 110, 300), profile(50, 55, 60))

    assert s.median == 110 / 55


def test_measured_speedup_range_is_the_widest_possible():
    s = measured_speedup(profile(100, 110, 120), profile(50, 55, 60))

    assert s.low == pytest.approx(100 / 60)
    assert s.high == pytest.approx(120 / 50)


def test_no_warnings_when_work_matches():
    assert work_mismatches({"a": 1000, "b": 1000}, {"a": ["1000"], "b": ["1000"]}) == []


def test_warns_on_different_problem_sizes():
    warnings = work_mismatches({"a": 1000, "b": 2000}, {"a": [], "b": []})

    assert len(warnings) == 1
    assert "problem sizes" in warnings[0]
    assert "a=1000" in warnings[0]
    assert "b=2000" in warnings[0]


def test_warns_on_different_program_arguments():
    warnings = work_mismatches({"a": 1000, "b": 1000}, {"a": ["1"], "b": ["2"]})

    assert len(warnings) == 1
    assert "program arguments" in warnings[0]


def test_warns_on_both():
    assert len(work_mismatches({"a": 1, "b": 2}, {"a": ["1"], "b": ["2"]})) == 2


def test_warns_on_different_params():
    warnings = work_mismatches(
        {"a": 1, "b": 1},
        {"a": [], "b": []},
        {"a": {"M": 1024, "N": 1024}, "b": {"M": 512, "N": 1024}},
    )

    assert len(warnings) == 1
    assert "kernel params" in warnings[0]
    assert "M=512" in warnings[0]


def test_params_in_another_order_match():
    params = {"a": {"M": 1, "N": 2}, "b": {"N": 2, "M": 1}}
    assert work_mismatches({"a": 1, "b": 1}, {"a": [], "b": []}, params) == []


def test_single_kernel_has_no_mismatch():
    assert work_mismatches({"a": 1}, {"a": ["1"]}) == []


# measured carveout vs the one the L1 size assumed


def prediction(carveout: int) -> PredictionOutput:
    zero = {f: 0 for f in PredictionOutput.__dataclass_fields__}
    return PredictionOutput(
        **{
            **zero,
            "shared_mem_carveout_bytes": carveout,
            "l1_cache_size_bytes": 96 * 1024 - carveout,
        }
    )


def measured_carveout(carveout: float | None) -> ProfileOutput:
    metrics = {} if carveout is None else {SHARED_CONFIG_METRIC: carveout}
    return ProfileOutput(
        runs=[KernelMeasurement(duration_ns=1.0, cycles=1.0, metrics=metrics)]
    )


def test_carveout_match_has_no_warning():
    predictions = {"tiled": prediction(32768)}
    assert carveout_mismatches(predictions, {"tiled": measured_carveout(32768.0)}) == []


def test_carveout_mismatch_warns():
    predictions = {"tiled": prediction(32768)}
    (warning,) = carveout_mismatches(predictions, {"tiled": measured_carveout(65536.0)})
    assert "65,536 B" in warning and "32,768 B" in warning
    assert "(65,536 B)" in warning  # the L1 size the model used


def test_carveout_not_reported_is_skipped():
    predictions = {"tiled": prediction(32768)}
    assert carveout_mismatches(predictions, {"tiled": measured_carveout(None)}) == []
