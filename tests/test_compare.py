"""Tests for gasp.compare — speedups and work-mismatch warnings."""

import pytest

from gasp.compare import measured_speedup, speedup, work_mismatches
from gasp.profiler import KernelMeasurement, ProfileOutput


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
