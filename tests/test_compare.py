"""Tests for gasp.compare — speedups and work-mismatch warnings."""

import dataclasses

import pytest

from gasp.compare import (
    carveout_mismatches,
    input_comparison_values,
    input_comparisons,
    measured_speedup,
    relative_error,
    relative_error_pct,
    speedup,
    work_mismatches,
)
from gasp.models.inputs import ModelInputs
from gasp.overrides import (
    INSTS_EXECUTED,
    L1_HIT_RATE_PCT,
    L2_READ_HITS,
    L2_READ_MISSES,
)
from gasp.predict import PredictionOutput
from gasp.profiler import (
    SHARED_CONFIG_METRIC,
    WARPS_ACTIVE_PCT_METRIC,
    KernelMeasurement,
    ProfileOutput,
)


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


# static model inputs vs ncu's


def test_relative_error():
    assert relative_error(110.0, 100.0) == "+10.0%"
    assert relative_error(1.0, 0.0) == "n/a"


def test_relative_error_pct():
    assert relative_error_pct(110.0, 100.0) == pytest.approx(10.0)
    assert relative_error_pct(1.0, 0.0) is None


def static_prediction(concurrent: float | None = 0.9) -> PredictionOutput:
    inputs = ModelInputs(
        total_inst_count=10.0,
        mem_inst_count=2.0,
        sfu_inst_count=0.0,
        sync_inst_count=0.0,
        divergent_inst_count=0.0,
        ilp=1.0,
        mlp=1.0,
        average_inst_latency=4.0,
        average_trans_warp=1.0,
        bank_conflict_count=0.0,
        l1_hit_ratio=0.5,
        l2_hit_ratio=0.25,
        l2_hit_ratio_concurrent=concurrent,
    )
    return dataclasses.replace(
        prediction(0), static_inputs=inputs, total_warps=100, occupancy=1.0
    )


# cache and counts groups only: coalescing and bank are left out
NCU = {
    L1_HIT_RATE_PCT: 0.0,
    L2_READ_HITS: 50.0,
    L2_READ_MISSES: 50.0,
    INSTS_EXECUTED: 1250.0,
    "smsp__inst_executed_op_global_ld.sum": 200.0,
    "smsp__inst_executed_op_global_st.sum": 0.0,
    "smsp__inst_executed_pipe_xu.sum": 0.0,
    WARPS_ACTIVE_PCT_METRIC: 80.0,
}


def comparison_rows(p: PredictionOutput) -> dict[str, list[str]]:
    return {row[0]: row[1:] for row in input_comparisons(p, NCU)}


def test_input_comparisons_rows():
    rows = comparison_rows(static_prediction())

    assert list(rows) == [
        "L1 hit ratio",
        "L2 hit ratio (whole kernel, GPUPerf)",
        "L2 hit ratio (concurrent, GASPerf)",
        "Instructions / warp",
        "Global mem instructions / warp",
        "SFU instructions / warp",
        "Occupancy (theoretical vs achieved)",
    ]
    assert rows["L2 hit ratio (whole kernel, GPUPerf)"] == [
        "0.250",
        "0.500",
        "-0.250",
        "-50.0%",
    ]
    assert rows["Instructions / warp"] == ["10.0", "12.5", "-2.5", "-20.0%"]
    assert rows["Occupancy (theoretical vs achieved)"] == [
        "1.000",
        "0.800",
        "+0.200",
        "+25.0%",
    ]


def test_input_comparisons_measured_zero_has_no_relative_error():
    rows = comparison_rows(static_prediction())

    assert rows["L1 hit ratio"] == ["0.500", "0.000", "+0.500", "n/a"]


def test_input_comparisons_without_concurrent_l2():
    rows = comparison_rows(static_prediction(concurrent=None))

    assert "L2 hit ratio (concurrent, GASPerf)" not in rows


def test_input_comparisons_without_static_inputs():
    p = dataclasses.replace(prediction(0), static_inputs=None)

    assert input_comparisons(p, NCU) == []


def test_input_comparison_values_match_the_rows():
    p = static_prediction()
    values = input_comparison_values(p, NCU)

    assert list(values) == [
        "l1_hit_ratio",
        "l2_hit_ratio",
        "l2_hit_ratio_concurrent",
        "total_inst_count",
        "mem_inst_count",
        "sfu_inst_count",
        "occupancy",
    ]
    assert len(values) == len(input_comparisons(p, NCU))
    assert values["total_inst_count"] == {
        "predicted": 10.0,
        "measured": 12.5,
        "error_pct": pytest.approx(-20.0),
    }
    assert values["l1_hit_ratio"]["error_pct"] is None  # measured 0


def test_input_comparison_values_without_static_inputs():
    p = dataclasses.replace(prediction(0), static_inputs=None)

    assert input_comparison_values(p, NCU) == {}
