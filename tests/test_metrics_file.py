"""Tests for gasp.metrics_file — saving and replaying ncu measurements."""

import json
from types import SimpleNamespace

import pytest

from gasp.metrics_file import (
    MetricsFileError,
    SavedKernel,
    SavedMetrics,
    check_matches,
    load_metrics,
    median_metrics,
    save_metrics,
)
from gasp.profiler import KernelMeasurement, ProfileOutput


def profile(*runs: dict[str, float]) -> ProfileOutput:
    return ProfileOutput(
        runs=[
            KernelMeasurement(duration_ns=10.0 * n, cycles=100.0 * n, metrics=m)
            for n, m in enumerate(runs, start=1)
        ]
    )


def saved() -> SavedMetrics:
    return SavedMetrics(
        gpu="rtx2080ti",
        arch="sm_75",
        kernels={
            "good": SavedKernel("_Zgood", 256, 3907, profile({"a.sum": 1.0})),
            "poor": SavedKernel("_Zpoor", 256, 3907, profile({"a.sum": 2.0})),
        },
    )


def spec(name: str, block_size: int = 256, grid_size: int = 3907):
    return SimpleNamespace(name=name, block_size=block_size, grid_size=grid_size)


def test_round_trip(tmp_path):
    path = str(tmp_path / "m.json")

    save_metrics(path, saved())

    assert load_metrics(path) == saved()


def test_file_keeps_raw_runs(tmp_path):
    path = tmp_path / "m.json"

    save_metrics(str(path), saved())
    data = json.loads(path.read_text())

    assert data["gpu"] == "rtx2080ti"
    assert data["kernels"]["good"]["runs"] == [
        {"duration_ns": 10.0, "cycles": 100.0, "metrics": {"a.sum": 1.0}}
    ]


def test_load_missing_file(tmp_path):
    with pytest.raises(MetricsFileError, match="cannot read"):
        load_metrics(str(tmp_path / "none.json"))


def test_load_invalid_json(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{not json")

    with pytest.raises(MetricsFileError, match="invalid JSON"):
        load_metrics(str(path))


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        '{"gpu": "a100"}',
        '{"gpu": "a", "arch": "b", "kernels": {"k": {"runs": []}}}',
    ],
)
def test_load_wrong_format(tmp_path, content):
    path = tmp_path / "m.json"
    path.write_text(content)

    with pytest.raises(MetricsFileError, match="format"):
        load_metrics(str(path))


def test_load_kernel_without_runs(tmp_path):
    path = tmp_path / "m.json"
    empty = SavedMetrics("a100", "sm_80", {"k": SavedKernel("_Zk", 256, 4, profile())})
    save_metrics(str(path), empty)

    with pytest.raises(MetricsFileError, match="no runs"):
        load_metrics(str(path))


def test_check_matches_passes():
    check_matches(saved(), "rtx2080ti", "sm_75", [spec("good"), spec("poor")])


@pytest.mark.parametrize(
    "gpu,arch,kernel,message",
    [
        ("a100", "sm_80", spec("good"), "profiled on rtx2080ti"),
        ("rtx2080ti", "sm_80", spec("good"), "profiled on rtx2080ti"),
        ("rtx2080ti", "sm_75", spec("other"), "no kernel named other"),
        ("rtx2080ti", "sm_75", spec("good", block_size=128), "<<<3907, 128>>>"),
        ("rtx2080ti", "sm_75", spec("good", grid_size=8), "<<<8, 256>>>"),
    ],
)
def test_check_matches_rejects(gpu, arch, kernel, message):
    with pytest.raises(MetricsFileError, match=message):
        check_matches(saved(), gpu, arch, [kernel])


def test_median_metrics_across_runs():
    p = profile({"a.sum": 5.0}, {"a.sum": 1.0}, {"a.sum": 3.0})

    assert median_metrics(p) == {"a.sum": 3.0}


def test_median_metrics_drops_a_metric_some_run_lacks():
    p = profile({"a.sum": 1.0, "b.pct": 50.0}, {"a.sum": 3.0})

    assert median_metrics(p) == {"a.sum": 2.0}
