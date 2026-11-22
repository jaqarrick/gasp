"""Tests for the ablation: the models' cap flags, run_ablation's records and
read_ablation's summary."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gasp.predict import predict
from gasp.profiler import KernelMeasurement, ProfileOutput
from tests.test_predict import prediction_input

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments" / "ablation")
)
import read_ablation
import run_ablation


def with_ilp_mlp(model: str, ilp: float, mlp: float):
    p = prediction_input()
    p.model = model
    p.dependency = replace(p.dependency, ilp=ilp, mlp=mlp)
    return predict(p, logger=MagicMock())


@pytest.mark.parametrize("model", ["GPUPerf", "GASPerf"])
def test_itilp_capped_when_ilp_times_n_exceeds_max(model):
    high = with_ilp_mlp(model, ilp=100.0, mlp=1.0)
    assert high.itilp_capped
    assert high.itilp == pytest.approx(high.itilp_max)

    low = with_ilp_mlp(model, ilp=0.01, mlp=1.0)
    assert not low.itilp_capped
    assert low.itilp < low.itilp_max


def test_gpuperf_itmlp_capped_by_peak_bandwidth():
    out = with_ilp_mlp("GPUPerf", ilp=1.0, mlp=1000.0)
    assert out.itmlp_capped
    assert out.itmlp == pytest.approx(out.mwp_peak_bw)


def test_gasperf_itmlp_never_capped():
    assert not with_ilp_mlp("GASPerf", ilp=1.0, mlp=1000.0).itmlp_capped


def measured(cycles: float) -> ProfileOutput:
    return ProfileOutput(
        runs=[KernelMeasurement(duration_ns=0, cycles=cycles, metrics={})]
    )


def kernel_result(name: str, model: str, t_gpu: float, capped: bool):
    prediction = SimpleNamespace(
        t_gpu=t_gpu,
        itilp=3.0,
        itilp_max=3.0 if capped else 5.0,
        itilp_capped=capped,
        itmlp=20.0,
        mwp_peak_bw=20.0,
        itmlp_capped=False,
        static_inputs=SimpleNamespace(ilp=2.0, mlp=1.5),
    )
    return SimpleNamespace(
        spec=SimpleNamespace(name=name),
        model=model,
        prediction=prediction,
        overrides="none",
        pico_error=False,
    )


def test_row_records_errors_speedups_and_caps():
    results = [
        kernel_result("good", "GASPerf", 110.0, capped=True),
        kernel_result("poor", "GASPerf", 220.0, capped=False),
    ]
    records = run_ablation.row_records(
        "all-static",
        results,
        {"good": measured(100.0), "poor": measured(250.0)},
        "good",
    )
    good, poor = records
    assert good["error_pct"] == pytest.approx(10.0)
    assert good["itilp_capped"] is True
    assert good["ilp"] == 2.0 and good["mlp"] == 1.5
    assert good["itilp_max"] == 3.0
    assert poor["error_pct"] == pytest.approx(-12.0)
    assert poor["itilp_capped"] is False
    # speedup of poor over good: predicted 0.5, measured 0.4
    assert poor["predicted_speedup"] == pytest.approx(0.5)
    assert poor["measured_speedup"] == pytest.approx(0.4)
    assert poor["speedup_error_pct"] == pytest.approx(25.0)
    assert set(run_ablation.COLUMNS) <= set(good)


def test_benchmark_of():
    assert (
        read_ablation.benchmark_of("benchmarks/bank_conflict/configs/x.toml")
        == "bank_conflict"
    )
    assert read_ablation.benchmark_of("other.toml") == "other"


def record(row, kernel, model, err, speedup_err, capped=False):
    return {
        "row": row,
        "kernel": kernel,
        "model": model,
        "error_pct": err,
        "speedup_error_pct": speedup_err,
        "itilp_capped": capped,
        "itmlp_capped": False,
    }


def write_batch(tmp_path: Path) -> Path:
    batch = tmp_path / "batch"
    docs = {
        "vecadd_a": (
            "benchmarks/vecadd/configs/a.toml",
            [
                record("all-static", "good", "GASPerf", 10.0, 0.0, capped=True),
                record("all-static", "poor", "GASPerf", -20.0, -5.0),
                record("all-static", "good", "GPUPerf", 1.0, 0.0),
                record("crude", "good", "GASPerf", 30.0, 0.0),
                record("crude", "poor", "GASPerf", 40.0, 10.0),
            ],
        ),
        "mm_b": (
            "benchmarks/mm/configs/b.toml",
            [
                record("all-static", "good", "GASPerf", 6.0, 0.0),
                record("all-static", "poor", "GASPerf", -2.0, 15.0, capped=True),
            ],
        ),
    }
    for name, (config, records) in docs.items():
        (batch / name).mkdir(parents=True)
        doc = {
            "gpu": "rtx2080ti",
            "config": config,
            "reference": "good",
            "rows": {"all-static": [], "crude": ["coalescing=crude", "bank=crude"]},
            "records": records,
        }
        (batch / name / "ablation.json").write_text(json.dumps(doc))
    (batch / "failed_config").mkdir()
    return batch


def load(batch: Path, model: str | None) -> list[dict]:
    records = []
    for d in sorted(p for p in batch.iterdir() if (p / "ablation.json").exists()):
        doc = json.loads((d / "ablation.json").read_text())
        records += read_ablation.records_of(d.name, doc, model)
    return records


def test_summary_per_benchmark_and_all(tmp_path):
    records = load(write_batch(tmp_path), "GASPerf")
    summary = read_ablation.summarize(records, ["all-static", "crude"])
    by_key = {(s["benchmark"], s["row"]): s for s in summary}

    vecadd = by_key[("vecadd", "all-static")]
    assert vecadd["kernels"] == 2
    assert vecadd["abs_err_mean"] == pytest.approx(15.0)
    # the reference kernel is left out of the speedup columns
    assert vecadd["speedups"] == 1
    assert vecadd["abs_speedup_err_mean"] == pytest.approx(5.0)
    assert vecadd["itilp_capped"] == 1

    everything = by_key[("all", "all-static")]
    assert everything["kernels"] == 4
    assert everything["abs_err_mean"] == pytest.approx((10 + 20 + 6 + 2) / 4)
    assert everything["abs_err_median"] == pytest.approx(8.0)
    assert everything["abs_speedup_err_median"] == pytest.approx(10.0)
    assert everything["itilp_capped"] == 2

    # benchmarks first, then all; rows in ROWS order within each
    assert [(s["benchmark"], s["row"]) for s in summary] == [
        ("mm", "all-static"),
        ("vecadd", "all-static"),
        ("vecadd", "crude"),
        ("all", "all-static"),
        ("all", "crude"),
    ]


def test_model_filter(tmp_path):
    batch = write_batch(tmp_path)
    assert {r["model"] for r in load(batch, "GPUPerf")} == {"GPUPerf"}
    assert {r["model"] for r in load(batch, None)} == {"GASPerf", "GPUPerf"}


def test_main_reports_missing_config(tmp_path, monkeypatch, capsys):
    batch = write_batch(tmp_path)
    monkeypatch.setattr(sys, "argv", ["read_ablation.py", "--csv", str(batch)])
    assert read_ablation.main() == 1
    out = capsys.readouterr()
    assert out.out.splitlines()[0] == ",".join(read_ablation.SUMMARY_COLUMNS)
    assert "skipped failed_config" in out.err


def kernel_settings(file: str) -> dict:
    return {
        "file": file,
        "kernel": file,
        "grid_dim": [64, 64],
        "block_dim": [16, 16],
        "problem_size": 1024,
        "footprint": 1,
        "reuse": None,
        "loop_iterations": None,
        "loop_counts": None,
        "params": {"n": 1024},
    }


def write_duplicate_batch(tmp_path: Path) -> Path:
    batch = tmp_path / "batch"
    configs = {
        "mm_a": [
            ("basic", 10.0, 0.0),
            ("tiled", 20.0, 5.0),
            ("uncoalesced", 30.0, 7.0),
        ],
        "mm_b": [("basic", 12.0, 0.0), ("tiled", 40.0, 9.0)],
    }
    for name, kernels in configs.items():
        (batch / name).mkdir(parents=True)
        doc = {
            "gpu": "rtx2080ti",
            "config": "benchmarks/mm/configs/x.toml",
            "reference": "basic",
            "rows": {"all-static": []},
            "records": [
                record("all-static", k, "GASPerf", e, s) for k, e, s in kernels
            ],
        }
        (batch / name / "ablation.json").write_text(json.dumps(doc))
        result = {
            "reference": "basic",
            "prediction": {"kernels": {k: kernel_settings(k) for k, _, _ in kernels}},
        }
        (batch / name / "result.json").write_text(json.dumps(result))
    return batch


def test_duplicates_are_marked_and_counted_once(tmp_path):
    records, missing, rows = read_ablation.load_batch(
        write_duplicate_batch(tmp_path), None
    )
    assert missing == []
    flags = {
        (r["config"], r["kernel"]): (r["duplicate_run"], r["duplicate_speedup"])
        for r in records
    }
    assert flags == {
        ("mm_a", "basic"): (False, False),
        ("mm_a", "tiled"): (False, False),
        ("mm_a", "uncoalesced"): (False, False),
        ("mm_b", "basic"): (True, False),
        ("mm_b", "tiled"): (True, True),
    }
    summary = read_ablation.summarize(records, rows)
    mm = next(s for s in summary if s["benchmark"] == "mm")
    assert mm["kernels"] == 3
    assert mm["abs_err_mean"] == pytest.approx(20.0)
    assert mm["speedups"] == 2
    assert mm["abs_speedup_err_mean"] == pytest.approx(6.0)


def test_no_result_json_means_no_duplicates(tmp_path):
    records, _, _ = read_ablation.load_batch(write_batch(tmp_path), None)
    assert not any(r["duplicate_run"] or r["duplicate_speedup"] for r in records)
