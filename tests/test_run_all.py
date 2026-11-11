"""Tests for the pure parts of hardware/microbenchmarks/run_all.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "hardware" / "microbenchmarks" / "run_all.py"
)
spec = importlib.util.spec_from_file_location("run_all", SCRIPT)
run_all = importlib.util.module_from_spec(spec)
# Register before executing: @dataclass looks the module up in sys.modules.
sys.modules["run_all"] = run_all
spec.loader.exec_module(run_all)

BASE = {
    "architecture": "turing",
    "chip": "rtx2080ti",
    "arch": "sm_75",
    "spec": {"warp_size": 32, "sms_per_gpu": 68},
    "profiled": {
        "dram_latency_cycles": 400,
        "bank_penalty_cycles": 1,
        "clock_rate_ghz": 1.35,
    },
}


def measured(value, clock=1.365):
    return {"value": value, "source": "hardware/runs/x.json", "sm_clock_ghz": clock}


def test_build_profile_overwrites_measured_and_copies_the_rest():
    profile, sources = run_all.build_profile(
        BASE, {"dram_latency_cycles": measured(406.36)}, clock_ghz=1.365
    )

    assert profile["profiled"]["dram_latency_cycles"] == 406.36
    assert profile["profiled"]["bank_penalty_cycles"] == 1
    assert profile["profiled"]["clock_rate_ghz"] == 1.365
    assert profile["spec"] == BASE["spec"]
    assert sources["bank_penalty_cycles"]["source"] == "copied from base profile"
    assert sources["dram_latency_cycles"]["source"] == "hardware/runs/x.json"


def test_build_profile_does_not_modify_the_base():
    run_all.build_profile(
        BASE, {"dram_latency_cycles": measured(406.36)}, clock_ghz=None
    )

    assert BASE["profiled"]["dram_latency_cycles"] == 400


def test_build_profile_without_clock_keeps_base_clock():
    profile, sources = run_all.build_profile(BASE, {}, clock_ghz=None)

    assert profile["profiled"]["clock_rate_ghz"] == 1.35
    assert sources["clock_rate_ghz"]["source"] == "copied from base profile"


def test_build_profile_rejects_fields_the_base_lacks():
    with pytest.raises(ValueError, match="sync_factor"):
        run_all.build_profile(BASE, {"sync_factor": measured(0.179)}, clock_ghz=None)


def test_dependency_order_puts_dependencies_first():
    order = run_all.dependency_order(["sync_cost", "dram_bandwidth", "dram_latency"])

    assert order.index("dram_latency") < order.index("sync_cost")
    assert order.index("dram_bandwidth") < order.index("sync_cost")


def test_dependency_order_leaves_out_unselected_dependencies():
    assert run_all.dependency_order(["sync_cost"]) == ["sync_cost"]


def test_every_profile_parameter_maps_to_a_profile_key():
    profile_keys = {
        "fp32_inst_latency_cycles",
        "fp64_inst_latency_cycles",
        "int32_inst_latency_cycles",
        "sfu_inst_latency_cycles",
        "l1_cache_hit_latency_cycles",
        "l2_cache_hit_latency_cycles",
        "dram_latency_cycles",
        "dram_bandwidth_bytes_per_cycle",
        "const_mem_latency_cycles",
        "shared_mem_load_latency_cycles",
        "bank_penalty_cycles",
        "shared_mem_store_latency_cycles",
        "shared_mem_load_throughput_cycles",
        "departure_delay_cycles",
        "sync_factor",
        "launch_overhead_base_cycles",
        "launch_overhead_per_thread_cycles",
    }
    mapped = {
        run_all.PROFILE_KEY.get(param, param)
        for bench in run_all.BENCHMARKS
        if bench.in_profile
        for param in bench.params
    }

    assert mapped == profile_keys


def test_sync_cost_run_json_is_found_and_its_clock_read(tmp_path):
    bench = run_all.BY_NAME["sync_cost"]
    bench_dir = tmp_path / bench.dir_name
    bench_dir.mkdir()
    (bench_dir / "sync_factor_cycles.rtx2080ti.run.json").write_text(
        '{"parameter": "sync_factor_cycles", "value": 0.179,'
        ' "clock": {"sm_clock_ghz": 1.35, "sm_clock_ghz_min": 1.349,'
        ' "sm_clock_ghz_max": 1.351}}'
    )

    path = run_all.run_json_path(bench_dir, bench.params[0], bench.aliases)

    assert path is not None
    assert run_all.read_value(path) == 0.179
    assert run_all.read_clock(path) == 1.35
