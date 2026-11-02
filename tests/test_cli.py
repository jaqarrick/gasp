"""Tests for gasp.hardware — HardwareProfile loading and validation."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gasp.cli import build_parser, resolve_traces
from gasp.cli.utils import GPU_PROFILES, HardwareProfile, load_hardware_profile

VALID_PROFILE = {
    "architecture": "ampere",
    "chip": "a100",
    "arch": "sm_80",
    "spec": {
        "warp_size": 32,
        "simd_width": 32,
        "sfu_width": 16,
        "sms_per_gpu": 108,
        "max_warps_per_sm": 64,
        "max_blocks_per_sm": 32,
        "max_registers_per_sm": 65536,
        "max_shared_mem_per_sm_bytes": 167936,
        "l2_cache_size_bytes": 41943040,
        "max_l1_cache_size_bytes": 196608,
    },
    "profiled": {
        "launch_overhead_cycles": 5000,
        "departure_delay_cycles": 6,
        "sync_factor_cycles": 24,
        "dram_latency_cycles": 400,
        "l1_cache_hit_latency_cycles": 33,
        "l2_cache_hit_latency_cycles": 200,
        "bank_penalty_cycles": 2,
        "sfu_inst_latency_cycles": 20,
        "fp32_inst_latency_cycles": 4,
        "int32_inst_latency_cycles": 4,
        "fp64_inst_latency_cycles": 8,
        "dram_bandwidth_bytes_per_cycle": 1555,
        "clock_rate_ghz": 1.41,
        "shared_mem_load_latency_cycles": 23,
        "shared_mem_store_latency_cycles": 19,
        "const_mem_latency_cycles": 4,
    },
}


@pytest.fixture
def logger():
    return MagicMock(spec=["info", "error", "warn", "debug"])


@pytest.fixture
def profile_file(tmp_path):
    """Write VALID_PROFILE to a temp JSON and return its path."""
    p = tmp_path / "a100.json"
    p.write_text(json.dumps(VALID_PROFILE))
    return p


class TestHardwareProfile:
    def test_construct_from_valid_kwargs(self):
        flat = {
            **VALID_PROFILE,
            **VALID_PROFILE["spec"],
            **VALID_PROFILE["profiled"],
        }
        flat.pop("spec")
        flat.pop("profiled")
        hp = HardwareProfile(**flat)

        assert hp.architecture == "ampere"
        assert hp.chip == "a100"
        assert hp.arch == "sm_80"
        assert hp.warp_size == 32
        assert hp.sms_per_gpu == 108
        assert hp.fp32_inst_latency_cycles == 4
        assert hp.clock_rate_ghz == 1.41

    def test_missing_field_raises(self):
        flat = {
            **VALID_PROFILE,
            **VALID_PROFILE["spec"],
            **VALID_PROFILE["profiled"],
        }
        flat.pop("spec")
        flat.pop("profiled")
        del flat["dram_latency_cycles"]

        with pytest.raises(TypeError):
            HardwareProfile(**flat)

    def test_extra_field_raises(self):
        flat = {
            **VALID_PROFILE,
            **VALID_PROFILE["spec"],
            **VALID_PROFILE["profiled"],
        }
        flat.pop("spec")
        flat.pop("profiled")
        flat["bogus_field"] = 42

        with pytest.raises(TypeError):
            HardwareProfile(**flat)


class TestGPUProfiles:
    def test_known_gpus(self):
        for gpu in ("a100", "h100", "b200"):
            assert gpu in GPU_PROFILES, f"{gpu} missing from GPU_PROFILES"

    def test_paths_are_json(self):
        for gpu, path in GPU_PROFILES.items():
            assert path.suffix == ".json", f"{gpu} profile is not .json"

    def test_paths_are_under_hardware(self):
        for gpu, path in GPU_PROFILES.items():
            assert "hardware" in path.parts, f"{gpu} path not under hardware/"


class TestLoadHardwareProfile:
    def test_loads_valid_profile(self, monkeypatch, profile_file, logger):
        monkeypatch.setitem(GPU_PROFILES, "a100", profile_file)

        hp = load_hardware_profile("a100", logger)

        assert isinstance(hp, HardwareProfile)
        assert hp.architecture == "ampere"
        assert hp.chip == "a100"
        assert hp.arch == "sm_80"
        assert hp.sms_per_gpu == 108
        assert hp.dram_latency_cycles == 400
        assert hp.clock_rate_ghz == 1.41
        logger.info.assert_called_once()

    def test_unknown_gpu_exits(self, logger):
        with pytest.raises(SystemExit):
            load_hardware_profile("v100_not_registered", logger)

        logger.error.assert_called_once()

    def test_all_spec_fields_loaded(self, monkeypatch, profile_file, logger):
        monkeypatch.setitem(GPU_PROFILES, "a100", profile_file)
        hp = load_hardware_profile("a100", logger)

        for field, value in VALID_PROFILE["spec"].items():
            assert getattr(hp, field) == value, f"spec.{field} mismatch"

    def test_all_profiled_fields_loaded(self, monkeypatch, profile_file, logger):
        monkeypatch.setitem(GPU_PROFILES, "a100", profile_file)
        hp = load_hardware_profile("a100", logger)

        for field, value in VALID_PROFILE["profiled"].items():
            assert getattr(hp, field) == value, f"profiled.{field} mismatch"

    def test_missing_json_field_raises(self, tmp_path, monkeypatch, logger):
        """JSON missing a profiled key → TypeError from dataclass."""
        bad = json.loads(json.dumps(VALID_PROFILE))
        del bad["profiled"]["dram_latency_cycles"]

        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        monkeypatch.setitem(GPU_PROFILES, "bad", p)

        with pytest.raises(TypeError):
            load_hardware_profile("bad", logger)

    def test_extra_json_field_raises(self, tmp_path, monkeypatch, logger):
        """Extra unexpected key in JSON → TypeError from dataclass."""
        bad = json.loads(json.dumps(VALID_PROFILE))
        bad["profiled"]["turbo_boost_factor"] = 1.5

        p = tmp_path / "extra.json"
        p.write_text(json.dumps(bad))
        monkeypatch.setitem(GPU_PROFILES, "extra", p)

        with pytest.raises(TypeError):
            load_hardware_profile("extra", logger)

    def test_file_not_found_raises(self, monkeypatch, logger):
        monkeypatch.setitem(GPU_PROFILES, "ghost", Path("/no/such/file.json"))

        with pytest.raises(FileNotFoundError):
            load_hardware_profile("ghost", logger)

    def test_malformed_json_raises(self, tmp_path, monkeypatch, logger):
        p = tmp_path / "bad.json"
        p.write_text("{ not valid json !!!")
        monkeypatch.setitem(GPU_PROFILES, "corrupt", p)

        with pytest.raises(json.JSONDecodeError):
            load_hardware_profile("corrupt", logger)

    def test_flattening_merges_correctly(self, monkeypatch, profile_file, logger):
        """spec and profiled are merged into a flat namespace — no collisions."""
        monkeypatch.setitem(GPU_PROFILES, "a100", profile_file)
        hp = load_hardware_profile("a100", logger)

        # top-level keys survive
        assert hp.architecture == "ampere"
        # spec keys survive
        assert hp.warp_size == 32
        # profiled keys survive
        assert hp.launch_overhead_cycles == 5000


# --log-traces / --trace-file

VECADD = "benchmarks/custom/vecadd.cu"


def parse(*argv):
    return build_parser().parse_args(["--gpu", "b200", VECADD, *argv])


def test_log_traces_defaults_to_off():
    args = parse()
    assert args.log_traces is None
    assert resolve_traces(args.log_traces, args.trace_file) == (None, "traces.log")


def test_log_traces_without_modules_is_all():
    assert parse("--log-traces").log_traces == {"all"}


def test_log_traces_comma_separated_modules():
    assert parse("--log-traces", "cfg, SASS,profiler").log_traces == {
        "cfg",
        "sass",
        "profiler",
    }


def test_log_traces_rejects_unknown_module():
    with pytest.raises(SystemExit):
        parse("--log-traces", "cfg,bogus")


def test_trace_file_alone_implies_all():
    args = parse("--trace-file", "out.log")
    assert resolve_traces(args.log_traces, args.trace_file) == ({"all"}, "out.log")


def test_trace_file_keeps_selected_modules():
    args = parse("--log-traces", "cfg", "--trace-file", "out.log")
    assert resolve_traces(args.log_traces, args.trace_file) == ({"cfg"}, "out.log")
