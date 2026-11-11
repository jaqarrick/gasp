"""Tests for gasp.hardware — HardwareProfile loading and validation."""

import dataclasses
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gasp.cli import (
    KernelResult,
    build_parser,
    build_specs,
    check_args,
    loop_counts_template,
    main,
    report_comparison,
    report_loops,
    report_measured,
    resolve_traces,
    unique_kernels,
)
from gasp.cli.config import ConfigError, KernelSpec
from gasp.cli.types import Args
from gasp.cli.utils import (
    GPU_PROFILES,
    HardwareProfile,
    load_hardware_profile,
    load_profile_file,
)
from gasp.common.constants import GPU_TO_ARCH
from gasp.dependency import LoopInfo
from gasp.predict import PredictionOutput
from gasp.profiler import KernelMeasurement, ProfileOutput

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
        "l1_shared_array_bytes": 196608,
        "shared_mem_carveouts_bytes": [
            0,
            8192,
            16384,
            32768,
            65536,
            102400,
            135168,
            167936,
        ],
    },
    "profiled": {
        "launch_overhead_base_cycles": 5000,
        "launch_overhead_per_thread_cycles": 0.002,
        "departure_delay_cycles": 6,
        "sync_factor": 24,
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
        "shared_mem_load_throughput_cycles": 1.0,
        "l1_cycles_per_request": 1.0,
        "l1_sectors_per_cycle_per_sm": 4.0,
        "l2_bandwidth_bytes_per_cycle": 5120,
        "const_mem_latency_cycles": 4,
    },
}


@pytest.fixture
def logger():
    return MagicMock(spec=["info", "error", "warn", "warning", "debug", "trace"])


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
        for gpu in ("a100", "h100", "b200", "l4", "v100", "rtx2080ti"):
            assert gpu in GPU_PROFILES, f"{gpu} missing from GPU_PROFILES"

    def test_paths_are_json(self):
        for gpu, path in GPU_PROFILES.items():
            assert path.suffix == ".json", f"{gpu} profile is not .json"

    def test_paths_are_under_hardware(self):
        for gpu, path in GPU_PROFILES.items():
            assert "hardware" in path.parts, f"{gpu} path not under hardware/"

    def test_every_profile_loads(self, logger):
        for gpu in GPU_PROFILES:
            hp = load_hardware_profile(gpu, logger)
            assert hp.chip == gpu
            assert hp.arch == GPU_TO_ARCH[gpu]


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
        """JSON missing a profiled key → error and exit."""
        bad = json.loads(json.dumps(VALID_PROFILE))
        del bad["profiled"]["dram_latency_cycles"]

        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        monkeypatch.setitem(GPU_PROFILES, "bad", p)

        with pytest.raises(SystemExit):
            load_hardware_profile("bad", logger)
        logger.error.assert_called_once()

    def test_extra_json_field_raises(self, tmp_path, monkeypatch, logger):
        """Extra unexpected key in JSON → error and exit."""
        bad = json.loads(json.dumps(VALID_PROFILE))
        bad["profiled"]["turbo_boost_factor"] = 1.5

        p = tmp_path / "extra.json"
        p.write_text(json.dumps(bad))
        monkeypatch.setitem(GPU_PROFILES, "extra", p)

        with pytest.raises(SystemExit):
            load_hardware_profile("extra", logger)
        logger.error.assert_called_once()

    def test_file_not_found_raises(self, monkeypatch, logger):
        monkeypatch.setitem(GPU_PROFILES, "ghost", Path("/no/such/file.json"))

        with pytest.raises(SystemExit):
            load_hardware_profile("ghost", logger)
        logger.error.assert_called_once()

    def test_malformed_json_raises(self, tmp_path, monkeypatch, logger):
        p = tmp_path / "bad.json"
        p.write_text("{ not valid json !!!")
        monkeypatch.setitem(GPU_PROFILES, "corrupt", p)

        with pytest.raises(SystemExit):
            load_hardware_profile("corrupt", logger)
        logger.error.assert_called_once()

    def test_flattening_merges_correctly(self, monkeypatch, profile_file, logger):
        """spec and profiled are merged into a flat namespace — no collisions."""
        monkeypatch.setitem(GPU_PROFILES, "a100", profile_file)
        hp = load_hardware_profile("a100", logger)

        # top-level keys survive
        assert hp.architecture == "ampere"
        # spec keys survive
        assert hp.warp_size == 32
        # profiled keys survive
        assert hp.launch_overhead_base_cycles == 5000
        assert hp.launch_overhead_per_thread_cycles == 0.002


def write_profile(tmp_path, data) -> Path:
    p = tmp_path / "custom.json"
    p.write_text(data if isinstance(data, str) else json.dumps(data))
    return p


def error_message(logger) -> str:
    logger.error.assert_called_once()
    return logger.error.call_args.args[0]


class TestLoadProfileFile:
    def test_loads_valid_file(self, profile_file, logger):
        hp = load_profile_file(profile_file, logger)
        assert hp.chip == "a100"
        assert hp.arch == "sm_80"
        logger.error.assert_not_called()
        logger.trace.assert_called()

    def test_missing_file_exits(self, tmp_path, logger):
        with pytest.raises(SystemExit):
            load_profile_file(tmp_path / "nope.json", logger)
        assert "not found" in error_message(logger)

    def test_invalid_json_exits(self, tmp_path, logger):
        with pytest.raises(SystemExit):
            load_profile_file(write_profile(tmp_path, "{not json"), logger)
        assert "not valid JSON" in error_message(logger)

    def test_non_object_exits(self, tmp_path, logger):
        with pytest.raises(SystemExit):
            load_profile_file(write_profile(tmp_path, [1, 2]), logger)

    def test_missing_section_exits(self, tmp_path, logger):
        data = dict(VALID_PROFILE)
        del data["spec"]
        with pytest.raises(SystemExit):
            load_profile_file(write_profile(tmp_path, data), logger)
        assert "'spec'" in error_message(logger)

    def test_missing_fields_all_reported(self, tmp_path, logger):
        data = json.loads(json.dumps(VALID_PROFILE))
        del data["profiled"]["dram_latency_cycles"]
        del data["spec"]["warp_size"]
        del data["chip"]
        with pytest.raises(SystemExit):
            load_profile_file(write_profile(tmp_path, data), logger)
        msg = error_message(logger)
        for name in ("dram_latency_cycles", "warp_size", "chip"):
            assert name in msg

    def test_unknown_field_exits(self, tmp_path, logger):
        data = json.loads(json.dumps(VALID_PROFILE))
        data["profiled"]["bogus_field"] = 1
        with pytest.raises(SystemExit):
            load_profile_file(write_profile(tmp_path, data), logger)
        assert "bogus_field" in error_message(logger)

    def test_wrong_type_exits(self, tmp_path, logger):
        data = json.loads(json.dumps(VALID_PROFILE))
        data["profiled"]["dram_latency_cycles"] = "fast"
        data["spec"]["shared_mem_carveouts_bytes"] = 65536
        with pytest.raises(SystemExit):
            load_profile_file(write_profile(tmp_path, data), logger)
        msg = error_message(logger)
        assert "dram_latency_cycles" in msg
        assert "shared_mem_carveouts_bytes" in msg

    def test_bool_is_not_a_number(self, tmp_path, logger):
        data = json.loads(json.dumps(VALID_PROFILE))
        data["spec"]["warp_size"] = True
        with pytest.raises(SystemExit):
            load_profile_file(write_profile(tmp_path, data), logger)

    def test_float_in_int_field_accepted(self, tmp_path, logger):
        data = json.loads(json.dumps(VALID_PROFILE))
        data["profiled"]["dram_latency_cycles"] = 406.36
        hp = load_profile_file(write_profile(tmp_path, data), logger)
        assert hp.dram_latency_cycles == 406.36


class TestGpuProfileFlags:
    def test_both_flags_warn_and_exit(self, monkeypatch, profile_file, capsys):
        monkeypatch.setattr(
            "sys.argv",
            ["gasp", "--gpu", "b200", "--profile", str(profile_file), VECADD],
        )
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1
        assert "mutually exclusive" in capsys.readouterr().err

    def test_neither_flag_exits(self):
        parser = build_parser()
        args = Args(**vars(parser.parse_intermixed_args([VECADD])))
        with pytest.raises(SystemExit):
            check_args(parser, args)

    def test_profile_alone_passes_check(self, profile_file):
        parser = build_parser()
        argv = ["--profile", str(profile_file), VECADD]
        args = Args(**vars(parser.parse_intermixed_args(argv)))
        check_args(parser, args)
        assert args.gpu is None


# --trace / --trace-file

VECADD = "benchmarks/vecadd/vecadd.cu"


STRIDE = "benchmarks/vecadd/vecadd_stride.cu"


def parse(*argv):
    return build_parser().parse_intermixed_args(["--gpu", "b200", VECADD, *argv])


def test_trace_defaults_to_off():
    args = parse()
    assert args.trace is None
    assert resolve_traces(args.trace, args.trace_file) == (None, "traces.log")


def test_trace_without_modules_is_all():
    assert parse("--trace").trace == {"all"}


def test_trace_comma_separated_modules():
    assert parse("--trace", "cfg, SASS,profiler").trace == {
        "cfg",
        "sass",
        "profiler",
    }


def test_trace_rejects_unknown_module():
    with pytest.raises(SystemExit):
        parse("--trace", "cfg,bogus")


def test_trace_file_alone_implies_all():
    args = parse("--trace-file", "out.log")
    assert resolve_traces(args.trace, args.trace_file) == ({"all"}, "out.log")


def test_trace_file_keeps_selected_modules():
    args = parse("--trace", "cfg", "--trace-file", "out.log")
    assert resolve_traces(args.trace, args.trace_file) == ({"cfg"}, "out.log")


# inputs and flag rules


def checked_args(*argv) -> Args:
    parser = build_parser()
    args = Args(**vars(parser.parse_intermixed_args(["--gpu", "b200", *argv])))
    check_args(parser, args)
    return args


def test_several_inputs():
    assert checked_args(VECADD, STRIDE).inputs == [(VECADD, None), (STRIDE, None)]


def test_input_selector():
    assert checked_args(f"{VECADD}::vecadd").inputs == [(VECADD, "vecadd")]


def test_inputs_can_be_mixed_with_flags():
    args = checked_args(VECADD, "--block-size", "128", STRIDE)

    assert args.inputs == [(VECADD, None), (STRIDE, None)]
    assert args.block_size == (128,)


def test_block_and_grid_take_several_dims():
    args = checked_args(VECADD, "--block-size", "16x16", "--grid-size", "64x64")
    [spec] = build_specs(args)

    assert (spec.block_dim, spec.grid_dim) == ((16, 16), (64, 64))
    assert (spec.block_size, spec.grid_size) == (256, 4096)


@pytest.mark.parametrize("value", ["0", "16x", "16x16x2x2", "x", "16x-1"])
def test_invalid_block_size_exits(value):
    with pytest.raises(SystemExit):
        checked_args(VECADD, "--block-size", value)


def test_params_are_repeatable():
    args = checked_args(VECADD, "--param", "M=1024", "--param", "N=512")
    [spec] = build_specs(args)

    assert spec.params == {"M": 1024, "N": 512}


@pytest.mark.parametrize("value", ["M", "M=", "=4", "M=x", "M=1.5"])
def test_invalid_param_exits(value):
    with pytest.raises(SystemExit):
        checked_args(VECADD, "--param", value)


@pytest.mark.parametrize("value", ["benchmarks/missing.cu", "README.md", f"{VECADD}::"])
def test_invalid_input_exits(value):
    with pytest.raises(SystemExit):
        checked_args(value)


def test_no_inputs_exits():
    with pytest.raises(SystemExit):
        checked_args()


def test_config_alone_is_accepted():
    assert checked_args("--config", "kernels.toml").config == "kernels.toml"


def test_config_with_input_file_exits():
    with pytest.raises(SystemExit):
        checked_args("--config", "kernels.toml", VECADD)


@pytest.mark.parametrize(
    "flag",
    [
        ("--block-size", "256"),
        ("--grid-size", "8"),
        ("--problem-size", "1000"),
        ("--footprint", "12000000"),
        ("--reuse", "0.5"),
        ("--loop-iterations", "8"),
        ("-I", "inc"),
        ("--nvcc-flags", "-O2"),
        ("--program-args", "1000"),
        ("--param", "M=1024"),
    ],
)
def test_config_with_per_kernel_flag_exits(flag):
    with pytest.raises(SystemExit):
        checked_args("--config", "kernels.toml", *flag)


@pytest.mark.parametrize(
    "flags",
    [
        ("--run", "--repeats", "5"),
        ("--run",),
        ("--reference", "good"),
        ("--verbose",),
    ],
)
def test_config_with_global_flags_is_accepted(flags):
    checked_args("--config", "kernels.toml", *flags)


def test_save_metrics_requires_run():
    with pytest.raises(SystemExit):
        checked_args(VECADD, "--save-metrics", "m.json")
    assert checked_args(VECADD, "--run", "--save-metrics", "m.json").save_metrics == (
        "m.json"
    )


def test_metrics_with_run_exits():
    with pytest.raises(SystemExit):
        checked_args(VECADD, "--run", "--metrics", "m.json")


def test_overrides_are_repeatable():
    args = checked_args(
        VECADD, "--override", "coalescing=crude", "--override", "bank=crude"
    )
    assert args.override == ["coalescing=crude", "bank=crude"]


def test_profiled_override_requires_metrics():
    with pytest.raises(SystemExit):
        checked_args(VECADD, "--override", "cache=profiled")
    checked_args(VECADD, "--metrics", "m.json", "--override", "cache=profiled")


@pytest.mark.parametrize(
    "values",
    [
        ("cache",),
        ("bogus=profiled",),
        ("cache=crude",),
        ("all=profiled", "cache=profiled"),
    ],
)
def test_invalid_override_exits(values):
    flags = [part for value in values for part in ("--override", value)]
    with pytest.raises(SystemExit):
        checked_args(VECADD, "--metrics", "m.json", *flags)


def test_config_with_metrics_and_overrides_is_accepted():
    checked_args(
        "--config", "kernels.toml", "--metrics", "m.json", "--override", "all=profiled"
    )
    checked_args("--config", "kernels.toml", "--run", "--save-metrics", "m.json")


def test_repeats_without_run_exits():
    with pytest.raises(SystemExit):
        checked_args(VECADD, "--repeats", "3")


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_invalid_repeats_exits(value):
    with pytest.raises(SystemExit):
        checked_args(VECADD, "--run", "--repeats", value)


def test_build_specs_applies_shared_flags_to_every_input():
    args = checked_args(
        VECADD,
        STRIDE,
        "--problem-size",
        "1000",
        "--block-size",
        "100",
        "--nvcc-flags",
        "--use_fast_math -O2",
        "--program-args",
        "1000",
        "-I",
        "inc",
    )
    vecadd, stride = build_specs(args)

    assert (vecadd.name, stride.name) == ("vecadd", "vecadd_stride")
    for spec in (vecadd, stride):
        assert spec.grid_size == 10
        assert spec.nvcc_flags == ["--use_fast_math", "-O2"]
        assert spec.program_args == ["1000"]
        assert spec.include_dirs == ["inc"]


# reports


def result(name: str, t_gpu: float, pico_error: bool = False) -> KernelResult:
    return KernelResult(
        spec=KernelSpec(
            name=name,
            file=f"{name}.cu",
            kernel=None,
            problem_size=1000,
            block_size=256,
            grid_size=4,
            block_dim=(256,),
            grid_dim=(4,),
            loop_iterations=None,
            loop_counts=None,
            include_dirs=[],
            nvcc_flags=[],
            program_args=[],
            params={},
        ),
        kernel_name=name,
        prediction=PredictionOutput(
            itilp=1.0,
            cwp=1.0,
            mwp_cp=1.0,
            mwp_peak_bw=1.0,
            itmlp=1.0,
            average_trans_warp=1.0,
            working_set_per_sm_bytes=0,
            working_set_total_bytes=0,
            l1_cache_size_bytes=0,
            shared_mem_carveout_bytes=0,
            reuse_fraction=0.0,
            l1_hit_ratio=0.0,
            l2_hit_ratio=0.0,
            amat=300.0,
            t_comp=10.0,
            t_mem=20.0,
            t_overlap=5.0,
            t_exec=25.0,
            launch_overhead=0.0,
            t_gpu=t_gpu,
        ),
        pico_error=pico_error,
    )


def measured(*cycles: float) -> ProfileOutput:
    return ProfileOutput(
        runs=[KernelMeasurement(duration_ns=1.0, cycles=c, metrics={}) for c in cycles]
    )


@pytest.fixture
def report_logger():
    return MagicMock(spec=["table", "warning"])


def table(logger):
    title, columns, rows = logger.table.call_args.args
    assert logger.table.call_args.kwargs["force"] is True
    return title, [dict(zip(columns, row)) for row in rows]


def test_comparison_speedup_is_relative_to_the_reference(report_logger):
    report_comparison(
        [result("good", 100.0), result("poor", 200.0)], "good", report_logger
    )
    title, rows = table(report_logger)

    assert "reference: good" in title
    assert [row["Speedup"] for row in rows] == ["1.00", "0.50"]
    assert rows[0]["Launch"] == "<<<4, 256>>>"
    assert rows[0]["T_comp"] == "10"
    assert rows[0]["T_mem"] == "20"
    assert rows[0]["T_overlap"] == "5"


def test_comparison_with_another_reference(report_logger):
    report_comparison(
        [result("good", 100.0), result("poor", 200.0)], "poor", report_logger
    )
    _, rows = table(report_logger)

    assert [row["Speedup"] for row in rows] == ["2.00", "1.00"]


def test_single_kernel_has_no_speedup(report_logger):
    report_comparison([result("only", 100.0)], "only", report_logger)
    title, rows = table(report_logger)

    assert "reference" not in title
    assert "Speedup" not in rows[0]
    assert rows[0]["T_GPU"] == "100"


def test_comparison_warns_when_pico_fell_back_on_some_kernels(report_logger):
    results = [result("good", 100.0), result("poor", 200.0, pico_error=True)]
    report_comparison(results, "good", report_logger)
    _, rows = table(report_logger)

    report_logger.warning.assert_called_once()
    assert [row["Pico"] for row in rows] == ["ok", "fallback"]


def test_comparison_shows_overrides(report_logger):
    overridden = dataclasses.replace(result("poor", 200.0), overrides="cache=profiled")
    report_comparison([result("good", 100.0), overridden], "good", report_logger)
    _, rows = table(report_logger)

    assert [row["Overrides"] for row in rows] == ["none", "cache=profiled"]


def test_comparison_no_warning_when_pico_fell_back_on_all(report_logger):
    results = [result("a", 100.0, pico_error=True), result("b", 200.0, pico_error=True)]
    report_comparison(results, "a", report_logger)

    report_logger.warning.assert_not_called()


def modelled(name: str, t_gpu: float, model: str) -> KernelResult:
    return dataclasses.replace(result(name, t_gpu), model=model)


# good and poor under both models; GASPerf predicts poor 4x slower, GPUPerf 2x
BOTH_MODELS = [
    modelled("good", 100.0, "GPUPerf"),
    modelled("good", 50.0, "GASPerf"),
    modelled("poor", 200.0, "GPUPerf"),
    modelled("poor", 200.0, "GASPerf"),
]


def test_comparison_has_a_row_per_kernel_and_model(report_logger):
    report_comparison(BOTH_MODELS, "good", report_logger)
    _, rows = table(report_logger)

    assert [(row["Kernel"], row["Model"]) for row in rows] == [
        ("good", "GPUPerf"),
        ("good", "GASPerf"),
        ("poor", "GPUPerf"),
        ("poor", "GASPerf"),
    ]


def test_comparison_speedup_is_within_each_model(report_logger):
    report_comparison(BOTH_MODELS, "good", report_logger)
    title, rows = table(report_logger)

    assert "within each model" in title
    assert [row["Speedup"] for row in rows] == ["1.00", "1.00", "0.50", "0.25"]


def test_single_kernel_under_both_models_has_no_speedup(report_logger):
    results = [modelled("only", 100.0, "GPUPerf"), modelled("only", 90.0, "GASPerf")]
    report_comparison(results, "only", report_logger)
    _, rows = table(report_logger)

    assert len(rows) == 2
    assert "Speedup" not in rows[0]


def test_measured_report_speedup_is_within_each_model(report_logger):
    profiles = {"good": measured(100), "poor": measured(125)}
    report_measured(BOTH_MODELS, profiles, "good", 1, report_logger)
    _, rows = table(report_logger)

    assert [row["Model"] for row in rows] == ["GPUPerf", "GASPerf"] * 2
    assert [row["Pred speedup"] for row in rows] == ["1.00", "1.00", "0.50", "0.25"]
    # the measurement is per kernel, the same under every model
    assert rows[2]["Meas speedup (range)"] == rows[3]["Meas speedup (range)"]


def test_unique_kernels_keeps_the_first_model_of_each():
    kept = unique_kernels(BOTH_MODELS)

    assert [(r.spec.name, r.model) for r in kept] == [
        ("good", "GPUPerf"),
        ("poor", "GPUPerf"),
    ]


def test_measured_report_compares_speedups(report_logger):
    results = [result("good", 100.0), result("poor", 200.0)]
    profiles = {"good": measured(90, 100, 110), "poor": measured(120, 125, 130)}
    report_measured(results, profiles, "good", 3, report_logger)
    title, rows = table(report_logger)

    assert "3 runs per kernel" in title
    assert rows[0]["Meas cycles (median)"] == "100"
    assert rows[0]["Spread (min-max)"] == "90-110"
    assert rows[0]["Meas speedup (range)"] == "1.00 (reference)"
    assert rows[1]["Pred speedup"] == "0.50"
    assert rows[1]["Meas speedup (range)"] == "0.80 (0.69-0.92)"


def test_measured_report_single_kernel_has_no_speedup(report_logger):
    report_measured(
        [result("only", 100.0)], {"only": measured(100)}, "only", 1, report_logger
    )
    title, rows = table(report_logger)

    assert "1 run per kernel" in title
    assert "Pred speedup" not in rows[0]
    assert rows[0]["Error"] == "+0.0%"


# loop table


MM_SOURCE = """\
__global__ void k(float *a, int K) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += a[k];
    }
}
"""


def mm_loops():
    def loop(header, insts, loads, parent=None):
        return LoopInfo(header, {header}, insts, loads, 0, parent)

    return [loop(".L_x_4", 70, 32), loop(".L_x_2", 24, 8), loop(".L_x_6", 10, 2)]


def loop_spec(tmp_path, **changes):
    source = tmp_path / "k.cu"
    source.write_text(MM_SOURCE)
    spec = dataclasses.replace(result("basic", 1.0).spec, file=str(source))
    return dataclasses.replace(spec, **changes)


def loop_table(logger):
    title, columns, rows = logger.table.call_args.args
    return title, [dict(zip(columns, row)) for row in rows]


def test_loop_table_shows_source_lines_and_counts(tmp_path, report_logger):
    spec = loop_spec(tmp_path, loop_counts=[64, 0, 0])
    lines = {h: (str(tmp_path / "k.cu"), 4) for h in (".L_x_4", ".L_x_2", ".L_x_6")}
    report_loops(spec, mm_loops(), lines, report_logger)
    title, rows = loop_table(report_logger)

    assert "3 in the compiled code" in title
    assert [r["#"] for r in rows] == ["1", "2", "3"]
    assert rows[0]["Source line"] == "k.cu:4  sum += a[k];"
    assert [r["Loads/trip"] for r in rows] == ["32 global", "8 global", "2 global"]
    assert [r["Count"] for r in rows] == ["64", "0", "0"]


def test_loop_table_uses_loop_iterations_or_question_marks(tmp_path, report_logger):
    report_loops(loop_spec(tmp_path, loop_iterations=8), mm_loops(), {}, report_logger)
    _, rows = loop_table(report_logger)
    assert [r["Count"] for r in rows] == ["8", "8", "8"]
    assert rows[0]["Source line"] == "unknown"

    report_loops(loop_spec(tmp_path, loop_counts=[1]), mm_loops(), {}, report_logger)
    _, rows = loop_table(report_logger)
    assert [r["Count"] for r in rows] == ["?", "?", "?"]


def test_loop_table_shows_nesting(tmp_path, report_logger):
    loops = [
        LoopInfo("outer", {"outer", "inner"}, 5, 0, 1, None),
        LoopInfo("inner", {"inner"}, 3, 0, 1, "outer"),
    ]
    report_loops(loop_spec(tmp_path), loops, {}, report_logger)
    _, rows = loop_table(report_logger)
    assert [r["Inside"] for r in rows] == ["-", "#1"]
    assert rows[1]["Loads/trip"] == "1 shared"


def test_no_loop_table_without_loops(tmp_path, report_logger):
    report_loops(loop_spec(tmp_path), [], {}, report_logger)
    report_logger.table.assert_not_called()


def test_loop_counts_template():
    assert loop_counts_template(3) == "loop_counts = [?, ?, ?]"


def test_build_specs_passes_footprint():
    (spec,) = build_specs(checked_args(VECADD, "--footprint", "12000000"))
    assert spec.footprint == 12_000_000


def test_build_specs_footprint_defaults_to_none():
    (spec,) = build_specs(checked_args(VECADD))
    assert spec.footprint is None


def test_build_specs_passes_reuse():
    (spec,) = build_specs(checked_args(VECADD, "--reuse", "0.94"))
    assert spec.reuse == 0.94


def test_reuse_out_of_range_is_rejected():
    with pytest.raises(ConfigError, match="reuse"):
        build_specs(checked_args(VECADD, "--reuse", "1.5"))
