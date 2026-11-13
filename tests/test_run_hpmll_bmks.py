"""Tests for the pure parts of scripts/run/run_hpmll_bmks.py."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run" / "run_hpmll_bmks.py"
spec = importlib.util.spec_from_file_location("run_hpmll_bmks", SCRIPT)
hp = importlib.util.module_from_spec(spec)
# Register before executing: @dataclass looks the module up in sys.modules.
sys.modules["run_hpmll_bmks"] = hp
spec.loader.exec_module(hp)

HPMLL = REPO / "hardware" / "microbenchmarks" / "hpmll"

# What each benchmark prints (format strings from the hpmll sources, filled in).
PRINTED = {
    "alu_lat_float": (
        "fp32 latency = 4.012207 (clk)\nTotal Clk number = 65736 \n",
        4.012207,
    ),
    "alu_lat_double": ("fp64 latency = 8.000000 (clk)\n", 8.0),
    "alu_lat_int32": ("int32 latency = 5.003418 (clk)\n", 5.003418),
    "l1_lat": (
        "L1 Latency  =      33.0254 cycles\nTotal Clk number = 1082195 \n",
        33.0254,
    ),
    "l2_lat": ("L2 Latency =     218.6106 cycles \n", 218.6106),
    "mem_lat": ("Mem latency =     406.1234 cycles \n", 406.1234),
    "shared_lat": ("Shared Memory Latency  = 23.000977 cycles\n", 23.000977),
    "shared_bw": ("Shared Memory Bandwidth = 127.873016 (byte/clk/SM)\n", 127.873016),
    "l1_bw_32f": ("L1 bandwidth = 63.700001 (byte/clk/SM)\n", 63.700001),
    "l1_bw_64f": ("L1 bandwidth = 64.1 (byte/clk/SM)\n", 64.1),
    "l1_bw_128": ("L1 bandwidth = 121.2 (byte/clk/SM)\n", 121.2),
    "l2_bw_32f": ("L2 bandwidth = 1622.199951 (byte/cycle)\n", 1622.199951),
    "l2_bw_64f": ("L2 bandwidth = 1700.5 (byte/cycle)\n", 1700.5),
    "l2_bw_128": ("L2 bandwidth = 1708.0 (byte/cycle)\n", 1708.0),
    "mem_bw": ("Mem BW= 421.5 (Byte/Clk)\nMem BW= 585.532410 (GB/sec)\n", 585.53241),
}

NCU_CSV = """==PROF== Connected to process 1234 (/x/l2_bw_32f)
==PROF== Profiling "init_data": 0%....50%....100% - 1 pass
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1234","l2_bw_32f","127.0.0.1","init_data","1","7","(1024, 1, 1)","(216, 1, 1)","0","7.5","Command line profiler metrics","sm__cycles_elapsed.avg.per_second","cycle/second","1,545,000,000"
"1","1234","l2_bw_32f","127.0.0.1","l2_bw","1","7","(1024, 1, 1)","(216, 1, 1)","0","7.5","Command line profiler metrics","gpc__cycles_elapsed.max","cycle","1,234,567"
"1","1234","l2_bw_32f","127.0.0.1","l2_bw","1","7","(1024, 1, 1)","(216, 1, 1)","0","7.5","Command line profiler metrics","sm__cycles_elapsed.avg.per_second","cycle/second","1,365,000,000"
"2","1234","l2_bw_32f","127.0.0.1","l2_bw","1","7","(1024, 1, 1)","(216, 1, 1)","0","7.5","Command line profiler metrics","sm__cycles_elapsed.avg.per_second","cycle/nsecond","1.37"
"3","1234","l2_bw_32f","127.0.0.1","l2_bw","1","7","(1024, 1, 1)","(216, 1, 1)","0","7.5","Command line profiler metrics","lts__t_sector_hit_rate.pct","%","99.87"
"""


def test_every_benchmark_has_a_printed_example():
    assert set(PRINTED) == set(hp.BY_NAME)


@pytest.mark.parametrize("name", sorted(PRINTED))
def test_printed_value_is_parsed(name):
    text, expected = PRINTED[name]
    assert hp.parse_printed(hp.BY_NAME[name], text) == pytest.approx(expected)


def test_mem_bw_reads_the_gb_per_s_line_not_bytes_per_clock():
    text, _ = PRINTED["mem_bw"]
    assert hp.parse_printed(hp.BY_NAME["mem_bw"], text) == pytest.approx(585.53241)


def test_missing_result_line_is_an_error():
    with pytest.raises(hp.BenchmarkError):
        hp.parse_printed(hp.BY_NAME["l1_lat"], "GPUassert: out of memory\n")


@pytest.mark.skipif(
    not (HPMLL / "RegularUnits").is_dir(), reason="hpmll submodule not checked out"
)
@pytest.mark.parametrize("name", sorted(PRINTED))
def test_pattern_matches_the_hpmll_source_format(name):
    """The regex must match the format string in the submodule's source, with
    the printf conversions filled in by a number."""
    bench = hp.BY_NAME[name]
    source = (HPMLL / bench.source).read_text()
    formats = re.findall(r'printf\("([^"]*)"', source)
    filled = [re.sub(r"%[-+ 0-9.]*[a-z]+", "12.5", f) for f in formats]
    assert any(re.search(bench.pattern, f) for f in filled), formats


def test_conversions():
    max_clock, clock = 1.545, 1.365
    assert hp.BY_NAME["shared_bw"].convert(64.0, clock, max_clock) == pytest.approx(2.0)
    assert hp.BY_NAME["l1_bw_128"].convert(128.0, clock, max_clock) == pytest.approx(
        4.0
    )
    # l2_bw printed bytes / (ns x max clock); the field wants bytes per measured cycle
    assert hp.BY_NAME["l2_bw_32f"].convert(1000.0, clock, max_clock) == pytest.approx(
        1000.0 * 1.545 / 1.365
    )
    # GB/s = bytes per ns
    assert hp.BY_NAME["mem_bw"].convert(585.5, clock, max_clock) == pytest.approx(
        585.5 / 1.365
    )
    assert hp.BY_NAME["l1_lat"].convert(33.0, clock, max_clock) == 33.0


def test_ncu_rows_skip_prof_lines_and_filter_by_kernel_and_metric():
    rows = hp.ncu_rows(NCU_CSV)
    assert len(rows) == 5
    clocks = hp.metric_values(rows, "l2_bw", hp.CLOCK_METRIC)
    assert clocks == pytest.approx(
        [1.365e9, 1.37e9]
    )  # init_data excluded, units converted
    assert hp.run_clock_ghz(rows, "l2_bw") == pytest.approx(1.3675)
    assert hp.metric_values(rows, "l2_bw", hp.L2_HIT) == pytest.approx([99.87])


def test_template_kernel_names_match():
    row = {
        "Kernel Name": "max_flops<float>",
        "Metric Name": hp.CLOCK_METRIC,
        "Metric Unit": "cycle/second",
        "Metric Value": "1365000000",
    }
    assert hp.metric_values([row], "max_flops", hp.CLOCK_METRIC) == [1.365e9]
    assert hp.metric_values([row], "max_flop", hp.CLOCK_METRIC) == []


def test_no_clock_rows_is_an_error():
    with pytest.raises(hp.BenchmarkError):
        hp.run_clock_ghz([], "l2_bw")


def test_unknown_unit_is_an_error():
    row = {
        "Kernel Name": "l2_bw",
        "Metric Name": hp.CLOCK_METRIC,
        "Metric Unit": "furlong",
    }
    with pytest.raises(hp.BenchmarkError):
        hp.metric_values([row], "l2_bw", hp.CLOCK_METRIC)


def test_makefile_flags(tmp_path):
    makefile = tmp_path / "Makefile"
    makefile.write_text(
        "GENCODE_SM70 := -gencode=arch=compute_70\n"
        "NVCC_FLGAS = -Xptxas -dlcm=cg -Xptxas -dscm=wt\n"
    )
    assert hp.makefile_flags(makefile) == ["-Xptxas", "-dlcm=cg", "-Xptxas", "-dscm=wt"]
    makefile.write_text("NVCC_FLGAS = #-Xptxas -dlcm=cv\n")
    assert hp.makefile_flags(makefile) == []
    assert hp.makefile_flags(tmp_path / "missing") == []


def test_checks():
    mem_lat = hp.BY_NAME["mem_lat"].check
    assert mem_lat.failure(2.0) is None
    assert "> 10%" in mem_lat.failure(48.0)
    l1 = hp.BY_NAME["l1_bw_32f"].check
    assert l1.failure(99.5) is None
    assert "< 95%" in l1.failure(80.0)


def test_mem_bw_skipped_without_12_gib():
    bench = hp.BY_NAME["mem_bw"]
    small = hp.Device("NVIDIA GeForce RTX 2080 Ti", "595", 10 * 1024**3, 1.545)
    big = hp.Device("NVIDIA A100", "595", 39 * 1024**3, 1.41)
    assert "12 GiB" in hp.skip_reason(bench, small)
    assert hp.skip_reason(bench, big) is None
    assert hp.skip_reason(hp.BY_NAME["l1_lat"], small) is None


def result(name, values, invalid=None):
    bench = hp.BY_NAME[name]
    return hp.Result(
        bench,
        values,
        [1.365] * len(values),
        values,
        [""] * len(values),
        invalid=invalid,
    )


def test_field_value_takes_highest_valid_shape():
    results = [
        result("l2_bw_32f", [1600.0, 1610.0, 1620.0]),
        result("l2_bw_64f", [1700.0, 1705.0, 1710.0]),
        result("l2_bw_128", [1900.0, 1900.0, 1900.0], invalid="spilled"),
    ]
    value, chosen = hp.field_value(results)
    assert value == 1705.0
    assert chosen.bench.name == "l2_bw_64f"


def test_field_value_none_when_nothing_valid():
    assert hp.field_value([result("mem_lat", [250.0], invalid="hit L2")]) is None
    assert hp.field_value([]) is None


def test_summarize_single_run_has_zero_std():
    assert hp.summarize([4.0]) == {
        "median": 4.0,
        "mean": 4.0,
        "std": 0.0,
        "min": 4.0,
        "max": 4.0,
    }


def test_run_json_has_the_fields_build_profile_reads():
    results = [result("l1_bw_32f", [2.0, 2.0]), result("l1_bw_128", [3.8, 3.9])]
    value, chosen = hp.field_value(results)
    context = {
        "gpu": "rtx2080ti",
        "gpu_name": "NVIDIA GeForce RTX 2080 Ti",
        "arch": "sm_75",
        "max_sm_clock_ghz": 1.545,
        "hpmll_commit": "ded6f57",
        "nvcc_version": "12.4",
        "driver": "595.104.02",
        "git_commit": "abc1234",
        "timestamp": "2026-11-13T00:00:00Z",
    }
    run = hp.run_json("l1_sectors_per_cycle_per_sm", results, chosen, value, context)
    assert run["parameter"] == "l1_sectors_per_cycle_per_sm"
    assert run["value"] == pytest.approx(3.85)
    assert run["clock"]["sm_clock_ghz"] == 1.365
    assert set(run["candidates"]) == {"l1_bw_32f", "l1_bw_128"}
    assert run["benchmark"] == "hpmll/RegularUnits/l1_bw_128/l1_bw_128.cu"


def test_every_field_is_a_profile_field():
    profile = REPO / "hardware" / "profiles" / "rtx2080ti.json"
    profiled = json.loads(profile.read_text())["profiled"]
    assert set(hp.FIELDS) <= set(profiled)
