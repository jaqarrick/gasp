"""Tests for gasp.profiler — ncu output parsing and command construction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gasp.profiler import (
    BLOCK_SIZE_METRIC,
    CYCLES_METRIC,
    DURATION_METRIC,
    GRID_SIZE_METRIC,
    METRICS,
    KernelMeasurement,
    ProfileInput,
    ProfileOutput,
    build_executable,
    dump_kernel_text,
    extract_sass_instructions,
    measure,
    parse_ncu_csv,
    profile_kernel,
    run_ncu,
    run_ncu_report,
)

PROFILE_CSV = (
    Path(__file__).parent / "fixtures" / "ncu" / "vecadd.turing.N1000000.profile.csv"
)

NCU_PREAMBLE = (
    "==PROF== Connected to process 461639 (/tmp/gasp-profile-x/kernel)\n"
    '==PROF== Profiling "vecadd" - 0: 0%....50%....100% - 9 passes\n'
    "N=1000000, block=256, grid=3907, errors=0\n"
    "==PROF== Disconnected from process 461639\n"
)


@pytest.fixture
def logger():
    logger = MagicMock(spec=["info", "error", "warning", "trace", "trace_enabled"])
    logger.trace_enabled.return_value = False
    return logger


@pytest.fixture
def profile_input(logger):
    return ProfileInput(
        name="vecadd_stride",
        input_file="benchmarks/vecadd/vecadd_stride.cu",
        arch="sm_80",
        kernel_name="_Z13vecadd_stridePfS_S_i",
        block_dim=(256,),
        grid_dim=(3907,),
        program_args=["1000000"],
        include_dirs=["inc"],
        extra_flags=["--use_fast_math"],
        repeats=3,
        logger=logger,
    )


def completed(stdout: str = "", returncode: int = 0) -> MagicMock:
    return MagicMock(returncode=returncode, stdout=stdout, stderr="")


def test_parse_ncu_csv_skips_preamble(logger):
    metrics = parse_ncu_csv(NCU_PREAMBLE + PROFILE_CSV.read_text(), logger)

    assert metrics["gpu__time_duration.avg"] == 21824.0
    assert metrics["smsp__inst_executed.sum"] == 468786.0
    assert metrics["lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"] == 0.0
    assert metrics["sm__warps_active.avg.pct_of_peak_sustained_active"] == 85.53
    assert len(metrics) == 9


def test_parse_ncu_csv_without_metrics_exits(logger):
    with pytest.raises(SystemExit):
        parse_ncu_csv(NCU_PREAMBLE, logger)
    logger.error.assert_called_once()


def test_parse_ncu_csv_header_only_exits(logger):
    header = PROFILE_CSV.read_text().splitlines()[0]
    with pytest.raises(SystemExit):
        parse_ncu_csv(NCU_PREAMBLE + header + "\n", logger)


def test_build_executable_command(profile_input):
    with patch("gasp.profiler.subprocess.run", return_value=completed()) as run:
        build_executable(profile_input, "/tmp/kernel")

    cmd = run.call_args.args[0]
    assert cmd[0] == "nvcc"
    assert "-arch=sm_80" in cmd
    assert "-Iinc" in cmd
    assert "--use_fast_math" in cmd
    assert "-DGASP_BLOCK_SIZE=256" in cmd
    assert "-DGASP_GRID_SIZE=3907" in cmd
    assert "-DGASP_BLOCK_DIM_X=256" in cmd
    assert "-DGASP_BLOCK_DIM_Y=1" in cmd
    assert "-DGASP_GRID_DIM_Z=1" in cmd
    assert cmd[-3:] == [profile_input.input_file, "-o", "/tmp/kernel"]


def test_build_executable_defines_2d_dims(profile_input):
    profile_input.block_dim = (16, 16)
    profile_input.grid_dim = (64, 64)
    with patch("gasp.profiler.subprocess.run", return_value=completed()) as run:
        build_executable(profile_input, "/tmp/kernel")

    cmd = run.call_args.args[0]
    assert "-DGASP_BLOCK_SIZE=256" in cmd
    assert "-DGASP_GRID_SIZE=4096" in cmd
    for macro in ("BLOCK_DIM_X=16", "BLOCK_DIM_Y=16", "BLOCK_DIM_Z=1"):
        assert f"-DGASP_{macro}" in cmd
    for macro in ("GRID_DIM_X=64", "GRID_DIM_Y=64", "GRID_DIM_Z=1"):
        assert f"-DGASP_{macro}" in cmd


def test_measure_2d_launch_compares_totals(profile_input):
    profile_input.block_dim = (16, 16)
    profile_input.grid_dim = (64, 64)
    with patch("gasp.profiler.run_ncu", return_value=launch_metrics(4096, 256)):
        measure(profile_input, "/tmp/kernel", trace=False)

    profile_input.logger.error.assert_not_called()


def test_build_executable_failure_exits(profile_input):
    with (
        patch("gasp.profiler.subprocess.run", return_value=completed(returncode=1)),
        pytest.raises(SystemExit),
    ):
        build_executable(profile_input, "/tmp/kernel")


def test_run_ncu_command(profile_input, logger):
    with patch(
        "gasp.profiler.subprocess.run",
        return_value=completed(NCU_PREAMBLE + PROFILE_CSV.read_text()),
    ) as run:
        run_ncu("/tmp/kernel", ["1000000"], profile_input.kernel_name, logger)

    cmd = run.call_args.args[0]
    assert cmd[0] == "ncu"
    assert "--csv" in cmd
    assert cmd[cmd.index("--kernel-name") + 1] == "_Z13vecadd_stridePfS_S_i"
    assert cmd[cmd.index("--kernel-name-base") + 1] == "mangled"
    assert cmd[cmd.index("--launch-count") + 1] == "1"
    metrics = cmd[cmd.index("--metrics") + 1].split(",")
    assert metrics == METRICS
    assert DURATION_METRIC in metrics and CYCLES_METRIC in metrics
    assert CYCLES_METRIC == "gpc__cycles_elapsed.max"
    assert cmd[-2:] == ["/tmp/kernel", "1000000"]


CUOBJDUMP_SASS = """\

Fatbin elf code:
================
arch = sm_75
code version = [1,7]

\tcode for sm_75
\t\tFunction : _Z13vecadd_stridePfS_S_i
\t.headerflags\t@"EF_CUDA_SM75 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM75)"
        /*0000*/                   MOV R1, c[0x0][0x28] ;                        /* 0x00000a0000017a02 */
                                                                                 /* 0x000fc40000000f00 */
        /*0010*/                   S2R R6, SR_CTAID.X ;                          /* 0x0000000000067919 */
                                                                                 /* 0x000e280000002500 */
.L_x_0:
        /*0020*/              @!P0 EXIT ;                                        /* 0x000000000000894d */
                                                                                 /* 0x000fea0003800000 */
\t\t..........
"""


def test_extract_sass_instructions_keeps_only_instructions():
    assert extract_sass_instructions(CUOBJDUMP_SASS) == [
        "/*0000*/ MOV R1, c[0x0][0x28] ;",
        "/*0010*/ S2R R6, SR_CTAID.X ;",
        "/*0020*/ @!P0 EXIT ;",
    ]


def test_dump_kernel_text_command(profile_input, logger):
    with (
        patch("gasp.profiler.shutil.which", return_value="/usr/bin/cuobjdump"),
        patch(
            "gasp.profiler.subprocess.run", return_value=completed(CUOBJDUMP_SASS)
        ) as run,
    ):
        text = dump_kernel_text("/tmp/kernel", profile_input.kernel_name, logger)

    assert run.call_args.args[0] == [
        "cuobjdump",
        "-sass",
        "-fun",
        "_Z13vecadd_stridePfS_S_i",
        "/tmp/kernel",
    ]
    assert text.splitlines()[0] == "/*0000*/ MOV R1, c[0x0][0x28] ;"


def test_dump_kernel_text_without_cuobjdump_warns(profile_input, logger):
    with patch("gasp.profiler.shutil.which", return_value=None):
        dump_kernel_text("/tmp/kernel", profile_input.kernel_name, logger)

    logger.warning.assert_called_once()


def test_run_ncu_report_command(profile_input, logger):
    with patch(
        "gasp.profiler.subprocess.run", return_value=completed("Section: ...")
    ) as run:
        report = run_ncu_report(
            "/tmp/kernel", ["1000000"], profile_input.kernel_name, logger
        )

    cmd = run.call_args.args[0]
    assert report == "Section: ..."
    assert cmd[cmd.index("--set") + 1] == "full"
    assert "--csv" not in cmd
    assert cmd[cmd.index("--kernel-name") + 1] == "_Z13vecadd_stridePfS_S_i"
    assert cmd[-2:] == ["/tmp/kernel", "1000000"]


def test_run_ncu_report_failure_does_not_exit(profile_input, logger):
    with patch("gasp.profiler.subprocess.run", return_value=completed(returncode=1)):
        report = run_ncu_report("/tmp/kernel", [], profile_input.kernel_name, logger)

    assert report.startswith("ncu --set full failed")
    logger.warning.assert_called_once()


def launch_metrics(grid_size: int, block_size: int, cycles: float = 26325.0):
    return {
        DURATION_METRIC: 21824.0,
        CYCLES_METRIC: cycles,
        GRID_SIZE_METRIC: float(grid_size),
        BLOCK_SIZE_METRIC: float(block_size),
    }


def test_measure_matching_launch(profile_input):
    with patch("gasp.profiler.run_ncu", return_value=launch_metrics(3907, 256)):
        m = measure(profile_input, "/tmp/kernel", trace=False)

    assert m.cycles == 26325.0
    profile_input.logger.error.assert_not_called()


def test_measure_launch_mismatch_exits(profile_input):
    # a program that launches something other than the requested <<<3907, 256>>>
    with (
        patch("gasp.profiler.run_ncu", return_value=launch_metrics(512, 256)),
        pytest.raises(SystemExit),
    ):
        measure(profile_input, "/tmp/kernel", trace=False)

    profile_input.logger.error.assert_called_once()


def test_measure_block_size_mismatch_exits(profile_input):
    with (
        patch("gasp.profiler.run_ncu", return_value=launch_metrics(3907, 128)),
        pytest.raises(SystemExit),
    ):
        measure(profile_input, "/tmp/kernel", trace=False)


def test_measure_missing_launch_metric_exits(profile_input):
    metrics = launch_metrics(3907, 256)
    del metrics[GRID_SIZE_METRIC]
    with (
        patch("gasp.profiler.run_ncu", return_value=metrics),
        pytest.raises(SystemExit),
    ):
        measure(profile_input, "/tmp/kernel", trace=False)


def profile(profile_input, metrics, trace_enabled=False):
    profile_input.logger.trace_enabled.return_value = trace_enabled
    with (
        patch("gasp.profiler.check_packages"),
        patch("gasp.profiler.build_executable") as build,
        patch("gasp.profiler.run_ncu", side_effect=metrics) as ncu,
        patch("gasp.profiler.trace_measurement") as trace,
    ):
        output = profile_kernel(profile_input)
    return output, build, ncu, trace


def test_profile_kernel_builds_once_and_runs_repeats(profile_input):
    metrics = [launch_metrics(3907, 256, c) for c in (300.0, 100.0, 200.0)]
    output, build, ncu, _ = profile(profile_input, metrics)

    assert build.call_count == 1
    assert ncu.call_count == 3
    assert [run.cycles for run in output.runs] == [300.0, 100.0, 200.0]


def test_profile_kernel_traces_only_the_first_run(profile_input):
    metrics = [launch_metrics(3907, 256) for _ in range(3)]
    _, _, _, trace = profile(profile_input, metrics, trace_enabled=True)

    assert trace.call_count == 1


def test_profile_kernel_checks_the_launch_on_every_run(profile_input):
    metrics = [launch_metrics(3907, 256), launch_metrics(512, 256)]
    with pytest.raises(SystemExit):
        profile(profile_input, metrics)


def test_profile_output_median_and_spread():
    output = ProfileOutput(
        runs=[
            KernelMeasurement(duration_ns=1.0, cycles=c, metrics={})
            for c in (300.0, 100.0, 200.0)
        ]
    )

    assert output.cycles == 200.0
    assert output.cycles_min == 100.0
    assert output.cycles_max == 300.0
