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
    ProfileInput,
    build_executable,
    dump_kernel_text,
    extract_sass_instructions,
    measure,
    parse_ncu_csv,
    run_ncu,
    run_ncu_report,
)

PROFILE_CSV = Path("benchmarks/custom/vecadd.turing.N1000000.profile.csv")

NCU_PREAMBLE = (
    "==PROF== Connected to process 461639 (/tmp/gasp-profile-x/parallel)\n"
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
        input_file="benchmarks/custom/vecadd_stride.cu",
        arch="sm_80",
        kernel_name="_Z13vecadd_stridePfS_S_i",
        block_size=256,
        grid_size=3907,
        program_args=["1000000"],
        include_dirs=["inc"],
        extra_flags=["--use_fast_math"],
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
        build_executable(profile_input, 1, 1, "/tmp/serial")

    cmd = run.call_args.args[0]
    assert cmd[0] == "nvcc"
    assert "-arch=sm_80" in cmd
    assert "-Iinc" in cmd
    assert "--use_fast_math" in cmd
    assert "-DGASP_BLOCK_SIZE=1" in cmd
    assert "-DGASP_GRID_SIZE=1" in cmd
    assert cmd[-3:] == [profile_input.input_file, "-o", "/tmp/serial"]


def test_build_executable_failure_exits(profile_input):
    with (
        patch("gasp.profiler.subprocess.run", return_value=completed(returncode=1)),
        pytest.raises(SystemExit),
    ):
        build_executable(profile_input, 1, 1, "/tmp/serial")


def test_run_ncu_command(profile_input, logger):
    with patch(
        "gasp.profiler.subprocess.run",
        return_value=completed(NCU_PREAMBLE + PROFILE_CSV.read_text()),
    ) as run:
        run_ncu("/tmp/parallel", ["1000000"], profile_input.kernel_name, logger)

    cmd = run.call_args.args[0]
    assert cmd[0] == "ncu"
    assert "--csv" in cmd
    assert cmd[cmd.index("--kernel-name") + 1] == "_Z13vecadd_stridePfS_S_i"
    assert cmd[cmd.index("--kernel-name-base") + 1] == "mangled"
    assert cmd[cmd.index("--launch-count") + 1] == "1"
    metrics = cmd[cmd.index("--metrics") + 1].split(",")
    assert metrics == METRICS
    assert DURATION_METRIC in metrics and CYCLES_METRIC in metrics
    assert cmd[-2:] == ["/tmp/parallel", "1000000"]


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
        text = dump_kernel_text("/tmp/serial", profile_input.kernel_name, logger)

    assert run.call_args.args[0] == [
        "cuobjdump",
        "-sass",
        "-fun",
        "_Z13vecadd_stridePfS_S_i",
        "/tmp/serial",
    ]
    assert text.splitlines()[0] == "/*0000*/ MOV R1, c[0x0][0x28] ;"


def test_dump_kernel_text_without_cuobjdump_warns(profile_input, logger):
    with patch("gasp.profiler.shutil.which", return_value=None):
        dump_kernel_text("/tmp/serial", profile_input.kernel_name, logger)

    logger.warning.assert_called_once()


def test_run_ncu_report_command(profile_input, logger):
    with patch(
        "gasp.profiler.subprocess.run", return_value=completed("Section: ...")
    ) as run:
        report = run_ncu_report(
            "/tmp/parallel", ["1000000"], profile_input.kernel_name, logger
        )

    cmd = run.call_args.args[0]
    assert report == "Section: ..."
    assert cmd[cmd.index("--set") + 1] == "full"
    assert "--csv" not in cmd
    assert cmd[cmd.index("--kernel-name") + 1] == "_Z13vecadd_stridePfS_S_i"
    assert cmd[-2:] == ["/tmp/parallel", "1000000"]


def test_run_ncu_report_failure_does_not_exit(profile_input, logger):
    with patch("gasp.profiler.subprocess.run", return_value=completed(returncode=1)):
        report = run_ncu_report("/tmp/parallel", [], profile_input.kernel_name, logger)

    assert report.startswith("ncu --set full failed")
    logger.warning.assert_called_once()


def launch_metrics(grid_size: int, block_size: int) -> dict[str, float]:
    return {
        DURATION_METRIC: 21824.0,
        CYCLES_METRIC: 26325.0,
        GRID_SIZE_METRIC: float(grid_size),
        BLOCK_SIZE_METRIC: float(block_size),
    }


def test_measure_matching_launch(profile_input):
    with (
        patch("gasp.profiler.build_executable"),
        patch("gasp.profiler.run_ncu", return_value=launch_metrics(1, 1)),
    ):
        m = measure(profile_input, 1, 1, "/tmp", "serial")

    assert m.cycles == 26325.0
    profile_input.logger.error.assert_not_called()


def test_measure_program_ignoring_launch_config_exits(profile_input):
    # vecadd.cu computes its own grid, so the one-thread build still launches 3907 blocks
    with (
        patch("gasp.profiler.build_executable"),
        patch("gasp.profiler.run_ncu", return_value=launch_metrics(3907, 256)),
        pytest.raises(SystemExit),
    ):
        measure(profile_input, 1, 1, "/tmp", "serial")

    profile_input.logger.error.assert_called_once()


def test_measure_grid_size_mismatch_exits(profile_input):
    # --grid-size 512 while the program launches ceil(N / block)
    with (
        patch("gasp.profiler.build_executable"),
        patch("gasp.profiler.run_ncu", return_value=launch_metrics(3907, 256)),
        pytest.raises(SystemExit),
    ):
        measure(profile_input, 256, 512, "/tmp", "parallel")


def test_measure_missing_launch_metric_exits(profile_input):
    metrics = launch_metrics(1, 1)
    del metrics[GRID_SIZE_METRIC]
    with (
        patch("gasp.profiler.build_executable"),
        patch("gasp.profiler.run_ncu", return_value=metrics),
        pytest.raises(SystemExit),
    ):
        measure(profile_input, 1, 1, "/tmp", "serial")
