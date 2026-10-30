"""Tests for gasp.profiler — ncu output parsing and command construction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gasp.profiler import (
    CYCLES_METRIC,
    DURATION_METRIC,
    METRICS,
    ProfileInput,
    build_executable,
    parse_ncu_csv,
    run_ncu,
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
    return MagicMock(spec=["info", "error", "warning"])


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
