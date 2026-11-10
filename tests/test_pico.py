"""Tests for gasp.pico — per-kernel costs and the fallback."""

import json
from unittest.mock import MagicMock, patch

import pytest

from gasp.pico import (
    PICO_ENV,
    PicoInput,
    parse_total_cost,
    perform_pico_analysis,
    pico_command,
)

KERNELS = [
    ("cname", "1"),
    ("ns::nsk", "2"),
    ("plain", "3"),
    ("tk<float>", "4"),
    ("tk<int>", "5"),
]


def pico_json(kernels=KERNELS) -> str:
    return json.dumps(
        {"kernels": [{"kernel_name": n, "total_cost": c} for n, c in kernels]}
    )


@pytest.mark.parametrize(
    "kernel,expected",
    [("_Z5plainPfS_", 3), ("_ZN2ns3nskEPf", 2), ("cname", 1)],
)
def test_parse_total_cost_selects_one_kernel(kernel, expected):
    assert parse_total_cost(pico_json(), kernel) == expected


def test_parse_total_cost_no_match_raises():
    with pytest.raises(ValueError):
        parse_total_cost(pico_json(), "_Z7missingPf")


def test_parse_total_cost_several_matches_raises_instead_of_summing():
    with pytest.raises(ValueError):
        parse_total_cost(pico_json(), "_Z2tkIfEvPT_")


def test_parse_total_cost_symbolic_raises():
    raw = pico_json([("plain", "Σ_{i | 0 ≤ i ≤ 7} 96")])
    with pytest.raises(ValueError):
        parse_total_cost(raw, "_Z5plainPfS_")


def test_parse_total_cost_missing_cost_raises():
    raw = json.dumps({"kernels": [{"kernel_name": "plain"}]})
    with pytest.raises(ValueError):
        parse_total_cost(raw, "_Z5plainPfS_")


@pytest.fixture
def pico_input(monkeypatch):
    monkeypatch.delenv(PICO_ENV, raising=False)
    return PicoInput(
        input_file="kernels.cu",
        kernel_name="_Z5plainPfS_",
        block_dim=(256,),
        grid_dim=(10,),
        problem_size=2560,
        bank_count=None,
        params={},
        logger=MagicMock(),
    )


def analyze(pico_input, result):
    with (
        patch("gasp.pico.shutil.which", return_value="/usr/bin/faial-cost"),
        patch("gasp.pico.subprocess.run", return_value=result),
    ):
        return perform_pico_analysis(pico_input)


def test_analysis_reads_the_selected_kernel(pico_input):
    output = analyze(pico_input, MagicMock(returncode=0, stdout=pico_json()))

    assert output.error is False
    assert output.uncoalesced_count == 3
    assert output.memory_accesses_count == 3


def test_analysis_falls_back_when_pico_fails(pico_input):
    output = analyze(pico_input, MagicMock(returncode=1, stdout="", stderr="rejected"))

    assert output.error is True
    assert output.uncoalesced_count == 0


def test_analysis_falls_back_on_a_symbolic_cost(pico_input):
    raw = pico_json([("plain", "Σ_{i | 0 ≤ i ≤ 7} 96")])
    output = analyze(pico_input, MagicMock(returncode=0, stdout=raw))

    assert output.error is True
    pico_input.logger.warning.assert_called_once()


def test_analysis_falls_back_without_pico(pico_input):
    with patch("gasp.pico.shutil.which", return_value=None):
        assert perform_pico_analysis(pico_input).error is True


def test_command_for_a_1d_launch_without_params(pico_input):
    assert pico_command(pico_input, "ua") == [
        "faial-cost",
        "-m",
        "ua",
        "-b",
        "256",
        "-g",
        "10",
        "--json",
        "kernels.cu",
    ]


def test_command_uses_the_path_from_the_environment(pico_input, monkeypatch):
    monkeypatch.setenv(PICO_ENV, "/opt/faial/bin/faial-cost")

    assert pico_command(pico_input, "ua")[0] == "/opt/faial/bin/faial-cost"


def test_command_ignores_an_empty_environment_path(pico_input, monkeypatch):
    monkeypatch.setenv(PICO_ENV, "")

    assert pico_command(pico_input, "ua")[0] == "faial-cost"


def test_analysis_looks_up_the_path_from_the_environment(pico_input, monkeypatch):
    monkeypatch.setenv(PICO_ENV, "/missing/faial-cost")
    with patch("gasp.pico.shutil.which", return_value=None) as which:
        assert perform_pico_analysis(pico_input).error is True

    which.assert_called_once_with("/missing/faial-cost")
    assert PICO_ENV in pico_input.logger.error.call_args.args[0]


def test_command_passes_dims_as_lists(pico_input):
    pico_input.block_dim = (16, 16)
    pico_input.grid_dim = (64, 64, 2)
    cmd = pico_command(pico_input, "ua")

    assert cmd[cmd.index("-b") + 1] == "[16,16]"
    assert cmd[cmd.index("-g") + 1] == "[64,64,2]"


def test_command_repeats_p_for_each_param(pico_input):
    pico_input.params = {"M": 1024, "N": 512, "K": 256}
    cmd = pico_command(pico_input, "count")

    assert cmd.count("-p") == 3
    pairs = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-p"]
    assert pairs == ["M=1024", "N=512", "K=256"]
    assert cmd[-2:] == ["--json", "kernels.cu"]


def test_analysis_runs_the_built_command(pico_input):
    pico_input.block_dim = (16, 16)
    pico_input.grid_dim = (64, 64)
    pico_input.params = {"M": 1024}

    result = MagicMock(returncode=0, stdout=pico_json())
    with (
        patch("gasp.pico.shutil.which", return_value="/usr/bin/faial-cost"),
        patch("gasp.pico.subprocess.run", return_value=result) as run,
    ):
        perform_pico_analysis(pico_input)

    for call in run.call_args_list:
        cmd = call.args[0]
        assert cmd == pico_command(pico_input, cmd[cmd.index("-m") + 1])
