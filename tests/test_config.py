"""Tests for gasp.cli.config — kernel specs from the command line and TOML."""

import pytest

from gasp.cli.config import (
    ConfigError,
    load_config,
    parse_input,
    specs_from_inputs,
)
from gasp.cli.constants import DEFAULT_BLOCK_SIZE, DEFAULT_PROBLEM_SIZE


@pytest.fixture
def cu(tmp_path):
    for name in ("good.cu", "poor.cu"):
        (tmp_path / name).write_text("")
    return tmp_path


def write_config(cu, text: str) -> str:
    path = cu / "kernels.toml"
    path.write_text(text.replace("DIR", str(cu)))
    return str(path)


TWO_KERNELS = """
[defaults]
problem_size = 1000
block_size = 100
loop_iterations = 8

[[kernel]]
name = "good"
file = "DIR/good.cu"

[[kernel]]
name = "poor"
file = "DIR/poor.cu"
loop_iterations = 16
block_size = 200
"""


def test_kernel_entry_wins_over_defaults(cu):
    good, poor = load_config(write_config(cu, TWO_KERNELS))

    assert good.loop_iterations == 8
    assert poor.loop_iterations == 16
    assert good.block_size == 100
    assert poor.block_size == 200


def test_grid_is_derived_from_each_kernels_settings(cu):
    good, poor = load_config(write_config(cu, TWO_KERNELS))

    assert good.grid_size == 10
    assert poor.grid_size == 5


def test_builtin_defaults_apply_last(cu):
    [spec] = load_config(write_config(cu, '[[kernel]]\nfile = "DIR/good.cu"\n'))

    assert spec.problem_size == DEFAULT_PROBLEM_SIZE
    assert spec.block_size == DEFAULT_BLOCK_SIZE
    assert spec.loop_iterations is None
    assert spec.program_args == []


def test_explicit_grid_is_kept(cu):
    text = '[defaults]\ngrid_size = 7\n[[kernel]]\nfile = "DIR/good.cu"\n'
    [spec] = load_config(write_config(cu, text))

    assert spec.grid_size == 7


def test_default_names(cu):
    text = """
[[kernel]]
file = "DIR/good.cu"

[[kernel]]
file = "DIR/poor.cu"
kernel = "poor_kernel"
"""
    good, poor = load_config(write_config(cu, text))

    assert good.name == "good"
    assert poor.name == "poor_kernel"
    assert poor.kernel == "poor_kernel"


def test_file_and_kernel_can_come_from_defaults(cu):
    text = """
[defaults]
file = "DIR/good.cu"

[[kernel]]
kernel = "a"

[[kernel]]
kernel = "b"
"""
    a, b = load_config(write_config(cu, text))

    assert a.file == b.file == f"{cu}/good.cu"
    assert (a.name, b.name) == ("a", "b")


def test_list_settings(cu):
    text = """
[[kernel]]
file = "DIR/good.cu"
include_dirs = ["inc"]
nvcc_flags = ["--use_fast_math"]
program_args = ["1000000"]
"""
    [spec] = load_config(write_config(cu, text))

    assert spec.include_dirs == ["inc"]
    assert spec.nvcc_flags == ["--use_fast_math"]
    assert spec.program_args == ["1000000"]


@pytest.mark.parametrize(
    "text,message",
    [
        ('[[kernel]]\nfile = "DIR/good.cu"\nbogus = 1\n', "bogus"),
        ('[bogus]\n[[kernel]]\nfile = "DIR/good.cu"\n', "bogus"),
        ('[defaults]\nname = "a"\n[[kernel]]\nfile = "DIR/good.cu"\n', "name"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nblock_size = 0\n', "block_size"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nblock_size = -1\n', "block_size"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nblock_size = "8"\n', "block_size"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nblock_size = true\n', "block_size"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nblock_size = []\n', "block_size"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nblock_size = [1, 2, 3, 4]\n', "block_size"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nblock_size = [16, 0]\n', "block_size"),
        ('[[kernel]]\nfile = "DIR/good.cu"\ngrid_size = [16, "8"]\n', "grid_size"),
        (
            '[[kernel]]\nfile = "DIR/good.cu"\nblock_size = [16, 16]\n',
            "needs grid_size",
        ),
        ('[[kernel]]\nfile = "DIR/good.cu"\nfootprint = 0\n', "footprint"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nfootprint = 1.5\n', "footprint"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nparams = 8\n', "params"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nparams = { M = "8" }\n', "params"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nparams = { M = true }\n', "params"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nparams = { M = 1.5 }\n', "params"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nprogram_args = "8"\n', "program_args"),
        ('[[kernel]]\nfile = "DIR/good.cu"\nprogram_args = [8]\n', "program_args"),
        ('[[kernel]]\nname = "a"\n', "missing file"),
        ('[[kernel]]\nfile = "DIR/missing.cu"\n', "file not found"),
        ('[[kernel]]\nfile = "DIR/kernels.toml"\n', "expected a .cu file"),
        ("[defaults]\nblock_size = 8\n", "no \\[\\[kernel\\]\\]"),
        ('[kernel]\nfile = "DIR/good.cu"\n', "\\[\\[kernel\\]\\]"),
        ("kernel = [1]\n", "\\[\\[kernel\\]\\]"),
        ("not toml", "invalid TOML"),
    ],
)
def test_invalid_config_raises(cu, text, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(cu, text))


def test_dims_as_arrays(cu):
    text = """
[[kernel]]
file = "DIR/good.cu"
block_size = [16, 16]
grid_size = [64, 64]
"""
    [spec] = load_config(write_config(cu, text))

    assert (spec.block_dim, spec.grid_dim) == ((16, 16), (64, 64))
    assert (spec.block_size, spec.grid_size) == (256, 4096)


def test_integer_dims_are_one_dimensional(cu):
    good, _ = load_config(write_config(cu, TWO_KERNELS))

    assert (good.block_dim, good.grid_dim) == ((100,), (10,))


def test_one_element_block_array_still_derives_the_grid(cu):
    text = "[defaults]\nproblem_size = 1000\nblock_size = [100]\n"
    text += '[[kernel]]\nfile = "DIR/good.cu"\n'
    [spec] = load_config(write_config(cu, text))

    assert spec.grid_dim == (10,)


def test_params_merge_by_name_over_defaults(cu):
    text = """
[defaults]
params = { M = 1024, N = 1024, K = 1024 }

[[kernel]]
name = "a"
file = "DIR/good.cu"

[[kernel]]
name = "b"
file = "DIR/poor.cu"
params = { K = 512, "blockIdx.x" = 0 }
"""
    a, b = load_config(write_config(cu, text))

    assert a.params == {"M": 1024, "N": 1024, "K": 1024}
    assert b.params == {"M": 1024, "N": 1024, "K": 512, "blockIdx.x": 0}


def test_params_default_to_empty(cu):
    good, _ = load_config(write_config(cu, TWO_KERNELS))

    assert good.params == {}


def test_loop_counts_on_an_entry(cu):
    text = """
[defaults]
loop_iterations = 8

[[kernel]]
name = "a"
file = "DIR/good.cu"
loop_counts = [64, 0, 0]

[[kernel]]
name = "b"
file = "DIR/poor.cu"
"""
    a, b = load_config(write_config(cu, text))

    assert a.loop_counts == [64, 0, 0]
    assert a.loop_iterations == 8
    assert b.loop_counts is None


def test_loop_counts_not_allowed_in_defaults(cu):
    text = '[defaults]\nloop_counts = [1]\n\n[[kernel]]\nfile = "DIR/good.cu"\n'
    with pytest.raises(ConfigError, match="unknown setting"):
        load_config(write_config(cu, text))


@pytest.mark.parametrize("value", ["[-1]", "[1.5]", "8", '["8"]', "[true]"])
def test_invalid_loop_counts_raise(cu, value):
    text = f'[[kernel]]\nfile = "DIR/good.cu"\nloop_counts = {value}\n'
    with pytest.raises(ConfigError, match="loop_counts"):
        load_config(write_config(cu, text))


def test_command_line_has_no_loop_counts(cu):
    (spec,) = specs_from_inputs([(str(cu / "good.cu"), None)], {})
    assert spec.loop_counts is None


def test_invalid_entry_error_names_the_kernel(cu):
    text = '[[kernel]]\nname = "poor"\nfile = "DIR/good.cu"\nblock_size = 0\n'
    with pytest.raises(ConfigError, match="kernel 'poor'"):
        load_config(write_config(cu, text))


def test_duplicate_names_raise(cu):
    text = """
[[kernel]]
name = "same"
file = "DIR/good.cu"

[[kernel]]
name = "same"
file = "DIR/poor.cu"
"""
    with pytest.raises(ConfigError, match="same"):
        load_config(write_config(cu, text))


def test_missing_config_raises(cu):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(str(cu / "missing.toml"))


def test_parse_input_file_only(cu):
    assert parse_input(f"{cu}/good.cu") == (f"{cu}/good.cu", None)


def test_parse_input_with_selector(cu):
    assert parse_input(f"{cu}/good.cu::kernel") == (f"{cu}/good.cu", "kernel")


def test_parse_input_keeps_namespaces_in_the_selector(cu):
    assert parse_input(f"{cu}/good.cu::ns::kernel") == (f"{cu}/good.cu", "ns::kernel")


@pytest.mark.parametrize("value", ["good.cu::", "missing.cu", "kernels.txt"])
def test_parse_input_invalid(cu, value):
    with pytest.raises(ConfigError):
        parse_input(f"{cu}/{value}")


def test_specs_from_inputs_share_settings(cu):
    good, poor = specs_from_inputs(
        [(f"{cu}/good.cu", None), (f"{cu}/poor.cu", "selected")],
        {"problem_size": 1000, "block_size": 100, "program_args": ["1000"]},
    )

    assert (good.name, poor.name) == ("good", "selected")
    assert poor.kernel == "selected"
    assert good.grid_size == poor.grid_size == 10
    assert good.program_args == poor.program_args == ["1000"]


def test_specs_from_inputs_duplicate_names_point_to_config(cu):
    with pytest.raises(ConfigError, match="--config"):
        specs_from_inputs([(f"{cu}/good.cu", None), (f"{cu}/good.cu", None)], {})


def test_specs_from_inputs_rejects_non_positive_sizes(cu):
    with pytest.raises(ConfigError, match="block_size"):
        specs_from_inputs([(f"{cu}/good.cu", None)], {"block_size": 0})


def test_footprint_from_defaults_and_entries(cu):
    text = """
[defaults]
footprint = 12000000

[[kernel]]
file = "DIR/good.cu"

[[kernel]]
file = "DIR/poor.cu"
footprint = 24000000
"""
    good, poor = load_config(write_config(cu, text))

    assert good.footprint == 12_000_000
    assert poor.footprint == 24_000_000


def test_footprint_defaults_to_none(cu):
    good, _ = load_config(write_config(cu, TWO_KERNELS))

    assert good.footprint is None
