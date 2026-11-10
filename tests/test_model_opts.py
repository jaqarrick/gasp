"""Tests for gasp.model_opts — the --model-opts names."""

import pytest

from gasp.model_opts import (
    NAMES,
    ModelOptions,
    ModelOptsError,
    format_model_opts,
    parse_model_opts,
)


def test_names_are_the_option_fields():
    assert NAMES == (
        "mem-classes",
        "amat-mem-l",
        "dram-floor",
        "l2-floor",
        "l1-floor",
        "shared-floor",
    )


def test_no_values_turn_nothing_on():
    assert parse_model_opts([]) == ModelOptions()


def test_one_name():
    assert parse_model_opts(["mem-classes"]) == ModelOptions(mem_classes=True)


def test_comma_lists_and_repeats_add_up():
    options = parse_model_opts(["mem-classes, amat-mem-l", "dram-floor"])

    assert options == ModelOptions(mem_classes=True, amat_mem_l=True, dram_floor=True)


def test_a_name_given_twice_is_on_once():
    assert parse_model_opts(["l1-floor", "l1-floor"]) == ModelOptions(l1_floor=True)


def test_all_turns_every_option_on():
    options = parse_model_opts(["all"])

    assert all(vars(options).values())


@pytest.mark.parametrize(
    "values", [["bogus"], ["mem_classes"], [""], ["mem-classes,"], ["all=on"]]
)
def test_parse_rejects(values):
    with pytest.raises(ModelOptsError):
        parse_model_opts(values)


def test_unknown_name_lists_the_choices():
    with pytest.raises(ModelOptsError, match="mem-classes.*all"):
        parse_model_opts(["bogus"])


@pytest.mark.parametrize(
    "values,label",
    [
        ([], "none"),
        (["dram-floor"], "dram-floor"),
        (["dram-floor", "mem-classes"], "mem-classes, dram-floor"),
        (["all"], "all"),
    ],
)
def test_format_model_opts(values, label):
    assert format_model_opts(parse_model_opts(values)) == label
