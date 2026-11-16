"""Tests for gasp.overrides — parsing --override and building the values."""

import pytest

from gasp.overrides import (
    GLOBAL_LOAD_REQUESTS,
    GLOBAL_LOAD_SECTORS,
    GLOBAL_LOADS_EXECUTED,
    GLOBAL_STORES_EXECUTED,
    GROUPS,
    INSTS_EXECUTED,
    L1_HIT_RATE_PCT,
    L2_READ_HITS,
    L2_READ_MISSES,
    PROFILED_METRICS,
    SFU_INSTS_EXECUTED,
    SHARED_LOAD_BANK_CONFLICTS,
    SHARED_STORE_BANK_CONFLICTS,
    ModelOverrides,
    OverrideError,
    build_overrides,
    format_overrides,
    parse_overrides,
    profiled_values,
    uses_profiled,
)
from gasp.profiler import METRICS

TOTAL_WARPS = 100


def metrics(**changes: float) -> dict[str, float]:
    values = {
        L1_HIT_RATE_PCT: 47.2,
        L2_READ_HITS: 20.0,
        L2_READ_MISSES: 60.0,
        GLOBAL_LOAD_SECTORS: 3200.0,
        GLOBAL_LOAD_REQUESTS: 100.0,
        SHARED_LOAD_BANK_CONFLICTS: 250.0,
        SHARED_STORE_BANK_CONFLICTS: 50.0,
        INSTS_EXECUTED: 1250.0,
        GLOBAL_LOADS_EXECUTED: 200.0,
        GLOBAL_STORES_EXECUTED: 100.0,
        SFU_INSTS_EXECUTED: 50.0,
    }
    return {**values, **changes}


# parsing


def test_parse_no_overrides():
    assert parse_overrides([]) == {}


def test_parse_several_groups():
    assert parse_overrides(["cache=profiled", "bank = crude"]) == {
        "cache": "profiled",
        "bank": "crude",
    }


def test_parse_all_names_every_group():
    assert parse_overrides(["all=profiled"]) == dict.fromkeys(GROUPS, "profiled")


@pytest.mark.parametrize(
    "values",
    [
        ["cache"],  # no source
        ["=profiled"],
        ["cache="],
        ["bogus=profiled"],  # unknown group
        ["cache=measured"],  # unknown source
        ["cache=crude"],  # cache has no crude level
        ["counts=crude"],
        ["all=crude"],
        ["all=static"],
        ["cache=profiled", "cache=profiled"],  # twice
        ["all=profiled", "bank=crude"],  # twice, through all
    ],
)
def test_parse_rejects(values):
    with pytest.raises(OverrideError):
        parse_overrides(values)


def test_uses_profiled():
    assert uses_profiled({"cache": "profiled"})
    assert not uses_profiled({"bank": "crude"})
    assert not uses_profiled({})


@pytest.mark.parametrize(
    "values,label",
    [
        ([], "none"),
        (["cache=static"], "none"),
        (["cache=profiled"], "cache=profiled"),
        (["bank=crude", "coalescing=crude"], "coalescing=crude, bank=crude"),
        (["all=profiled"], "all=profiled"),
    ],
)
def test_format_overrides(values, label):
    assert format_overrides(parse_overrides(values)) == label


# values


def test_every_profiled_metric_is_collected_by_run():
    assert set(PROFILED_METRICS) <= set(METRICS)


def test_no_choices_override_nothing():
    assert build_overrides({}, None, TOTAL_WARPS) == ModelOverrides()
    assert build_overrides({"cache": "static"}, metrics(), TOTAL_WARPS) == (
        ModelOverrides()
    )


def test_profiled_cache():
    o = build_overrides({"cache": "profiled"}, metrics(), TOTAL_WARPS)

    assert o == ModelOverrides(
        l1_hit_ratio=pytest.approx(0.472), l2_hit_ratio=pytest.approx(0.25)
    )


def test_profiled_cache_without_l2_lookups():
    m = metrics(**{L2_READ_HITS: 0.0, L2_READ_MISSES: 0.0})

    assert build_overrides({"cache": "profiled"}, m, TOTAL_WARPS).l2_hit_ratio == 0.0


def test_profiled_coalescing_is_lines_per_request():
    # 32 sectors per request = 8 lines of 4 sectors
    o = build_overrides({"coalescing": "profiled"}, metrics(), TOTAL_WARPS)

    assert o == ModelOverrides(average_trans_warp=8.0)


@pytest.mark.parametrize(
    "sectors,requests",
    [
        (200.0, 100.0),  # under one line per request
        (0.0, 0.0),  # no global loads
    ],
)
def test_profiled_coalescing_is_at_least_one(sectors, requests):
    m = metrics(**{GLOBAL_LOAD_SECTORS: sectors, GLOBAL_LOAD_REQUESTS: requests})

    o = build_overrides({"coalescing": "profiled"}, m, TOTAL_WARPS)

    assert o.average_trans_warp == 1.0


def test_profiled_bank_is_per_warp():
    o = build_overrides({"bank": "profiled"}, metrics(), TOTAL_WARPS)

    assert o == ModelOverrides(bank_conflict_count=3.0)


def test_profiled_counts_are_per_warp():
    o = build_overrides({"counts": "profiled"}, metrics(), TOTAL_WARPS)

    assert o == ModelOverrides(
        total_inst_count=12.5, mem_inst_count=3.0, sfu_inst_count=0.5
    )


def test_crude_needs_no_metrics():
    o = build_overrides({"coalescing": "crude", "bank": "crude"}, None, TOTAL_WARPS)

    assert o == ModelOverrides(average_trans_warp=1.0, bank_conflict_count=0.0)


def test_all_profiled_sets_every_field():
    o = build_overrides(parse_overrides(["all=profiled"]), metrics(), TOTAL_WARPS)

    assert None not in vars(o).values()


def test_missing_metric_is_named():
    m = metrics()
    del m[L1_HIT_RATE_PCT]

    with pytest.raises(OverrideError, match=L1_HIT_RATE_PCT):
        build_overrides({"cache": "profiled"}, m, TOTAL_WARPS)


def test_missing_metric_of_another_group_is_ignored():
    m = metrics()
    del m[L1_HIT_RATE_PCT]

    build_overrides({"counts": "profiled"}, m, TOTAL_WARPS)


def test_profiled_values_match_all_profiled():
    o = build_overrides(parse_overrides(["all=profiled"]), metrics(), TOTAL_WARPS)

    assert profiled_values(metrics(), TOTAL_WARPS) == vars(o)


def test_profiled_values_leave_out_a_group_with_a_missing_metric():
    m = metrics()
    del m[L1_HIT_RATE_PCT]
    values = profiled_values(m, TOTAL_WARPS)

    assert "l1_hit_ratio" not in values
    assert "l2_hit_ratio" not in values
    assert values["total_inst_count"] == 12.5


def test_profiled_without_metrics_file():
    with pytest.raises(OverrideError, match="metrics file"):
        build_overrides({"cache": "profiled"}, None, TOTAL_WARPS)
