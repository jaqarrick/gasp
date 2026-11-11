"""Tests for gasp.cache — piecewise capacity hit rate (Jarmusch Table III)."""

import pytest

from gasp.cache import get_cache_hit_rate_from_working_set as hit_rate
from gasp.cache import (
    get_capacity_regime,
    get_concurrent_working_set,
    get_footprint_reuse_fraction,
    get_hit_ratios,
    get_hit_ratios_given_l1,
    get_l1_size,
    get_reuse_fraction,
)

C = 100_000


@pytest.mark.parametrize(
    "working_set,expected",
    [
        (0, 1.0),  # nothing to miss
        (int(0.8 * C) - 1, 1.0),  # just under the transition: fully resident
        (int(0.9 * C), 0.5),  # halfway through the transition, linear (alpha = 1)
        (C, 0.0),  # end of the transition
        (2 * C, 0.5),  # streaming: only the cache-sized part hits
        (10 * C, 0.1),
    ],
)
def test_hit_rate(working_set, expected):
    assert hit_rate(working_set, C) == pytest.approx(expected)


@pytest.mark.parametrize(
    "reusable,accesses,expected",
    [(0, 3, 0.0), (3, 6, 0.5), (10, 10, 1.0), (0, 0, 0.0)],
)
def test_reuse_fraction(reusable, accesses, expected):
    assert get_reuse_fraction(reusable, accesses) == expected


@pytest.mark.parametrize(
    "r,cap1,cap2,h1,h2",
    [
        (0.0, 1.0, 1.0, 0.0, 0.0),  # all first touches: nothing hits anywhere
        (0.5, 1.0, 1.0, 0.5, 0.0),  # L1 keeps every reuse, L2 sees first touches only
        (1.0, 1.0, 1.0, 1.0, 0.0),  # everything hits L1, no misses left
        (1.0, 0.0, 1.0, 0.0, 1.0),  # too big for L1, fits in L2
        (1.0, 0.0, 0.0, 0.0, 0.0),  # too big for both
        # half reusable, L1 keeps half of that: h1 = 0.25, and of the 0.75 that
        # miss L1, the 0.25 reusable ones all hit L2
        (0.5, 0.5, 1.0, 0.25, 0.25 / 0.75),
    ],
)
def test_hit_ratios(r, cap1, cap2, h1, h2):
    assert get_hit_ratios(r, cap1, cap2) == pytest.approx((h1, h2))


def test_hit_rate_no_cache_misses():
    assert hit_rate(0, 0) == 0.0
    assert hit_rate(1024, 0) == 0.0


@pytest.mark.parametrize(
    "working_set,size,expected",
    [
        (0, C, "resident"),
        (int(0.8 * C), C, "transition"),
        (C, C, "transition"),
        (C + 1, C, "streaming"),
        (10, 0, "no cache"),
    ],
)
def test_capacity_regime(working_set, size, expected):
    assert get_capacity_regime(working_set, size) == expected


@pytest.mark.parametrize(
    "footprint,accessed,expected",
    [
        (12_000_000, 12_000_000, 0.0),  # vecadd good: every byte is a first touch
        (12_000_000, 24_000_000, 0.5),  # vecadd poor: each byte twice
        (8, 8_000, 0.999),  # heavy reuse
        (100, 50, 0.0),  # footprint larger than accesses: clamped, no reuse
        (100, 0, 0.0),  # no accesses
    ],
)
def test_footprint_reuse_fraction(footprint, accessed, expected):
    assert get_footprint_reuse_fraction(footprint, accessed) == pytest.approx(expected)


@pytest.mark.parametrize(
    "h1,r,cap2,expected_h2",
    [
        (0.94, 0.9985, 1.0, (0.9985 - 0.94) / 0.06),  # what L1 misses of the reuse
        (0.94, 0.9985, 0.5, (0.9985 - 0.94) * 0.5 / 0.06),
        (0.5, 0.5, 1.0, 0.0),  # L1 already holds all the reuse
        (0.5, 0.2, 1.0, 0.0),  # given h1 above r: no reuse left for L2
        (1.0, 1.0, 1.0, 0.0),  # nothing misses L1
        (0.0, 0.0, 1.0, 0.0),
    ],
)
def test_hit_ratios_given_l1(h1, r, cap2, expected_h2):
    got_h1, got_h2 = get_hit_ratios_given_l1(h1, r, cap2)
    assert got_h1 == h1
    assert got_h2 == pytest.approx(expected_h2)


KB = 1024
TURING = (96 * KB, [32 * KB, 64 * KB])
AMPERE = (
    192 * KB,
    [0, 8 * KB, 16 * KB, 32 * KB, 64 * KB, 100 * KB, 132 * KB, 164 * KB],
)


@pytest.mark.parametrize(
    "array,carveouts,shared_needed,expected",
    [
        (*TURING, 0, (64 * KB, 32 * KB)),  # Turing always reserves 32 KB of shared
        (*TURING, 8 * KB, (64 * KB, 32 * KB)),  # mm_tiled: 4 blocks x 2 KB
        (*TURING, 32 * KB + 1, (32 * KB, 64 * KB)),  # one byte over: next split
        (*TURING, 128 * KB, (32 * KB, 64 * KB)),  # over every option: the largest
        (*AMPERE, 0, (192 * KB, 0)),  # no shared memory: the whole array is L1
        (*AMPERE, 8 * KB, (184 * KB, 8 * KB)),
        (*AMPERE, 9 * KB, (176 * KB, 16 * KB)),
        (64 * KB, [], 0, (64 * KB, 0)),  # no carveout list: the whole array
    ],
)
def test_l1_size(array, carveouts, shared_needed, expected):
    assert get_l1_size(array, carveouts, shared_needed) == expected


def test_l1_size_options_need_not_be_sorted():
    assert get_l1_size(96 * KB, [64 * KB, 32 * KB], 0) == (64 * KB, 32 * KB)


def test_concurrent_working_set_is_the_resident_share():
    assert get_concurrent_working_set(12_000_000, 272, 4096) == round(
        12_000_000 * 272 / 4096
    )


def test_concurrent_working_set_never_exceeds_the_total():
    assert get_concurrent_working_set(1_000, 400, 40) == 1_000
    assert get_concurrent_working_set(1_000, 4, 0) == 1_000
