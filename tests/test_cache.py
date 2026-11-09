"""Tests for gasp.cache — piecewise capacity hit rate (Jarmusch Table III)."""

import pytest

from gasp.cache import get_cache_hit_rate_from_working_set as hit_rate
from gasp.cache import get_capacity_regime, get_hit_ratios, get_reuse_fraction

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
