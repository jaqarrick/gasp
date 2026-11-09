"""Tests for gasp.cache — piecewise capacity hit rate (Jarmusch Table III)."""

import pytest

from gasp.cache import get_cache_hit_rate_from_working_set as hit_rate

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


def test_hit_rate_no_cache_misses():
    assert hit_rate(0, 0) == 0.0
    assert hit_rate(1024, 0) == 0.0
