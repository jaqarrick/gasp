TRANSITION_START = 0.8  # from jarmusch


def get_capacity_regime(working_set_size_bytes: int, cache_size_bytes: int) -> str:
    """Which branch of the capacity model applies, for traces."""
    if cache_size_bytes <= 0:
        return "no cache"
    if working_set_size_bytes < TRANSITION_START * cache_size_bytes:
        return "resident"
    if working_set_size_bytes <= cache_size_bytes:
        return "transition"
    return "streaming"


def get_cache_hit_rate_from_working_set(
    working_set_size_bytes: int,
    cache_size_bytes: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Piecewise hit rate from working set vs capacity (Jarmusch [2026], Table III)."""
    W = working_set_size_bytes
    C = cache_size_bytes

    if C <= 0:
        return 0.0

    transition_start = TRANSITION_START * C

    if W < transition_start:
        # Fully resident
        return 1.0
    elif W <= C:
        # Transition: power-law falloff
        frac = (W - transition_start) / (C - transition_start)
        return beta * (1.0 - frac) ** alpha
    else:
        # Streaming: only the cache-sized portion hits
        return C / W


def get_l1_size(
    array_bytes: int, carveouts_bytes: list[int], shared_needed_bytes: int
) -> tuple[int, int]:
    """L1 bytes per SM, and the smallest shared-memory carveout that fits the resident blocks."""
    options = sorted(carveouts_bytes)
    if not options:
        return max(0, array_bytes), 0
    carveout = next((c for c in options if c >= shared_needed_bytes), options[-1])
    return max(0, array_bytes - carveout), carveout


def get_concurrent_working_set(
    working_set_total_bytes: int, resident_blocks: int, total_blocks: int
) -> int:
    """Working set of the blocks resident at the same time: working set x resident / total blocks."""
    if total_blocks <= 0:
        return working_set_total_bytes
    share = min(1.0, resident_blocks / total_blocks)
    return round(working_set_total_bytes * share)


def get_reuse_fraction(reusable_access_count: int, global_access_count: int) -> float:
    """Share of global accesses that can hit in any cache"""
    if global_access_count <= 0:
        return 0.0
    return reusable_access_count / global_access_count


def get_footprint_reuse_fraction(footprint_bytes: int, bytes_accessed: int) -> float:
    """Share of accessed bytes that are not first touches (compulsory misses)."""
    if bytes_accessed <= 0:
        return 0.0
    return 1.0 - min(1.0, footprint_bytes / bytes_accessed)


def get_hit_ratios(
    reuse_fraction: float, l1_capacity_hit: float, l2_capacity_hit: float
) -> tuple[float, float]:
    r = reuse_fraction  # share of addresses with any hit
    h1 = r * l1_capacity_hit
    l1_misses = 1.0 - h1
    if l1_misses <= 0:
        return h1, 0.0
    h2 = r * (1.0 - l1_capacity_hit) * l2_capacity_hit / l1_misses
    return h1, h2


def get_hit_ratios_given_l1(
    l1_hit_ratio: float, reuse_fraction: float, l2_capacity_hit: float
) -> tuple[float, float]:
    h1 = l1_hit_ratio
    l1_misses = 1.0 - h1
    if l1_misses <= 0:
        return h1, 0.0
    reusable_misses = max(0.0, reuse_fraction - h1)
    return h1, reusable_misses * l2_capacity_hit / l1_misses
