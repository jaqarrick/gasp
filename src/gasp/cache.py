def get_cache_hit_rate_from_working_set(
    working_set_size_bytes: int,
    cache_size_bytes: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Piecewise cache hit rate from working set size vs cache capacity.
    Adapted from Jarmusch [2026], Table III (there for the MI300A last-level cache)
    """
    W = working_set_size_bytes
    C = cache_size_bytes

    if C <= 0:
        return 0.0

    # Transition zone starts at ~80% of cache capacity
    transition_start = 0.8 * C

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
