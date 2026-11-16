"""Launch dimensions: a block or grid shape of 1 to 3 positive sizes (x, y, z)."""

import math

Dims = tuple[int, ...]

MAX_DIMS = 3


def check_dims(dims: Dims) -> Dims:
    if not 1 <= len(dims) <= MAX_DIMS:
        raise ValueError(f"expected 1 to {MAX_DIMS} dimensions, got {len(dims)}")
    if any(type(d) is not int or d < 1 for d in dims):
        raise ValueError("every dimension must be a positive integer")
    return dims


def parse_dims(text: str) -> Dims:
    """`256` or `16x16` or `8x8x4`."""
    try:
        dims = tuple(int(part) for part in text.lower().split("x"))
    except ValueError:
        message = f"expected a size such as 256 or 16x16, got '{text}'"
        raise ValueError(message) from None
    return check_dims(dims)


def total(dims: Dims) -> int:
    return math.prod(dims)


def format_dims(dims: Dims) -> str:
    return "x".join(str(d) for d in dims)


def padded(dims: Dims) -> Dims:
    """x, y and z, with missing dimensions set to 1."""
    return dims + (1,) * (MAX_DIMS - len(dims))
