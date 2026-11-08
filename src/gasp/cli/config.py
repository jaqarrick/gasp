import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gasp.launch import Dims, check_dims, total

from .constants import DEFAULT_BLOCK_SIZE, DEFAULT_PROBLEM_SIZE

INT_SETTINGS = ("problem_size", "loop_iterations")
DIM_SETTINGS = ("block_size", "grid_size")
LIST_SETTINGS = ("include_dirs", "nvcc_flags", "program_args")
STR_SETTINGS = ("file", "kernel")
PARAMS = "params"
# trips of each loop in the compiled kernel, so only on a kernel entry
LOOP_COUNTS = "loop_counts"
DEFAULT_KEYS = {*INT_SETTINGS, *DIM_SETTINGS, *LIST_SETTINGS, *STR_SETTINGS, PARAMS}
ENTRY_KEYS = {*DEFAULT_KEYS, "name", LOOP_COUNTS}


class ConfigError(ValueError):
    pass


@dataclass
class KernelSpec:
    name: str
    file: str
    kernel: str | None  # selector, None if the file has to hold a single kernel
    problem_size: int
    block_size: int  # total threads per block
    grid_size: int  # total blocks
    block_dim: Dims
    grid_dim: Dims
    loop_iterations: int | None
    # trips per entry for each loop, in code order; wins over loop_iterations
    loop_counts: list[int] | None
    include_dirs: list[str]
    nvcc_flags: list[str]
    program_args: list[str]
    params: dict[str, int]  # kernel arguments, passed to Pico


def check_cu_file(path: str) -> str:
    if not path.endswith(".cu"):
        raise ConfigError(f"expected a .cu file, got '{path}'")
    if not os.path.isfile(path):
        raise ConfigError(f"file not found: {path}")
    return path


def parse_input(value: str) -> tuple[str, str | None]:
    """Split `file.cu` or `file.cu::kernel` into the file and the kernel selector."""
    file, separator, kernel = value.partition("::")
    if separator and not kernel:
        raise ConfigError(f"missing a kernel name after '::' in '{value}'")
    return check_cu_file(file), kernel or None


def _as_dims(value: Any) -> Dims:
    """A dimension setting: a positive integer, or an array of 1 to 3 of them."""
    dims = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    return check_dims(dims)


def _validate_settings(settings: dict[str, Any], where: str, allowed: set[str]) -> None:
    unknown = settings.keys() - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown setting(s) {', '.join(sorted(unknown))}")

    for key, value in settings.items():
        if key in INT_SETTINGS:
            if type(value) is not int or value <= 0:
                raise ConfigError(f"{where}: {key} must be a positive integer")
        elif key in DIM_SETTINGS:
            try:
                _as_dims(value)
            except ValueError:
                raise ConfigError(
                    f"{where}: {key} must be a positive integer "
                    "or an array of 1 to 3 positive integers"
                ) from None
        elif key == PARAMS:
            if not isinstance(value, dict) or not all(
                isinstance(name, str) and name and type(v) is int
                for name, v in value.items()
            ):
                raise ConfigError(
                    f"{where}: {key} must be a table of names to integers"
                )
        elif key == LOOP_COUNTS:
            if not isinstance(value, list) or not all(
                type(v) is int and v >= 0 for v in value
            ):
                raise ConfigError(
                    f"{where}: {key} must be an array of non-negative integers"
                )
        elif key in LIST_SETTINGS:
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise ConfigError(f"{where}: {key} must be an array of strings")
        elif not isinstance(value, str) or not value:
            raise ConfigError(f"{where}: {key} must be a non-empty string")


def _build_spec(
    name: str, file: str, settings: dict[str, Any], where: str
) -> KernelSpec:
    problem_size = settings.get("problem_size", DEFAULT_PROBLEM_SIZE)
    block_dim = _as_dims(settings.get("block_size", DEFAULT_BLOCK_SIZE))
    if "grid_size" in settings:
        grid_dim = _as_dims(settings["grid_size"])
    elif len(block_dim) == 1:
        grid_dim = (math.ceil(problem_size / total(block_dim)),)
    else:
        raise ConfigError(
            f"{where}: a block_size with several dimensions needs grid_size, "
            "because the grid can't be derived from problem_size"
        )
    return KernelSpec(
        name=name,
        file=file,
        kernel=settings.get("kernel"),
        problem_size=problem_size,
        block_size=total(block_dim),
        grid_size=total(grid_dim),
        block_dim=block_dim,
        grid_dim=grid_dim,
        loop_iterations=settings.get("loop_iterations"),
        loop_counts=settings.get(LOOP_COUNTS),
        include_dirs=settings.get("include_dirs", []),
        nvcc_flags=settings.get("nvcc_flags", []),
        program_args=settings.get("program_args", []),
        params=dict(settings.get(PARAMS, {})),
    )


def _check_unique_names(specs: list[KernelSpec], hint: str) -> None:
    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            raise ConfigError(f"more than one kernel is named '{spec.name}'{hint}")
        seen.add(spec.name)


def _default_name(file: str, kernel: str | None) -> str:
    return kernel or Path(file).stem


def specs_from_inputs(
    inputs: list[tuple[str, str | None]], settings: dict[str, Any]
) -> list[KernelSpec]:
    """Kernels given on the command line, which all share the same settings."""
    _validate_settings(settings, "command line", DEFAULT_KEYS - set(STR_SETTINGS))
    specs = [
        _build_spec(
            _default_name(file, kernel),
            file,
            {**settings, "kernel": kernel},
            "command line",
        )
        for file, kernel in inputs
    ]
    _check_unique_names(specs, ", use --config to give them different names")
    return specs


def load_config(path: str) -> list[KernelSpec]:
    """Kernels from a TOML file. An entry's settings win over [defaults], and an
    entry's params are merged name by name over those in [defaults]."""
    try:
        with open(path, "rb") as f:
            config = tomllib.load(f)
    except OSError as e:
        raise ConfigError(f"cannot read config {path}: {e.strerror}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {path}: {e}") from e

    unknown = config.keys() - {"defaults", "kernel"}
    if unknown:
        raise ConfigError(f"{path}: unknown section(s) {', '.join(sorted(unknown))}")

    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ConfigError(f"{path}: [defaults] must be a table")
    _validate_settings(defaults, "[defaults]", DEFAULT_KEYS)

    entries = config.get("kernel", [])
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ConfigError(f"{path}: kernels must be given as [[kernel]] entries")
    if not entries:
        raise ConfigError(f"{path}: no [[kernel]] entries")

    specs = []
    for index, entry in enumerate(entries, start=1):
        where = f"kernel '{entry['name']}'" if "name" in entry else f"kernel #{index}"
        _validate_settings(entry, where, ENTRY_KEYS)
        settings = {**defaults, **entry}
        if PARAMS in defaults or PARAMS in entry:
            settings[PARAMS] = {**defaults.get(PARAMS, {}), **entry.get(PARAMS, {})}
        if "file" not in settings:
            raise ConfigError(f"{where}: missing file")
        try:
            file = check_cu_file(settings["file"])
        except ConfigError as e:
            raise ConfigError(f"{where}: {e}") from e
        name = settings.get("name", _default_name(file, settings.get("kernel")))
        specs.append(_build_spec(name, file, settings, where))

    _check_unique_names(specs, f" in {path}")
    return specs
