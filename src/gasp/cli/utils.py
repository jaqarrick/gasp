import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path

from gasp.dependency import LatencyTable
from gasp.logger import Logger
from gasp.parser.types import OpCategory

PROFILES_DIR = Path("hardware/profiles")

GPU_PROFILES = {
    "b200": PROFILES_DIR / "b200.placeholder.json",
    "h100": PROFILES_DIR / "h100.placeholder.json",
    "a100": PROFILES_DIR / "a100.placeholder.json",
    "l4": PROFILES_DIR / "l4.placeholder.json",
    "v100": PROFILES_DIR / "v100.placeholder.json",
    # measured with hardware/microbenchmarks (runs in hardware/runs/rtx2080ti/)
    "rtx2080ti": PROFILES_DIR / "rtx2080ti.json",
}


@dataclass
class HardwareProfile:
    architecture: str
    chip: str
    arch: str

    # spec
    warp_size: int
    simd_width: int
    sfu_width: int
    sms_per_gpu: int
    max_warps_per_sm: int
    max_blocks_per_sm: int
    max_registers_per_sm: int
    max_shared_mem_per_sm_bytes: int
    l2_cache_size_bytes: int
    # one on-chip array per SM, split between shared memory and L1
    l1_shared_array_bytes: int
    # shared-memory sizes the driver can pick from, ascending
    shared_mem_carveouts_bytes: list[int]

    # profiled
    launch_overhead_base_cycles: float
    launch_overhead_per_thread_cycles: float
    departure_delay_cycles: int
    sync_factor: float
    dram_latency_cycles: int
    l1_cache_hit_latency_cycles: int
    l2_cache_hit_latency_cycles: int
    bank_penalty_cycles: int
    sfu_inst_latency_cycles: int
    fp32_inst_latency_cycles: int
    int32_inst_latency_cycles: int
    fp64_inst_latency_cycles: int
    dram_bandwidth_bytes_per_cycle: int
    clock_rate_ghz: float
    shared_mem_load_latency_cycles: int
    shared_mem_store_latency_cycles: int
    # SM cycles per warp shared load with the shared-memory pipe saturated
    shared_mem_load_throughput_cycles: float
    const_mem_latency_cycles: int


def load_hardware_profile(gpu: str, logger: Logger) -> HardwareProfile:
    path = GPU_PROFILES.get(gpu)

    if path is None:
        logger.error(f"No profile defined for gpu: {gpu}")
        sys.exit(1)

    return load_profile_file(path, logger)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _has_type(value: object, kind: object) -> bool:
    if kind is str:
        return isinstance(value, str)
    if kind in (int, float):
        # measured values may be fractional even where the model uses ints
        return _is_number(value)
    # list[int]
    return isinstance(value, list) and all(_is_number(v) for v in value)


def _fail(logger: Logger, msg: str) -> None:
    logger.error(msg)
    sys.exit(1)


def load_profile_file(path: str | Path, logger: Logger) -> HardwareProfile:
    path = Path(path)
    logger.info(f"loading hardware profile from {path}")

    if not path.is_file():
        _fail(logger, f"hardware profile not found: {path}")

    try:
        with open(path) as f:
            profile = json.load(f)
    except json.JSONDecodeError as e:
        _fail(
            logger,
            f"hardware profile {path} is not valid JSON "
            f"(line {e.lineno}, column {e.colno}): {e.msg}",
        )
    if not isinstance(profile, dict):
        _fail(logger, f"hardware profile {path} must be a JSON object")

    for section in ("spec", "profiled"):
        if not isinstance(profile.get(section), dict):
            _fail(
                logger,
                f"hardware profile {path} needs a '{section}' object",
            )
    spec = profile.pop("spec")
    profiled = profile.pop("profiled")
    flat = {**profile, **spec, **profiled}

    expected = {f.name: f.type for f in fields(HardwareProfile)}
    missing = [name for name in expected if name not in flat]
    if missing:
        _fail(
            logger,
            f"hardware profile {path} is missing field(s): {', '.join(missing)}",
        )
    unknown = [name for name in flat if name not in expected]
    if unknown:
        _fail(
            logger,
            f"hardware profile {path} has unknown field(s): {', '.join(unknown)}",
        )
    wrong = [
        f"{name} (expected {getattr(kind, '__name__', kind)}, got {flat[name]!r})"
        for name, kind in expected.items()
        if not _has_type(flat[name], kind)
    ]
    if wrong:
        _fail(
            logger,
            f"hardware profile {path} has field(s) of the wrong type: "
            f"{', '.join(wrong)}",
        )

    hp = HardwareProfile(**flat)
    logger.trace("config", f"hardware profile {path}: chip {hp.chip}, arch {hp.arch}")
    for name in expected:
        logger.trace("config", f"  {name} = {getattr(hp, name)}")
    return hp


def build_latency_table(hp: HardwareProfile) -> LatencyTable:
    return LatencyTable(
        by_category={
            OpCategory.FP32: hp.fp32_inst_latency_cycles,
            OpCategory.FP64: hp.fp64_inst_latency_cycles,
            OpCategory.INT: hp.int32_inst_latency_cycles,
            OpCategory.SFU: hp.sfu_inst_latency_cycles,
            OpCategory.SHARED_LOAD: hp.shared_mem_load_latency_cycles,
            OpCategory.SHARED_STORE: hp.shared_mem_store_latency_cycles,
            OpCategory.CONST_LOAD: hp.const_mem_latency_cycles,
        },
        default=hp.fp32_inst_latency_cycles,
    )
