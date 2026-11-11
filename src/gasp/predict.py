from dataclasses import asdict, dataclass, replace

from gasp.cache import (
    get_cache_hit_rate_from_working_set,
    get_capacity_regime,
    get_concurrent_working_set,
    get_footprint_reuse_fraction,
    get_hit_ratios,
    get_hit_ratios_given_l1,
    get_l1_size,
    get_reuse_fraction,
)
from gasp.common.constants import SECTORS_PER_LINE
from gasp.dependency import DependencyOutput
from gasp.logger import Logger
from gasp.models import DEFAULT_MODEL, MODELS
from gasp.models.gpuperf.types import HardwareParameters, LaunchParameters
from gasp.models.inputs import ModelInputs
from gasp.occupancy import OccupancyOutput
from gasp.overrides import ModelOverrides
from gasp.pico import PicoOutput


@dataclass
class PredictionInput:
    dependency: DependencyOutput
    occupancy: OccupancyOutput
    pico: PicoOutput
    hardware: HardwareParameters
    launch: LaunchParameters
    footprint_bytes: int | None = None
    # share of global accesses that hit L1 on data the same block just loaded
    # (--reuse), None to use L1's capacity term
    block_reuse: float | None = None
    # values that replace static model inputs (--override), None for none
    overrides: ModelOverrides | None = None
    # which of MODELS to run
    model: str = DEFAULT_MODEL


def apply_overrides(
    static: ModelInputs, overrides: ModelOverrides | None, logger: Logger
) -> ModelInputs:
    """The inputs the model runs on: static, with every overridden field replaced."""
    if overrides is None:
        return static
    changed = {k: v for k, v in asdict(overrides).items() if v is not None}
    for name, value in changed.items():
        logger.trace(
            "predict", f"override {name}: static {getattr(static, name):g} → {value:g}"
        )
    # a measured L2 hit ratio replaces both models' estimates
    if overrides.l2_hit_ratio is not None:
        changed["l2_hit_ratio_concurrent"] = overrides.l2_hit_ratio
    return replace(static, **changed)


@dataclass
class PredictionOutput:
    itilp: float
    cwp: float
    mwp_cp: float
    mwp_peak_bw: float
    itmlp: float
    average_trans_warp: float
    working_set_per_sm_bytes: int
    working_set_total_bytes: int
    l1_cache_size_bytes: int
    shared_mem_carveout_bytes: int
    reuse_fraction: float
    l1_hit_ratio: float
    l2_hit_ratio: float
    amat: float
    t_comp: float
    t_mem: float
    t_overlap: float
    t_exec: float
    launch_overhead: float
    t_gpu: float
    o_cfdiv: float = 0.0
    shared_pipe_time: float = 0.0
    o_shmem: float = 0.0
    l1_time: float = 0.0
    model: str = DEFAULT_MODEL


def get_average_trans_warp(p: PicoOutput, mem_inst_count: int) -> float:
    """128-byte transactions per warp memory access, 1 = coalesced.

    For now we output sectors per transaction
    ua -> uncoalesced access count
    count -> total memory inst per warp

    So we return the average ua/count

    PICO's output is in sectors
    GPUPerf wants cache lines

    4 sectors (32B)= 1 cache line (128B)
    """
    if p.error or mem_inst_count == 0 or p.memory_accesses_count == 0:
        return 1.0
    sectors_per_access = p.uncoalesced_count / p.memory_accesses_count
    return max(1.0, sectors_per_access / SECTORS_PER_LINE)


def get_average_inst_latency(d: DependencyOutput) -> float:
    """Average latency of the non-SFU instructions, which W_parallel and CWP
    charge. SFU instructions are charged only in O_SFU (Eq. 9). A kernel of SFU
    instructions only keeps the average over all of them."""
    non_sfu = d.total_inst_count - d.sfu_inst_count
    if non_sfu <= 0:
        return d.total_latency / d.total_inst_count
    return (d.total_latency - d.sfu_latency) / non_sfu


def predict(input: PredictionInput, logger: Logger) -> PredictionOutput:
    d = input.dependency
    occ = input.occupancy
    p = input.pico
    h = input.hardware
    l = input.launch

    # 1. static inputs

    if p.error or d.mem_inst_count == 0:
        uncoalesced_count = 0
        bank_conflict_count = 0
    else:
        uncoalesced_count = p.uncoalesced_count
        bank_conflict_count = p.bank_conflict_count

    # working set: the blocks resident on one SM share its L1, the whole grid shares L2
    bytes_per_block = d.global_bytes_per_thread * l.block_size
    working_set_per_sm = bytes_per_block * occ.blocks_per_sm
    working_set_total = bytes_per_block * l.grid_size
    l1_size, carveout = get_l1_size(
        h.l1_shared_array_bytes,
        h.shared_mem_carveouts_bytes,
        occ.blocks_per_sm * l.shared_mem_per_block_bytes,
    )
    l1_capacity_hit = get_cache_hit_rate_from_working_set(working_set_per_sm, l1_size)
    l2_capacity_hit = get_cache_hit_rate_from_working_set(
        working_set_total, h.l2_cache_size_bytes
    )
    # first touches miss at every level, whatever the working set
    bytes_accessed = d.global_bytes_accessed_per_thread * l.block_size * l.grid_size
    if input.footprint_bytes is not None and bytes_accessed > 0:
        # whole-kernel reuse, so sharing between threads counts too
        reuse_fraction = get_footprint_reuse_fraction(
            input.footprint_bytes, bytes_accessed
        )
        working_set_total = input.footprint_bytes
        l2_capacity_hit = get_cache_hit_rate_from_working_set(
            working_set_total, h.l2_cache_size_bytes
        )
        reuse_source = (
            f"footprint {input.footprint_bytes:,} B / accessed {bytes_accessed:,} B"
        )
    else:
        # per-thread fallback: loop accesses and repeats within a block
        reuse_fraction = get_reuse_fraction(
            d.reusable_access_count, d.global_access_count
        )
        reuse_source = (
            f"per thread {d.reusable_access_count}/{d.global_access_count} reusable"
        )
    # GASPerf's L2 term: only the blocks resident at the same time compete for
    # L2, so data reused while they run hits even when the whole kernel's data
    # is larger than L2
    working_set_concurrent = get_concurrent_working_set(
        working_set_total, occ.blocks_per_sm * occ.active_sms, l.grid_size
    )
    l2_concurrent_hit = get_cache_hit_rate_from_working_set(
        working_set_concurrent, h.l2_cache_size_bytes
    )
    l2_trace = (
        f"L2 cap {l2_capacity_hit:.3f}"
        f" ({get_capacity_regime(working_set_total, h.l2_cache_size_bytes)}),"
        f" concurrent {working_set_concurrent:,} B cap {l2_concurrent_hit:.3f}"
        f" ({get_capacity_regime(working_set_concurrent, h.l2_cache_size_bytes)})"
    )
    if input.block_reuse is not None:
        # given by the user (--reuse): replaces L1's capacity term, which only
        # sees totals and misses data reused within a block right after loading
        l1_hit_ratio, l2_hit_ratio = get_hit_ratios_given_l1(
            input.block_reuse, reuse_fraction, l2_capacity_hit
        )
        _, l2_hit_ratio_concurrent = get_hit_ratios_given_l1(
            input.block_reuse, reuse_fraction, l2_concurrent_hit
        )
        l1_trace = f"L1 from --reuse {input.block_reuse:g}"
    else:
        l1_hit_ratio, l2_hit_ratio = get_hit_ratios(
            reuse_fraction, l1_capacity_hit, l2_capacity_hit
        )
        _, l2_hit_ratio_concurrent = get_hit_ratios(
            reuse_fraction, l1_capacity_hit, l2_concurrent_hit
        )
        l1_trace = (
            f"L1 cap {l1_capacity_hit:.3f}"
            f" ({get_capacity_regime(working_set_per_sm, l1_size)})"
        )
    logger.trace(
        "predict",
        f"cache: L1 {l1_size:,} B (carveout {carveout:,} B);"
        f" r = {reuse_fraction:.4f} ({reuse_source}); {l1_trace};"
        f" {l2_trace}; h1 {l1_hit_ratio:.3f}, h2 {l2_hit_ratio:.3f},"
        f" h2 concurrent {l2_hit_ratio_concurrent:.3f}",
    )

    static = ModelInputs(
        total_inst_count=d.total_inst_count,
        mem_inst_count=d.mem_inst_count,
        sfu_inst_count=d.sfu_inst_count,
        sync_inst_count=d.sync_inst_count,
        divergent_inst_count=d.divergent_inst_count,
        ilp=d.ilp,
        mlp=d.mlp,
        average_inst_latency=get_average_inst_latency(d),
        average_trans_warp=get_average_trans_warp(p, d.mem_inst_count),
        bank_conflict_count=bank_conflict_count,
        l1_hit_ratio=l1_hit_ratio,
        l2_hit_ratio=l2_hit_ratio,
        shared_load_inst_count=d.shared_load_inst_count,
        shared_store_inst_count=d.shared_store_inst_count,
        l2_hit_ratio_concurrent=l2_hit_ratio_concurrent,
    )

    # 2. overrides: the only place a static input is replaced

    m = apply_overrides(static, input.overrides, logger)

    # 3. the model, from those inputs

    logger.trace("predict", f"model: {input.model}")
    run = MODELS[input.model].run
    out = run(
        m, occ, l, h, working_set_per_sm, working_set_total, uncoalesced_count, logger
    )
    t_exec = out.t_exec
    # launch overhead fit by launch_overhead_bench.cu: base + per thread launched
    threads = l.block_size * l.grid_size
    launch_overhead = h.launch_overhead_base + h.launch_overhead_per_thread * threads
    t_gpu = t_exec + launch_overhead

    logger.info("Prediction completed")
    return PredictionOutput(
        itilp=out.itilp,
        cwp=out.cwp,
        mwp_cp=out.mwp_cp,
        mwp_peak_bw=out.mwp_peak_bw,
        itmlp=out.itmlp,
        average_trans_warp=m.average_trans_warp,
        working_set_per_sm_bytes=working_set_per_sm,
        working_set_total_bytes=working_set_total,
        l1_cache_size_bytes=l1_size,
        shared_mem_carveout_bytes=carveout,
        reuse_fraction=reuse_fraction,
        l1_hit_ratio=m.l1_hit_ratio,
        # the one the model used: GASPerf's is against the concurrent blocks
        l2_hit_ratio=(
            out.l2_hit_ratio if out.l2_hit_ratio is not None else m.l2_hit_ratio
        ),
        amat=out.amat,
        t_comp=out.t_comp,
        t_mem=out.t_mem,
        t_overlap=out.t_overlap,
        t_exec=t_exec,
        launch_overhead=launch_overhead,
        t_gpu=t_gpu,
        o_cfdiv=out.o_cfdiv,
        shared_pipe_time=out.shared_pipe_time,
        o_shmem=out.o_shmem,
        l1_time=out.l1_time,
        model=input.model,
    )
