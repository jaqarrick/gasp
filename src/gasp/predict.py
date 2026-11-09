from dataclasses import dataclass

from gasp.cache import (
    get_cache_hit_rate_from_working_set,
    get_capacity_regime,
    get_footprint_reuse_fraction,
    get_hit_ratios,
    get_hit_ratios_given_l1,
    get_l1_size,
    get_reuse_fraction,
)
from gasp.dependency import DependencyOutput
from gasp.gpuperf import GPUPerf
from gasp.gpuperf.types import (
    HardwareParameters,
    KernelParameters,
    LaunchParameters,
    OccupancyParams,
)
from gasp.logger import Logger
from gasp.occupancy import OccupancyOutput
from gasp.pico import PicoOutput

# bytes per thread per memory instruction, for MWP_peak_bw
WORD_BYTES = 4
SECTORS_PER_LINE = 4  # for Pico <=> GPUPerf conversion


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


def predict(input: PredictionInput, logger: Logger) -> PredictionOutput:
    d = input.dependency
    occ = input.occupancy
    p = input.pico
    h = input.hardware
    l = input.launch
    n = occ.active_warps_per_sm

    avg_inst_lat = d.total_latency / d.total_inst_count

    # Eqs. 4-5
    itilp_max = avg_inst_lat / (h.warp_size / h.simd_width)
    itilp = min(d.ilp * n, itilp_max)

    # CWP = Mem_L / (Comp_cycles / #mem_insts), capped at N
    # Comp_cycles: one warp's compute time, dependency-limited by ILP
    comp_cycles = d.total_inst_count * avg_inst_lat / d.ilp
    if d.mem_inst_count > 0:
        cwp = min(h.dram_latency / (comp_cycles / d.mem_inst_count), n)
    else:
        cwp = n

    # Eq. 15
    mwp_cp = min(max(1, cwp - 1), occ.mwp)

    # Eq. 14, MWP_peak_bw as in Hong-Kim: bandwidth / (per-warp bandwidth x active SMs)
    bw_per_warp = h.warp_size * WORD_BYTES / h.dram_latency
    mwp_peak_bw = h.dram_bandwidth / (bw_per_warp * occ.active_sms)
    itmlp = min(d.mlp * mwp_cp, mwp_peak_bw)

    average_trans_warp = get_average_trans_warp(p, d.mem_inst_count)
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
    l2_trace = (
        f"L2 cap {l2_capacity_hit:.3f}"
        f" ({get_capacity_regime(working_set_total, h.l2_cache_size_bytes)})"
    )
    if input.block_reuse is not None:
        # given by the user (--reuse): replaces L1's capacity term, which only
        # sees totals and misses data reused within a block right after loading
        l1_hit_ratio, l2_hit_ratio = get_hit_ratios_given_l1(
            input.block_reuse, reuse_fraction, l2_capacity_hit
        )
        l1_trace = f"L1 from --reuse {input.block_reuse:g}"
    else:
        l1_hit_ratio, l2_hit_ratio = get_hit_ratios(
            reuse_fraction, l1_capacity_hit, l2_capacity_hit
        )
        l1_trace = (
            f"L1 cap {l1_capacity_hit:.3f}"
            f" ({get_capacity_regime(working_set_per_sm, l1_size)})"
        )
    logger.trace(
        "predict",
        f"cache: L1 {l1_size:,} B (carveout {carveout:,} B);"
        f" r = {reuse_fraction:.4f} ({reuse_source}); {l1_trace};"
        f" {l2_trace}; h1 {l1_hit_ratio:.3f}, h2 {l2_hit_ratio:.3f}",
    )

    k = KernelParameters(
        total_inst_count=d.total_inst_count,
        mem_inst_count=d.mem_inst_count,
        sfu_inst_count=d.sfu_inst_count,
        sync_inst_count=d.sync_inst_count,
        uncoalesced_count=uncoalesced_count,
        divergent_inst_count=0,
        bank_conflict_count=bank_conflict_count,
        ilp=d.ilp,
        mlp=d.mlp,
        average_inst_latency=avg_inst_lat,
        working_set_per_sm_bytes=working_set_per_sm,
        working_set_total_bytes=working_set_total,
    )
    o = OccupancyParams(
        active_warps=n,
        active_sms=occ.active_sms,
        total_warps=occ.total_warps,
        warps_per_block=occ.warps_per_block,
        blocks_per_sm=occ.blocks_per_sm,
        num_reps=occ.num_reps,
        itilp=itilp,
        itmlp=itmlp,
        cwp=cwp,
        mwp=occ.mwp,
        mwp_cp=mwp_cp,
        average_trans_warp=average_trans_warp,
        l1_hit_ratio=l1_hit_ratio,
        l2_hit_ratio=l2_hit_ratio,
    )

    logger.trace("predict", f"kernel parameters: {k}")
    logger.trace("predict", f"occupancy parameters: {o}")

    model = GPUPerf(k, o, l, h, logger)
    t_exec = model.predict_total_exec_time_cycles()
    # launch overhead fit by launch_overhead_bench.cu: base + per thread launched
    threads = l.block_size * l.grid_size
    launch_overhead = h.launch_overhead_base + h.launch_overhead_per_thread * threads
    t_gpu = t_exec + launch_overhead

    logger.info("Prediction completed")
    return PredictionOutput(
        itilp=itilp,
        cwp=cwp,
        mwp_cp=mwp_cp,
        mwp_peak_bw=mwp_peak_bw,
        itmlp=itmlp,
        average_trans_warp=average_trans_warp,
        working_set_per_sm_bytes=working_set_per_sm,
        working_set_total_bytes=working_set_total,
        l1_cache_size_bytes=l1_size,
        shared_mem_carveout_bytes=carveout,
        reuse_fraction=reuse_fraction,
        l1_hit_ratio=l1_hit_ratio,
        l2_hit_ratio=l2_hit_ratio,
        amat=model.amat,
        t_comp=model.t_comp,
        t_mem=model.t_mem,
        t_overlap=model.t_overlap,
        t_exec=t_exec,
        launch_overhead=launch_overhead,
        t_gpu=t_gpu,
    )
