from dataclasses import asdict, dataclass, field, replace

from gasp.cache import (
    get_cache_hit_rate_from_working_set,
    get_capacity_regime,
    get_footprint_reuse_fraction,
    get_hit_ratios,
    get_hit_ratios_given_l1,
    get_l1_size,
    get_reuse_fraction,
)
from gasp.common.constants import SECTORS_PER_LINE, WORD_BYTES
from gasp.dependency import DependencyOutput
from gasp.gpuperf import BOUND_LATENCY, GPUPerf, get_mean_amat
from gasp.gpuperf.types import (
    HardwareParameters,
    KernelParameters,
    LaunchParameters,
    MemClass,
    OccupancyParams,
)
from gasp.logger import Logger
from gasp.model_opts import ModelOptions, format_model_opts
from gasp.occupancy import OccupancyOutput
from gasp.overrides import ModelOverrides
from gasp.pico import PicoOutput

# T_mem is set by the DRAM bandwidth cap inside ITMLP (Eq. 14)
BOUND_ITMLP_CAP = "DRAM bandwidth (ITMLP cap)"


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
    # changes to the model's formulas (--model-opts), None for none
    options: ModelOptions | None = None


@dataclass
class ModelInputs:
    """Every input the model takes from static analysis, per warp where it's
    a count. predict() fills this from the static analyses, applies the
    overrides to it once, and derives everything else from the result."""

    total_inst_count: float
    mem_inst_count: float
    sfu_inst_count: float
    sync_inst_count: float
    divergent_inst_count: float
    ilp: float
    mlp: float
    average_inst_latency: float
    average_trans_warp: float
    bank_conflict_count: float
    l1_hit_ratio: float
    l2_hit_ratio: float
    # global stores, part of mem_inst_count
    store_inst_count: float = 0.0
    # L2 hit ratio of global stores, which never hit L1
    store_l2_hit_ratio: float = 0.0
    shared_load_inst_count: float = 0.0
    shared_store_inst_count: float = 0.0


def get_mem_classes(m: ModelInputs, options: ModelOptions) -> list[MemClass]:
    """The access classes T_mem sums over: one for every global access, or
    loads and stores apart, where stores never hit L1."""
    everything = MemClass("global", m.mem_inst_count, m.l1_hit_ratio, m.l2_hit_ratio)
    if not options.mem_classes:
        return [everything]
    stores = min(max(m.store_inst_count, 0.0), m.mem_inst_count)
    loads = m.mem_inst_count - stores
    classes = []
    if loads > 0:
        classes.append(MemClass("loads", loads, m.l1_hit_ratio, m.l2_hit_ratio))
    if stores > 0:
        classes.append(MemClass("stores", stores, 0.0, m.store_l2_hit_ratio))
    return classes or [everything]


def get_skipped_floors(options: ModelOptions, h: HardwareParameters) -> list[str]:
    """Floors that are on but can't be applied: `option (missing profile fields)`."""
    needs = {
        "l1-floor": {"l1_throughput_sectors_per_cycle_per_sm": h.l1_sector_rate},
        "l2-floor": {"l2_bandwidth_bytes_per_cycle": h.l2_bandwidth},
        "shared-floor": {
            "shared_mem_load_throughput_cycles": h.shared_load_throughput,
            "shared_mem_store_throughput_cycles": h.shared_store_throughput,
        },
    }
    skipped = []
    for name, rates in needs.items():
        missing = [key for key, value in rates.items() if value is None]
        if getattr(options, name.replace("-", "_")) and missing:
            skipped.append(f"{name} (no {', '.join(missing)})")
    return skipped


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
    mwp: float = 0.0
    # Mem_L in MWP and CWP: the DRAM latency, or AMAT with amat-mem-l
    mem_l: float = 0.0
    # AMAT per access class, by class name
    class_amats: dict[str, float] = field(default_factory=dict)
    # which term sets T_mem: latency, or a throughput limit
    t_mem_bound: str = BOUND_LATENCY
    # W_parallel is set by the shared-memory floor
    shared_floor_binds: bool = False
    # floors that were on but left out: the profile has no rate for them
    skipped_floors: list[str] = field(default_factory=list)
    model_opts: str = "none"


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
    options = input.options or ModelOptions()

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
    # stores skip L1. A repeated store is taken to find its line still in L2,
    # whatever the footprint: the repeat follows the first store closely
    store_l2_hit_ratio = reuse_fraction

    static = ModelInputs(
        total_inst_count=d.total_inst_count,
        mem_inst_count=d.mem_inst_count,
        sfu_inst_count=d.sfu_inst_count,
        sync_inst_count=d.sync_inst_count,
        divergent_inst_count=d.divergent_inst_count,
        ilp=d.ilp,
        mlp=d.mlp,
        average_inst_latency=d.total_latency / d.total_inst_count,
        average_trans_warp=get_average_trans_warp(p, d.mem_inst_count),
        bank_conflict_count=bank_conflict_count,
        l1_hit_ratio=l1_hit_ratio,
        l2_hit_ratio=l2_hit_ratio,
        store_inst_count=d.global_store_inst_count,
        store_l2_hit_ratio=store_l2_hit_ratio,
        shared_load_inst_count=d.shared_load_inst_count,
        shared_store_inst_count=d.shared_store_inst_count,
    )

    # 2. overrides: the only place a static input is replaced

    m = apply_overrides(static, input.overrides, logger)

    # 3. the model, from those inputs

    # Eqs. 4-5
    itilp_max = m.average_inst_latency / (h.warp_size / h.simd_width)
    itilp = min(m.ilp * n, itilp_max)

    mem_classes = get_mem_classes(m, options)
    logger.trace(
        "predict", f"model options: {format_model_opts(options)}; {mem_classes}"
    )

    skipped_floors = get_skipped_floors(options, h)
    if skipped_floors:
        logger.warning(
            "The hardware profile has no rate for these model options, so they "
            f"are left out: {'; '.join(skipped_floors)}"
        )

    # Mem_L, the memory wait in MWP and CWP: the DRAM latency as in Hong-Kim
    # and GPUPerf, or what an access costs on average once caches hit
    if options.amat_mem_l:
        mem_l = get_mean_amat(h, m.average_trans_warp, mem_classes)
        mwp = min(mem_l / h.departure_delay, n)
    else:
        mem_l = h.dram_latency
        mwp = occ.mwp

    # CWP = Mem_L / (Comp_cycles / #mem_insts), capped at N
    # Comp_cycles: one warp's compute time, dependency-limited by ILP
    comp_cycles = m.total_inst_count * m.average_inst_latency / m.ilp
    if m.mem_inst_count > 0:
        cwp = min(mem_l / (comp_cycles / m.mem_inst_count), n)
    else:
        cwp = n

    # Eq. 15
    mwp_cp = min(max(1, cwp - 1), mwp)

    # Eq. 14, MWP_peak_bw as in Hong-Kim: bandwidth / (per-warp bandwidth x active SMs)
    bw_per_warp = h.warp_size * WORD_BYTES / h.dram_latency
    mwp_peak_bw = h.dram_bandwidth / (bw_per_warp * occ.active_sms)
    if options.dram_floor:
        # the bandwidth limit is a floor on T_mem instead, on DRAM bytes only
        itmlp = m.mlp * mwp_cp
    else:
        itmlp = min(m.mlp * mwp_cp, mwp_peak_bw)

    k = KernelParameters(
        total_inst_count=m.total_inst_count,
        mem_inst_count=m.mem_inst_count,
        sfu_inst_count=m.sfu_inst_count,
        sync_inst_count=m.sync_inst_count,
        uncoalesced_count=uncoalesced_count,
        divergent_inst_count=m.divergent_inst_count,
        bank_conflict_count=m.bank_conflict_count,
        ilp=m.ilp,
        mlp=m.mlp,
        average_inst_latency=m.average_inst_latency,
        working_set_per_sm_bytes=working_set_per_sm,
        working_set_total_bytes=working_set_total,
        mem_classes=mem_classes,
        global_load_inst_count=max(
            0.0, m.mem_inst_count - min(m.store_inst_count, m.mem_inst_count)
        ),
        shared_load_inst_count=m.shared_load_inst_count,
        shared_store_inst_count=m.shared_store_inst_count,
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
        mwp=mwp,
        mwp_cp=mwp_cp,
        average_trans_warp=m.average_trans_warp,
        l1_hit_ratio=m.l1_hit_ratio,
        l2_hit_ratio=m.l2_hit_ratio,
    )

    logger.trace("predict", f"kernel parameters: {k}")
    logger.trace("predict", f"occupancy parameters: {o}")

    model = GPUPerf(k, o, l, h, logger, options)
    t_exec = model.predict_total_exec_time_cycles()
    t_mem_bound = model.t_mem_bound
    if (
        t_mem_bound == BOUND_LATENCY
        and not options.dram_floor
        and m.mem_inst_count > 0
        and mwp_peak_bw < m.mlp * mwp_cp
    ):
        t_mem_bound = BOUND_ITMLP_CAP
    logger.trace(
        "predict",
        f"T_mem terms: {model.t_mem_terms}, set by {t_mem_bound};"
        f" shared floor {model.shared_floor:.0f}"
        f" ({'binds' if model.shared_floor_binds else 'does not bind'})",
    )
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
        average_trans_warp=m.average_trans_warp,
        working_set_per_sm_bytes=working_set_per_sm,
        working_set_total_bytes=working_set_total,
        l1_cache_size_bytes=l1_size,
        shared_mem_carveout_bytes=carveout,
        reuse_fraction=reuse_fraction,
        l1_hit_ratio=m.l1_hit_ratio,
        l2_hit_ratio=m.l2_hit_ratio,
        amat=model.amat,
        t_comp=model.t_comp,
        t_mem=model.t_mem,
        t_overlap=model.t_overlap,
        t_exec=t_exec,
        launch_overhead=launch_overhead,
        t_gpu=t_gpu,
        o_cfdiv=model.o_cfdiv,
        mwp=mwp,
        mem_l=mem_l,
        class_amats=model.class_amats,
        t_mem_bound=t_mem_bound,
        shared_floor_binds=model.shared_floor_binds,
        skipped_floors=skipped_floors,
        model_opts=format_model_opts(options),
    )
