from dataclasses import dataclass

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


@dataclass
class PredictionInput:
    dependency: DependencyOutput
    occupancy: OccupancyOutput
    pico: PicoOutput
    hardware: HardwareParameters
    launch: LaunchParameters


@dataclass
class PredictionOutput:
    itilp: float
    cwp: float
    mwp_cp: float
    mwp_peak_bw: float
    itmlp: float
    average_trans_warp: float
    t_comp: float
    t_mem: float
    t_overlap: float
    t_exec: float
    t_gpu: float
    t_one_thread: float
    speedup: float


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

    if p.error or d.mem_inst_count == 0:
        average_trans_warp = 1.0
        uncoalesced_count = 0
        bank_conflict_count = 0
    else:
        average_trans_warp = max(
            1.0, p.uncoalesced_count / (d.mem_inst_count * occ.total_warps)
        )
        uncoalesced_count = p.uncoalesced_count
        bank_conflict_count = p.bank_conflict_count

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
        working_set_size=0,
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
        cache_hit_ratio=0.0,
    )

    logger.trace("predict", f"kernel parameters: {k}")
    logger.trace("predict", f"occupancy parameters: {o}")

    model = GPUPerf(k, o, l, h, logger)
    t_exec = model.predict_total_exec_time_cycles()
    t_gpu = t_exec + h.launch_overhead

    # one thread runs every thread's work back to back, no overlap
    per_thread = d.total_latency + d.mem_inst_count * model.amat
    t_one_thread = per_thread * l.grid_size * l.block_size

    logger.info("Prediction completed")
    return PredictionOutput(
        itilp=itilp,
        cwp=cwp,
        mwp_cp=mwp_cp,
        mwp_peak_bw=mwp_peak_bw,
        itmlp=itmlp,
        average_trans_warp=average_trans_warp,
        t_comp=model.t_comp,
        t_mem=model.t_mem,
        t_overlap=model.t_overlap,
        t_exec=t_exec,
        t_gpu=t_gpu,
        t_one_thread=t_one_thread,
        speedup=t_one_thread / t_gpu,
    )
