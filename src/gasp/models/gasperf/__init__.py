# GASPerf: our performance model. It starts as a copy of GPUPerf (gasp.models.gpuperf)
# and is ours to change; GPUPerf stays as published. Edit freely.
# Changes from GPUPerf: shared-memory and FP64 pipe limits in T_comp (O_shmem,
# O_fp64: one floor at the busier pipe's time); an L1
# throughput limit on T_mem; the L2 hit ratio against the blocks resident at the
# same time; when compute-bound (CWP < MWP), ITMLP = min(MLP x MWP, MWP_peak_bw)
# instead of Eq. 15's MWP_cp. A launch overhead term for T_GPU exists but is off
# (INCLUDE_LAUNCH_OVERHEAD).
from gasp.common.constants import SECTORS_PER_LINE
from gasp.logger import Logger
from gasp.models.gpuperf.types import HardwareParameters, LaunchParameters
from gasp.models.inputs import WORD_BYTES, ModelInputs
from gasp.occupancy import OccupancyOutput

from .types import KernelParameters, ModelOutput, OccupancyParams

# Off: on the pairs measured so far, every kernel in a pair launches the same
# grid, so the term only pulls predicted speedups toward 1 (helping or hurting by
# about a point). When off, T_GPU = T_exec and the reported launch overhead is 0.
INCLUDE_LAUNCH_OVERHEAD = True


def non_sfu_inst_count(total_inst_count: float, sfu_inst_count: float) -> float:
    """#insts as in GPUPerf: instructions per warp excluding SFU ones (Table 1).
    SFU cost enters only through O_SFU (Eq. 9)."""
    return max(total_inst_count - sfu_inst_count, 0.0)


class GASPerf:
    def __init__(
        self,
        k_params: KernelParameters,
        o_params: OccupancyParams,
        l_params: LaunchParameters,
        h_params: HardwareParameters,
        logger: Logger,
    ) -> None:
        self.k = k_params
        self.l = l_params
        self.h = h_params
        self.o = o_params
        self.logger = logger

    def predict_total_exec_time_cycles(self):
        self.t_comp = self._get_t_comp()
        self.t_mem = self._get_t_mem()
        self.t_overlap = self._get_t_overlap(self.t_mem, self.t_comp)
        t_exec = self.t_comp + self.t_mem - self.t_overlap
        return t_exec

    def predict_launch_overhead_cycles(self):
        """Launch overhead fit by launch_overhead_bench.cu: base + per thread
        launched. Not in GPUPerf."""
        threads = self.l.block_size * self.l.grid_size
        return self.h.launch_overhead_base + self.h.launch_overhead_per_thread * threads

    def _get_t_comp(self):
        k = self.k
        h = self.h
        o = self.o
        # divergent instructions are part of total_inst_count; charged in O_CFdiv
        # instead of W_parallel, at the same cost per instruction
        insts = non_sfu_inst_count(k.total_inst_count, k.sfu_inst_count)
        divergent = min(k.divergent_inst_count, insts)
        effective_throughput = k.average_inst_latency / o.itilp

        # w_parallel
        inst_per_sm = ((insts - divergent) * o.total_warps) / o.active_sms
        w_parallel = inst_per_sm * effective_throughput

        # w_serial = o_sync + o_sfu + o_cfdiv + o_bank
        mem_ratio = k.mem_inst_count / insts if insts > 0 else 0.0
        f_sync = h.sync_factor * h.dram_latency * mem_ratio
        o_sync = ((k.sync_inst_count * o.total_warps) / o.active_sms) * f_sync

        # with no other instructions the SFU share is all of them
        sfu_ratio = k.sfu_inst_count / insts if insts > 0 else 1.0
        f_sfu = min(
            [
                max([sfu_ratio - (h.sfu_width / h.simd_width), 0]),
                1,
            ]
        )
        sfu_throughput = h.warp_size / h.sfu_width
        o_sfu = (
            ((k.sfu_inst_count * o.total_warps) / o.active_sms) * sfu_throughput * f_sfu
        )

        o_cfdiv = ((divergent * o.total_warps) / o.active_sms) * effective_throughput
        self.o_cfdiv = o_cfdiv

        # Calibration experiment 1
        # Shared-memory pipe limit
        shared_insts = k.shared_load_inst_count + k.shared_store_inst_count
        self.shared_pipe_time = (
            (shared_insts * o.total_warps) / o.active_sms
        ) * h.shared_mem_throughput

        # Calibration experiment 2
        # FP64 pipe limit
        self.fp64_pipe_time = ((k.fp64_inst_count * o.total_warps) / o.active_sms) * (
            h.warp_size / h.fp64_width
        )
        # the pipes run side by side, so only the busiest one can set T_comp
        issue_time = w_parallel + o_cfdiv
        busiest_pipe_time = max(self.shared_pipe_time, self.fp64_pipe_time)
        o_pipe = max(0.0, busiest_pipe_time - issue_time)
        # reported split: O_shmem as without the FP64 limit, O_fp64 the rest
        # tbd do we want max or the sum?
        self.o_shmem = max(0.0, self.shared_pipe_time - issue_time)
        self.o_fp64 = o_pipe - self.o_shmem
        # Pico's count is per warp (extra shared accesses, N - 1 for an N-way
        # conflict), so it scales by warps per SM like the other W_serial terms.
        o_bank = (
            (k.bank_conflict_count * o.total_warps) / o.active_sms
        ) * h.bank_penalty

        w_serial = o_sync + o_sfu + o_cfdiv + o_bank + o_pipe

        return w_parallel + w_serial

    def _get_t_mem(self):
        k = self.k
        o = self.o
        h = self.h
        requests_per_sm = (k.mem_inst_count * o.total_warps) / o.active_sms
        self.amat = self._get_amat()
        t_mem_latency = requests_per_sm / o.itmlp * self.amat

        # L1 throughput limit
        sectors_per_request = o.average_trans_warp * SECTORS_PER_LINE
        # Adjustment - check for strided access pattern
        lines_per_request = sectors_per_request if o.average_trans_warp > 1 else 1.0
        cycles_per_request = max(
            h.l1_cycles_per_request,
            lines_per_request * SECTORS_PER_LINE / h.l1_sectors_per_cycle,
        )
        self.l1_time = requests_per_sm * cycles_per_request
        return max(t_mem_latency, self.l1_time)

    def _get_t_overlap(self, t_memory: float, t_comp: float) -> float:
        o = self.o
        overlap_zeta = 1 if o.cwp <= o.mwp else 0
        N = o.active_warps
        f_overlap = (N - overlap_zeta) / N
        t_overlap: float = min([t_comp * f_overlap, t_memory])
        return t_overlap

    def _get_amat(self):
        o = self.o
        return amat(self.h, o.l1_hit_ratio, o.l2_hit_ratio, o.average_trans_warp)


def average_dram_latency(h: HardwareParameters, average_trans_warp: float) -> float:
    """Eq. 13: avg DRAM lat = DRAM lat + (avg trans warp - 1) x delta."""
    return h.dram_latency + (average_trans_warp - 1) * h.departure_delay


def amat(
    h: HardwareParameters,
    l1_hit_ratio: float,
    l2_hit_ratio: float,
    average_trans_warp: float,
) -> float:
    """Eq. 12 over two cache levels. Every latency is end-to-end, so a miss
    costs the next level's latency with no lookup added on top: all misses
    gives exactly the average DRAM latency."""
    dram = average_dram_latency(h, average_trans_warp)
    l1_miss_latency = l2_hit_ratio * h.l2_hit_latency + (1 - l2_hit_ratio) * dram
    return l1_hit_ratio * h.l1_hit_latency + (1 - l1_hit_ratio) * l1_miss_latency


def run(
    m: ModelInputs,
    occ: OccupancyOutput,
    l: LaunchParameters,
    h: HardwareParameters,
    working_set_per_sm: int,
    working_set_total: int,
    uncoalesced_count: int,
    logger: Logger,
) -> ModelOutput:
    """GASPerf on one kernel's inputs: the derived parameters, then the times."""
    n = occ.active_warps_per_sm
    insts = non_sfu_inst_count(m.total_inst_count, m.sfu_inst_count)

    # Eqs. 4-5
    itilp_max = m.average_inst_latency / (h.warp_size / h.simd_width)
    itilp = min(m.ilp * n, itilp_max)

    l2_hit_ratio = (
        m.l2_hit_ratio_concurrent
        if m.l2_hit_ratio_concurrent is not None
        else m.l2_hit_ratio
    )
    avg_dram_lat = average_dram_latency(h, m.average_trans_warp)
    amat_cycles = amat(h, m.l1_hit_ratio, l2_hit_ratio, m.average_trans_warp)

    # Eqs. A.5-A.7: MWP = min(avg DRAM lat / delta, MWP_peak_bw, N), with BW per
    # warp one transaction (warp_size x 4 B) per avg DRAM lat
    bw_per_warp = h.warp_size * WORD_BYTES / avg_dram_lat
    mwp_peak_bw = h.dram_bandwidth / (bw_per_warp * occ.active_sms)
    mwp = min(avg_dram_lat / h.departure_delay, mwp_peak_bw, n)

    # Eqs. A.1-A.4, per warp: CWP = min((mem_cycles + comp_cycles) / comp_cycles, N),
    # comp_cycles = #insts x avg inst lat / ITILP, mem_cycles = #mem insts x AMAT / MLP
    comp_cycles = insts * m.average_inst_latency / itilp
    mem_cycles = m.mem_inst_count * amat_cycles / m.mlp
    cwp = min((mem_cycles + comp_cycles) / comp_cycles, n) if comp_cycles > 0 else n

    # Eq. 15
    mwp_cp = min(max(1, cwp - 1), mwp)

    # Compute-bound (CWP < MWP, Hong-Kim's MWP > CWP case): Eq. 15 sets MWP_cp to
    # CWP - 1 (at least 1), so a compute-heavy kernel's requests are charged
    # nearly one at a time, which can make T_mem larger
    # than T_comp in a kernel the model itself calls compute-bound. Here the
    # warps' requests overlap up to MWP (capped at N) and DRAM bandwidth, still
    # limited by MLP. Strict <: CWP = MWP = N keeps Eq. 15. GPUPerf unchanged.
    if cwp < mwp:
        itmlp = min(m.mlp * mwp, mwp_peak_bw)
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
        shared_load_inst_count=m.shared_load_inst_count,
        shared_store_inst_count=m.shared_store_inst_count,
        fp64_inst_count=m.fp64_inst_count,
        ilp=m.ilp,
        mlp=m.mlp,
        average_inst_latency=m.average_inst_latency,
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
        mwp=mwp,
        mwp_cp=mwp_cp,
        average_trans_warp=m.average_trans_warp,
        l1_hit_ratio=m.l1_hit_ratio,
        l2_hit_ratio=l2_hit_ratio,
    )
    logger.trace("predict", f"GASPerf kernel parameters: {k}")
    logger.trace("predict", f"GASPerf occupancy parameters: {o}")

    model = GASPerf(k, o, l, h, logger)
    t_exec = model.predict_total_exec_time_cycles()

    # Include Alvani's launch overhead
    launch_overhead = (
        model.predict_launch_overhead_cycles() if INCLUDE_LAUNCH_OVERHEAD else 0.0
    )
    t_gpu = t_exec + launch_overhead
    logger.trace(
        "predict",
        f"GASPerf T_GPU = T_exec {t_exec:.0f} + launch overhead {launch_overhead:.0f}",
    )
    return ModelOutput(
        itilp=itilp,
        cwp=cwp,
        mwp=mwp,
        mwp_cp=mwp_cp,
        mwp_peak_bw=mwp_peak_bw,
        itmlp=itmlp,
        amat=model.amat,
        o_cfdiv=model.o_cfdiv,
        t_comp=model.t_comp,
        t_mem=model.t_mem,
        t_overlap=model.t_overlap,
        t_exec=t_exec,
        launch_overhead=launch_overhead,
        t_gpu=t_gpu,
        shared_pipe_time=model.shared_pipe_time,
        o_shmem=model.o_shmem,
        fp64_pipe_time=model.fp64_pipe_time,
        o_fp64=model.o_fp64,
        l1_time=model.l1_time,
        l2_hit_ratio=l2_hit_ratio,
    )
