# This module contains the GPUPerf formula for execution time
from gasp.common.constants import SECTORS_PER_LINE, WORD_BYTES
from gasp.logger import Logger
from gasp.model_opts import ModelOptions

from .types import (
    HardwareParameters,
    KernelParameters,
    LaunchParameters,
    MemClass,
    OccupancyParams,
)

# what T_mem is set by (GPUPerf.t_mem_bound)
BOUND_LATENCY = "latency"
BOUND_L1 = "L1 rate"
BOUND_L2 = "L2 bandwidth"
BOUND_DRAM = "DRAM bandwidth"


def get_amat(
    h: HardwareParameters, average_trans_warp: float, h1: float, h2: float
) -> float:
    # every latency is end-to-end, so a miss costs the next level's latency
    # with no lookup added on top: all misses gives exactly the DRAM latency
    average_dram_latency = h.dram_latency + (average_trans_warp - 1) * h.departure_delay
    l1_miss_latency = h2 * h.l2_hit_latency + (1 - h2) * average_dram_latency
    return h1 * h.l1_hit_latency + (1 - h1) * l1_miss_latency


def get_mean_amat(
    h: HardwareParameters, average_trans_warp: float, classes: list[MemClass]
) -> float:
    """AMAT over every access class, weighted by instruction count."""
    amats = [
        get_amat(h, average_trans_warp, c.l1_hit_ratio, c.l2_hit_ratio) for c in classes
    ]
    total = sum(c.inst_count for c in classes)
    if len(classes) == 1 or total <= 0:
        return amats[0]
    return sum(c.inst_count * a for c, a in zip(classes, amats)) / total


class GPUPerf:
    def __init__(
        self,
        k_params: KernelParameters,
        o_params: OccupancyParams,
        l_params: LaunchParameters,
        h_params: HardwareParameters,
        logger: Logger,
        options: ModelOptions | None = None,
    ) -> None:
        self.opts = options or ModelOptions()
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

    def _get_t_comp(self):
        k = self.k
        h = self.h
        o = self.o
        # divergent instructions are part of total_inst_count; charged in O_CFdiv
        # instead of W_parallel, at the same cost per instruction
        divergent = min(k.divergent_inst_count, k.total_inst_count)
        effective_throughput = k.average_inst_latency / o.itilp

        # w_parallel
        inst_per_sm = ((k.total_inst_count - divergent) * o.total_warps) / o.active_sms
        w_parallel = inst_per_sm * effective_throughput

        # w_serial = o_sync + o_sfu + o_cfdiv + o_bank
        f_sync = (
            h.sync_factor * h.dram_latency * (k.mem_inst_count / k.total_inst_count)
        )
        o_sync = ((k.sync_inst_count * o.total_warps) / o.active_sms) * f_sync

        f_sfu = min(
            [
                max(
                    [
                        (k.sfu_inst_count / k.total_inst_count)
                        - (h.sfu_width / h.simd_width),
                        0,
                    ]
                ),
                1,
            ]
        )
        sfu_throughput = h.warp_size / h.sfu_width
        o_sfu = (
            ((k.sfu_inst_count * o.total_warps) / o.active_sms) * sfu_throughput * f_sfu
        )

        # the shared-memory pipe serves a fixed number of warp accesses per cycle
        self.shared_floor = 0.0
        if (
            self.opts.shared_floor
            and h.shared_load_throughput is not None
            and h.shared_store_throughput is not None
        ):
            shared_cycles_per_warp = (
                k.shared_load_inst_count * h.shared_load_throughput
                + k.shared_store_inst_count * h.shared_store_throughput
            )
            self.shared_floor = (shared_cycles_per_warp * o.total_warps) / o.active_sms
        self.shared_floor_binds = self.shared_floor > w_parallel
        w_parallel = max(w_parallel, self.shared_floor)

        o_cfdiv = ((divergent * o.total_warps) / o.active_sms) * effective_throughput
        self.o_cfdiv = o_cfdiv
        # Pico's count is per warp (extra shared accesses, N - 1 for an N-way
        # conflict), so it scales by warps per SM like the other W_serial terms.
        o_bank = (
            (k.bank_conflict_count * o.total_warps) / o.active_sms
        ) * h.bank_penalty

        w_serial = o_sync + o_sfu + o_cfdiv + o_bank

        return w_parallel + w_serial

    def _get_t_mem(self):
        k = self.k
        o = self.o
        h = self.h
        opts = self.opts
        tw = o.average_trans_warp

        self.class_amats = {
            c.name: self._get_amat(c.l1_hit_ratio, c.l2_hit_ratio)
            for c in k.mem_classes
        }
        self.amat = get_mean_amat(h, tw, k.mem_classes)
        latency = sum(
            (c.inst_count * o.total_warps)
            / (o.active_sms * o.itmlp)
            * self.class_amats[c.name]
            for c in k.mem_classes
        )

        # floors: traffic / the rate of the level that serves it. An
        # uncoalesced access moves average_trans_warp lines
        bytes_per_access = h.warp_size * WORD_BYTES * tw
        bytes_missing_l1 = sum(
            c.inst_count * o.total_warps * bytes_per_access * (1 - c.l1_hit_ratio)
            for c in k.mem_classes
        )
        bytes_missing_l2 = sum(
            c.inst_count
            * o.total_warps
            * bytes_per_access
            * (1 - c.l1_hit_ratio)
            * (1 - c.l2_hit_ratio)
            for c in k.mem_classes
        )
        terms = {BOUND_LATENCY: latency}
        if opts.l1_floor and h.l1_sector_rate:
            load_sectors_per_sm = (
                (k.global_load_inst_count * o.total_warps)
                / o.active_sms
                * (tw * SECTORS_PER_LINE)
            )
            terms[BOUND_L1] = load_sectors_per_sm / h.l1_sector_rate
        if opts.l2_floor and h.l2_bandwidth:
            terms[BOUND_L2] = bytes_missing_l1 / h.l2_bandwidth
        if opts.dram_floor and h.dram_bandwidth > 0:
            terms[BOUND_DRAM] = bytes_missing_l2 / h.dram_bandwidth
        self.t_mem_terms = terms
        # the first of equal terms wins, so latency when nothing else is larger
        self.t_mem_bound = max(terms, key=lambda name: terms[name])
        return terms[self.t_mem_bound]

    def _get_t_overlap(self, t_memory: float, t_comp: float) -> float:
        o = self.o
        overlap_zeta = 1 if o.cwp <= o.mwp else 0
        N = o.active_warps
        f_overlap = (N - overlap_zeta) / N
        t_overlap: float = min([t_comp * f_overlap, t_memory])
        return t_overlap

    def _get_amat(self, h1: float | None = None, h2: float | None = None):
        """AMAT for the given hit ratios, by default the kernel-wide ones."""
        o = self.o
        return get_amat(
            self.h,
            o.average_trans_warp,
            o.l1_hit_ratio if h1 is None else h1,
            o.l2_hit_ratio if h2 is None else h2,
        )
