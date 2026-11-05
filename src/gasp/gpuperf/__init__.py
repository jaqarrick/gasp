# This module contains the GPUPerf formula for execution time
from gasp.logger import Logger

from .types import (
    HardwareParameters,
    KernelParameters,
    LaunchParameters,
    OccupancyParams,
)


class GPUPerf:
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

    def _get_t_comp(self):
        k = self.k
        h = self.h
        o = self.o
        # w_parallel
        inst_per_sm = (k.total_inst_count * o.total_warps) / o.active_sms
        effective_throughput = k.average_inst_latency / o.itilp
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

        # todo: implement our own model for overhead / bank conflict overhead since GPUPerf uses emulation
        o_cfdiv = k.divergent_inst_count * k.average_inst_latency
        o_bank = k.bank_conflict_count * h.bank_penalty

        w_serial = o_sync + o_sfu + o_cfdiv + o_bank

        return w_parallel + w_serial

    def _get_t_mem(self):
        k = self.k
        o = self.o
        mem_requests_per_sm = (k.mem_inst_count * o.total_warps) / (
            o.active_sms * o.itmlp
        )
        amat = self._get_amat()
        return mem_requests_per_sm * amat

    def _get_t_overlap(self, t_memory: float, t_comp: float) -> float:
        o = self.o
        overlap_zeta = 1 if o.cwp <= o.mwp else 0
        N = o.active_warps
        f_overlap = (N - overlap_zeta) / N
        t_overlap: float = min([t_comp * f_overlap, t_memory])
        return t_overlap

    def _get_amat(self):
        # note: average memory access time involves cache hit ratio, which will be performed in a different module
        average_dram_latency = (
            self.h.dram_latency
            + (self.o.average_trans_warp - 1) * self.h.departure_delay
        )
        miss_ratio = 1 - self.o.cache_hit_ratio
        amat = average_dram_latency * miss_ratio + self.h.cache_hit_latency
        return amat
