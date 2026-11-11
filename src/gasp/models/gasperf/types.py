"""GASPerf's own parameter types, copied from gasp.models.gpuperf.types. The
hardware and launch inputs are shared and stay in gasp.models.gpuperf.types."""

from dataclasses import dataclass


@dataclass
class KernelParameters:
    """Inputs from Parser + CFG modules"""

    # per warp; floats because profiled overrides are averages over warps
    total_inst_count: float
    mem_inst_count: float
    sfu_inst_count: float
    sync_inst_count: float
    uncoalesced_count: int
    divergent_inst_count: float
    bank_conflict_count: float
    shared_load_inst_count: float
    shared_store_inst_count: float
    ilp: float
    mlp: float
    average_inst_latency: float
    working_set_per_sm_bytes: int  # feeds the L1 hit rate
    working_set_total_bytes: int  # feeds the L2 hit rate


@dataclass
class OccupancyParams:
    """All these inputs are from the occupancy / simulation module"""

    active_warps: int  # N — active warps per SM
    active_sms: int  # SMs that receive at least one block
    total_warps: int  # grid_size * warps_per_block
    warps_per_block: int  # ceil(block_size / warp_size)
    blocks_per_sm: int  # from resource limits
    num_reps: int  # scheduling rounds
    # actually lets calculate these two inside the module
    itilp: float  # min(ilp * N, itilp_max)
    itmlp: float  # min(mlp * mwp_cp, mwp_peak_bw)

    # need to compute in module as well
    cwp: float  # Mem_L / (comp_cycles / mem_inst_count)
    mwp: float  # min(Mem_L / delta, N)
    mwp_cp: float  # min(max(1, cwp - 1), mwp)
    average_trans_warp: (
        float  # uncoalesced_count / (mem_inst_count * total_warps), floor 1
    )
    l1_hit_ratio: float  # from the cache model (gasp.cache)
    l2_hit_ratio: float  # of the accesses that miss L1


@dataclass
class ModelOutput:
    """What one model run gives predict(): its derived parameters and times."""

    itilp: float
    cwp: float
    mwp_cp: float
    mwp_peak_bw: float
    itmlp: float
    amat: float
    o_cfdiv: float
    t_comp: float
    t_mem: float
    t_overlap: float
    t_exec: float
    # shared loads + stores per SM x cycles each, and what it adds to T_comp
    shared_pipe_time: float = 0.0
    o_shmem: float = 0.0
    # global requests per SM x L1 cycles each; T_mem is at least this
    l1_time: float = 0.0
    # the L2 hit ratio GASPerf used (the concurrent-blocks one)
    l2_hit_ratio: float | None = None
