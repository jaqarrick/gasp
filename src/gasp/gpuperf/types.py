from dataclasses import dataclass


@dataclass
class KernelParameters:
    """Inputs from Parser + CFG modules"""

    total_inst_count: int
    mem_inst_count: int
    sfu_inst_count: int
    sync_inst_count: int
    uncoalesced_count: int
    divergent_inst_count: int
    bank_conflict_count: int
    ilp: float
    mlp: float
    average_inst_latency: float
    working_set_size: int


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
    cache_hit_ratio: float  # implement in cache model


@dataclass
class HardwareParameters:
    """From hardware spec and microbenchmarks"""

    warp_size: int  # e.g. 32
    simd_width: int
    sfu_width: int
    departure_delay: float
    sync_factor: (
        int  # fudge factor- GPUPerf uses 64 "based on microarchitecture simulations"
    )
    dram_latency: float
    cache_hit_latency: float  # todo: distinguish L1 and L2
    bank_penalty: float
    launch_overhead: float
    dram_bandwidth: float  # bytes per cycle, whole GPU


@dataclass
class LaunchParameters:
    block_size: int  # threads per block
    grid_size: int  # grid size in blocks
    problem_size: int  # number of elements or threads launched
    registers_per_thread: int
    shared_mem_per_block_bytes: int
