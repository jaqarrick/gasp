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
    divergent_inst_count: int
    bank_conflict_count: float
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
class HardwareParameters:
    """From hardware spec and microbenchmarks"""

    warp_size: int  # e.g. 32
    simd_width: int
    sfu_width: int
    departure_delay: float
    # Γ, dimensionless. GPUPerf used 64 "based on microarchitecture simulations";
    # hardware/microbenchmarks/sync_cost.cu measures it per GPU
    sync_factor: float
    dram_latency: float
    # end-to-end latencies, like dram_latency: a hit costs only its own latency
    l1_hit_latency: float
    l2_hit_latency: float
    # L1 gets what the shared-memory carveout leaves of this array
    l1_shared_array_bytes: int
    shared_mem_carveouts_bytes: list[int]
    l2_cache_size_bytes: int
    bank_penalty: float
    launch_overhead_base: float  # cycles per launch
    launch_overhead_per_thread: float  # cycles per launched thread
    dram_bandwidth: float  # bytes per cycle, whole GPU


@dataclass
class LaunchParameters:
    block_size: int  # threads per block
    grid_size: int  # grid size in blocks
    problem_size: int  # number of elements or threads launched
    registers_per_thread: int
    shared_mem_per_block_bytes: int
