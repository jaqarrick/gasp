from dataclasses import dataclass
from math import ceil, floor

from gasp.logger import Logger


@dataclass
class OccupancyInput:
    # launch params
    block_size: int
    grid_size: int

    # hardware params
    warp_size: int
    sms_per_gpu: int
    max_warps_per_sm: int
    max_blocks_per_sm: int
    max_registers_per_sm: int
    max_shared_mem_per_sm_bytes: int
    dram_latency_cycles: int
    departure_delay_cycles: int

    # resource usage
    registers_per_thread: int
    shared_memory_per_block: int


@dataclass
class OccupancyOutput:
    active_warps_per_sm: int
    total_warps: int
    active_sms: int
    num_reps: int
    mwp: float
    warps_per_block: int
    blocks_per_sm: int


def calculate_occupancy(input: OccupancyInput, logger: Logger) -> OccupancyOutput:
    warps_per_block = ceil(input.block_size / input.warp_size)

    # three resource limits
    max_blocks_by_reg = floor(
        input.max_registers_per_sm
        / (input.registers_per_thread * input.warp_size * warps_per_block)
    )
    max_blocks_by_smem = (
        input.max_blocks_per_sm
        if input.shared_memory_per_block == 0
        else floor(input.max_shared_mem_per_sm_bytes / input.shared_memory_per_block)
    )
    max_blocks_by_warps = floor(input.max_warps_per_sm / warps_per_block)
    blocks_per_sm = min(
        max_blocks_by_reg,
        max_blocks_by_smem,
        max_blocks_by_warps,
        input.max_blocks_per_sm,
    )

    N = blocks_per_sm * warps_per_block

    total_blocks = input.grid_size
    total_warps = input.grid_size * warps_per_block
    active_sms = min(input.sms_per_gpu, total_blocks)
    num_reps = ceil(total_blocks / (blocks_per_sm * active_sms))

    mwp = min(input.dram_latency_cycles / input.departure_delay_cycles, N)

    logger.trace(
        "occupancy",
        f"warps/block={warps_per_block} "
        f"block limits: registers={max_blocks_by_reg} shared={max_blocks_by_smem} "
        f"warps={max_blocks_by_warps} hw={input.max_blocks_per_sm} "
        f"-> blocks/SM={blocks_per_sm}",
    )
    logger.info("Occupancy calculations completed")
    return OccupancyOutput(
        active_warps_per_sm=N,
        total_warps=total_warps,
        active_sms=active_sms,
        num_reps=num_reps,
        mwp=mwp,
        warps_per_block=warps_per_block,
        blocks_per_sm=blocks_per_sm,
    )
