import argparse
import math
import os
import shlex
import sys

from gasp import __version__
from gasp.cfg import gen_cfg_from_raw
from gasp.common.constants import GPU_TO_ARCH
from gasp.cuda_toolkit import CUDAExtractor
from gasp.dependency import analyze
from gasp.gpuperf.types import HardwareParameters, LaunchParameters
from gasp.logger import Logger
from gasp.occupancy import OccupancyInput, calculate_occupancy
from gasp.parser import get_parser
from gasp.pico import PicoInput, perform_pico_analysis
from gasp.predict import PredictionInput, predict
from gasp.profiler import ProfileInput, profile_kernel

from .constants import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_LOOP_ITERATIONS,
    DEFAULT_PROBLEM_SIZE,
)
from .types import Args
from .utils import build_latency_table, load_hardware_profile


def cu_file(path: str) -> str:
    if not path.endswith(".cu"):
        raise argparse.ArgumentTypeError(f"expected a .cu file, got '{path}'")
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gasp", description="GASP: GPU Analytical Static Predictor"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gpu", help="built-in GPU name (e.g., b200)")
    group.add_argument("--profile", help="path to custom hardware profile JSON")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("input", type=cu_file, metavar="FILE", help="path to .cu file")

    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Number of threads per block to be launched",
    )
    parser.add_argument("--grid-size", type=int, default=None, help="Size of the grid")
    parser.add_argument(
        "--problem-size", type=int, default=None, help="Size of the input"
    )
    parser.add_argument(
        "--loop-iterations",
        type=int,
        default=None,
        help="Iterations per loop in the kernel (applied to every loop)",
    )
    parser.add_argument(
        "-I",
        "--include-dir",
        action="append",
        default=[],
        help="Include directory passed to nvcc (repeatable)",
    )
    parser.add_argument(
        "--nvcc-flags",
        default=None,
        help='Extra flags passed to nvcc, e.g. "--use_fast_math -Xptxas -dlcm=cg"',
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Compile and run the program under ncu with one thread and with the "
        "full launch, and compare the measured speedup with the prediction",
    )
    parser.add_argument(
        "--program-args",
        default=None,
        help='Arguments passed to the program when run with --run, e.g. "1000000"',
    )

    return parser


def relative_error(predicted: float, measured: float) -> str:
    if measured == 0:
        return "n/a"
    return f"{(predicted - measured) / measured * 100:+.1f}%"


def main() -> None:
    parser = build_parser()
    raw = parser.parse_args()
    args = Args(**vars(raw))
    verbose = args.verbose
    profile = args.profile
    gpu = args.gpu.lower()
    input_file = args.input
    grid_size = args.grid_size
    block_size = args.block_size
    problem_size = args.problem_size
    loop_iterations = args.loop_iterations
    include_dirs = args.include_dir
    nvcc_flags = shlex.split(args.nvcc_flags) if args.nvcc_flags else []
    program_args = shlex.split(args.program_args) if args.program_args else []

    logger = Logger(verbose)
    logger.info("Verbose mode enabled")

    if block_size is None:
        logger.info(
            f"No argument for --block-size supplied. Defaulting to {DEFAULT_BLOCK_SIZE}"
        )
        block_size = DEFAULT_BLOCK_SIZE

    if problem_size is None:
        logger.info(
            f"No argument for --problem-size supplied. Defaulting to {DEFAULT_PROBLEM_SIZE}"
        )
        problem_size = DEFAULT_PROBLEM_SIZE

    if grid_size is None:
        grid_size = math.ceil(problem_size / block_size)
        logger.info(
            f"No argument for --grid_size supplied. Defaulting to ceil(problem_size / block_size) = {grid_size}"
        )

    if loop_iterations is None:
        logger.info(
            f"No argument for --loop-iterations supplied. Defaulting to {DEFAULT_LOOP_ITERATIONS}"
        )
        loop_iterations = DEFAULT_LOOP_ITERATIONS

    arch = GPU_TO_ARCH.get(gpu.lower())

    if arch is None:
        logger.error(f"No architecture defined for GPU {gpu}")
        sys.exit(1)

    logger.table(
        "Configuration",
        ["Parameter", "Value"],
        [
            ["GPU", str(gpu or "Not provided")],
            ["Profile", str(profile or "Not provided")],
            ["Input", input_file],
            ["Block Size", str(block_size)],
            ["Grid Size", str(grid_size)],
            ["Problem Size", str(problem_size)],
            ["Loop Iterations", str(loop_iterations)],
            ["Verbose", str(verbose)],
        ],
    )

    logger.info(f"Extracting SASS and resource usage from {input_file}")

    extractor = CUDAExtractor(
        input_file,
        arch,
        logger,
        include_dirs=include_dirs,
        extra_flags=nvcc_flags,
    )
    result = extractor.process()
    resources = result["resources"]

    cfg = gen_cfg_from_raw(raw_cfg=result["raw_cfg"])
    # print_cfg(cfg)
    # print(result["resources"])
    # {'_Z9addvectorPiS_S_i:': {'reg': 12, 'shared': 0, 'stack': 0, 'local': 0}}
    # for now we assume a single kernel
    kernel_name, kernel = next(iter(resources.items()))
    registers_per_thread = kernel["reg"]
    shared_memory_per_block = kernel["shared"]

    hp = load_hardware_profile(gpu, logger)
    occupancy_results = calculate_occupancy(
        OccupancyInput(
            block_size=block_size,
            grid_size=grid_size,
            warp_size=hp.warp_size,
            sms_per_gpu=hp.sms_per_gpu,
            max_warps_per_sm=hp.max_warps_per_sm,
            max_blocks_per_sm=hp.max_blocks_per_sm,
            max_registers_per_sm=hp.max_registers_per_sm,
            max_shared_mem_per_sm_bytes=hp.max_shared_mem_per_sm_bytes,
            dram_latency_cycles=hp.dram_latency_cycles,
            departure_delay_cycles=hp.departure_delay_cycles,
            registers_per_thread=registers_per_thread,
            shared_memory_per_block=shared_memory_per_block,
        ),
        logger=logger,
    )
    logger.table(
        "Occupancy",
        ["Parameter", "Value"],
        [
            ["Active Warps/SM", str(occupancy_results.active_warps_per_sm)],
            ["Total Warps", str(occupancy_results.total_warps)],
            ["Active SMs", str(occupancy_results.active_sms)],
            ["Num Reps", str(occupancy_results.num_reps)],
            ["MWP", f"{occupancy_results.mwp:.2f}"],
        ],
    )

    pico_results = perform_pico_analysis(
        PicoInput(
            logger=logger,
            grid_size=grid_size,
            input_file=input_file,
            block_size=block_size,
            problem_size=problem_size,
            bank_count=None,
            parameters=[],
        )
    )

    if pico_results.error:
        logger.error("PICO error")
        logger.info("Continuing without PICO results")
    else:
        logger.table(
            "PICO Results",
            ["Result", "Value"],
            [
                ["Bank Conflicts", str(pico_results.bank_conflict_count)],
                ["Uncoalesced Count", str(pico_results.uncoalesced_count)],
                ["Active Threads", str(pico_results.active_threads)],
                ["Memory Access Count", str(pico_results.memory_accesses_count)],
            ],
        )

    sass_parser = get_parser(cfg=cfg, kernel_name=kernel_name, arch=arch, logger=logger)

    instructions = sass_parser.parse()

    logger.info("SASS has been parsed")

    dependency_results = analyze(
        cfg=cfg,
        all_instructions=instructions,
        ni=loop_iterations,
        latency=build_latency_table(hp),
        logger=logger,
    )
    logger.table(
        "Dependency Analysis",
        ["Parameter", "Value"],
        [
            ["ILP", f"{dependency_results.ilp:.2f}"],
            ["MLP", f"{dependency_results.mlp:.2f}"],
            ["Total Insts", str(dependency_results.total_inst_count)],
            ["Mem Insts", str(dependency_results.mem_inst_count)],
            ["SFU Insts", str(dependency_results.sfu_inst_count)],
            ["Sync Insts", str(dependency_results.sync_inst_count)],
        ],
    )

    prediction = predict(
        PredictionInput(
            dependency=dependency_results,
            occupancy=occupancy_results,
            pico=pico_results,
            hardware=HardwareParameters(
                warp_size=hp.warp_size,
                simd_width=hp.simd_width,
                sfu_width=hp.sfu_width,
                departure_delay=hp.departure_delay_cycles,
                sync_factor=hp.sync_factor_cycles,
                dram_latency=hp.dram_latency_cycles,
                cache_hit_latency=hp.l1_cache_hit_latency_cycles,
                bank_penalty=hp.bank_penalty_cycles,
                launch_overhead=hp.launch_overhead_cycles,
                dram_bandwidth=hp.dram_bandwidth_bytes_per_cycle,
            ),
            launch=LaunchParameters(
                block_size=block_size,
                grid_size=grid_size,
                problem_size=problem_size,
                registers_per_thread=registers_per_thread,
                shared_mem_per_block_bytes=shared_memory_per_block,
            ),
        ),
        logger=logger,
    )
    logger.table(
        "Prediction",
        ["Parameter", "Value"],
        [
            ["ITILP", f"{prediction.itilp:.2f}"],
            ["CWP", f"{prediction.cwp:.2f}"],
            ["MWP_cp", f"{prediction.mwp_cp:.2f}"],
            ["MWP_peak_bw", f"{prediction.mwp_peak_bw:.2f}"],
            ["ITMLP", f"{prediction.itmlp:.2f}"],
            ["Avg Trans/Warp", f"{prediction.average_trans_warp:.2f}"],
            ["T_comp (cycles)", f"{prediction.t_comp:.0f}"],
            ["T_mem (cycles)", f"{prediction.t_mem:.0f}"],
            ["T_overlap (cycles)", f"{prediction.t_overlap:.0f}"],
            ["T_exec (cycles)", f"{prediction.t_exec:.0f}"],
            ["T_GPU (cycles)", f"{prediction.t_gpu:.0f}"],
            ["T_one_thread (cycles)", f"{prediction.t_one_thread:.0f}"],
            ["Speedup", f"{prediction.speedup:.2f}"],
        ],
    )

    if not args.run:
        return

    measured = profile_kernel(
        ProfileInput(
            input_file=input_file,
            arch=arch,
            kernel_name=kernel_name.rstrip(":"),
            block_size=block_size,
            grid_size=grid_size,
            program_args=program_args,
            include_dirs=include_dirs,
            extra_flags=nvcc_flags,
            logger=logger,
        )
    )
    logger.table(
        "Predicted vs Measured",
        ["Parameter", "Predicted", "Measured", "Error"],
        [
            [
                "T_one_thread (cycles)",
                f"{prediction.t_one_thread:.0f}",
                f"{measured.serial.cycles:.0f}",
                relative_error(prediction.t_one_thread, measured.serial.cycles),
            ],
            [
                "T_GPU (cycles)",
                f"{prediction.t_gpu:.0f}",
                f"{measured.parallel.cycles:.0f}",
                relative_error(prediction.t_gpu, measured.parallel.cycles),
            ],
            [
                "Speedup",
                f"{prediction.speedup:.2f}",
                f"{measured.speedup:.2f}",
                relative_error(prediction.speedup, measured.speedup),
            ],
            ["One-thread duration (ns)", "", f"{measured.serial.duration_ns:.0f}", ""],
            ["Full-run duration (ns)", "", f"{measured.parallel.duration_ns:.0f}", ""],
        ],
        force=True,
    )


if __name__ == "__main__":
    main()
