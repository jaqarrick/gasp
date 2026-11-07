import argparse
import shlex
import sys
from dataclasses import dataclass

from gasp import __version__
from gasp.cfg import filter_cfg, format_cfg, gen_cfg_from_raw
from gasp.common.constants import GPU_TO_ARCH
from gasp.compare import measured_speedup, speedup, work_mismatches
from gasp.cuda_toolkit import CUDAExtractor
from gasp.dependency import LoopIterationsRequiredError, analyze
from gasp.gpuperf.types import HardwareParameters, LaunchParameters
from gasp.kernels import resolve_kernel
from gasp.logger import DEFAULT_TRACE_FILE, TRACE_MODULES, Logger
from gasp.occupancy import OccupancyInput, calculate_occupancy
from gasp.parser import get_parser
from gasp.pico import PicoInput, perform_pico_analysis
from gasp.predict import PredictionInput, PredictionOutput, predict
from gasp.profiler import ProfileInput, ProfileOutput, profile_kernel

from .config import (
    ConfigError,
    KernelSpec,
    load_config,
    parse_input,
    specs_from_inputs,
)
from .constants import DEFAULT_REPEATS
from .types import Args
from .utils import HardwareProfile, build_latency_table, load_hardware_profile


@dataclass
class KernelResult:
    spec: KernelSpec
    kernel_name: str  # mangled
    prediction: PredictionOutput
    pico_error: bool


def kernel_input(value: str) -> tuple[str, str | None]:
    try:
        return parse_input(value)
    except ConfigError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got '{value}'")
    return number


def trace_modules(value: str) -> set[str]:
    modules = {m.strip().lower() for m in value.split(",") if m.strip()}
    valid = {*TRACE_MODULES, "all"}
    unknown = modules - valid
    if not modules or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown trace module(s) {', '.join(sorted(unknown)) or repr(value)}, "
            f"expected a comma-separated list of: {', '.join(sorted(valid))}"
        )
    return modules


def resolve_traces(
    trace: set[str] | None, trace_file: str | None
) -> tuple[set[str] | None, str]:
    # --trace-file on its own traces every module
    if trace is None and trace_file is not None:
        trace = {"all"}
    return trace, trace_file or DEFAULT_TRACE_FILE


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
    parser.add_argument(
        "--trace",
        type=trace_modules,
        nargs="?",
        const={"all"},
        default=None,
        metavar="MODULES",
        help="Write traces to the trace file, optionally only for a "
        f"comma-separated list of modules ({', '.join(TRACE_MODULES)}). "
        "Defaults to all",
    )
    parser.add_argument(
        "--trace-file",
        default=None,
        help=f"File traces are written to (default: ./{DEFAULT_TRACE_FILE}). "
        "Implies --trace all if --trace is not given",
    )
    parser.add_argument(
        "inputs",
        type=kernel_input,
        nargs="*",
        metavar="INPUT",
        help="Kernels to compare, each a .cu file or file.cu::kernel. "
        "A file holding several kernels needs the kernel name",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="TOML file listing the kernels and their settings. "
        "Can't be combined with input files or per-kernel flags",
    )
    parser.add_argument(
        "--reference",
        default=None,
        metavar="NAME",
        help="Kernel the others are compared with (default: the first kernel)",
    )

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
        help="Iterations per loop in the kernel (applied to every loop). "
        "Required if the kernel contains loops",
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
        help="Compile and run each program under ncu, and compare the measured "
        "cycles and speedup with the prediction",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=None,
        metavar="N",
        help=f"ncu runs per kernel with --run (default: {DEFAULT_REPEATS})",
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


def check_args(parser: argparse.ArgumentParser, args: Args) -> None:
    if args.repeats is not None and not args.run:
        parser.error("--repeats requires --run")

    if args.config is None:
        if not args.inputs:
            parser.error("give at least one input file, or --config")
        return

    if args.inputs:
        parser.error("--config can't be combined with input files")
    per_kernel_flags = {
        "--block-size": args.block_size,
        "--grid-size": args.grid_size,
        "--problem-size": args.problem_size,
        "--loop-iterations": args.loop_iterations,
        "--include-dir": args.include_dir or None,
        "--nvcc-flags": args.nvcc_flags,
        "--program-args": args.program_args,
    }
    used = [flag for flag, value in per_kernel_flags.items() if value is not None]
    if used:
        parser.error(f"--config can't be combined with {', '.join(used)}")


def build_specs(args: Args) -> list[KernelSpec]:
    if args.config is not None:
        return load_config(args.config)

    settings = {
        "problem_size": args.problem_size,
        "block_size": args.block_size,
        "grid_size": args.grid_size,
        "loop_iterations": args.loop_iterations,
        "include_dirs": args.include_dir or None,
        "nvcc_flags": shlex.split(args.nvcc_flags) if args.nvcc_flags else None,
        "program_args": shlex.split(args.program_args) if args.program_args else None,
    }
    return specs_from_inputs(
        args.inputs, {k: v for k, v in settings.items() if v is not None}
    )


def analyze_kernel(
    spec: KernelSpec,
    arch: str,
    hp: HardwareProfile,
    warn_shared_program: bool,
    logger: Logger,
) -> KernelResult:
    name = spec.name
    logger.trace("config", f"=== kernel {name} ===")
    logger.table(
        f"Configuration: {name}",
        ["Parameter", "Value"],
        [
            ["Input", spec.file],
            ["Kernel", spec.kernel or "Not provided"],
            ["Block Size", str(spec.block_size)],
            ["Grid Size", str(spec.grid_size)],
            ["Problem Size", str(spec.problem_size)],
            ["Loop Iterations", str(spec.loop_iterations or "Not provided")],
        ],
        trace="config",
    )

    logger.info(f"Extracting SASS and resource usage from {spec.file}")

    extractor = CUDAExtractor(
        spec.file,
        arch,
        logger,
        include_dirs=spec.include_dirs,
        extra_flags=spec.nvcc_flags,
    )
    result = extractor.process()
    # {'_Z9addvectorPiS_S_i': {'reg': 12, 'shared': 0, 'stack': 0, 'local': 0}}
    resources = result["resources"]

    try:
        kernel_name = resolve_kernel(spec.kernel, resources)
        cfg = filter_cfg(gen_cfg_from_raw(raw_cfg=result["raw_cfg"]), kernel_name)
    except ValueError as e:
        logger.error(f"{name}: {spec.file}: {e}")
        sys.exit(1)
    logger.trace("cfg", format_cfg(cfg))

    if warn_shared_program and len(resources) > 1:
        logger.warning(
            f"{name}: {spec.file} holds several kernels. The GASP_* macros and program "
            "arguments apply to the whole program, and the other kernels still run "
            "and can affect the measured one. Separate files avoid this"
        )

    kernel = resources[kernel_name]
    registers_per_thread = kernel["reg"]
    shared_memory_per_block = kernel["shared"]

    occupancy_results = calculate_occupancy(
        OccupancyInput(
            block_size=spec.block_size,
            grid_size=spec.grid_size,
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
        f"Occupancy: {name}",
        ["Parameter", "Value"],
        [
            ["Active Warps/SM", str(occupancy_results.active_warps_per_sm)],
            ["Total Warps", str(occupancy_results.total_warps)],
            ["Active SMs", str(occupancy_results.active_sms)],
            ["Num Reps", str(occupancy_results.num_reps)],
            ["MWP", f"{occupancy_results.mwp:.2f}"],
        ],
        trace="occupancy",
    )

    pico_results = perform_pico_analysis(
        PicoInput(
            logger=logger,
            grid_size=spec.grid_size,
            input_file=spec.file,
            kernel_name=kernel_name,
            block_size=spec.block_size,
            problem_size=spec.problem_size,
            bank_count=None,
            parameters=[],
        )
    )

    if pico_results.error:
        logger.error(f"{name}: PICO error")
        logger.info("Continuing without PICO results")
    else:
        logger.table(
            f"PICO Results: {name}",
            ["Result", "Value"],
            [
                ["Bank Conflicts", str(pico_results.bank_conflict_count)],
                ["Uncoalesced Count", str(pico_results.uncoalesced_count)],
                ["Active Threads", str(pico_results.active_threads)],
                ["Memory Access Count", str(pico_results.memory_accesses_count)],
            ],
            trace="pico",
        )

    sass_parser = get_parser(cfg=cfg, kernel_name=kernel_name, arch=arch, logger=logger)

    instructions = sass_parser.parse()

    logger.info("SASS has been parsed")

    try:
        dependency_results = analyze(
            cfg=cfg,
            all_instructions=instructions,
            loop_iterations=spec.loop_iterations,
            latency=build_latency_table(hp),
            logger=logger,
        )
    except LoopIterationsRequiredError as e:
        logger.error(
            f"{name}: {e}. Rerun with --loop-iterations <N>, "
            "or set loop_iterations in the config"
        )
        sys.exit(1)
    logger.table(
        f"Dependency Analysis: {name}",
        ["Parameter", "Value"],
        [
            ["ILP", f"{dependency_results.ilp:.2f}"],
            ["MLP", f"{dependency_results.mlp:.2f}"],
            ["Total Insts", str(dependency_results.total_inst_count)],
            ["Mem Insts", str(dependency_results.mem_inst_count)],
            ["SFU Insts", str(dependency_results.sfu_inst_count)],
            ["Sync Insts", str(dependency_results.sync_inst_count)],
        ],
        trace="dependency",
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
                block_size=spec.block_size,
                grid_size=spec.grid_size,
                problem_size=spec.problem_size,
                registers_per_thread=registers_per_thread,
                shared_mem_per_block_bytes=shared_memory_per_block,
            ),
        ),
        logger=logger,
    )
    logger.table(
        f"Prediction: {name}",
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
        ],
        trace="predict",
    )
    return KernelResult(
        spec=spec,
        kernel_name=kernel_name,
        prediction=prediction,
        pico_error=pico_results.error,
    )


def report_comparison(
    results: list[KernelResult], reference: str, logger: Logger
) -> None:
    compare = len(results) > 1
    t_reference = next(r.prediction.t_gpu for r in results if r.spec.name == reference)

    if len({r.pico_error for r in results}) > 1:
        logger.warning(
            "Pico fell back on some kernels only, so these speedups compare different "
            "coalescing assumptions"
        )

    columns = ["Kernel", "Launch", "T_comp", "T_mem", "T_overlap", "T_GPU"]
    if compare:
        columns.append("Speedup")
    columns.append("Pico")

    rows = []
    for r in results:
        p = r.prediction
        row = [
            r.spec.name,
            f"<<<{r.spec.grid_size}, {r.spec.block_size}>>>",
            f"{p.t_comp:.0f}",
            f"{p.t_mem:.0f}",
            f"{p.t_overlap:.0f}",
            f"{p.t_gpu:.0f}",
        ]
        if compare:
            row.append(f"{speedup(t_reference, p.t_gpu):.2f}")
        row.append("fallback" if r.pico_error else "ok")
        rows.append(row)

    title = (
        f"Predicted comparison (reference: {reference})" if compare else "Prediction"
    )
    logger.table(f"{title}, cycles", columns, rows, force=True, trace="predict")


def report_measured(
    results: list[KernelResult],
    measured: dict[str, ProfileOutput],
    reference: str,
    repeats: int,
    logger: Logger,
) -> None:
    compare = len(results) > 1
    t_reference = next(r.prediction.t_gpu for r in results if r.spec.name == reference)

    columns = [
        "Kernel",
        "Pred T_GPU",
        "Meas cycles (median)",
        "Spread (min-max)",
        "Error",
    ]
    if compare:
        columns += ["Pred speedup", "Meas speedup (range)", "Error"]

    rows = []
    for r in results:
        m = measured[r.spec.name]
        t_gpu = r.prediction.t_gpu
        row = [
            r.spec.name,
            f"{t_gpu:.0f}",
            f"{m.cycles:.0f}",
            f"{m.cycles_min:.0f}-{m.cycles_max:.0f}",
            relative_error(t_gpu, m.cycles),
        ]
        if compare and r.spec.name == reference:
            row += ["1.00", "1.00 (reference)", ""]
        elif compare:
            predicted = speedup(t_reference, t_gpu)
            s = measured_speedup(measured[reference], m)
            row += [
                f"{predicted:.2f}",
                f"{s.median:.2f} ({s.low:.2f}-{s.high:.2f})",
                relative_error(predicted, s.median),
            ]
        rows.append(row)

    runs = f"{repeats} run{'s' if repeats != 1 else ''} per kernel"
    title = (
        f"Predicted vs measured (reference: {reference}, {runs})"
        if compare
        else f"Predicted vs measured ({runs})"
    )
    logger.table(title, columns, rows, force=True, trace="profiler")


def main() -> None:
    parser = build_parser()
    raw = parser.parse_intermixed_args()
    args = Args(**vars(raw))
    check_args(parser, args)

    profile = args.profile
    gpu = args.gpu.lower()
    repeats = args.repeats or DEFAULT_REPEATS

    traces, trace_file = resolve_traces(args.trace, args.trace_file)
    logger = Logger(args.verbose, traces=traces, trace_file=trace_file)
    logger.info("Verbose mode enabled")
    logger.trace("config", f"command: {shlex.join(sys.argv)}")
    logger.trace("config", f"gpu: {gpu}, profile: {profile or 'Not provided'}")

    try:
        specs = build_specs(args)
    except ConfigError as e:
        logger.error(str(e))
        sys.exit(1)

    names = [spec.name for spec in specs]
    reference = args.reference or names[0]
    if reference not in names:
        logger.error(
            f"--reference {reference} is not one of the kernels: {', '.join(names)}"
        )
        sys.exit(1)

    arch = GPU_TO_ARCH.get(gpu)
    if arch is None:
        logger.error(f"No architecture defined for GPU {gpu}")
        sys.exit(1)

    for warning in work_mismatches(
        {spec.name: spec.problem_size for spec in specs},
        {spec.name: spec.program_args for spec in specs},
    ):
        logger.warning(warning)

    hp = load_hardware_profile(gpu, logger)
    results = [
        analyze_kernel(spec, arch, hp, warn_shared_program=args.run, logger=logger)
        for spec in specs
    ]
    report_comparison(results, reference, logger)

    if not args.run:
        return

    measured = {
        r.spec.name: profile_kernel(
            ProfileInput(
                name=r.spec.name,
                input_file=r.spec.file,
                arch=arch,
                kernel_name=r.kernel_name,
                block_size=r.spec.block_size,
                grid_size=r.spec.grid_size,
                program_args=r.spec.program_args,
                include_dirs=r.spec.include_dirs,
                extra_flags=r.spec.nvcc_flags,
                repeats=repeats,
                logger=logger,
            )
        )
        for r in results
    }
    report_measured(results, measured, reference, repeats, logger)


if __name__ == "__main__":
    main()
