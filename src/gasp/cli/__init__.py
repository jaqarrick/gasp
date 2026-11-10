import argparse
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from gasp import __version__
from gasp.cfg import filter_cfg, format_cfg, gen_cfg_from_raw
from gasp.common.constants import GPU_TO_ARCH
from gasp.compare import (
    carveout_mismatches,
    measured_speedup,
    speedup,
    work_mismatches,
)
from gasp.cuda_toolkit import CUDAExtractor, LabelLines
from gasp.dependency import (
    DependencyOutput,
    LoopCountMismatchError,
    LoopInfo,
    LoopIterationsRequiredError,
    analyze,
    describe_loops,
)
from gasp.gpuperf.types import HardwareParameters, LaunchParameters
from gasp.kernels import resolve_kernel
from gasp.launch import Dims, format_dims, parse_dims
from gasp.logger import DEFAULT_TRACE_FILE, TRACE_MODULES, Logger
from gasp.metrics_file import (
    MetricsFileError,
    SavedKernel,
    SavedMetrics,
    check_matches,
    load_metrics,
    median_metrics,
    save_metrics,
)
from gasp.occupancy import OccupancyInput, OccupancyOutput, calculate_occupancy
from gasp.overrides import (
    GROUPS,
    OverrideError,
    build_overrides,
    format_overrides,
    parse_overrides,
    uses_profiled,
)
from gasp.parser import get_parser
from gasp.pico import PicoInput, PicoOutput, perform_pico_analysis
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
from .utils import (
    GPU_PROFILES,
    HardwareProfile,
    build_latency_table,
    load_hardware_profile,
    load_profile_file,
)


@dataclass
class PreparedKernel:
    """One kernel's static analysis: everything predict() needs, computed once.
    predict_kernel can then be called on it again for each set of overrides."""

    spec: KernelSpec
    kernel_name: str  # mangled
    dependency: DependencyOutput
    occupancy: OccupancyOutput
    pico: PicoOutput
    hardware: HardwareParameters
    launch: LaunchParameters


@dataclass
class KernelResult:
    spec: KernelSpec
    kernel_name: str  # mangled
    prediction: PredictionOutput
    pico_error: bool
    overrides: str = "none"  # label, see overrides.format_overrides


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


def dims(value: str) -> Dims:
    try:
        return parse_dims(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def kernel_param(value: str) -> tuple[str, int]:
    name, separator, number = value.partition("=")
    try:
        if not separator or not name.strip():
            raise ValueError
        return name.strip(), int(number)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected NAME=INTEGER, such as M=1024, got '{value}'"
        ) from None


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

    # exclusive, but checked in main() so passing both warns instead of a usage error
    parser.add_argument("--gpu", help="built-in GPU name (e.g., b200)")
    parser.add_argument("--profile", help="path to custom hardware profile JSON")
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
        type=dims,
        default=None,
        help="Threads per block, such as 256 or 16x16",
    )
    parser.add_argument(
        "--grid-size",
        type=dims,
        default=None,
        help="Blocks in the grid, such as 3907 or 64x64. "
        "Required when --block-size has several dimensions",
    )
    parser.add_argument(
        "--problem-size", type=int, default=None, help="Size of the input"
    )
    parser.add_argument(
        "--footprint",
        type=int,
        default=None,
        help="Distinct global bytes the kernel reads and writes, such as 12000000 "
        "for vecadd with 1000000 floats. Without it, the cache model estimates "
        "reuse per thread",
    )
    parser.add_argument(
        "--reuse",
        type=float,
        default=None,
        help="Share of global accesses (0 to 1) that hit L1 on data the same "
        "block just loaded, such as 0.94 for a 16x16 matrix multiply where 16 "
        "threads read each value. Sets the L1 hit ratio directly",
    )
    parser.add_argument(
        "--loop-iterations",
        type=int,
        default=None,
        help="Iterations per loop in the kernel (applied to every loop). "
        "Required if the kernel contains loops, unless the config entry sets "
        "loop_counts",
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
    parser.add_argument(
        "--param",
        type=kernel_param,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Value of a kernel argument passed to Pico, such as M=1024 (repeatable)",
    )
    parser.add_argument(
        "--save-metrics",
        default=None,
        metavar="FILE",
        help="With --run: save the ncu measurements to FILE (JSON), to replay "
        "later with --metrics",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        metavar="FILE",
        help="Use the measurements in FILE (from --save-metrics) instead of "
        "running ncu. Needs no GPU. Can't be combined with --run",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="GROUP=SOURCE",
        help="Replace a group of the model's static inputs (repeatable). Groups: "
        f"{', '.join(GROUPS)}, or all. Sources: profiled (from --metrics), or "
        "crude for coalescing and bank (no Pico: coalesced, no conflicts)",
    )

    return parser


def format_params(params: dict[str, int]) -> str:
    return ", ".join(f"{name}={value}" for name, value in params.items())


def relative_error(predicted: float, measured: float) -> str:
    if measured == 0:
        return "n/a"
    return f"{(predicted - measured) / measured * 100:+.1f}%"


def check_args(parser: argparse.ArgumentParser, args: Args) -> None:
    if args.gpu is None and args.profile is None:
        parser.error("give --gpu or --profile")

    if args.repeats is not None and not args.run:
        parser.error("--repeats requires --run")

    if args.save_metrics is not None and not args.run:
        parser.error("--save-metrics requires --run")
    if args.metrics is not None and args.run:
        parser.error("--metrics replays a saved --run, so it can't be combined with it")
    try:
        choices = parse_overrides(args.override)
    except OverrideError as e:
        parser.error(f"--override: {e}")
    if uses_profiled(choices) and args.metrics is None:
        parser.error("--override with a profiled source requires --metrics")

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
        "--footprint": args.footprint,
        "--reuse": args.reuse,
        "--loop-iterations": args.loop_iterations,
        "--include-dir": args.include_dir or None,
        "--nvcc-flags": args.nvcc_flags,
        "--program-args": args.program_args,
        "--param": args.param or None,
    }
    used = [flag for flag, value in per_kernel_flags.items() if value is not None]
    if used:
        parser.error(f"--config can't be combined with {', '.join(used)}")


def build_specs(args: Args) -> list[KernelSpec]:
    if args.config is not None:
        return load_config(args.config)

    settings = {
        "problem_size": args.problem_size,
        "footprint": args.footprint,
        "reuse": args.reuse,
        "block_size": list(args.block_size) if args.block_size else None,
        "grid_size": list(args.grid_size) if args.grid_size else None,
        "loop_iterations": args.loop_iterations,
        "include_dirs": args.include_dir or None,
        "nvcc_flags": shlex.split(args.nvcc_flags) if args.nvcc_flags else None,
        "program_args": shlex.split(args.program_args) if args.program_args else None,
        "params": dict(args.param) if args.param else None,
    }
    return specs_from_inputs(
        args.inputs, {k: v for k, v in settings.items() if v is not None}
    )


def format_loop_counts(counts: list[int] | None) -> str:
    return "Not provided" if counts is None else str(counts)


def loop_source(
    header: str, label_lines: LabelLines, source_lines: list[str], source_file: str
) -> str:
    """file:line of a loop's first instruction, with the source text if it's ours."""
    location = label_lines.get(header)
    if location is None:
        return "unknown"
    file, line = location
    text = f"{Path(file).name}:{line}"
    if Path(file).name == Path(source_file).name and 0 < line <= len(source_lines):
        code = " ".join(source_lines[line - 1].split())
        text += f"  {code[:60]}"
    return text


def loop_rows(
    loops: list[LoopInfo],
    counts: list[str],
    label_lines: LabelLines,
    source_lines: list[str],
    source_file: str,
) -> list[list[str]]:
    number = {loop.header: i for i, loop in enumerate(loops, start=1)}
    rows = []
    for loop, count in zip(loops, counts, strict=True):
        loads = []
        if loop.global_loads:
            loads.append(f"{loop.global_loads} global")
        if loop.shared_loads:
            loads.append(f"{loop.shared_loads} shared")
        rows.append(
            [
                str(number[loop.header]),
                loop_source(loop.header, label_lines, source_lines, source_file),
                f"#{number[loop.parent]}" if loop.parent else "-",
                str(loop.inst_count),
                ", ".join(loads) or "-",
                count,
            ]
        )
    return rows


def report_loops(
    spec: KernelSpec,
    loops: list[LoopInfo],
    label_lines: LabelLines,
    logger: Logger,
    force: bool = False,
) -> None:
    """Print the kernel's loops in code order, the order loop_counts follows."""
    if not loops:
        return
    if spec.loop_counts is not None and len(spec.loop_counts) == len(loops):
        counts = [str(c) for c in spec.loop_counts]
    elif spec.loop_iterations is not None and spec.loop_counts is None:
        counts = [str(spec.loop_iterations)] * len(loops)
    else:
        counts = ["?"] * len(loops)
    try:
        with open(spec.file, encoding="utf-8", errors="replace") as f:
            source_lines = f.read().splitlines()
    except OSError:
        source_lines = []
    logger.table(
        f"Loops: {spec.name} ({len(loops)} in the compiled code, in loop_counts order)",
        ["#", "Source line", "Inside", "Insts", "Loads/trip", "Count"],
        loop_rows(loops, counts, label_lines, source_lines, spec.file),
        force=force,
        trace="dependency",
    )


def loop_counts_template(count: int) -> str:
    return f"loop_counts = [{', '.join(['?'] * count)}]"


def prepare_kernel(
    spec: KernelSpec,
    arch: str,
    hp: HardwareProfile,
    warn_shared_program: bool,
    logger: Logger,
) -> PreparedKernel:
    """The static analysis of one kernel: nvcc, CFG, occupancy, Pico, SASS
    parser and dependency analysis. The slow part, done once per kernel."""
    name = spec.name
    logger.trace("config", f"=== kernel {name} ===")
    logger.table(
        f"Configuration: {name}",
        ["Parameter", "Value"],
        [
            ["Input", spec.file],
            ["Kernel", spec.kernel or "Not provided"],
            ["Block Size", format_dims(spec.block_dim)],
            ["Grid Size", format_dims(spec.grid_dim)],
            ["Problem Size", str(spec.problem_size)],
            ["Footprint (bytes)", str(spec.footprint or "Not provided")],
            ["Reuse", "Not provided" if spec.reuse is None else f"{spec.reuse:g}"],
            ["Loop Iterations", str(spec.loop_iterations or "Not provided")],
            ["Loop Counts", format_loop_counts(spec.loop_counts)],
            ["Kernel Params", format_params(spec.params) or "Not provided"],
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
            input_file=spec.file,
            kernel_name=kernel_name,
            block_dim=spec.block_dim,
            grid_dim=spec.grid_dim,
            problem_size=spec.problem_size,
            bank_count=None,
            params=spec.params,
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

    loops = describe_loops(cfg, instructions)
    label_lines = result.get("label_lines", {})
    if loops and spec.loop_counts is not None and spec.loop_iterations is not None:
        logger.warning(
            f"{name}: loop_counts is set, so loop_iterations = "
            f"{spec.loop_iterations} is ignored"
        )

    try:
        dependency_results = analyze(
            cfg=cfg,
            all_instructions=instructions,
            loop_iterations=spec.loop_iterations,
            latency=build_latency_table(hp),
            logger=logger,
            loop_counts=spec.loop_counts,
        )
    except LoopCountMismatchError as e:
        report_loops(spec, loops, label_lines, logger, force=True)
        logger.error(
            f"{name}: loop_counts has {e.given} number(s), but the compiled kernel "
            f"has {len(loops)} loop(s), listed above in code order. Set "
            f"{loop_counts_template(len(loops))} on its [[kernel]] entry"
        )
        sys.exit(1)
    except LoopIterationsRequiredError:
        report_loops(spec, loops, label_lines, logger, force=True)
        logger.error(
            f"{name}: the compiled kernel has {len(loops)} loop(s), listed above in "
            f"code order, and no trip counts. Set "
            f"{loop_counts_template(len(loops))} on its [[kernel]] entry in a "
            "config, one count per loop, or give every loop the same count with "
            "--loop-iterations <N> or loop_iterations"
        )
        sys.exit(1)
    report_loops(spec, loops, label_lines, logger)
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
            ["Divergent Insts", str(dependency_results.divergent_inst_count)],
        ],
        trace="dependency",
    )

    if spec.footprint is None:
        logger.info(
            f"{name}: no footprint given, so the cache model estimates reuse per thread"
        )

    return PreparedKernel(
        spec=spec,
        kernel_name=kernel_name,
        dependency=dependency_results,
        occupancy=occupancy_results,
        pico=pico_results,
        hardware=HardwareParameters(
            warp_size=hp.warp_size,
            simd_width=hp.simd_width,
            sfu_width=hp.sfu_width,
            departure_delay=hp.departure_delay_cycles,
            sync_factor=hp.sync_factor,
            dram_latency=hp.dram_latency_cycles,
            l1_hit_latency=hp.l1_cache_hit_latency_cycles,
            l2_hit_latency=hp.l2_cache_hit_latency_cycles,
            l1_shared_array_bytes=hp.l1_shared_array_bytes,
            shared_mem_carveouts_bytes=hp.shared_mem_carveouts_bytes,
            l2_cache_size_bytes=hp.l2_cache_size_bytes,
            bank_penalty=hp.bank_penalty_cycles,
            launch_overhead_base=hp.launch_overhead_base_cycles,
            launch_overhead_per_thread=hp.launch_overhead_per_thread_cycles,
            dram_bandwidth=hp.dram_bandwidth_bytes_per_cycle,
        ),
        launch=LaunchParameters(
            block_size=spec.block_size,
            grid_size=spec.grid_size,
            problem_size=spec.problem_size,
            registers_per_thread=registers_per_thread,
            shared_mem_per_block_bytes=shared_memory_per_block,
        ),
    )


def predict_kernel(
    prepared: PreparedKernel,
    choices: dict[str, str],
    saved: SavedMetrics | None,
    logger: Logger,
) -> KernelResult:
    """Run the model on a prepared kernel, with the chosen overrides.

    choices: {group: source} from parse_overrides, empty for all static.
    saved: the metrics file, needed when a source is profiled."""
    spec = prepared.spec
    name = spec.name
    label = format_overrides(choices)
    metrics = median_metrics(saved.kernels[name].profile) if saved else None
    try:
        overrides = build_overrides(choices, metrics, prepared.occupancy.total_warps)
    except OverrideError as e:
        logger.error(f"{name}: {e}")
        sys.exit(1)
    logger.trace("predict", f"=== {name}: overrides {label} ===")

    prediction = predict(
        PredictionInput(
            footprint_bytes=spec.footprint,
            block_reuse=spec.reuse,
            dependency=prepared.dependency,
            occupancy=prepared.occupancy,
            pico=prepared.pico,
            hardware=prepared.hardware,
            launch=prepared.launch,
            overrides=overrides,
        ),
        logger=logger,
    )
    logger.table(
        f"Prediction: {name} (overrides: {label})",
        ["Parameter", "Value"],
        [
            ["ITILP", f"{prediction.itilp:.2f}"],
            ["CWP", f"{prediction.cwp:.2f}"],
            ["MWP_cp", f"{prediction.mwp_cp:.2f}"],
            ["MWP_peak_bw", f"{prediction.mwp_peak_bw:.2f}"],
            ["ITMLP", f"{prediction.itmlp:.2f}"],
            ["Avg Trans/Warp", f"{prediction.average_trans_warp:.2f}"],
            ["Working set/SM (bytes)", str(prediction.working_set_per_sm_bytes)],
            ["Working set total (bytes)", str(prediction.working_set_total_bytes)],
            ["L1 size (bytes)", str(prediction.l1_cache_size_bytes)],
            ["Shared carveout (bytes)", str(prediction.shared_mem_carveout_bytes)],
            ["Reuse fraction", f"{prediction.reuse_fraction:.3f}"],
            ["L1 hit ratio", f"{prediction.l1_hit_ratio:.3f}"],
            ["L2 hit ratio", f"{prediction.l2_hit_ratio:.3f}"],
            ["AMAT (cycles)", f"{prediction.amat:.1f}"],
            ["O_CFdiv (cycles)", f"{prediction.o_cfdiv:.0f}"],
            ["T_comp (cycles)", f"{prediction.t_comp:.0f}"],
            ["T_mem (cycles)", f"{prediction.t_mem:.0f}"],
            ["T_overlap (cycles)", f"{prediction.t_overlap:.0f}"],
            ["T_exec (cycles)", f"{prediction.t_exec:.0f}"],
            ["Launch overhead (cycles)", f"{prediction.launch_overhead:.0f}"],
            ["T_GPU (cycles)", f"{prediction.t_gpu:.0f}"],
        ],
        trace="predict",
    )
    return KernelResult(
        spec=spec,
        kernel_name=prepared.kernel_name,
        prediction=prediction,
        pico_error=prepared.pico.error,
        overrides=label,
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
    columns += ["Pico", "Overrides"]

    rows = []
    for r in results:
        p = r.prediction
        row = [
            r.spec.name,
            f"<<<{format_dims(r.spec.grid_dim)}, {format_dims(r.spec.block_dim)}>>>",
            f"{p.t_comp:.0f}",
            f"{p.t_mem:.0f}",
            f"{p.t_overlap:.0f}",
            f"{p.t_gpu:.0f}",
        ]
        if compare:
            row.append(f"{speedup(t_reference, p.t_gpu):.2f}")
        row += ["fallback" if r.pico_error else "ok", r.overrides]
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


def load_hardware(
    gpu: str | None, profile: str | None, logger: Logger
) -> tuple[str, str, HardwareProfile]:
    """GPU name, arch and hardware profile from --gpu or --profile."""
    if gpu is not None and profile is not None:
        logger.warning("--gpu and --profile are mutually exclusive; pass only one")
        sys.exit(1)

    if profile is not None:
        hp = load_profile_file(profile, logger)
        gpu = hp.chip
        arch = hp.arch
        source = profile
    else:
        gpu = gpu.lower()
        arch = GPU_TO_ARCH.get(gpu)
        if arch is None:
            logger.error(f"No architecture defined for GPU {gpu}")
            sys.exit(1)
        hp = load_hardware_profile(gpu, logger)
        source = str(GPU_PROFILES[gpu])
    logger.trace("config", f"gpu: {gpu}, profile: {source}, arch: {arch}")
    return gpu, arch, hp


def load_saved_metrics(
    path: str, gpu: str, arch: str, specs: list[KernelSpec], logger: Logger
) -> SavedMetrics:
    """A metrics file, checked to be a profile of these kernels on this GPU."""
    try:
        saved = load_metrics(path)
        check_matches(saved, gpu, arch, specs)
    except MetricsFileError as e:
        logger.error(str(e))
        sys.exit(1)
    return saved


def profile_kernels(
    results: list[KernelResult], arch: str, repeats: int, logger: Logger
) -> dict[str, ProfileOutput]:
    return {
        r.spec.name: profile_kernel(
            ProfileInput(
                name=r.spec.name,
                input_file=r.spec.file,
                arch=arch,
                kernel_name=r.kernel_name,
                block_dim=r.spec.block_dim,
                grid_dim=r.spec.grid_dim,
                program_args=r.spec.program_args,
                include_dirs=r.spec.include_dirs,
                extra_flags=r.spec.nvcc_flags,
                repeats=repeats,
                logger=logger,
            )
        )
        for r in results
    }


def main() -> None:
    parser = build_parser()
    raw = parser.parse_intermixed_args()
    args = Args(**vars(raw))
    check_args(parser, args)

    traces, trace_file = resolve_traces(args.trace, args.trace_file)
    logger = Logger(args.verbose, traces=traces, trace_file=trace_file)
    logger.info("Verbose mode enabled")
    logger.trace("config", f"command: {shlex.join(sys.argv)}")

    gpu, arch, hp = load_hardware(args.gpu, args.profile, logger)

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

    for warning in work_mismatches(
        {spec.name: spec.problem_size for spec in specs},
        {spec.name: spec.program_args for spec in specs},
        {spec.name: spec.params for spec in specs},
    ):
        logger.warning(warning)

    choices = parse_overrides(args.override)  # validated in check_args
    saved = None
    if args.metrics is not None:
        saved = load_saved_metrics(args.metrics, gpu, arch, specs, logger)

    prepared = [
        prepare_kernel(spec, arch, hp, warn_shared_program=args.run, logger=logger)
        for spec in specs
    ]
    results = [predict_kernel(p, choices, saved, logger) for p in prepared]
    report_comparison(results, reference, logger)

    if args.run:
        repeats = args.repeats or DEFAULT_REPEATS
        measured = profile_kernels(results, arch, repeats, logger)
    elif saved is not None:
        measured = {name: saved.kernels[name].profile for name in names}
        repeats = len(measured[reference].runs)
    else:
        return

    report_measured(results, measured, reference, repeats, logger)
    predictions = {r.spec.name: r.prediction for r in results}
    for warning in carveout_mismatches(predictions, measured):
        logger.warning(warning)

    if args.save_metrics is not None:
        try:
            save_metrics(
                args.save_metrics,
                SavedMetrics(
                    gpu=gpu,
                    arch=arch,
                    kernels={
                        r.spec.name: SavedKernel(
                            kernel_name=r.kernel_name,
                            block_size=r.spec.block_size,
                            grid_size=r.spec.grid_size,
                            profile=measured[r.spec.name],
                        )
                        for r in results
                    },
                ),
            )
        except MetricsFileError as e:
            logger.error(str(e))
            sys.exit(1)
        logger.success(f"Saved ncu metrics to {args.save_metrics}")


if __name__ == "__main__":
    main()
