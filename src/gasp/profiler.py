import csv
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from gasp.logger import Logger, format_table

REQUIRED: list[str] = ["nvcc", "ncu"]

DURATION_METRIC = "gpu__time_duration.sum"
CYCLES_METRIC = "sm__cycles_elapsed.max"
METRICS: list[str] = [
    DURATION_METRIC,
    CYCLES_METRIC,
    "smsp__inst_executed.sum",
    "smsp__inst_executed_op_global_ld.sum",
    "smsp__inst_executed_op_global_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]

# cuobjdump -sass instruction line: /*0040*/  IMAD R2, R3, R4, RZ ;  /* 0x... */
_SASS_INSTRUCTION = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?;)")


@dataclass
class ProfileInput:
    input_file: str
    arch: str
    kernel_name: str
    block_size: int
    grid_size: int
    program_args: list[str]
    include_dirs: list[str]
    extra_flags: list[str]
    logger: Logger


@dataclass
class KernelMeasurement:
    duration_ns: float
    cycles: float
    metrics: dict[str, float]


@dataclass
class ProfileOutput:
    serial: KernelMeasurement
    parallel: KernelMeasurement
    speedup: float


def check_packages(logger: Logger) -> None:
    missing = [t for t in REQUIRED if not shutil.which(t)]
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error("Install the CUDA Toolkit with Nsight Compute")
        sys.exit(1)


def build_executable(
    i: ProfileInput, block_size: int, grid_size: int, out_path: str
) -> None:
    cmd = [
        "nvcc",
        f"-arch={i.arch}",
        *(f"-I{d}" for d in i.include_dirs),
        *i.extra_flags,
        f"-DGASP_BLOCK_SIZE={block_size}",
        f"-DGASP_GRID_SIZE={grid_size}",
        i.input_file,
        "-o",
        out_path,
    ]
    i.logger.info(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        i.logger.error(f"nvcc failed:\n{result.stderr}")
        sys.exit(1)


def _csv_start(lines: list[str]) -> int | None:
    return next(
        (n for n, line in enumerate(lines) if line.startswith(('"ID",', "ID,"))),
        None,
    )


def parse_ncu_csv(text: str, logger: Logger) -> dict[str, float]:
    # ncu prints ==PROF== lines and the program's own stdout before the CSV
    lines = text.splitlines()
    start = _csv_start(lines)
    if start:
        logger.trace("profiler", "--- program output ---\n" + "\n".join(lines[:start]))
    if start is None:
        logger.error(f"No ncu metrics found in output:\n{text}")
        sys.exit(1)

    rows = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    metrics: dict[str, float] = {}
    for row in rows:
        name = row.get("Metric Name")
        value = row.get("Metric Value")
        if not name or value is None:
            continue
        try:
            metrics[name] = float(value.replace(",", ""))
        except ValueError:
            logger.warning(f"Skipping non-numeric ncu value {name} = {value}")

    if not metrics:
        logger.error(f"No ncu metrics found in output:\n{text}")
        sys.exit(1)
    return metrics


def run_ncu(
    exe: str, program_args: list[str], kernel_name: str, logger: Logger
) -> dict[str, float]:
    cmd = [
        "ncu",
        "--csv",
        "--print-units",
        "base",
        "--kernel-name-base",
        "mangled",
        "--kernel-name",
        kernel_name,
        "--launch-count",
        "1",
        "--metrics",
        ",".join(METRICS),
        exe,
        *program_args,
    ]
    logger.info(f"Profiling: {' '.join(cmd)}")
    logger.trace("profiler", f"ncu command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        logger.error(f"ncu failed:\n{result.stderr}\n{result.stdout}")
        sys.exit(1)
    return parse_ncu_csv(result.stdout, logger)


def run_ncu_report(
    exe: str, program_args: list[str], kernel_name: str, logger: Logger
) -> str:
    """Human-readable `ncu --set full` report, for traces only."""
    cmd = [
        "ncu",
        "--set",
        "full",
        "--kernel-name-base",
        "mangled",
        "--kernel-name",
        kernel_name,
        "--launch-count",
        "1",
        exe,
        *program_args,
    ]
    logger.trace("profiler", f"ncu report command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        logger.warning("ncu --set full failed, see the trace file")
        return f"ncu --set full failed:\n{result.stderr}\n{result.stdout}"
    return result.stdout


def extract_sass_instructions(text: str) -> list[str]:
    """Instruction lines of cuobjdump -sass output, without encodings or headers."""
    instructions = []
    for line in text.splitlines():
        m = _SASS_INSTRUCTION.match(line)
        if m:
            instructions.append(f"/*{m.group(1)}*/ {m.group(2)}")
    return instructions


def dump_kernel_text(exe: str, kernel_name: str, logger: Logger) -> str:
    """The kernel's .text as SASS instructions, for traces only."""
    if not shutil.which("cuobjdump"):
        logger.warning("cuobjdump not found, skipping kernel .text trace")
        return "cuobjdump not found"
    cmd = ["cuobjdump", "-sass", "-fun", kernel_name, exe]
    result = subprocess.run(cmd, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        logger.warning("cuobjdump failed, skipping kernel .text trace")
        return f"cuobjdump failed:\n{result.stderr}"
    instructions = extract_sass_instructions(result.stdout)
    if not instructions:
        return f"no instructions found for {kernel_name}"
    return "\n".join(instructions)


def trace_measurement(
    i: ProfileInput,
    exe: str,
    label: str,
    metrics: dict[str, float],
) -> None:
    logger = i.logger
    logger.trace(
        "profiler",
        f"=== {label}: kernel .text ===\n"
        + dump_kernel_text(exe, i.kernel_name, logger),
    )
    rows = [[name, f"{value:,.2f}"] for name, value in sorted(metrics.items())]
    logger.trace(
        "profiler",
        f"=== {label}: collected metrics ===\n"
        + format_table(["Metric", "Value"], rows),
    )
    logger.info(f"Running ncu --set full on the {label} run for the trace")
    logger.trace(
        "profiler",
        f"=== {label}: ncu --set full ===\n"
        + run_ncu_report(exe, i.program_args, i.kernel_name, logger),
    )


def measure(
    i: ProfileInput, block_size: int, grid_size: int, workdir: str, name: str
) -> KernelMeasurement:
    exe = os.path.join(workdir, name)
    label = f"{name} <<<{grid_size}, {block_size}>>>"
    build_executable(i, block_size, grid_size, exe)
    i.logger.trace("profiler", f"=== {label}: {i.kernel_name} ===")
    metrics = run_ncu(exe, i.program_args, i.kernel_name, i.logger)
    for required in (DURATION_METRIC, CYCLES_METRIC):
        if required not in metrics:
            i.logger.error(f"ncu did not report {required} for {name} run")
            sys.exit(1)
    if i.logger.trace_enabled("profiler"):
        trace_measurement(i, exe, label, metrics)
    return KernelMeasurement(
        duration_ns=metrics[DURATION_METRIC],
        cycles=metrics[CYCLES_METRIC],
        metrics=metrics,
    )


def profile_kernel(i: ProfileInput) -> ProfileOutput:
    check_packages(i.logger)
    with tempfile.TemporaryDirectory(prefix="gasp-profile-") as workdir:
        i.logger.info("Profiling one-thread run <<<1, 1>>>")
        serial = measure(i, 1, 1, workdir, "serial")
        i.logger.info(f"Profiling full run <<<{i.grid_size}, {i.block_size}>>>")
        parallel = measure(i, i.block_size, i.grid_size, workdir, "parallel")

    return ProfileOutput(
        serial=serial,
        parallel=parallel,
        speedup=serial.cycles / parallel.cycles,
    )
