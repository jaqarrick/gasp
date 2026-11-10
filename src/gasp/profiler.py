import csv
import io
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from gasp.launch import Dims, format_dims, padded, total
from gasp.logger import Logger, format_table
from gasp.overrides import PROFILED_METRICS

REQUIRED: list[str] = ["nvcc", "ncu"]

DURATION_METRIC = "gpu__time_duration.sum"
CYCLES_METRIC = "gpc__cycles_elapsed.max"
GRID_SIZE_METRIC = "launch__grid_size"
BLOCK_SIZE_METRIC = "launch__block_size"
SHARED_CONFIG_METRIC = "launch__shared_mem_config_size"
METRICS: list[str] = [
    DURATION_METRIC,
    CYCLES_METRIC,
    GRID_SIZE_METRIC,
    BLOCK_SIZE_METRIC,
    SHARED_CONFIG_METRIC,
    # what profiled overrides are computed from
    *PROFILED_METRICS,
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]

# cuobjdump -sass instruction line: /*0040*/  IMAD R2, R3, R4, RZ ;  /* 0x... */
_SASS_INSTRUCTION = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?;)")


@dataclass
class ProfileInput:
    name: str
    input_file: str
    arch: str
    kernel_name: str
    block_dim: Dims
    grid_dim: Dims
    program_args: list[str]
    include_dirs: list[str]
    extra_flags: list[str]
    repeats: int
    logger: Logger

    @property
    def block_size(self) -> int:
        return total(self.block_dim)

    @property
    def grid_size(self) -> int:
        return total(self.grid_dim)

    @property
    def launch(self) -> str:
        return f"<<<{format_dims(self.grid_dim)}, {format_dims(self.block_dim)}>>>"


def launch_macros(i: ProfileInput) -> list[str]:
    """GASP_BLOCK_SIZE / GASP_GRID_SIZE are totals; the *_DIM_* macros give x, y, z."""
    macros = [f"-DGASP_BLOCK_SIZE={i.block_size}", f"-DGASP_GRID_SIZE={i.grid_size}"]
    for prefix, dims in (("BLOCK", i.block_dim), ("GRID", i.grid_dim)):
        for axis, size in zip("XYZ", padded(dims), strict=True):
            macros.append(f"-DGASP_{prefix}_DIM_{axis}={size}")
    return macros


@dataclass
class KernelMeasurement:
    duration_ns: float
    cycles: float
    metrics: dict[str, float]


@dataclass
class ProfileOutput:
    runs: list[KernelMeasurement]

    @property
    def cycles(self) -> float:
        return statistics.median(run.cycles for run in self.runs)

    @property
    def shared_mem_config_bytes(self) -> int | None:
        """Carveout ncu reported on the first run, None if it didn't report one."""
        value = self.runs[0].metrics.get(SHARED_CONFIG_METRIC) if self.runs else None
        return None if value is None else int(value)

    @property
    def cycles_min(self) -> float:
        return min(run.cycles for run in self.runs)

    @property
    def cycles_max(self) -> float:
        return max(run.cycles for run in self.runs)


def check_packages(logger: Logger) -> None:
    missing = [t for t in REQUIRED if not shutil.which(t)]
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error("Install the CUDA Toolkit with Nsight Compute")
        sys.exit(1)


def build_executable(i: ProfileInput, out_path: str) -> None:
    cmd = [
        "nvcc",
        f"-arch={i.arch}",
        *(f"-I{d}" for d in i.include_dirs),
        *i.extra_flags,
        *launch_macros(i),
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


def measure(i: ProfileInput, exe: str, trace: bool) -> KernelMeasurement:
    label = f"{i.name} {i.launch}"
    i.logger.trace("profiler", f"=== {label}: {i.kernel_name} ===")
    metrics = run_ncu(exe, i.program_args, i.kernel_name, i.logger)
    for required in (
        DURATION_METRIC,
        CYCLES_METRIC,
        GRID_SIZE_METRIC,
        BLOCK_SIZE_METRIC,
    ):
        if required not in metrics:
            i.logger.error(f"ncu did not report {required} for {i.name}")
            sys.exit(1)
    # a program that ignores the GASP_* macros launches its own config. ncu reports
    # totals, so a different shape with the same totals isn't caught
    launched_grid = int(metrics[GRID_SIZE_METRIC])
    launched_block = int(metrics[BLOCK_SIZE_METRIC])
    if (launched_grid, launched_block) != (i.grid_size, i.block_size):
        i.logger.error(
            f"{i.name} launched <<<{launched_grid}, {launched_block}>>>, "
            f"expected {i.launch} ({i.grid_size} blocks of {i.block_size} threads). "
            "The program must take its launch config from the GASP_* macros, or the "
            "requested grid and block size must match what it launches "
            "(see benchmarks/README.md)"
        )
        sys.exit(1)
    if trace and i.logger.trace_enabled("profiler"):
        trace_measurement(i, exe, label, metrics)
    return KernelMeasurement(
        duration_ns=metrics[DURATION_METRIC],
        cycles=metrics[CYCLES_METRIC],
        metrics=metrics,
    )


def profile_kernel(i: ProfileInput) -> ProfileOutput:
    check_packages(i.logger)
    runs = []
    with tempfile.TemporaryDirectory(prefix="gasp-profile-") as workdir:
        exe = os.path.join(workdir, "kernel")
        build_executable(i, exe)
        for run in range(i.repeats):
            i.logger.info(f"Profiling {i.name} {i.launch}, run {run + 1}/{i.repeats}")
            # ncu --set full is slow, so only the first run gets the full trace
            runs.append(measure(i, exe, trace=run == 0))
    return ProfileOutput(runs=runs)
