import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass

from gasp.kernels import kernel_base_name, source_base_name
from gasp.launch import Dims
from gasp.logger import Logger


@dataclass
class PicoInput:
    input_file: str
    kernel_name: str
    block_dim: Dims
    grid_dim: Dims
    problem_size: int
    bank_count: int | None
    params: dict[str, int]  # kernel arguments, each passed as -p NAME=VALUE
    logger: Logger


@dataclass
class PicoOutput:
    uncoalesced_count: int
    bank_conflict_count: int
    active_threads: int
    memory_accesses_count: int
    error: bool


# faial-cost is found on PATH unless GASP_FAIAL_COST gives its path
PICO_ENV = "GASP_FAIAL_COST"
PICO_DEFAULT = "faial-cost"


def pico_executable() -> str:
    return os.environ.get(PICO_ENV) or PICO_DEFAULT


def format_dim(dims: Dims) -> str:
    """faial-cost's form: `256` for one dimension, `[16,16]` for several."""
    if len(dims) == 1:
        return str(dims[0])
    return "[" + ",".join(str(d) for d in dims) + "]"


def pico_command(i: PicoInput, cmd: str) -> list[str]:
    command = [pico_executable(), "-m", cmd, "-b", format_dim(i.block_dim)]
    command += ["-g", format_dim(i.grid_dim)]
    for name, value in i.params.items():
        command += ["-p", f"{name}={value}"]
    return command + ["--json", i.input_file]


def parse_total_cost(raw: str, kernel_name: str) -> int:
    """Cost of one kernel. Raises ValueError unless exactly one entry matches it."""
    base = kernel_base_name(kernel_name)
    data = json.loads(raw)
    matches = [
        kernel
        for kernel in data.get("kernels", [])
        if source_base_name(kernel.get("kernel_name", "")) == base
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one result for {base}, got {len(matches)}")
    if "total_cost" not in matches[0]:
        raise ValueError(f"no cost reported for {base}")
    return int(matches[0]["total_cost"])


def run_pico(i: PicoInput, cmd: str, results: dict) -> None:
    path = pico_command(i, cmd)
    i.logger.trace("pico", f"command: {' '.join(path)}")

    result = subprocess.run(
        path,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        # the caller falls back when a result is missing
        i.logger.error(f"pico failed:\n{result.stderr}")
        return
    i.logger.trace("pico", f"{cmd} result: {result.stdout.strip()}")
    results[cmd] = result.stdout


def _fallback() -> PicoOutput:
    return PicoOutput(
        uncoalesced_count=0,
        bank_conflict_count=0,
        active_threads=0,
        memory_accesses_count=0,
        error=True,
    )


def perform_pico_analysis(input: PicoInput) -> PicoOutput:
    logger = input.logger
    pico = pico_executable()
    if shutil.which(pico) is None:
        if PICO_ENV in os.environ:
            logger.error(f"pico: {PICO_ENV}={pico} is not an executable file.")
        else:
            logger.error(
                f"pico ({pico}) is not on PATH. Run scripts/install_faial.sh, then "
                f"add its bin directory to PATH or set {PICO_ENV} to the "
                f"faial-cost path."
            )
        return _fallback()

    commands = ["bc", "ua", "count", "active"]
    results: dict[str, str] = {}
    threads = []

    for cmd in commands:
        t = threading.Thread(target=run_pico, args=(input, cmd, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if len(results) != len(commands):
        return _fallback()

    try:
        return PicoOutput(
            uncoalesced_count=parse_total_cost(results["ua"], input.kernel_name),
            bank_conflict_count=parse_total_cost(results["bc"], input.kernel_name),
            active_threads=parse_total_cost(results["active"], input.kernel_name),
            memory_accesses_count=parse_total_cost(results["count"], input.kernel_name),
            error=False,
        )
    except ValueError as e:
        # a symbolic cost, e.g. a loop bounded by a kernel argument Pico wasn't given,
        # or a kernel Pico doesn't report exactly once
        logger.warning(f"pico returned no usable cost: {e}")
        return _fallback()
