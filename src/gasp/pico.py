import json
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass

from gasp.logger import Logger


@dataclass
class PicoInput:
    input_file: str
    block_size: int
    grid_size: int
    problem_size: int
    bank_count: int | None
    parameters: list[dict[str, int]] | None
    logger: Logger


@dataclass
class PicoOutput:
    uncoalesced_count: int
    bank_conflict_count: int
    active_threads: int
    memory_accesses_count: int
    error: bool


pico = "faial-cost"


def parse_total_cost(raw: str) -> int:
    data = json.loads(raw)
    total = 0
    for kernel in data["kernels"]:
        if "total_cost" in kernel:
            total += int(kernel["total_cost"])
    return total


def run_pico(i: PicoInput, cmd: str, results: dict) -> None:
    params = []
    for d in i.parameters:
        for k, v in d.items():
            params.append(f"{k}={v}")

    path = [pico, "-m", cmd, "-b", str(i.block_size), "-g", str(i.grid_size)]
    if params:
        path += ["-p", *params]
    path += ["--json", i.input_file]

    result = subprocess.run(
        path,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        i.logger.error(f"pico failed:\n{result.stderr}")
        sys.exit(1)
    results[cmd] = result.stdout


def perform_pico_analysis(input: PicoInput) -> PicoOutput:
    logger = input.logger
    if shutil.which(pico) is None:
        logger.error(f"pico ({pico}) is not available on current system.")
        return PicoOutput(
            uncoalesced_count=0,
            bank_conflict_count=0,
            active_threads=0,
            memory_accesses_count=0,
            error=True,
        )

    commands = ["bc", "ua", "count", "active"]
    results: dict[str, str] = {}
    threads = []

    for cmd in commands:
        t = threading.Thread(target=run_pico, args=(input, cmd, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    bc_json = results["bc"]
    ua_json = results["ua"]
    count_json = results["count"]
    active_warps_json = results["active"]

    try:
        return PicoOutput(
            uncoalesced_count=parse_total_cost(ua_json),
            bank_conflict_count=parse_total_cost(bc_json),
            active_threads=parse_total_cost(active_warps_json),
            memory_accesses_count=parse_total_cost(count_json),
            error=False,
        )
    except ValueError as e:
        # symbolic cost, e.g. a loop bounded by a kernel argument Pico wasn't given
        logger.warning(f"pico returned a non-numeric cost: {e}")
        return PicoOutput(
            uncoalesced_count=0,
            bank_conflict_count=0,
            active_threads=0,
            memory_accesses_count=0,
            error=True,
        )
