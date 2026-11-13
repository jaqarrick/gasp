"""Run every microbenchmark at base clock and assemble a hardware profile.

Each benchmark is compiled with nvcc and run under
`ncu --clock-control base --cache-control none`, so its cycle counts are taken at
the same locked base clock as GASP's `--run` validation (see README "Base clock").
Launch overhead runs through run_launch_overhead.py, which calls ncu itself.

Output, for `--gpu rtx2080ti --version 1.2`:

    hardware/runs/rtx2080ti/1_2/
        build/                          compiled benchmarks
        <parameter>/                    one directory per benchmark: run JSON, run.log, ncu.log
        rtx2080ti.1_2.json              the profile (load it with `gasp --profile <path>`)
        rtx2080ti.1_2.sources.json      where every profile value came from

Benchmarks run in dependency order: shared_memory_store_cycles reads the shared
load latency, and sync_cost needs DRAM latency and bandwidth. A dependency is
taken from this run, or from an earlier run in the same version directory.
Fields no benchmark measures (the spec section) are copied from a base profile.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PROFILES_DIR = REPO_ROOT / "hardware" / "profiles"

NCU_FLAGS = [
    "--clock-control",
    "base",
    "--cache-control",
    "none",  # keep caches warm across launches, as when run directly
    "--metrics",
    "gpc__cycles_elapsed.max",  # one cheap metric, so each kernel runs once
]
CLOCK_TOLERANCE = 0.02  # warn if a run's clock is more than 2% from the median

# Run parameter -> profile key, where they differ.
PROFILE_KEY = {
    "launch_base_cycles": "launch_overhead_base_cycles",
    "launch_per_thread_cycles": "launch_overhead_per_thread_cycles",
    "sync_factor_cycles": "sync_factor",
}


@dataclass
class Benchmark:
    name: str
    source: str
    params: list[str]  # run parameters it writes (first one names its directory)
    flags: bool = True  # takes --gpu / --out-dir; the fp32/fp64/int32/sfu ones do not
    deps: list[str] = field(default_factory=list)
    in_profile: bool = True  # shared_mem_store_throughput_cycles has no profile field
    aliases: list[str] = field(default_factory=list)  # other names its run JSON may use
    # profile key -> how to read it from the run JSON, for benchmarks whose
    # profile values aren't their "value" (empty: "value" under the param's key)
    derive: dict[str, Callable[[dict], float]] = field(default_factory=dict)

    @property
    def dir_name(self) -> str:
        return "launch_overhead" if self.name == "launch_overhead" else self.params[0]

    @property
    def profile_keys(self) -> list[str]:
        if not self.in_profile:
            return []
        if self.derive:
            return list(self.derive)
        return [PROFILE_KEY.get(p, p) for p in self.params]


# l1_throughput.cu: a narrow request costs a fixed time whatever its sectors
# (the 32-bit shapes, 2 or 4 sectors, give the same request rate); wide ones
# are limited by bandwidth (the 64- and 128-bit shapes give the same sector rate)
L1_NARROW_SHAPE = "32bit_4sectors"
L1_WIDE_SHAPES = ("64bit_8sectors", "128bit_16sectors")


def l1_cycles_per_request(run: dict) -> float:
    return round(1.0 / run["plateaus"][L1_NARROW_SHAPE]["requests_per_cycle"], 4)


def l1_sectors_per_cycle(run: dict) -> float:
    rates = [run["plateaus"][s]["sectors_per_cycle"] for s in L1_WIDE_SHAPES]
    return round(statistics.mean(rates), 4)


BENCHMARKS = [
    Benchmark(
        "fp32_latency", "fp32_latency.cu", ["fp32_inst_latency_cycles"], flags=False
    ),
    Benchmark(
        "fp64_latency", "fp64_latency.cu", ["fp64_inst_latency_cycles"], flags=False
    ),
    Benchmark(
        "int32_latency", "int32_latency.cu", ["int32_inst_latency_cycles"], flags=False
    ),
    Benchmark(
        "sfu_latency", "sfu_latency.cu", ["sfu_inst_latency_cycles"], flags=False
    ),
    Benchmark("l1_latency", "l1_latency.cu", ["l1_cache_hit_latency_cycles"]),
    Benchmark("l2_latency", "l2_latency.cu", ["l2_cache_hit_latency_cycles"]),
    Benchmark("dram_latency", "dram_latency.cu", ["dram_latency_cycles"]),
    Benchmark(
        "dram_bandwidth", "dram_bandwidth.cu", ["dram_bandwidth_bytes_per_cycle"]
    ),
    Benchmark(
        "constant_memory_load", "constant_memory_load.cu", ["const_mem_latency_cycles"]
    ),
    Benchmark(
        "shared_memory_load",
        "shared_memory_load.cu",
        ["shared_mem_load_latency_cycles"],
    ),
    Benchmark("bank_conflict", "bank_conflict.cu", ["bank_penalty_cycles"]),
    Benchmark(
        "shared_memory_store_cycles",
        "shared_memory_store_cycles.cu",
        ["shared_mem_store_latency_cycles"],
        deps=["shared_memory_load"],
    ),
    Benchmark(
        "shared_memory_store_throughput",
        "shared_memory_store_throughput.cu",
        ["shared_mem_store_throughput_cycles"],
        in_profile=False,
    ),
    Benchmark(
        "shared_memory_load_throughput",
        "shared_memory_load_throughput.cu",
        ["shared_mem_load_throughput_cycles"],
    ),
    Benchmark(
        "l1_throughput",
        "l1_throughput.cu",
        ["l1_throughput_sectors_per_cycle_per_sm"],
        derive={
            "l1_cycles_per_request": l1_cycles_per_request,
            "l1_sectors_per_cycle_per_sm": l1_sectors_per_cycle,
        },
    ),
    Benchmark("l2_bandwidth", "l2_bandwidth.cu", ["l2_bandwidth_bytes_per_cycle"]),
    Benchmark("departure_delay", "departure_delay.cu", ["departure_delay_cycles"]),
    Benchmark(
        "sync_cost",
        "sync_cost.cu",
        ["sync_factor_cycles"],
        deps=["dram_latency", "dram_bandwidth"],
        aliases=["sync_factor"],
    ),
    Benchmark(
        "launch_overhead",
        "launch_overhead_bench.cu",
        ["launch_base_cycles", "launch_per_thread_cycles"],
    ),
]
BY_NAME = {b.name: b for b in BENCHMARKS}


class BenchmarkError(Exception):
    pass


# ---- Helpers ------------------------------------------------------------------


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run_logged(command: list[str], log_path: Path, cwd: Path | None = None) -> None:
    """Run a command, saving its stdout and stderr to log_path."""
    log(f"  $ {' '.join(command)}")
    with log_path.open("a") as out:
        out.write(f"$ {' '.join(command)}\n")
        out.flush()
        result = subprocess.run(
            command, check=False, stdout=out, stderr=subprocess.STDOUT, cwd=cwd
        )
    if result.returncode != 0:
        raise BenchmarkError(f"exit code {result.returncode}, see {log_path}")


def detect_arch() -> str:
    """sm_XX from the GPU's compute capability (nvidia-smi reports e.g. 7.5)."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    cap = result.stdout.strip().splitlines()[0] if result.returncode == 0 else ""
    if not cap or "." not in cap:
        raise SystemExit(
            "Could not read the compute capability from nvidia-smi; pass --arch sm_XX"
        )
    major, minor = cap.split(".")
    return f"sm_{major}{minor}"


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.stdout.strip() or "unknown"


def run_json_path(
    bench_dir: Path, param: str, aliases: list[str] | None = None
) -> Path | None:
    """The run JSON for param (or an alias) in bench_dir, whatever GPU name it used."""
    for name in [param, *(aliases or [])]:
        matches = sorted(bench_dir.glob(f"{name}.*.run.json")) + sorted(
            bench_dir.glob(f"{name}_*.json")  # run_launch_overhead.py naming
        )
        if matches:
            return matches[0]
    return None


def read_value(path: Path) -> float:
    return float(json.loads(path.read_text())["value"])


def read_clock(path: Path) -> float | None:
    data = json.loads(path.read_text())
    clock = data.get("clock", {}).get("sm_clock_ghz")
    if clock is None:
        clock = data.get("derived", {}).get("sm_clock_ghz_median")  # dram_bandwidth
    return float(clock) if clock else None


def dependency_order(names: list[str]) -> list[str]:
    """names, with every dependency listed before the benchmarks that need it."""
    ordered: list[str] = []

    def visit(name: str) -> None:
        if name in ordered:
            return
        for dep in BY_NAME[name].deps:
            if dep in names:
                visit(dep)
        ordered.append(name)

    for name in names:
        visit(name)
    return ordered


# ---- Running one benchmark ------------------------------------------------------


def l1_bytes_per_sm(gpu: str) -> int:
    """L1 left for a kernel with no shared memory: the unified array minus the
    smallest carveout, from the GPU's profile spec (64 KB on the RTX 2080 Ti)."""
    spec = json.loads(default_base_profile(gpu).read_text())["spec"]
    array = spec["l1_shared_array_bytes"]
    carveouts = spec.get("shared_mem_carveouts_bytes") or [0]
    return array - min(carveouts)


def compile_benchmark(
    bench: Benchmark, build_dir: Path, arch: str, log_path: Path
) -> Path:
    binary = build_dir / bench.name
    run_logged(
        [
            "nvcc",
            "-O3",
            "-std=c++17",
            f"-arch={arch}",
            str(SCRIPT_DIR / bench.source),
            "-o",
            str(binary),
        ],
        log_path,
    )
    return binary


def dependency_value(root: Path, dep: str) -> tuple[float, Path]:
    bench = BY_NAME[dep]
    path = run_json_path(root / bench.dir_name, bench.params[0], bench.aliases)
    if path is None:
        raise BenchmarkError(
            f"needs {dep}, which has no result in {root}; run it first (or drop --only)"
        )
    return read_value(path), path


def run_benchmark(bench: Benchmark, root: Path, gpu: str, arch: str) -> None:
    bench_dir = root / bench.dir_name
    bench_dir.mkdir(parents=True, exist_ok=True)
    log_path = bench_dir / "run.log"
    log_path.write_text("")
    binary = compile_benchmark(bench, root / "build", arch, log_path)

    if bench.name == "launch_overhead":
        # Calls ncu itself (one process per launch config), so it is not wrapped.
        run_logged(
            [
                sys.executable,
                str(SCRIPT_DIR / "run_launch_overhead.py"),
                "--binary",
                str(binary),
                "--output-dir",
                str(bench_dir),
                "--raw-dir",
                str(bench_dir / "raw"),
                "--gpu",
                gpu,
                "--clock-control",
                "base",
            ],
            log_path,
        )
        return

    args: list[str] = []
    if bench.flags:
        args = ["--gpu", gpu, "--out-dir", str(bench_dir)]
    if bench.name == "shared_memory_store_cycles":
        # The benchmark reads the load latency from its own out-dir.
        _, load_json = dependency_value(root, "shared_memory_load")
        shutil.copy(
            load_json, bench_dir / f"shared_mem_load_latency_cycles.{gpu}.run.json"
        )
    if bench.name == "l1_throughput":
        args += ["--l1-bytes", str(l1_bytes_per_sm(gpu))]
    if bench.name == "sync_cost":
        latency, _ = dependency_value(root, "dram_latency")
        bandwidth, _ = dependency_value(root, "dram_bandwidth")
        args += ["--dram-latency", repr(latency), "--dram-bandwidth", repr(bandwidth)]

    ncu = ["ncu", *NCU_FLAGS, "--log-file", str(bench_dir / "ncu.log")]
    # cwd = build dir: the fp32/fp64/int32/sfu benchmarks write next to the executable.
    run_logged([*ncu, str(binary), *args], log_path, cwd=root / "build")

    if not bench.flags:
        produced = run_json_path(root / "build", bench.params[0])
        if produced is None:
            raise BenchmarkError(f"no {bench.params[0]} JSON next to {binary}")
        shutil.move(produced, bench_dir / f"{bench.params[0]}.{gpu}.run.json")


# ---- Profile -----------------------------------------------------------------------


def default_base_profile(gpu: str) -> Path:
    measured = PROFILES_DIR / f"{gpu}.json"
    if (
        measured.exists() and measured.stat().st_size > 4
    ):  # a100.json is empty, b200.json is {}
        return measured
    return PROFILES_DIR / f"{gpu}.placeholder.json"


def build_profile(
    base: dict, measured: dict[str, dict], clock_ghz: float | None
) -> tuple[dict, dict]:
    """Copy base, overwrite measured fields; return (profile, sources).

    measured maps a profile key to {"value": float, "source": str, "sm_clock_ghz": float|None}.
    """
    profile = json.loads(json.dumps(base))
    profiled = profile["profiled"]
    sources: dict[str, dict] = {}
    unknown = sorted(set(measured) - set(profiled))
    if unknown:
        raise ValueError(
            f"measured fields not in the base profile: {', '.join(unknown)}"
        )
    for key in profiled:
        if key in measured:
            profiled[key] = measured[key]["value"]
            sources[key] = measured[key]
        elif key == "clock_rate_ghz" and clock_ghz is not None:
            profiled[key] = round(clock_ghz, 4)
            sources[key] = {
                "value": profiled[key],
                "source": "median SM clock of the runs",
            }
        else:
            sources[key] = {
                "value": profiled[key],
                "source": "copied from base profile",
            }
    return profile, sources


# ---- Main --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--gpu", required=True, help="GPU key, e.g. rtx2080ti (names the run directory)"
    )
    parser.add_argument(
        "--version", required=True, help="run label, e.g. 1.2 (directory 1_2)"
    )
    parser.add_argument(
        "--arch", help="nvcc -arch (default: from the GPU's compute capability)"
    )
    parser.add_argument(
        "--base-profile",
        help="profile to copy unmeasured fields from "
        "(default: hardware/profiles/<gpu>.json, else <gpu>.placeholder.json)",
    )
    parser.add_argument(
        "--only",
        help="comma-separated benchmarks to run: " + ", ".join(BY_NAME),
    )
    parser.add_argument(
        "--force", action="store_true", help="reuse an existing version directory"
    )
    args = parser.parse_args()

    for tool in ("nvcc", "ncu", "nvidia-smi"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH")

    names = list(BY_NAME)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in BY_NAME]
        if unknown:
            raise SystemExit(f"unknown benchmark(s): {', '.join(unknown)}")
    names = dependency_order(names)

    version = args.version.replace(".", "_")
    root = REPO_ROOT / "hardware" / "runs" / args.gpu / version
    if root.exists() and not args.force:
        raise SystemExit(f"{root} exists; pass --force to run into it again")
    (root / "build").mkdir(parents=True, exist_ok=True)

    base_path = (
        Path(args.base_profile) if args.base_profile else default_base_profile(args.gpu)
    )
    if not base_path.exists():
        raise SystemExit(f"base profile {base_path} not found; pass --base-profile")
    arch = args.arch or detect_arch()
    log(f"GPU {args.gpu}, {arch}, version {version}, base profile {base_path}")
    log(f"Order: {', '.join(names)}")

    failed: dict[str, str] = {}
    missing = [n for n in names if not (SCRIPT_DIR / BY_NAME[n].source).exists()]
    for name in missing:
        log(f"\n== {name}: SKIPPED, {BY_NAME[name].source} is not in {SCRIPT_DIR}")
    for name in names:
        if name in missing:
            continue
        bench = BY_NAME[name]
        blocked = [d for d in bench.deps if d in failed]
        if blocked:
            failed[name] = f"skipped: {', '.join(blocked)} failed"
            log(f"\n== {name}: {failed[name]}")
            continue
        log(f"\n== {name}")
        try:
            run_benchmark(bench, root, args.gpu, arch)
        except BenchmarkError as exc:
            failed[name] = str(exc)
            log(f"  FAILED: {exc}")

    # ---- Collect every result in the version directory (this run and earlier ones)
    measured: dict[str, dict] = {}
    clocks: dict[str, float] = {}
    for bench in BENCHMARKS:
        if bench.name in failed:
            continue
        for param in bench.params:
            path = run_json_path(root / bench.dir_name, param, bench.aliases)
            if path is None:
                continue
            clock = read_clock(path)
            if clock is not None:
                clocks[param] = clock
            if not bench.in_profile:
                continue
            source = str(path.relative_to(REPO_ROOT))
            if bench.derive:
                run = json.loads(path.read_text())
                for name, plateau in run.get("plateaus", {}).items():
                    if not plateau.get("plateau_ok", True):
                        log(f"  WARNING: {bench.name} shape {name} has no plateau")
                for key, read in bench.derive.items():
                    measured[key] = {
                        "value": read(run),
                        "source": f"{source} ({read.__name__})",
                        "sm_clock_ghz": clock,
                    }
            else:
                measured[PROFILE_KEY.get(param, param)] = {
                    "value": read_value(path),
                    "source": source,
                    "sm_clock_ghz": clock,
                }

    clock_ghz = statistics.median(clocks.values()) if clocks else None
    if clock_ghz is not None:
        log(f"\nSM clock: median {clock_ghz:.4f} GHz over {len(clocks)} runs")
        for param, clock in sorted(clocks.items()):
            if abs(clock - clock_ghz) > CLOCK_TOLERANCE * clock_ghz:
                log(
                    f"  WARNING: {param} ran at {clock:.4f} GHz, not the base clock; rerun it"
                )

    base = json.loads(base_path.read_text())
    try:
        profile, sources = build_profile(base, measured, clock_ghz)
    except ValueError as exc:
        raise SystemExit(
            f"{exc} ({base_path}); fix the base profile or pass --base-profile"
        )
    copied = [
        k for k, v in sources.items() if v["source"] == "copied from base profile"
    ]

    profile_path = root / f"{args.gpu}.{version}.json"
    sources_path = root / f"{args.gpu}.{version}.sources.json"
    if failed:
        log("\nNot writing the profile, because some benchmarks failed:")
        for name, reason in failed.items():
            log(f"  {name}: {reason}")
        return 1
    profile_path.write_text(json.dumps(profile, indent=4) + "\n")
    sources_path.write_text(
        json.dumps(
            {
                "version": args.version,
                "gpu": args.gpu,
                "arch": arch,
                "base_profile": str(base_path.relative_to(REPO_ROOT))
                if base_path.is_relative_to(REPO_ROOT)
                else str(base_path),
                "git_commit": git_commit(),
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "fields": sources,
            },
            indent=4,
        )
        + "\n"
    )
    log(f"\nProfile: {profile_path}\nSources: {sources_path}")
    log(f"Copied from {base_path.name} (not measured): {', '.join(copied) or 'none'}")
    if missing:
        log(f"Not run (source missing): {', '.join(missing)}; their fields were copied")
    # Launch overhead records no clock, but run_launch_overhead.py locks base clock itself.
    launch_keys = {PROFILE_KEY[p] for p in BY_NAME["launch_overhead"].params}
    clocked = {
        key
        for bench in BENCHMARKS
        if any(p in clocks for p in bench.params)
        for key in bench.profile_keys
    }
    no_clock = sorted(set(measured) - clocked - launch_keys)
    if no_clock:
        log(f"No clock recorded (check these ran at base clock): {', '.join(no_clock)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
