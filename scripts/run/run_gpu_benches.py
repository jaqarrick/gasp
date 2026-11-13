"""scripts/run/run_gpu_benches.py: run gpu-benches' gpu-stream and gpu-latency, and
compare their DRAM bandwidth and L1/L2/DRAM latency with GASP's profile values.

This is a cross-check, not a profile source. gpu-benches times kernels from the host
(cudaEvent) and prints GB/s or nanoseconds turned into cycles with the clock NVML
reports, so it is not run under ncu (ncu's replay would be inside the timing).
Values are converted to base-clock units with the base profile's clock_rate_ghz:

    gpu-stream   bytes/cycle = read GB/s / base GHz. Exact: a DRAM-bound kernel's
                 GB/s does not depend on the SM clock.
    gpu-latency  cycles = ns x base GHz. Exact only when the SM clock was at base
                 during the run (lock it with `sudo nvidia-smi -lgc <MHz>,<MHz>`).
                 Otherwise the part of the latency that runs on the SM/L2 clock is
                 scaled wrongly. The SM clock is sampled during each run and a
                 warning is printed when it is not at base.

Output, for `--gpu rtx2080ti --version 1.2`:

    hardware/runs/rtx2080ti/1_2/gpu-benches/
        build/                              binaries and the patched gpu-latency copy
        gpu-stream/   build.log run.log sm_clock.csv
        gpu-latency/  build.log run.log sm_clock.csv
        rtx2080ti.1_2.gpu-benches.json      parsed tables, picked values, comparison

gpu-latency is built from a copy whose size sweep stops at --latency-max-l2 x L2
(the original goes to 2 GB). The submodule is never edited.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GASP_BM_DIR = REPO_ROOT / "hardware" / "microbenchmarks" / "gasp-bm"
DEFAULT_GPU_BENCHES_DIR = REPO_ROOT / "hardware" / "microbenchmarks" / "gpu-benches"
SCHEMA_VERSION = "1.0.0"


def _load_run_all():
    """gasp-bm's run_all.py, for its helpers."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_all", GASP_BM_DIR / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_all"] = module  # @dataclass looks the module up while it runs
    spec.loader.exec_module(module)
    return module


mb = _load_run_all()

# gpu-stream allocates four double buffers of this many elements (main.cu)
STREAM_BUFFER_ELEMS = 256 * 1024 * 1024 + 2
STREAM_BUFFERS = 4
STREAM_HEADROOM = 512 * 1024**2  # left free for the CUDA context
STREAM_KERNELS = ["init", "read", "scale", "triad", "3pt", "5pt"]

# gpu-latency: one 128-byte cache line per chase step; the sweep bound in main.cu
LATENCY_LINE_BYTES = 128
LATENCY_SWEEP_BOUND = "LEN < (1 << 24)"

# Windows of the latency curve each field is taken from, by working-set size
L1_WINDOW_FRACTION = 0.5  # sizes up to half the L1
L2_WINDOW_L1_MULTIPLE = 4  # from 4x the L1 ...
L2_WINDOW_FRACTION = 0.5  # ... to half the L2
DRAM_WINDOW_L2_MULTIPLES = (4.0, 8.0)  # gasp-bm's dram_latency ring is 4x L2

SM_CLOCK_SAMPLE_MS = 100


class BenchmarkError(Exception):
    pass


# ---- Parsing gpu-benches output --------------------------------------------------

NUMBER = r"[-+]?\d+(?:\.\d+)?"
STREAM_ROW = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+(\d+)\s+({NUMBER})%\s*\|\s*GB/s:((?:\s+{NUMBER})+)\s*$"
)
LATENCY_ROW = re.compile(rf"^\s*(\d+)\s+(\d+)((?:\s+{NUMBER}){{6}})\s*$")


def parse_stream(text: str) -> list[dict]:
    """gpu-stream's rows: launch shape and GB/s per kernel."""
    rows = []
    for line in text.splitlines():
        match = STREAM_ROW.match(line)
        if not match:
            continue
        rates = [float(x) for x in match.group(5).split()]
        if len(rates) != len(STREAM_KERNELS):
            raise BenchmarkError(f"expected {len(STREAM_KERNELS)} GB/s values: {line!r}")
        rows.append(
            {
                "block_size": int(match.group(1)),
                "threads": int(match.group(2)),
                "blocks_per_sm": int(match.group(3)),
                "occupancy_pct": float(match.group(4)),
                "gb_per_s": dict(zip(STREAM_KERNELS, rates, strict=True)),
            }
        )
    return rows


def parse_latency(text: str) -> list[dict]:
    """gpu-latency's rows: working set and time per load (median of 21 runs).

    A row is: iters, NVML clock (MHz), working set (kB), time (ms), then cycles per
    load as mean-ish value, median, 5th and 95th percentile, all at the NVML clock.
    """
    rows = []
    for line in text.splitlines():
        match = LATENCY_ROW.match(line)
        if not match:
            continue
        kb, _ms, value, median, p5, p95 = (float(x) for x in match.group(3).split())
        clock_mhz = int(match.group(2))
        if clock_mhz <= 0:
            raise BenchmarkError(f"gpu-latency printed clock {clock_mhz}: {line!r}")
        rows.append(
            {
                "iters": int(match.group(1)),
                "nvml_clock_mhz": clock_mhz,
                "bytes": round(kb * 1024),
                "cycles_at_nvml_clock": {"value": value, "median": median, "p5": p5, "p95": p95},
                "ns_median": median / clock_mhz * 1e3,
            }
        )
    return rows


# ---- Picking values ----------------------------------------------------------------


def stream_peak(rows: list[dict]) -> dict:
    """The row with the highest read GB/s (the occupancy sweep's plateau)."""
    if not rows:
        raise BenchmarkError("gpu-stream printed no result rows")
    return max(rows, key=lambda r: r["gb_per_s"]["read"])


def latency_windows(l1_bytes: int, l2_bytes: int) -> dict[str, tuple[float, float]]:
    """Working-set size windows (bytes, inclusive) for each latency field."""
    low, high = DRAM_WINDOW_L2_MULTIPLES
    return {
        "l1_cache_hit_latency_cycles": (0, L1_WINDOW_FRACTION * l1_bytes),
        "l2_cache_hit_latency_cycles": (
            L2_WINDOW_L1_MULTIPLE * l1_bytes,
            L2_WINDOW_FRACTION * l2_bytes,
        ),
        "dram_latency_cycles": (low * l2_bytes, high * l2_bytes),
    }


def window_ns(rows: list[dict], window: tuple[float, float]) -> tuple[float, list[dict]]:
    """Median ns per load over the rows whose working set is inside window."""
    low, high = window
    inside = [r for r in rows if low <= r["bytes"] <= high]
    if not inside:
        raise BenchmarkError(f"no gpu-latency sizes between {low:.0f} and {high:.0f} bytes")
    return statistics.median(r["ns_median"] for r in inside), inside


def pct_diff(value: float, reference: float) -> float | None:
    return None if reference == 0 else round((value - reference) / reference * 100, 2)


# ---- Building and running ----------------------------------------------------------


@dataclass
class Bench:
    name: str  # directory in gpu-benches
    std: str  # -std= from its Makefile
    nvml: bool  # links libnvidia-ml (gpu-clock.cuh)


BENCHES = [
    Bench("gpu-stream", "c++20", nvml=False),
    Bench("gpu-latency", "c++11", nvml=True),
]
BY_NAME = {b.name: b for b in BENCHES}


def cuda_home() -> Path:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise SystemExit("nvcc not found on PATH")
    return Path(nvcc).resolve().parents[1]


def nvml_link_flags() -> list[str]:
    """-L for libnvidia-ml.so: the CUDA toolkit's stub, when there is one."""
    flags = []
    for stubs in (cuda_home() / "lib64" / "stubs", cuda_home() / "targets" / "x86_64-linux" / "lib" / "stubs"):
        if (stubs / "libnvidia-ml.so").exists():
            flags.append(f"-L{stubs}")
    return [*flags, "-lnvidia-ml"]


def patch_latency_bound(source: str, max_len: int) -> str:
    """gpu-latency's main.cu with its size sweep stopped at max_len cache lines."""
    if source.count(LATENCY_SWEEP_BOUND) != 1:
        raise BenchmarkError(f"expected one {LATENCY_SWEEP_BOUND!r} in gpu-latency/main.cu")
    return source.replace(LATENCY_SWEEP_BOUND, f"LEN < {max_len}")


def build(bench: Bench, suite_dir: Path, build_dir: Path, arch: str, log_path: Path, max_len: int | None) -> Path:
    src_dir = suite_dir / bench.name
    source = src_dir / "main.cu"
    if max_len is not None:
        copy = build_dir / "src" / bench.name / "main.cu"
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(patch_latency_bound(source.read_text(), max_len))
        source = copy
    binary = build_dir / bench.name
    command = [
        "nvcc",
        f"-std={bench.std}",
        "-O3",
        f"-arch={arch}",
        # the copy's `#include "../X"` resolves against the original directory
        f"-I{src_dir}",
        str(source),
        "-o",
        str(binary),
        *(nvml_link_flags() if bench.nvml else []),
    ]
    mb.run_logged(command, log_path)
    return binary


def run_sampling_clock(binary: Path, log_path: Path, clock_path: Path) -> list[float]:
    """Run binary (output to log_path) while nvidia-smi samples the SM clock; the
    samples in MHz."""
    with clock_path.open("w") as clock_out:
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm",
                "--format=csv,noheader,nounits",
                f"-lms={SM_CLOCK_SAMPLE_MS}",
            ],
            stdout=clock_out,
            stderr=subprocess.DEVNULL,
        )
        try:
            mb.run_logged([str(binary)], log_path, cwd=binary.parent)
        finally:
            sampler.terminate()
            sampler.wait()
    samples = []
    for line in clock_path.read_text().splitlines():
        try:
            samples.append(float(line.strip()))
        except ValueError:
            continue
    return samples


def clock_summary(samples: list[float], base_ghz: float) -> dict:
    if not samples:
        return {"samples": 0}
    return {
        "samples": len(samples),
        "min_mhz": min(samples),
        "median_mhz": statistics.median(samples),
        "max_mhz": max(samples),
        "at_base": all(abs(s / 1000 - base_ghz) <= mb.CLOCK_TOLERANCE * base_ghz for s in samples),
    }


def free_memory_bytes() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise SystemExit("nvidia-smi could not query the GPU's free memory")
    return int(float(result.stdout.splitlines()[0])) * 1024**2


def git_short(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


# ---- Comparison --------------------------------------------------------------------


def reference_profiles(gpu: str, version: str, base_path: Path) -> dict[str, Path]:
    """Profiles to compare with: the base profile, and this version's run_all and
    hpmll profiles when they exist."""
    run_dir = REPO_ROOT / "hardware" / "runs" / gpu / version
    candidates = {
        "base": base_path,
        "gasp-bm": run_dir / f"{gpu}.{version}.json",
        "hpmll": run_dir / "hpmll" / f"{gpu}.{version}.hpmll.json",
    }
    return {name: path for name, path in candidates.items() if path.exists()}


def compare(values: dict[str, float], references: dict[str, Path]) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for key, value in values.items():
        row: dict = {"gpu-benches": round(value, 4)}
        for name, path in references.items():
            ref = json.loads(path.read_text()).get("profiled", {}).get(key)
            if isinstance(ref, (int, float)):
                row[name] = {"value": ref, "diff_pct": pct_diff(value, ref)}
        table[key] = row
    return table


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


# ---- Main --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gpu", required=True, help="GPU key, e.g. rtx2080ti")
    parser.add_argument("--version", required=True, help="run label, e.g. 1.2 (directory 1_2)")
    parser.add_argument("--arch", help="nvcc -arch (default: from the GPU's compute capability)")
    parser.add_argument(
        "--base-profile",
        help="profile for the base clock, L1/L2 sizes and comparison "
        "(default: hardware/profiles/<gpu>.json, else <gpu>.placeholder.json)",
    )
    parser.add_argument("--only", help="comma-separated benchmarks: " + ", ".join(BY_NAME))
    parser.add_argument(
        "--latency-max-l2",
        type=float,
        default=DRAM_WINDOW_L2_MULTIPLES[1],
        help="stop gpu-latency's size sweep at this multiple of L2 "
        f"(default {DRAM_WINDOW_L2_MULTIPLES[1]:g}; the DRAM window is "
        f"{DRAM_WINDOW_L2_MULTIPLES[0]:g}x to {DRAM_WINDOW_L2_MULTIPLES[1]:g}x L2)",
    )
    parser.add_argument(
        "--gpu-benches-dir", type=Path, default=DEFAULT_GPU_BENCHES_DIR, help="gpu-benches checkout"
    )
    parser.add_argument("--force", action="store_true", help="reuse an existing output directory")
    args = parser.parse_args()

    for tool in ("nvcc", "nvidia-smi"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH")
    suite_dir = args.gpu_benches_dir.resolve()
    if not (suite_dir / "gpu-stream" / "main.cu").exists():
        raise SystemExit(
            f"{suite_dir} has no gpu-stream/main.cu; fetch the submodule:\n"
            "  git submodule update --init hardware/microbenchmarks/gpu-benches"
        )
    if args.latency_max_l2 < DRAM_WINDOW_L2_MULTIPLES[0]:
        raise SystemExit(
            f"--latency-max-l2 must be at least {DRAM_WINDOW_L2_MULTIPLES[0]:g}, "
            "the start of the DRAM window"
        )

    names = list(BY_NAME)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in BY_NAME]
        if unknown:
            raise SystemExit(f"unknown benchmark(s): {', '.join(unknown)}")

    version = args.version.replace(".", "_")
    root = REPO_ROOT / "hardware" / "runs" / args.gpu / version / "gpu-benches"
    if root.exists() and not args.force:
        raise SystemExit(f"{root} exists; pass --force to run into it again")
    (root / "build").mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_profile) if args.base_profile else mb.default_base_profile(args.gpu)
    if not base_path.exists():
        raise SystemExit(f"base profile {base_path} not found; pass --base-profile")
    base = json.loads(base_path.read_text())
    try:
        base_ghz = float(base["profiled"]["clock_rate_ghz"])
        spec = base["spec"]
        l2_bytes = int(spec["l2_cache_size_bytes"])
        # gpu-latency asks for the smallest shared-memory carveout, so L1 gets the rest
        l1_bytes = int(spec["l1_shared_array_bytes"]) - min(spec["shared_mem_carveouts_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{base_path} lacks {exc}; it needs profiled.clock_rate_ghz, "
                         "spec.l2_cache_size_bytes, spec.l1_shared_array_bytes and "
                         "spec.shared_mem_carveouts_bytes") from exc
    arch = args.arch or mb.detect_arch()
    mb.log(
        f"GPU {args.gpu}, {arch}, version {version}, base clock {base_ghz:.3f} GHz, "
        f"L1 {l1_bytes // 1024} KiB, L2 {l2_bytes / 1024**2:.2f} MiB, base profile {base_path}"
    )

    output: dict = {
        "schema_version": SCHEMA_VERSION,
        "suite": "gpu-benches",
        "gpu": args.gpu,
        "arch": arch,
        "base_clock_ghz": base_ghz,
        "base_profile": relative(base_path),
        "benchmarks": {},
    }
    values: dict[str, float] = {}
    failed: dict[str, str] = {}
    warnings: list[str] = []

    for name in names:
        bench = BY_NAME[name]
        bench_dir = root / name
        bench_dir.mkdir(exist_ok=True)
        mb.log(f"\n== {name}")
        try:
            if name == "gpu-stream":
                need = STREAM_BUFFERS * STREAM_BUFFER_ELEMS * 8 + STREAM_HEADROOM
                free = free_memory_bytes()
                if free < need:
                    raise BenchmarkError(
                        f"needs {need / 1024**3:.1f} GiB free, the GPU has {free / 1024**3:.1f} GiB"
                    )
                max_len = None
            else:
                max_len = math.ceil(args.latency_max_l2 * l2_bytes / LATENCY_LINE_BYTES)
                mb.log(f"  building from a copy with the sweep stopped at {max_len} lines "
                       f"({max_len * LATENCY_LINE_BYTES / 1024**2:.1f} MiB)")
            build_log = bench_dir / "build.log"
            build_log.write_text("")
            binary = build(bench, suite_dir, root / "build", arch, build_log, max_len)
            run_log = bench_dir / "run.log"
            run_log.write_text("")
            samples = run_sampling_clock(binary, run_log, bench_dir / "sm_clock.csv")
            clock = clock_summary(samples, base_ghz)
            text = run_log.read_text()

            if name == "gpu-stream":
                rows = parse_stream(text)
                peak = stream_peak(rows)
                gbs = peak["gb_per_s"]["read"]
                values["dram_bandwidth_bytes_per_cycle"] = gbs / base_ghz
                mb.log(
                    f"  read peak {gbs:.0f} GB/s at {peak['threads']} threads "
                    f"({peak['occupancy_pct']:.1f}% occupancy) = "
                    f"{gbs / base_ghz:.2f} bytes/cycle at base"
                )
                output["benchmarks"][name] = {
                    "conversion": "read GB/s / base clock GHz (bytes per base-clock cycle)",
                    "peak": peak,
                    "rows": rows,
                    "sm_clock": clock,
                }
            else:
                rows = parse_latency(text)
                picked = {}
                for key, window in latency_windows(l1_bytes, l2_bytes).items():
                    ns, inside = window_ns(rows, window)
                    values[key] = ns * base_ghz
                    picked[key] = {
                        "ns": round(ns, 3),
                        "cycles_at_base": round(ns * base_ghz, 4),
                        "window_bytes": [round(window[0]), round(window[1])],
                        "sizes_in_window": len(inside),
                    }
                    mb.log(f"  {key} = {ns:.1f} ns = {ns * base_ghz:.2f} cycles at base "
                           f"({len(inside)} sizes)")
                if not clock.get("at_base", False):
                    warnings.append(
                        f"gpu-latency: SM clock {clock.get('min_mhz')}-{clock.get('max_mhz')} MHz, "
                        f"not locked at {base_ghz * 1000:.0f}; its cycle values are approximate"
                    )
                output["benchmarks"][name] = {
                    "conversion": "median ns per load in the window x base clock GHz; exact "
                    "only if the SM clock was at base during the run",
                    "picked": picked,
                    "source_changes": {
                        "sweep_bound": {"gpu-benches": LATENCY_SWEEP_BOUND, "used": f"LEN < {max_len}"}
                    },
                    "rows": rows,
                    "sm_clock": clock,
                }
        except (mb.BenchmarkError, BenchmarkError) as exc:
            failed[name] = str(exc)
            mb.log(f"  FAILED: {exc}")

    references = reference_profiles(args.gpu, version, base_path)
    output["comparison"] = compare(values, references)
    output["reference_profiles"] = {k: relative(p) for k, p in references.items()}
    output["failed"] = failed
    output["warnings"] = warnings
    output["gpu_benches_commit"] = git_short(suite_dir)
    output["git_commit"] = mb.git_commit()
    output["timestamp"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_path = root / f"{args.gpu}.{version}.gpu-benches.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")

    mb.log("\nComparison (gpu-benches vs " + ", ".join(references) + "):")
    for key, row in output["comparison"].items():
        refs = "  ".join(
            f"{n} {r['value']:.4g} ({r['diff_pct']:+.1f}%)" if r["diff_pct"] is not None else f"{n} {r['value']}"
            for n, r in row.items()
            if n != "gpu-benches"
        )
        mb.log(f"  {key}: {row['gpu-benches']:.4g}   {refs}")
    for w in warnings:
        mb.log(f"WARNING: {w}")
    mb.log(f"\nResults: {out_path}")
    if failed:
        for name, reason in failed.items():
            mb.log(f"FAILED {name}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
