"""scripts/run/run_hpmll_bmks.py: run the HPMLL microbenchmarks that match GASP profile fields, and build a profile """
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import re
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GASP_BM_DIR = REPO_ROOT / "hardware" / "microbenchmarks" / "gasp-bm"
DEFAULT_HPMLL_DIR = REPO_ROOT / "hardware" / "microbenchmarks" / "hpmll"


def _load_run_all():
    """gasp-bm's run_all.py, for its profile building and helpers."""
    spec = importlib.util.spec_from_file_location("run_all", GASP_BM_DIR / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_all"] = module  # @dataclass looks the module up while it runs
    spec.loader.exec_module(module)
    return module


mb = _load_run_all()
SCHEMA_VERSION = "1.0.0"

CLOCK_METRIC = "sm__cycles_elapsed.avg.per_second"
L1_HIT = "l1tex__t_sector_hit_rate.pct"
L2_HIT = "lts__t_sector_hit_rate.pct"
NCU_FLAGS = [
    "--clock-control",
    "base",
    "--cache-control",
    "none",
    "--target-processes",
    "all",
    "--csv",
    "--print-units",
    "base",
    "--print-kernel-base",
    "function",
]


TIMED_METRICS = ["gpc__cycles_elapsed.max", CLOCK_METRIC]

WARP_LOAD_BYTES = 128  # one 32-bit load per lane of a warp
SECTOR_BYTES = 32
# mem_bw allocates six float arrays of ARRAY_SIZE (2^29, 2 GiB each: 12 GiB)
#  On a GPU without the memory, it is built from a copy with a smaller power-of-two ARRAY_SIZE
MEM_BW_ARRAYS = 6
MEM_BW_ARRAY_SIZE = 536870912
MEM_BW_HEADROOM = 512 * 1024**2  # left free for the CUDA context
MEM_BW_MIN_L2_MULTIPLE = 8  # a smaller array must still be at least 8x the L2
# we need to increase this because hpmll only uses 8 trips. since one of those is the first
# l1 load (i.e. the first DRAM trip), the average is brought down. Increasing this iteration count
# Makes 256 trips
L1_LAT_ITERS = 1048576


@dataclass
class Check:
    """Cache hit rate a result needs to be valid: metric >= low, or <= high."""

    metric: str
    low: float | None = None
    high: float | None = None
    why: str = ""

    def failure(self, rate: float) -> str | None:
        if self.low is not None and rate < self.low:
            return f"{self.metric} = {rate:.1f}% < {self.low:.0f}% ({self.why})"
        if self.high is not None and rate > self.high:
            return f"{self.metric} = {rate:.1f}% > {self.high:.0f}% ({self.why})"
        return None


@dataclass
class HpmllBench:
    name: str  # directory under RegularUnits/
    kernel: str  # timed kernel's function name (for its ncu rows)
    pattern: str  # regex for the printed result; group 1 is the number
    field: str  # profile key
    unit: str
    # (printed value, measured SM clock GHz, max SM clock GHz) -> field value
    convert: Callable[[float, float, float], float]
    conversion: str  # how convert works, for the run JSON
    check: Check | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def source(self) -> str:
        return f"RegularUnits/{self.name}/{self.name}.cu"


def as_printed(x: float, _clock: float, _max_clock: float) -> float:
    return x


def shared_load_cycles(x: float, _clock: float, _max_clock: float) -> float:
    return WARP_LOAD_BYTES / x


def l1_sectors(x: float, _clock: float, _max_clock: float) -> float:
    return x / SECTOR_BYTES


def l2_bytes_per_cycle(x: float, clock: float, max_clock: float) -> float:
    # l2_bw prints bytes / (globaltimer ns x deviceProp.clockRate); undo the max clock
    return x * max_clock / clock


def gb_per_s_to_bytes_per_cycle(x: float, clock: float, _max_clock: float) -> float:
    return x / clock  # GB/s is bytes per ns


NUMBER = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
L1_CHECK = Check(L1_HIT, low=95.0, why="the array must stay in L1")
L2_CHECK = Check(L2_HIT, low=95.0, why="the array must stay in L2")

BENCHES = [
    HpmllBench(
        "alu_lat_float",
        "max_flops",
        rf"fp32 latency\s*=\s*{NUMBER}",
        "fp32_inst_latency_cycles",
        "cycles",
        as_printed,
        "printed value (dependent fma.rn.f32 chain)",
    ),
    HpmllBench(
        "alu_lat_double",
        "max_flops",
        rf"fp64 latency\s*=\s*{NUMBER}",
        "fp64_inst_latency_cycles",
        "cycles",
        as_printed,
        "printed value (dependent fma.rn.f64 chain)",
    ),
    HpmllBench(
        "alu_lat_int32",
        "max_flops",
        rf"int32 latency\s*=\s*{NUMBER}",
        "int32_inst_latency_cycles",
        "cycles",
        as_printed,
        "printed value (dependent mad.lo.s32 chain)",
        notes=["measures IMAD (mad.lo.s32), not IADD3"],
    ),
    HpmllBench(
        "l1_lat",
        "l1_lat",
        rf"L1 Latency\s*=\s*{NUMBER}",
        "l1_cache_hit_latency_cycles",
        "cycles",
        as_printed,
        "printed value (ld.global.ca pointer chase, 32 KB array, 256 passes)",
        # 256 passes leave one cold pass in 256 (>= 99.6% hits); at 99%, misses
        # at L2 latency add at most ~2 cycles
        check=Check(L1_HIT, low=99.0, why="the chase must hit L1"),
        notes=[f"built with ITERS {L1_LAT_ITERS} (hpmll: 32768) so the cold first pass is negligible"],
    ),
    HpmllBench(
        "l2_lat",
        "l2_lat",
        rf"L2 Latency\s*=\s*{NUMBER}",
        "l2_cache_hit_latency_cycles",
        "cycles",
        as_printed,
        "printed value (ld.global.cg pointer chase, 32 KB array)",
        check=Check(L2_HIT, low=90.0, why="the chase must hit L2"),
    ),
    HpmllBench(
        "mem_lat",
        "mem_lat",
        rf"Mem latency\s*=\s*{NUMBER}",
        "dram_latency_cycles",
        "cycles",
        as_printed,
        "printed value (ld.global.cg chase over 256 KB, after writing an 8 MB array)",
        check=Check(
            L2_HIT, high=10.0, why="the chase must miss L2; an L2 of 8 MB or more keeps it"
        ),
    ),
    HpmllBench(
        "shared_lat",
        "shared_lat",
        rf"Shared Memory Latency\s*=\s*{NUMBER}",
        "shared_mem_load_latency_cycles",
        "cycles",
        as_printed,
        "printed value (ld.shared pointer chase)",
        notes=[
            "each load in the chain waits on a shift that computes its address, "
            "so this is shared load latency plus about 4-5 cycles"
        ],
    ),
    HpmllBench(
        "shared_bw",
        "shared_bw",
        rf"Shared Memory Bandwidth\s*=\s*{NUMBER}",
        "shared_mem_load_throughput_cycles",
        "cycles",
        shared_load_cycles,
        f"{WARP_LOAD_BYTES} / printed bytes per clock per SM",
    ),
    *[
        HpmllBench(
            name,
            "l1_bw",
            rf"L1 bandwidth\s*=\s*{NUMBER}",
            "l1_sectors_per_cycle_per_sm",
            "sectors/cycle/SM",
            l1_sectors,
            f"printed bytes per clock per SM / {SECTOR_BYTES}; the field takes the "
            "highest valid shape",
            check=L1_CHECK,
        )
        for name in ("l1_bw_32f", "l1_bw_64f", "l1_bw_128")
    ],
    *[
        HpmllBench(
            name,
            "l2_bw",
            rf"L2 bandwidth\s*=\s*{NUMBER}",
            "l2_bandwidth_bytes_per_cycle",
            "bytes/cycle",
            l2_bytes_per_cycle,
            "printed value x max SM clock / measured SM clock (it converts globaltimer "
            "ns with deviceProp.clockRate, the max clock); the field takes the highest "
            "valid shape",
            check=L2_CHECK,
            notes=["only the last of the benchmark's launches is timed"],
        )
        for name in ("l2_bw_32f", "l2_bw_64f", "l2_bw_128")
    ],
    HpmllBench(
        "mem_bw",
        "mem_bw",
        rf"Mem BW\s*=\s*{NUMBER}\s*\(GB/sec\)",
        "dram_bandwidth_bytes_per_cycle",
        "bytes/cycle",
        gb_per_s_to_bytes_per_cycle,
        "printed GB/s (bytes per ns) / measured SM clock",
        notes=["counts 5 reads and 1 write per element"],
    ),
]
BY_NAME = {b.name: b for b in BENCHES}
FIELDS = list(dict.fromkeys(b.field for b in BENCHES))


class BenchmarkError(Exception):
    pass

def parse_printed(bench: HpmllBench, stdout: str) -> float:
    match = re.search(bench.pattern, stdout)
    if match is None:
        raise BenchmarkError(f"no line matching {bench.pattern!r} in the output")
    return float(match.group(1))


# ncu unit -> factor to base units (cycles per second, percent)
UNIT_SCALE = {
    "": 1.0,
    "%": 1.0,
    "hz": 1.0,
    "cycle/second": 1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
    "cycle/usecond": 1e6,
    "cycle/nsecond": 1e9,
}


def ncu_rows(text: str) -> list[dict[str, str]]:
    """The metric rows of an ncu --csv log (ncu's own ==PROF== lines skipped)."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID"')), None)
    if start is None:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def metric_values(rows: list[dict[str, str]], kernel: str, metric: str) -> list[float]:
    """Every value of metric for launches of kernel, in base units."""
    values = []
    for row in rows:
        if row.get("Metric Name") != metric:
            continue
        name = row.get("Kernel Name", "")
        if name != kernel and not re.match(rf"(?:void )?{re.escape(kernel)}\b", name):
            continue
        unit = row.get("Metric Unit", "").strip().lower()
        if unit not in UNIT_SCALE:
            raise BenchmarkError(f"unexpected unit {unit!r} for {metric}")
        values.append(float(row["Metric Value"].replace(",", "")) * UNIT_SCALE[unit])
    return values


def run_clock_ghz(rows: list[dict[str, str]], kernel: str) -> float:
    values = metric_values(rows, kernel, CLOCK_METRIC)
    if not values:
        raise BenchmarkError(f"ncu recorded no {CLOCK_METRIC} for kernel {kernel}")
    return statistics.median(values) / 1e9


def makefile_flags(makefile: Path) -> list[str]:
    """The benchmark's NVCC_FLGAS (sic) from its Makefile, comments dropped."""
    if not makefile.exists():
        return []
    for line in makefile.read_text().splitlines():
        match = re.match(r"\s*NVCC_FLGAS\s*:?=(.*)", line)
        if match:
            return match.group(1).split("#")[0].split()
    return []


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


# ---- Device ------------------------------------------------------------------------


@dataclass
class Device:
    name: str
    driver: str
    memory_free_bytes: int
    max_sm_clock_ghz: float


def query_device() -> Device:
    fields = "name,driver_version,memory.free,clocks.max.sm"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise SystemExit("nvidia-smi could not query the GPU")
    name, driver, free_mib, max_mhz = (
        v.strip() for v in result.stdout.splitlines()[0].split(",")
    )
    return Device(name, driver, int(float(free_mib)) * 1024**2, float(max_mhz) / 1000)


def git_short(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def nvcc_version() -> str:
    result = subprocess.run(["nvcc", "--version"], check=False, capture_output=True, text=True)
    match = re.search(r"release (\d+\.\d+)", result.stdout)
    return match.group(1) if match else "unknown"


def mem_bw_array_size(free_bytes: int) -> int:
    """Largest power-of-two ARRAY_SIZE, up to hpmll's, whose six arrays fit in free_bytes."""
    size = MEM_BW_ARRAY_SIZE
    while size > 1 and MEM_BW_ARRAYS * size * 4 + MEM_BW_HEADROOM > free_bytes:
        size //= 2
    return size


def source_changes(bench: HpmllBench, device: Device) -> dict[str, int]:
    """#defines to change in a copy of the source before building ({}: build it as is)."""
    if bench.name == "l1_lat":
        return {"ITERS": L1_LAT_ITERS}
    if bench.name != "mem_bw":
        return {}
    size = mem_bw_array_size(device.memory_free_bytes)
    return {} if size == MEM_BW_ARRAY_SIZE else {"ARRAY_SIZE": size}


def skip_reason(bench: HpmllBench, device: Device, l2_bytes: int) -> str | None:
    if bench.name == "mem_bw":
        size = mem_bw_array_size(device.memory_free_bytes)
        if size * 4 < MEM_BW_MIN_L2_MULTIPLE * l2_bytes:
            return (
                f"{device.memory_free_bytes / 1024**3:.1f} GiB free fits arrays of "
                f"{size * 4 / 1024**2:.0f} MiB, less than {MEM_BW_MIN_L2_MULTIPLE}x the "
                f"{l2_bytes / 1024**2:.1f} MiB L2"
            )
    return None


def patch_defines(source: str, defines: dict[str, int]) -> str:
    """source with each `#define NAME value` line set to the new value."""
    for name, value in defines.items():
        pattern = re.compile(rf"^(#define\s+{re.escape(name)}\s+)\S+", re.M)
        source, n = pattern.subn(rf"\g<1>{value}", source)
        if n != 1:
            raise BenchmarkError(f"expected one #define {name} in the source, found {n}")
    return source



@dataclass
class Result:
    bench: HpmllBench
    printed: list[float]
    clocks_ghz: list[float]
    values: list[float]
    stdout_lines: list[str]
    hit_rates: dict[str, float] = field(default_factory=dict)
    invalid: str | None = None
    source_changes: dict[str, dict] = field(default_factory=dict)  # name -> {hpmll, used}


def build(
    bench: HpmllBench,
    hpmll_dir: Path,
    build_dir: Path,
    arch: str,
    log_path: Path,
    defines: dict[str, int],
) -> Path:
    """Compile the benchmark from the submodule, or from a patched copy in
    build_dir/src/ when defines is not empty (the submodule is never edited)."""
    src_dir = hpmll_dir / "RegularUnits" / bench.name
    source = src_dir / f"{bench.name}.cu"
    if defines:
        copy = build_dir / "src" / bench.name / source.name
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(patch_defines(source.read_text(), defines))
        source = copy
    binary = build_dir / bench.name
    command = [
        "nvcc",
        *makefile_flags(src_dir / "Makefile"),
        f"-arch={arch}",
        str(source),
        "-o",
        str(binary),
        "-lcudart",
    ]
    mb.run_logged(command, log_path)
    return binary


def run_once(binary: Path, metrics: list[str], log_path: Path, csv_path: Path) -> tuple[str, list]:
    command = ["ncu", *NCU_FLAGS, "--metrics", ",".join(metrics), "--log-file", str(csv_path), str(binary)]
    mb.run_logged(command, log_path, cwd=binary.parent)
    return log_path.read_text(), ncu_rows(csv_path.read_text())


def original_defines(source: str, names: list[str]) -> dict[str, int]:
    values = {}
    for name in names:
        match = re.search(rf"^#define\s+{re.escape(name)}\s+(\d+)", source, re.M)
        if match:
            values[name] = int(match.group(1))
    return values


def run_bench(
    bench: HpmllBench,
    hpmll_dir: Path,
    root: Path,
    arch: str,
    reps: int,
    max_clock: float,
    defines: dict[str, int],
) -> Result:
    bench_dir = root / bench.name
    bench_dir.mkdir(parents=True, exist_ok=True)
    build_log = bench_dir / "build.log"
    build_log.write_text("")
    binary = build(bench, hpmll_dir, root / "build", arch, build_log, defines)

    printed, clocks, values, lines = [], [], [], []
    for i in range(reps):
        log_path = bench_dir / f"run-{i}.log"
        log_path.write_text("")
        stdout, rows = run_once(binary, TIMED_METRICS, log_path, bench_dir / f"ncu-{i}.csv")
        x = parse_printed(bench, stdout)
        clock = run_clock_ghz(rows, bench.kernel)
        printed.append(x)
        clocks.append(clock)
        values.append(bench.convert(x, clock, max_clock))
        lines.append(re.search(bench.pattern, stdout).group(0))
    result = Result(bench, printed, clocks, values, lines)
    if defines:
        before = original_defines((hpmll_dir / bench.source).read_text(), list(defines))
        result.source_changes = {
            name: {"hpmll": before.get(name), "used": value} for name, value in defines.items()
        }

    if bench.check is not None:
        log_path = bench_dir / "check.log"
        log_path.write_text("")
        _, rows = run_once(binary, [bench.check.metric], log_path, bench_dir / "check.csv")
        rates = metric_values(rows, bench.kernel, bench.check.metric)
        if not rates:
            raise BenchmarkError(f"ncu recorded no {bench.check.metric} for {bench.kernel}")
        rate = statistics.median(rates)
        result.hit_rates[bench.check.metric] = rate
        result.invalid = bench.check.failure(rate)
    return result


def field_value(results: list[Result]) -> tuple[float, Result] | None:
    """The field's value from its valid results: the highest median (several
    shapes measure one bandwidth), or None when none is valid."""
    valid = [r for r in results if r.invalid is None]
    if not valid:
        return None
    best = max(valid, key=lambda r: statistics.median(r.values))
    return statistics.median(best.values), best


def run_json(
    key: str, results: list[Result], chosen: Result, value: float, context: dict
) -> dict:
    def entry(r: Result) -> dict:
        return {
            "benchmark": f"hpmll/{r.bench.source}",
            "stats": {k: round(v, 4) for k, v in summarize(r.values).items()},
            "values": [round(v, 4) for v in r.values],
            "printed_values": r.printed,
            "printed_lines": r.stdout_lines,
            "sm_clock_ghz_samples": [round(c, 4) for c in r.clocks_ghz],
            "conversion": r.bench.conversion,
            "hit_rates_pct": {k: round(v, 2) for k, v in r.hit_rates.items()},
            "valid": r.invalid is None,
            "invalid_reason": r.invalid,
            "source_changes": r.source_changes,
            "notes": r.bench.notes,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": f"hpmll/{chosen.bench.source}",
        "parameter": key,
        "value": round(value, 4),
        "unit": chosen.bench.unit,
        "gpu": context["gpu"],
        "gpu_name": context["gpu_name"],
        "arch": context["arch"],
        "stats": {k: round(v, 4) for k, v in summarize(chosen.values).items()},
        "clock": {
            "sm_clock_ghz": round(statistics.median(chosen.clocks_ghz), 4),
            "max_sm_clock_ghz": context["max_sm_clock_ghz"],
        },
        "candidates": {r.bench.name: entry(r) for r in results},
        "reps": len(chosen.values),
        "hpmll_commit": context["hpmll_commit"],
        "nvcc_version": context["nvcc_version"],
        "driver_version": context["driver"],
        "git_commit": context["git_commit"],
        "timestamp": context["timestamp"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gpu", required=True, help="GPU key, e.g. rtx2080ti")
    parser.add_argument("--version", required=True, help="run label, e.g. 1.2 (directory 1_2)")
    parser.add_argument("--arch", help="nvcc -arch (default: from the GPU's compute capability)")
    parser.add_argument(
        "--base-profile",
        help="profile to copy unmeasured fields from "
        "(default: hardware/profiles/<gpu>.json, else <gpu>.placeholder.json)",
    )
    parser.add_argument("--only", help="comma-separated benchmarks: " + ", ".join(BY_NAME))
    parser.add_argument("--reps", type=int, default=5, help="runs per benchmark (default 5)")
    parser.add_argument("--hpmll-dir", type=Path, default=DEFAULT_HPMLL_DIR, help="hpmll checkout")
    parser.add_argument("--force", action="store_true", help="reuse an existing output directory")
    args = parser.parse_args()

    if args.reps < 1:
        raise SystemExit("--reps must be at least 1")
    for tool in ("nvcc", "ncu", "nvidia-smi"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH")
    hpmll_dir = args.hpmll_dir.resolve()
    if not (hpmll_dir / "RegularUnits").is_dir():
        raise SystemExit(
            f"{hpmll_dir} has no RegularUnits/; add the submodule:\n"
            "  git submodule add git@github.com:HPMLL/NVIDIA-Hopper-Benchmark.git "
            "hardware/microbenchmarks/hpmll"
        )

    names = list(BY_NAME)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in BY_NAME]
        if unknown:
            raise SystemExit(f"unknown benchmark(s): {', '.join(unknown)}")

    version = args.version.replace(".", "_")
    root = REPO_ROOT / "hardware" / "runs" / args.gpu / version / "hpmll"
    if root.exists() and not args.force:
        raise SystemExit(f"{root} exists; pass --force to run into it again")
    (root / "build").mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_profile) if args.base_profile else mb.default_base_profile(args.gpu)
    if not base_path.exists():
        raise SystemExit(f"base profile {base_path} not found; pass --base-profile")
    arch = args.arch or mb.detect_arch()
    device = query_device()
    l2_bytes = json.loads(base_path.read_text())["spec"]["l2_cache_size_bytes"]
    mb.log(
        f"GPU {args.gpu} ({device.name}), {arch}, version {version}, {args.reps} reps, "
        f"max SM clock {device.max_sm_clock_ghz:.3f} GHz, base profile {base_path}"
    )

    results: dict[str, Result] = {}
    failed: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for name in names:
        bench = BY_NAME[name]
        reason = skip_reason(bench, device, l2_bytes)
        if reason:
            skipped[name] = reason
            mb.log(f"\n== {name}: SKIPPED, {reason}")
            continue
        defines = source_changes(bench, device)
        mb.log(f"\n== {name}")
        for define, value in defines.items():
            mb.log(f"  building from a copy with {define} {value}")
        try:
            result = run_bench(
                bench, hpmll_dir, root, arch, args.reps, device.max_sm_clock_ghz, defines
            )
        except (mb.BenchmarkError, BenchmarkError) as exc:
            failed[name] = str(exc)
            mb.log(f"  FAILED: {exc}")
            continue
        results[name] = result
        median = statistics.median(result.values)
        mb.log(f"  {bench.field} = {median:.4f} {bench.unit}")
        if result.invalid:
            mb.log(f"  INVALID: {result.invalid}")

    context = {
        "gpu": args.gpu,
        "gpu_name": device.name,
        "arch": arch,
        "max_sm_clock_ghz": round(device.max_sm_clock_ghz, 4),
        "hpmll_commit": git_short(hpmll_dir),
        "nvcc_version": nvcc_version(),
        "driver": device.driver,
        "git_commit": mb.git_commit(),
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # One run JSON per field, from this run's results
    measured: dict[str, dict] = {}
    not_measured: dict[str, str] = {}
    for key in FIELDS:
        benches = [b.name for b in BENCHES if b.field == key and b.name in names]
        if not benches:
            continue
        field_results = [results[n] for n in benches if n in results]
        picked = field_value(field_results)
        if picked is None:
            reasons = [f"{n}: {failed.get(n) or skipped.get(n) or results[n].invalid}" for n in benches]
            not_measured[key] = "; ".join(reasons)
            continue
        value, chosen = picked
        field_dir = root / key
        field_dir.mkdir(exist_ok=True)
        path = field_dir / f"{key}.{args.gpu}.run.json"
        path.write_text(json.dumps(run_json(key, field_results, chosen, value, context), indent=2) + "\n")
        measured[key] = {
            "value": round(value, 4),
            "source": f"{path.relative_to(REPO_ROOT)} (hpmll {chosen.bench.name})",
            "sm_clock_ghz": round(statistics.median(chosen.clocks_ghz), 4),
        }

    clocks = [c for r in results.values() for c in r.clocks_ghz]
    clock_ghz = statistics.median(clocks) if clocks else None
    if clock_ghz is not None:
        mb.log(f"\nSM clock: median {clock_ghz:.4f} GHz over {len(clocks)} runs")
        for r in results.values():
            c = statistics.median(r.clocks_ghz)
            if abs(c - clock_ghz) > mb.CLOCK_TOLERANCE * clock_ghz:
                mb.log(f"  WARNING: {r.bench.name} ran at {c:.4f} GHz, not the base clock")

    if failed:
        mb.log("\nNot writing the profile, because some benchmarks failed:")
        for name, reason in failed.items():
            mb.log(f"  {name}: {reason}")
        return 1

    base = json.loads(base_path.read_text())
    try:
        profile, sources = mb.build_profile(base, measured, clock_ghz)
    except ValueError as exc:
        raise SystemExit(f"{exc} ({base_path}); fix the base profile or pass --base-profile")
    for key, reason in not_measured.items():
        sources[key]["hpmll_not_used"] = reason

    profile_path = root / f"{args.gpu}.{version}.hpmll.json"
    sources_path = root / f"{args.gpu}.{version}.hpmll.sources.json"
    profile_path.write_text(json.dumps(profile, indent=4) + "\n")
    sources_path.write_text(
        json.dumps(
            {
                "version": args.version,
                "gpu": args.gpu,
                "arch": arch,
                "suite": "hpmll",
                "hpmll_commit": context["hpmll_commit"],
                "base_profile": str(base_path.relative_to(REPO_ROOT))
                if base_path.is_relative_to(REPO_ROOT)
                else str(base_path),
                "git_commit": context["git_commit"],
                "timestamp": context["timestamp"],
                "fields": sources,
            },
            indent=4,
        )
        + "\n"
    )
    copied = [k for k, v in sources.items() if v["source"] == "copied from base profile"]
    mb.log(f"\nProfile: {profile_path}\nSources: {sources_path}")
    mb.log(f"Measured by hpmll: {', '.join(measured) or 'none'}")
    for key, reason in not_measured.items():
        mb.log(f"Not measured, copied from {base_path.name}: {key} ({reason})")
    others = [k for k in copied if k not in not_measured]
    mb.log(f"Copied from {base_path.name} (hpmll has no benchmark): {', '.join(others) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
