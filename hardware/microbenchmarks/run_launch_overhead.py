# This script runs ncu on launch_overhead_bench for 4 configurations
# (<<<1,1>>>, <<<1,256>>>, <<<32,256>>>, <<<512,256>>>), repeated --reps times.
# Each benchmark process launches the empty kernel --warmup times and then once
# more; ncu profiles every launch, so each run yields a cold measurement (the
# first launch) and a warm measurement (the last launch).
# For every repetition it fits launch_overhead = base + per_thread * (grid*block)
# by least squares. The warm fits give the reported value (median over
# repetitions); the cold fits are kept alongside for comparison.
# One JSON file per fitted parameter is written to hardware/runs/, named after
# the GPU so results from different GPUs never overwrite each other.
# Every run must report the same GPU (UUID, name, compute capability, SM count),
# otherwise the script aborts.
# CUDA compilation and update to the GASP profile must be done manually.

# Run the script with this command (Change path as needed): python3 hardware/microbenchmarks/run_launch_overhead.py --binary ./launch_overhead_bench

"""Profile launch_overhead_bench with ncu and fit its per-thread launch model."""

import argparse
import csv
import io
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = "1"  # TODO: set to the schema_version agreed on Discord
BENCHMARK = "launch_overhead_bench.cu"
METRICS = (
    "sm__cycles_elapsed.max",
    "sm__cycles_active.max",
)
CONFIGS = (
    (1, 1),
    (1, 32),
    (1, 256),
    (1, 1024),
    (32, 256),
    (128, 256),
    (256, 256),
    (512, 256),
)
GPU_INFO_PREFIX = "GPU_INFO "
GPU_INFO_REQUIRED = (
    "bench_version",
    "device",
    "name",
    "uuid",
    "compute_capability",
    "num_sms",
    "l2_cache_bytes",
    "max_warps_per_sm",
    "shared_mem_per_sm_bytes",
    "cuda_runtime",
    "nvcc",
)
# Fields that must be identical in every run, otherwise the runs used different GPUs.
GPU_ID_FIELDS = ("uuid", "name", "compute_capability", "num_sms")
# (JSON parameter name, unit, index into the (base, slope, r_squared) fit tuple)
PARAMETERS = (
    ("launch_base_cycles", "cycles", 0),
    ("launch_per_thread_cycles", "cycles/thread", 1),
)


class GpuMismatchError(RuntimeError):
    """Two runs of the benchmark reported different GPUs."""


def run(
    command: list[str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, cwd=cwd)


def run_text(command: list[str], cwd: Path | None = None) -> str:
    """Stdout of an optional-metadata command, or '' if it can't run or fails."""
    try:
        result = run(command, cwd=cwd)
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "unknown_gpu"


def architecture_name(compute_capability: str) -> str:
    """Map CUDA compute capability to a GPU architecture family."""
    major, minor = map(int, compute_capability.split("."))

    if major == 7 and minor == 0:
        return "volta"
    if major == 7 and minor == 5:
        return "turing"
    if major == 8:
        return "ampere"
    if major == 9:
        return "hopper"
    if major in (10, 12):
        return "blackwell"

    return "unknown"


def parse_gpu_info(*texts: str) -> dict:
    """Find the benchmark's GPU_INFO line in the given outputs and decode it."""
    for text in texts:
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith(GPU_INFO_PREFIX):
                continue
            try:
                info = json.loads(line[len(GPU_INFO_PREFIX) :])
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed GPU_INFO line: {line!r}") from exc
            missing = [key for key in GPU_INFO_REQUIRED if key not in info]
            if missing:
                raise RuntimeError(
                    f"GPU_INFO is missing {missing}. Recompile launch_overhead_bench.cu."
                )
            return info
    raise RuntimeError(
        "No GPU_INFO line found in the benchmark output. "
        "Recompile launch_overhead_bench.cu with the current source."
    )


def check_same_gpu(reference: dict, info: dict, label: str) -> None:
    differences = [
        f"{field}: {reference[field]!r} vs {info[field]!r}"
        for field in GPU_ID_FIELDS
        if reference[field] != info[field]
    ]
    if differences:
        raise GpuMismatchError(
            f"{label} ran on a different GPU than the first run ("
            + "; ".join(differences)
            + "). Results from different GPUs must not be mixed; rerun on one GPU."
        )


def to_float(raw: str | None, metric: str) -> float:
    text = (raw or "").strip().replace(",", "")
    try:
        value = float(text)
    except ValueError as exc:
        raise RuntimeError(f"Can't parse {metric} value {raw!r}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite value for {metric}: {raw!r}")
    return value


def parse_launch_metrics(csv_output: str) -> list[dict[str, float]]:
    """Return one {metric: value} dict per profiled launch, in launch order.

    Accepts both ncu CSV layouts: one row per metric ("Metric Name" and
    "Metric Value" columns) and one row per launch (one column per metric).
    """
    lines = csv_output.splitlines()

    for header_index, line in enumerate(lines):
        try:
            header = next(csv.reader([line]))
        except (csv.Error, StopIteration):
            continue

        columns = [column.strip() for column in header]
        lowered = [column.lower() for column in columns]
        if "id" not in lowered:
            continue
        long_layout = "metric name" in lowered and "metric value" in lowered
        wide_layout = all(metric in columns for metric in METRICS)
        if not (long_layout or wide_layout):
            continue

        launches: dict[int, dict[str, float]] = {}
        reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
        for raw_row in reader:
            row = {(key or "").strip(): value for key, value in raw_row.items()}
            raw_id = row.get("ID")
            if not isinstance(raw_id, str) or not raw_id.strip().isdigit():
                continue  # the units row, or ncu's ==PROF== trailer lines
            found = launches.setdefault(int(raw_id.strip()), {})
            if long_layout:
                name = (row.get("Metric Name") or "").strip()
                if name in METRICS:
                    found[name] = to_float(row.get("Metric Value"), name)
            else:
                for metric in METRICS:
                    found[metric] = to_float(row.get(metric), metric)

        incomplete = [
            launch_id
            for launch_id, found in launches.items()
            if any(metric not in found for metric in METRICS)
        ]
        if not launches or incomplete:
            raise RuntimeError(
                f"Missing {METRICS} for launch IDs {incomplete or 'all'}. "
                "Check the saved ncu CSV output and metric support."
            )
        return [launches[launch_id] for launch_id in sorted(launches)]

    raise RuntimeError(
        "Could not find a recognizable ncu CSV table (an ID column plus either "
        "Metric Name/Metric Value columns or one column per metric). "
        "Inspect the saved raw output; the installed ncu may format --page raw "
        "differently."
    )


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Fit y = base + slope*x; return base, slope, and R-squared."""
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise RuntimeError("Cannot fit line: all thread counts are identical.")

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    base = mean_y - slope * mean_x

    total_ss = sum((y - mean_y) ** 2 for y in ys)
    residual_ss = sum((y - (base + slope * x)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 if total_ss == 0 else 1.0 - residual_ss / total_ss
    return base, slope, r_squared


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def launch_record(
    launch_index: int, metrics: dict[str, float]
) -> dict[str, float | int]:
    elapsed = metrics[METRICS[0]]
    active = metrics[METRICS[1]]
    return {
        "launch_index": launch_index,
        "cycles_elapsed_max": elapsed,
        "cycles_active_max": active,
        "launch_overhead_cycles": elapsed - active,
    }


def profile_one(
    ncu: str,
    binary: Path,
    grid: int,
    block: int,
    args: argparse.Namespace,
    raw_dir: Path,
    label: str,
) -> tuple[dict, dict, dict]:
    """Profile one benchmark process; return (cold record, warm record, GPU info)."""
    log_path = raw_dir / f"{label}.csv"
    log_path.unlink(missing_ok=True)
    command = [
        ncu,
        "--csv",
        "--page",
        "raw",
        "--log-file",  # keeps ncu's CSV separate from the benchmark's own stdout
        str(log_path),
        "--cache-control",
        args.cache_control,
        "--clock-control",
        args.clock_control,
        "--metrics",
        ",".join(METRICS),
        str(binary),
        str(grid),
        str(block),
        str(args.warmup),
    ]
    result = run(command)

    (raw_dir / f"{label}.stdout.txt").write_text(result.stdout)
    (raw_dir / f"{label}.stderr.txt").write_text(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"ncu failed for {grid}x{block} (exit code {result.returncode}):\n"
            f"{result.stderr}\nRaw output saved under {raw_dir}/"
        )

    csv_text = log_path.read_text() if log_path.is_file() else result.stdout
    try:
        launches = parse_launch_metrics(csv_text)
        info = parse_gpu_info(result.stdout, csv_text)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not process {grid}x{block}: {exc}\nRaw output saved under {raw_dir}/"
        ) from exc

    expected = args.warmup + 1
    if len(launches) != expected:
        raise RuntimeError(
            f"{grid}x{block}: expected {expected} profiled launches "
            f"({args.warmup} warmup + 1), ncu reported {len(launches)}. "
            f"Raw output saved under {raw_dir}/"
        )
    return launch_record(0, launches[0]), launch_record(args.warmup, launches[-1]), info


def run_experiment(args: argparse.Namespace) -> int:
    ncu = shutil.which("ncu")
    if ncu is None:
        raise RuntimeError("ncu is not on PATH.")

    binary = Path(args.binary)
    if not binary.is_file():
        raise RuntimeError(f"benchmark executable not found: {binary}")
    binary = (
        binary.resolve()
    )  # Path("./x") prints as "x", which would be searched on PATH

    # Probe run (no kernel launch): identifies the GPU every later run must match.
    try:
        probe = run([str(binary), "--info"])
    except OSError as exc:
        raise RuntimeError(f"Could not run {binary}: {exc}") from exc
    if probe.returncode != 0:
        raise RuntimeError(f"{binary} --info failed:\n{probe.stderr}")
    reference_gpu = parse_gpu_info(probe.stdout)

    gpu_name = reference_gpu["name"]
    gpu_slug = slugify(gpu_name)
    gpu_key = args.gpu or gpu_slug
    arch = "sm_" + reference_gpu["compute_capability"].replace(".", "")
    architecture = architecture_name(reference_gpu["compute_capability"])

    device_info = {
        "name": gpu_name,
        "sms": reference_gpu["num_sms"],
        "max_warps_per_sm": reference_gpu["max_warps_per_sm"],
        "l2_cache_bytes": reference_gpu["l2_cache_bytes"],
        "shared_mem_per_sm_bytes": reference_gpu["shared_mem_per_sm_bytes"],
    }
    raw_dir = Path(args.raw_dir) / gpu_slug
    raw_dir.mkdir(parents=True, exist_ok=True)

    listing = run_text(["nvidia-smi", "-L"])
    visible_gpus = sum(1 for line in listing.splitlines() if line.startswith("GPU "))
    if visible_gpus > 1:
        print(
            f"Warning: {visible_gpus} GPUs are visible; the benchmark uses device "
            f"{reference_gpu['device']} ({gpu_name}). Every run is checked against it.",
            file=sys.stderr,
        )

    # Reps in the outer loop so each repetition is a complete fit over all configs.
    runs: list[dict] = []
    for rep in range(args.reps):
        for grid, block in CONFIGS:
            label = f"grid_{grid}_block_{block}_rep{rep}"
            print(f"[rep {rep + 1}/{args.reps}] {grid}x{block}", flush=True)
            cold, warm, info = profile_one(
                ncu, binary, grid, block, args, raw_dir, label
            )
            check_same_gpu(reference_gpu, info, label)
            runs.append(
                {
                    "rep": rep,
                    "grid": grid,
                    "block": block,
                    "total_threads": grid * block,
                    "cold": cold,
                    "warm": warm,
                }
            )

    def fits_for(kind: str) -> list[tuple[float, float, float]]:
        fits = []
        for rep in range(args.reps):
            points = [r for r in runs if r["rep"] == rep]
            xs = [p["total_threads"] for p in points]
            ys = [p[kind]["launch_overhead_cycles"] for p in points]
            fits.append(fit_line(xs, ys))
        return fits

    warm_fits = fits_for("warm")
    cold_fits = fits_for("cold")

    warmup_effect = []
    for grid, block in CONFIGS:
        mine = [r for r in runs if (r["grid"], r["block"]) == (grid, block)]
        cold_median = statistics.median(
            r["cold"]["launch_overhead_cycles"] for r in mine
        )
        warm_median = statistics.median(
            r["warm"]["launch_overhead_cycles"] for r in mine
        )
        warmup_effect.append(
            {
                "grid": grid,
                "block": block,
                "total_threads": grid * block,
                "cold_median_cycles": cold_median,
                "warm_median_cycles": warm_median,
                "warm_minus_cold_cycles": warm_median - cold_median,
            }
        )

    script_dir = Path(__file__).resolve().parent
    driver = run_text(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    ).splitlines()
    driver_version = driver[0].strip() if driver else "unknown"
    git_commit = (
        run_text(["git", "rev-parse", "--short", "HEAD"], cwd=script_dir) or "unknown"
    )
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    measurement = {
        "method": (
            "ncu: launch overhead = sm__cycles_elapsed.max - sm__cycles_active.max "
            "per kernel launch; least-squares fit against grid*block per repetition; "
            "value = median over repetitions of the warm-launch fit"
        ),
        "metric_elapsed": METRICS[0],
        "metric_active": METRICS[1],
        "launch_used_for_value": "warm",
        "profiler": run_text([ncu, "--version"]),
        "ncu_cache_control": args.cache_control,
        "ncu_clock_control": args.clock_control,
        "configs": [
            {"grid": grid, "block": block, "total_threads": grid * block}
            for grid, block in CONFIGS
        ],
        "warmup_launches": args.warmup,
        "reps": args.reps,
        "device_index": reference_gpu["device"],
        "gpu_uuid": reference_gpu["uuid"],
        "visible_gpu_count": visible_gpus or None,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for parameter, unit, index in PARAMETERS:
        warm_values = [fit[index] for fit in warm_fits]
        cold_values = [fit[index] for fit in cold_fits]
        warm_stats = summarize(warm_values)
        cold_stats = summarize(cold_values)
        report = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": BENCHMARK,
            "benchmark_version": str(reference_gpu["bench_version"]),
            "parameter": parameter,
            "value": warm_stats["median"],
            "unit": unit,
            "gpu": gpu_key,
            "architecture": architecture,
            "arch": arch,
            "device": device_info,
            "gpu_name": gpu_name,
            "num_sms": reference_gpu["num_sms"],
            "l2_cache_bytes": reference_gpu["l2_cache_bytes"],
            # per_rep_values : this parameter from each repetition's warm fit; stats are computed over these
            # r_squared_per_rep : R-squared of each of those fits
            # cold_start     : same quantities from the first (cold) launch of each process
            # warmup_effect  : per-config median launch overhead, cold vs warm
            # runs           : every ncu measurement (launch_overhead_cycles = elapsed - active)
            "per_rep_values": warm_values,
            "r_squared_per_rep": [fit[2] for fit in warm_fits],
            "cold_start": {
                "value": cold_stats["median"],
                "stats": cold_stats,
                "per_rep_values": cold_values,
            },
            "warmup_effect": warmup_effect,
            "runs": runs,
            "measurement": measurement,
            "cuda_version": reference_gpu["cuda_runtime"],
            "nvcc_version": reference_gpu["nvcc"],
            "driver_version": driver_version,
            "git_commit": git_commit,
            "timestamp": timestamp,
        }
        output = output_dir / f"{parameter}_{gpu_slug}.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        written.append((parameter, unit, warm_stats, cold_stats, output))

    print(
        f"\nGPU: {gpu_name} ({arch}, {reference_gpu['num_sms']} SMs); "
        f"{args.reps} reps, {args.warmup} warmup launch(es)"
    )
    print("Median launch overhead per config (cycles):")
    print(f"{'config':>10} {'cold':>12} {'warm':>12} {'warm-cold':>12}")
    for entry in warmup_effect:
        config = f"{entry['grid']}x{entry['block']}"
        print(
            f"{config:>10} {entry['cold_median_cycles']:>12.1f} "
            f"{entry['warm_median_cycles']:>12.1f} {entry['warm_minus_cold_cycles']:>12.1f}"
        )
    for parameter, unit, warm_stats, cold_stats, output in written:
        print(
            f"{parameter} = {warm_stats['median']:.12g} {unit} "
            f"(cold: {cold_stats['median']:.12g}) -> {output}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        default="./launch_overhead_bench",
        help="Path to the compiled CUDA benchmark",
    )
    parser.add_argument(
        "--output-dir",
        default="hardware/runs",
        help="Directory for the result JSON files (one per fitted parameter, "
        "named after the GPU)",
    )
    parser.add_argument(
        "--raw-dir",
        default="launch_overhead_ncu_raw",
        help="Parent directory for raw ncu output (a subdirectory per GPU is created)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Repetitions of the whole config sweep (the median is reported)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup launches before the warm measured launch (at least 1)",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Short GPU key for the JSON 'gpu' field (default: slug of the GPU name)",
    )
    parser.add_argument(
        "--cache-control",
        choices=("none", "all"),
        default="none",
        help="ncu --cache-control. 'all' (ncu's own default) flushes caches before "
        "every launch, which hides any effect of the warmup launch",
    )
    parser.add_argument(
        "--clock-control",
        choices=("base", "none"),
        default="base",
        help="ncu --clock-control (ncu's own default is base)",
    )
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be at least 1")
    if args.warmup < 1:
        parser.error("--warmup must be at least 1")

    try:
        return run_experiment(args)
    except GpuMismatchError as exc:
        print(f"Error: GPU mismatch. {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
