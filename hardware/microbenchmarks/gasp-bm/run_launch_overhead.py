# Profiles launch_overhead_bench with ncu over 8 grid/block configs, --reps times.
# Each process launches an empty kernel --warmup times plus once more; ncu captures
# the first (cold) and last (warm) launches.
#
# Per repetition, fits: launch_overhead = base + per_thread * (grid * block).
# Reports the median cold fit across reps (aborts on a negative cold slope) and
# writes one JSON file per parameter to hardware/runs/.
#
# Usage: python3 hardware/microbenchmarks/run_launch_overhead.py --binary ./launch_overhead_bench

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

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "1"
BENCHMARK = "launch_overhead_bench.cu"
METRICS = ("sm__cycles_elapsed.max", "sm__cycles_active.max")
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
GPU_ID_FIELDS = ("uuid", "name", "compute_capability", "num_sms")
PARAMETERS = (
    ("launch_base_cycles", "cycles", 0),
    ("launch_per_thread_cycles", "cycles/thread", 1),
)
ARCH_MAP = {
    (7, 0): "volta",
    (7, 5): "turing",
    (8, 0): "ampere",
    (8, 6): "ampere",
    (8, 9): "ada",
    (9, 0): "hopper",
    (10, 0): "blackwell",
    (12, 0): "blackwell",
}


def cmd_stdout(command: list[str], cwd: Path | None = None) -> str:
    """Run a command and return its stripped stdout or "" on failure"""
    try:
        res = subprocess.run(
            command, check=False, text=True, capture_output=True, cwd=cwd
        )
        return res.stdout.strip() if res.returncode == 0 else ""
    except OSError:
        return ""


def parse_gpu_info(*texts: str) -> dict:
    """Find and validate the benchmark's GPU_INFO {json}"""
    for line in (ln.strip() for t in texts for ln in t.splitlines()):
        if not line.startswith(GPU_INFO_PREFIX):
            continue
        try:
            info = json.loads(line[len(GPU_INFO_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed GPU_INFO line: {line!r}") from exc
        missing = [k for k in GPU_INFO_REQUIRED if k not in info]
        if missing:
            raise RuntimeError(f"GPU_INFO missing {missing}. Recompile {BENCHMARK}.")
        return info
    raise RuntimeError(f"No GPU_INFO line found. Recompile {BENCHMARK}.")


def check_same_gpu(reference: dict, info: dict, label: str) -> None:
    """Fail if a profiled run landed on a different GPU than the reference"""
    diffs = [
        f"{k}: {reference[k]!r} vs {info[k]!r}"
        for k in GPU_ID_FIELDS
        if reference[k] != info[k]
    ]
    if diffs:
        raise RuntimeError(f"{label} ran on a different GPU ({'; '.join(diffs)}).")


def parse_launch_metrics(csv_output: str) -> list[dict[str, float]]:
    """Parse ncu --csv --page raw output into one {metric: value} dict per launch ID.
    This handles both the long layout (Metric Name / Metric Value rows) and the wide
    layout (one column per metric).
    """
    lines = csv_output.splitlines()
    for idx, line in enumerate(lines):
        try:
            header = [c.strip() for c in next(csv.reader([line]))]
        except (csv.Error, StopIteration):
            continue
        lowered = [c.lower() for c in header]
        long_fmt = "metric name" in lowered and "metric value" in lowered
        wide_fmt = all(m in header for m in METRICS)
        if "id" not in lowered or not (long_fmt or wide_fmt):
            continue

        launches: dict[int, dict[str, float]] = {}
        for raw_row in csv.DictReader(io.StringIO("\n".join(lines[idx:]))):
            row = {(k or "").strip(): v for k, v in raw_row.items()}
            raw_id = (row.get("ID") or "").strip()
            if not raw_id.isdigit():
                continue
            entry = launches.setdefault(int(raw_id), {})
            names = (
                [(row.get("Metric Name") or "").strip()] if long_fmt else list(METRICS)
            )
            for m in (n for n in names if n in METRICS):
                raw_val = (row.get("Metric Value" if long_fmt else m) or "").strip()
                try:
                    val = float(raw_val.replace(",", ""))
                except ValueError as exc:
                    raise RuntimeError(f"Can't parse {m} value {raw_val!r}") from exc
                if not math.isfinite(val):
                    raise RuntimeError(f"Non-finite value for {m}: {raw_val!r}")
                entry[m] = val

        incomplete = [
            i for i, d in launches.items() if any(m not in d for m in METRICS)
        ]
        if not launches or incomplete:
            raise RuntimeError(
                f"Missing {METRICS} for launch IDs {incomplete or 'all'}."
            )
        return [launches[i] for i in sorted(launches)]
    raise RuntimeError("Could not find a recognizable ncu CSV table in output.")


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares fit y = base + slope * x; returns (base, slope, r_squared)."""
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        raise RuntimeError("Cannot fit line: all thread counts are identical.")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    base = my - slope * mx
    total = sum((y - my) ** 2 for y in ys)
    resid = sum((y - (base + slope * x)) ** 2 for x, y in zip(xs, ys))
    return base, slope, 1.0 if total == 0 else 1.0 - resid / total


def summarize(values: list[float]) -> dict[str, float]:
    """Median/mean/std/min/max of a list (std is 0 for a single value)."""
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def profile_one(
    ncu, binary, grid, block, args, raw_dir, label
) -> tuple[dict, dict, dict]:
    """Run one config under ncu; return (cold record, warm record, GPU_INFO)"""
    log_path = raw_dir / f"{label}.csv"
    log_path.unlink(missing_ok=True)
    cmd = [
        ncu,
        "--csv",
        "--page",
        "raw",
        "--log-file",
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
    res = subprocess.run(cmd, check=False, text=True, capture_output=True)
    (raw_dir / f"{label}.stdout.txt").write_text(res.stdout)
    (raw_dir / f"{label}.stderr.txt").write_text(res.stderr)
    if res.returncode != 0:
        raise RuntimeError(
            f"ncu failed for {grid}x{block} (exit {res.returncode}):\n{res.stderr}"
        )

    csv_text = log_path.read_text() if log_path.is_file() else res.stdout
    launches = parse_launch_metrics(csv_text)
    info = parse_gpu_info(res.stdout, csv_text)
    if len(launches) != args.warmup + 1:
        raise RuntimeError(
            f"{grid}x{block}: expected {args.warmup + 1} launches, ncu reported {len(launches)}."
        )

    def record(idx: int, m: dict[str, float]) -> dict:
        return {
            "launch_index": idx,
            "cycles_elapsed_max": m[METRICS[0]],
            "cycles_active_max": m[METRICS[1]],
            "launch_overhead_cycles": m[METRICS[0]] - m[METRICS[1]],
        }

    return record(0, launches[0]), record(args.warmup, launches[-1]), info


def run_sweep(ncu, binary, ref_gpu, args, raw_dir) -> list[dict]:
    """Profile every config --reps times; return one run record per (rep, config)"""
    runs = []
    for rep in range(args.reps):
        for grid, block in CONFIGS:
            print(f"[rep {rep + 1}/{args.reps}] {grid}x{block}", flush=True)
            label = f"grid_{grid}_block_{block}_rep{rep}"
            cold, warm, info = profile_one(
                ncu, binary, grid, block, args, raw_dir, label
            )
            check_same_gpu(ref_gpu, info, label)
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
    return runs


def run_experiment(args: argparse.Namespace) -> int:
    """Runs the full pipeline: sweep, fit per rep, check cold slope, write JSON, and prints summary."""
    ncu = shutil.which("ncu")
    if not ncu:
        raise RuntimeError("ncu is not on PATH.")
    binary = Path(args.binary)
    if not binary.is_file():
        raise RuntimeError(f"Benchmark executable not found: {binary}")
    binary = binary.resolve()

    probe = subprocess.run(
        [str(binary), "--info"], check=False, text=True, capture_output=True
    )
    if probe.returncode != 0:
        raise RuntimeError(f"{binary} --info failed:\n{probe.stderr}")
    ref_gpu = parse_gpu_info(probe.stdout)

    gpu_name = ref_gpu["name"]
    gpu_slug = re.sub(r"[^a-z0-9]+", "_", gpu_name.lower()).strip("_") or "unknown_gpu"
    major, minor = map(int, ref_gpu["compute_capability"].split("."))
    device_info = {
        "name": gpu_name,
        "sms": ref_gpu["num_sms"],
        "max_warps_per_sm": ref_gpu["max_warps_per_sm"],
        "l2_cache_bytes": ref_gpu["l2_cache_bytes"],
        "shared_mem_per_sm_bytes": ref_gpu["shared_mem_per_sm_bytes"],
    }
    raw_dir = Path(args.raw_dir) / gpu_slug
    raw_dir.mkdir(parents=True, exist_ok=True)

    visible_gpus = sum(
        1
        for ln in cmd_stdout(["nvidia-smi", "-L"]).splitlines()
        if ln.startswith("GPU ")
    )
    if visible_gpus > 1:
        print(
            f"Warning: {visible_gpus} GPUs visible; using device {ref_gpu['device']} ({gpu_name}).",
            file=sys.stderr,
        )

    runs = run_sweep(ncu, binary, ref_gpu, args, raw_dir)

    def fits(kind: str) -> list[tuple[float, float, float]]:
        """One (base, slope, r^2) fit per rep, using the cold or warm launches"""
        return [
            fit_line(
                [r["total_threads"] for r in runs if r["rep"] == rep],
                [r[kind]["launch_overhead_cycles"] for r in runs if r["rep"] == rep],
            )
            for rep in range(args.reps)
        ]

    cold_fits, warm_fits = fits("cold"), fits("warm")

    # Per-config median overhead, cold vs warm
    warmup_effect = []
    for grid, block in CONFIGS:
        sub = [r for r in runs if (r["grid"], r["block"]) == (grid, block)]
        cold = statistics.median(r["cold"]["launch_overhead_cycles"] for r in sub)
        warm = statistics.median(r["warm"]["launch_overhead_cycles"] for r in sub)
        warmup_effect.append(
            {
                "grid": grid,
                "block": block,
                "total_threads": grid * block,
                "cold_median_cycles": cold,
                "warm_median_cycles": warm,
                "warm_minus_cold_cycles": warm - cold,
            }
        )

    # A negative per-thread cost is non-physical: stop instead of reporting it.
    cold_slope = statistics.median(f[1] for f in cold_fits)
    if cold_slope < 0:
        medians = ", ".join(
            f"{e['grid']}x{e['block']}={e['cold_median_cycles']:.0f}"
            for e in warmup_effect
        )
        raise RuntimeError(
            f"Negative cold per-thread slope ({cold_slope:.6g} cycles/thread); aborting. "
            f"Cold medians: {medians}"
        )

    driver_out = cmd_stdout(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    ).splitlines()
    measurement = {
        "method": (
            "ncu: launch overhead = sm__cycles_elapsed.max - sm__cycles_active.max "
            "per kernel launch; least-squares fit against grid*block per repetition; "
            "value = median over repetitions of the cold-launch (first launch) fit"
        ),
        "metric_elapsed": METRICS[0],
        "metric_active": METRICS[1],
        "launch_used_for_value": "cold",
        "profiler": cmd_stdout([ncu, "--version"]),
        "ncu_cache_control": args.cache_control,
        "ncu_clock_control": args.clock_control,
        "configs": [
            {"grid": g, "block": b, "total_threads": g * b} for g, b in CONFIGS
        ],
        "warmup_launches": args.warmup,
        "reps": args.reps,
        "device_index": ref_gpu["device"],
        "gpu_uuid": ref_gpu["uuid"],
        "visible_gpu_count": visible_gpus or None,
    }
    common = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "benchmark_version": str(ref_gpu["bench_version"]),
    }
    tail = {
        "cuda_version": ref_gpu["cuda_runtime"],
        "nvcc_version": ref_gpu["nvcc"],
        "driver_version": driver_out[0].strip() if driver_out else "unknown",
        "git_commit": cmd_stdout(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent
        )
        or "unknown",
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for param, unit, idx in PARAMETERS:
        cold_vals = [f[idx] for f in cold_fits]
        warm_vals = [f[idx] for f in warm_fits]
        cold_stats, warm_stats = summarize(cold_vals), summarize(warm_vals)
        report = {
            **common,
            "parameter": param,
            "value": cold_stats["median"],
            "unit": unit,
            "gpu": args.gpu or gpu_slug,
            "architecture": ARCH_MAP.get((major, minor), "unknown"),
            "arch": f"sm_{major}{minor}",
            "device": device_info,
            "gpu_name": gpu_name,
            "num_sms": ref_gpu["num_sms"],
            "l2_cache_bytes": ref_gpu["l2_cache_bytes"],
            "stats": cold_stats,
            "per_rep_values": cold_vals,
            "r_squared_per_rep": [f[2] for f in cold_fits],
            "warm_launch": {
                "value": warm_stats["median"],
                "stats": warm_stats,
                "per_rep_values": warm_vals,
                "r_squared_per_rep": [f[2] for f in warm_fits],
            },
            "warmup_effect": warmup_effect,
            "runs": runs,
            "measurement": measurement,
            **tail,
        }
        out_path = output_dir / f"{param}_{gpu_slug}.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        written.append(
            (param, unit, cold_stats["median"], warm_stats["median"], out_path)
        )

    print(
        f"\nGPU: {gpu_name} (sm_{major}{minor}, {ref_gpu['num_sms']} SMs); "
        f"{args.reps} reps, {args.warmup} warmup launch(es)"
    )
    print(f"{'config':>10} {'cold':>12} {'warm':>12} {'warm-cold':>12}")
    for e in warmup_effect:
        cfg = f"{e['grid']}x{e['block']}"
        print(
            f"{cfg:>10} {e['cold_median_cycles']:>12.1f} "
            f"{e['warm_median_cycles']:>12.1f} {e['warm_minus_cold_cycles']:>12.1f}"
        )
    for param, unit, cold_med, warm_med, out_path in written:
        print(f"{param} = {cold_med:.12g} {unit} (warm: {warm_med:.12g}) -> {out_path}")
    return 0


def main() -> int:
    """Parse CLI args and run the experiment"""
    p = argparse.ArgumentParser(
        description="Profile launch_overhead_bench with ncu and fit its per-thread launch model."
    )
    p.add_argument(
        "--binary", default="./launch_overhead_bench", help="Compiled benchmark path"
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "hardware" / "runs"),
        help="Output JSON directory",
    )
    p.add_argument(
        "--raw-dir",
        default=str(REPO_ROOT / "launch_overhead_ncu_raw"),
        help="Raw ncu logs directory",
    )
    p.add_argument(
        "--reps", type=int, default=5, help="Repetitions of the config sweep"
    )
    p.add_argument(
        "--warmup", type=int, default=1, help="Warmup launches per process (>= 1)"
    )
    p.add_argument("--gpu", default=None, help="Short GPU key for JSON output")
    p.add_argument(
        "--cache-control",
        choices=("none", "all"),
        default="none",
        help="ncu --cache-control",
    )
    p.add_argument(
        "--clock-control",
        choices=("base", "none"),
        default="base",
        help="ncu --clock-control",
    )
    args = p.parse_args()
    if args.reps < 1:
        p.error("--reps must be at least 1")
    if args.warmup < 1:
        p.error("--warmup must be at least 1")
    try:
        return run_experiment(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
