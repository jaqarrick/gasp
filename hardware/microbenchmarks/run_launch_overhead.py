# This is a script that runs ncu for 4 configurations (<<<1,1>>>, <<<1,256>>>, <<<32,256>>>, <<<512,256>>>)
# saves each profiler output, calculates differences + least-squares fit, and writes results to JSON in hardware/runs/
# CUDA compilation and update to the GASP profile must be done manually.

# Run the script with this command (Change path as needed): python3 hardware/microbenchmarks/run_launch_overhead.py --binary ./launch_overhead_bench

"""Profile launch_overhead_bench with ncu and fit its per-thread launch model."""

import argparse
import csv
import io
import json
import math
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

METRICS = (
    "sm__cycles_elapsed.max",
    "sm__cycles_active.max",
)
CONFIGS = ((1, 1), (1, 256), (32, 256), (512, 256))


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def parse_metric_values(csv_output: str) -> dict[str, float]:
    """Read the ncu --csv --page raw table and return its two requested metrics."""
    lines = csv_output.splitlines()

    for header_index, line in enumerate(lines):
        try:
            header = next(csv.reader([line]))
        except csv.Error:
            continue

        normalized = [column.strip().lower() for column in header]
        if "metric name" not in normalized or "metric value" not in normalized:
            continue

        table = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
        values: dict[str, list[float]] = {metric: [] for metric in METRICS}

        for row in table:
            metric_name = (row.get("Metric Name") or "").strip()
            if metric_name not in values:
                continue

            raw_value = (row.get("Metric Value") or "").strip().replace(",", "")
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise RuntimeError(
                    f"Can't parse {metric_name} value {raw_value!r}"
                ) from exc

            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite value for {metric_name}: {raw_value}")
            values[metric_name].append(value)

        result: dict[str, float] = {}
        for metric, samples in values.items():
            if len(samples) != 1:
                raise RuntimeError(
                    f"Expected one {metric} row, found {len(samples)}. "
                    "Check the saved ncu CSV output and metric support."
                )
            result[metric] = samples[0]
        return result

    raise RuntimeError(
        "Could not find Metric Name / Metric Value columns in ncu CSV output. "
        "Inspect the saved raw output; the installed ncu may format --page raw "
        "differently."
    )


def fit_line(points: list[dict[str, float]]) -> tuple[float, float, float]:
    """Fit y = base + slope*x; return base, slope, and R-squared."""
    xs = [point["total_threads"] for point in points]
    ys = [point["overhead_cycles"] for point in points]
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        default="./launch_overhead_bench",
        help="Path to the compiled CUDA benchmark",
    )
    parser.add_argument(
        "--output",
        default="hardware/runs/launch_overhead_results.json",
        help="Path for the combined calibration results JSON",
    )
    parser.add_argument(
        "--raw-dir",
        default="launch_overhead_ncu_raw",
        help="Directory for per-configuration ncu stdout/stderr",
    )
    args = parser.parse_args()

    ncu = shutil.which("ncu")
    if ncu is None:
        print("Error: ncu is not on PATH.", file=sys.stderr)
        return 1

    binary = Path(args.binary)
    if not binary.is_file():
        print(f"Error: benchmark executable not found: {binary}", file=sys.stderr)
        return 1

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    measurements: list[dict[str, float | int]] = []

    for grid, block in CONFIGS:
        label = f"grid_{grid}_block_{block}"
        command = [
            ncu,
            "--csv",
            "--page",
            "raw",
            "--metrics",
            ",".join(METRICS),
            str(binary),
            str(grid),
            str(block),
        ]
        result = run(command)

        (raw_dir / f"{label}.csv").write_text(result.stdout)
        (raw_dir / f"{label}.stderr.txt").write_text(result.stderr)

        if result.returncode != 0:
            print(f"ncu failed for {grid}x{block}.", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            print(f"Raw output saved under {raw_dir}/", file=sys.stderr)
            return result.returncode or 1

        try:
            metrics = parse_metric_values(result.stdout)
        except RuntimeError as exc:
            print(f"Could not process {grid}x{block}: {exc}", file=sys.stderr)
            print(f"Raw output saved under {raw_dir}/", file=sys.stderr)
            return 1

        elapsed = metrics[METRICS[0]]
        active = metrics[METRICS[1]]
        measurements.append(
            {
                "grid": grid,
                "block": block,
                "total_threads": grid * block,
                "cycles_elapsed_max": elapsed,
                "cycles_active_max": active,
                "overhead_cycles": elapsed - active,
            }
        )

    base, per_thread, r_squared = fit_line(measurements)

    version = run([ncu, "--version"])
    gpu = run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap",
            "--format=csv,noheader",
        ]
    )
    if version.returncode != 0 or gpu.returncode != 0:
        print("Could not read ncu version or GPU info.", file=sys.stderr)
        print(version.stderr, gpu.stderr, file=sys.stderr)
        return 1

    gpu_fields = [field.strip() for field in gpu.stdout.splitlines()[0].split(",", 1)]
    if len(gpu_fields) != 2:
        print(f"Unexpected nvidia-smi output: {gpu.stdout!r}", file=sys.stderr)
        return 1

    report = {
        "benchmark": "launch_overhead_bench.cu",
        "method": (
            "overhead_cycles = sm__cycles_elapsed.max - "
            "sm__cycles_active.max; least-squares fit against grid * block"
        ),
        "gpu_name": gpu_fields[0],
        "compute_capability": gpu_fields[1],
        "ncu_version": version.stdout.strip(),
        "timestamp": datetime.now(UTC).isoformat(),
        "measurements": measurements,
        "fit": {
            "launch_base_cycles": base,
            "launch_per_thread_cycles": per_thread,
            "r_squared": r_squared,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote calibration results to {output}")
    print(f"launch_base_cycles       = {base:.8f}")
    print(f"launch_per_thread_cycles = {per_thread:.12f}")
    print(f"R²                       = {r_squared:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
