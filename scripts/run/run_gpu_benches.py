"""scripts/run/run_gpu_benches.py: measure the profile's DRAM bandwidth with gpu-benches'
gpu-stream (RRZE-HPC), and compare it with other profiles.

gpu-stream is built and run as published, without ncu: it times from the host, which
ncu's replay would distort. The value is the highest read-kernel GB/s over its
occupancy sweep, divided by the base profile's clock_rate_ghz. A DRAM-bound kernel's
GB/s does not depend on the SM clock, so this is exact without a clock lock.

The value is written as a run JSON where gasp-bm's run_all.py reads it, and
run_all.py calls this script for dram_bandwidth_bytes_per_cycle, so
`make microbenchmarks` builds the whole profile. Run on its own
(`make microbenchmarks-gpu-benches`) it also prints a comparison with the base
profile and this version's gasp-bm and HPMLL profiles.

Output, for `--gpu rtx2080ti --version 1.2`:

    hardware/runs/rtx2080ti/1_2/
        dram_bandwidth_bytes_per_cycle/dram_bandwidth_bytes_per_cycle.rtx2080ti.run.json
        gpu-benches/
            build/gpu-stream
            gpu-stream/  build.log run.log sm_clock.csv
            rtx2080ti.1_2.gpu-benches.json  parsed rows and the comparison
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GASP_BM_DIR = REPO_ROOT / "hardware" / "microbenchmarks" / "gasp-bm"
DEFAULT_GPU_BENCHES_DIR = REPO_ROOT / "hardware" / "microbenchmarks" / "gpu-benches"
SCHEMA_VERSION = "3.0.0"
PARAM = "dram_bandwidth_bytes_per_cycle"


def _load_run_all():
    """gasp-bm's run_all.py, for its logging, arch and base-profile helpers."""
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
STREAM_STD = "c++20"  # gpu-stream/Makefile

SM_CLOCK_SAMPLE_MS = 100


class BenchmarkError(Exception):
    pass


# ---- Parsing ---------------------------------------------------------------------

NUMBER = r"[-+]?\d+(?:\.\d+)?"
STREAM_ROW = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+(\d+)\s+({NUMBER})%\s*\|\s*GB/s:((?:\s+{NUMBER})+)\s*$"
)


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


def stream_peak(rows: list[dict]) -> dict:
    """The row with the highest read GB/s (the occupancy sweep's plateau)."""
    if not rows:
        raise BenchmarkError("gpu-stream printed no result rows")
    return max(rows, key=lambda r: r["gb_per_s"]["read"])


def pct_diff(value: float, reference: float) -> float | None:
    return None if reference == 0 else round((value - reference) / reference * 100, 2)


# ---- Running -----------------------------------------------------------------------


def run_sampling_clock(command: list[str], log_path: Path, clock_path: Path, cwd: Path) -> list[float]:
    """Run command (output to log_path) while nvidia-smi samples the SM clock; the
    samples in MHz (recorded for reference: the value does not use them)."""
    with clock_path.open("w") as clock_out:
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                *mb.nvidia_smi_gpu(),
                "--query-gpu=clocks.sm",
                "--format=csv,noheader,nounits",
                f"-lms={SM_CLOCK_SAMPLE_MS}",
            ],
            stdout=clock_out,
            stderr=subprocess.DEVNULL,
        )
        try:
            mb.run_logged(command, log_path, cwd=cwd)
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


def free_memory_bytes() -> int:
    result = subprocess.run(
        ["nvidia-smi", *mb.nvidia_smi_gpu(), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
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


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


# ---- Comparison --------------------------------------------------------------------


def reference_profiles(gpu: str, version: str, base_path: Path) -> dict[str, Path]:
    """Profiles to compare with: the base profile, and this version's gasp-bm and
    HPMLL profiles when they exist."""
    run_dir = REPO_ROOT / "hardware" / "runs" / gpu / version
    candidates = {
        "base": base_path,
        "gasp-bm": run_dir / f"{gpu}.{version}.json",
        "hpmll": run_dir / "hpmll" / f"{gpu}.{version}.hpmll.json",
    }
    return {name: path for name, path in candidates.items() if path.exists()}


def compare(key: str, value: float, references: dict[str, Path]) -> dict:
    row: dict = {"gpu-benches": round(value, 4)}
    for name, path in references.items():
        ref = json.loads(path.read_text()).get("profiled", {}).get(key)
        if isinstance(ref, (int, float)):
            row[name] = {"value": ref, "diff_pct": pct_diff(value, ref)}
    return row


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
        help="profile for the base clock and comparison "
        "(default: hardware/profiles/<gpu>.json, else <gpu>.placeholder.json)",
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
    src_dir = suite_dir / "gpu-stream"
    if not (src_dir / "main.cu").exists():
        raise SystemExit(
            f"{suite_dir} has no gpu-stream/main.cu; fetch the submodule:\n"
            "  git submodule update --init hardware/microbenchmarks/gpu-benches"
        )

    version = args.version.replace(".", "_")
    root = REPO_ROOT / "hardware" / "runs" / args.gpu / version
    out = root / "gpu-benches"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to run into it again")
    (out / "build").mkdir(parents=True, exist_ok=True)
    bench_dir = out / "gpu-stream"
    bench_dir.mkdir(exist_ok=True)

    base_path = Path(args.base_profile) if args.base_profile else mb.default_base_profile(args.gpu)
    if not base_path.exists():
        raise SystemExit(f"base profile {base_path} not found; pass --base-profile")
    try:
        base_ghz = float(json.loads(base_path.read_text())["profiled"]["clock_rate_ghz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{base_path} has no usable profiled.clock_rate_ghz ({exc})") from exc
    arch = args.arch or mb.detect_arch()
    mb.log(f"GPU {args.gpu}, {arch}, version {version}, base clock {base_ghz:.3f} GHz, base profile {base_path}")

    commit = git_short(suite_dir)
    git_commit = mb.git_commit()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    mb.log("\n== gpu-stream")
    try:
        need = STREAM_BUFFERS * STREAM_BUFFER_ELEMS * 8 + STREAM_HEADROOM
        free = free_memory_bytes()
        if free < need:
            raise BenchmarkError(f"needs {need / 1024**3:.1f} GiB free, the GPU has {free / 1024**3:.1f} GiB")
        build_log = bench_dir / "build.log"
        build_log.write_text("")
        binary = out / "build" / "gpu-stream"
        mb.run_logged(
            ["nvcc", f"-std={STREAM_STD}", "-O3", f"-arch={arch}", str(src_dir / "main.cu"), "-o", str(binary)],
            build_log,
        )
        run_log = bench_dir / "run.log"
        run_log.write_text("")
        samples = run_sampling_clock([str(binary)], run_log, bench_dir / "sm_clock.csv", binary.parent)
        rows = parse_stream(run_log.read_text())
        peak = stream_peak(rows)
    except (mb.BenchmarkError, BenchmarkError) as exc:
        mb.log(f"  FAILED: {exc}")
        return 1

    gbs = peak["gb_per_s"]["read"]
    value = gbs / base_ghz
    mb.log(
        f"  read peak {gbs:.0f} GB/s at {peak['threads']} threads "
        f"({peak['occupancy_pct']:.1f}% occupancy) = {value:.2f} bytes/cycle at base"
    )
    clock = {
        # GB/s is clock-independent, so the run clock does not enter the value
        "sm_clock_ghz": None,
        "conversion": f"read GB/s / base profile clock_rate_ghz ({base_ghz} GHz)",
        "sampled_mhz": [min(samples), statistics.median(samples), max(samples)] if samples else [],
    }
    run_json = root / PARAM / f"{PARAM}.{args.gpu}.run.json"
    run_json.parent.mkdir(parents=True, exist_ok=True)
    run_json.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": "gpu-benches/gpu-stream/main.cu",
                "parameter": PARAM,
                "value": round(value, 4),
                "unit": "bytes/cycle",
                "gpu": args.gpu,
                "arch": arch,
                "method": "read kernel (one 8-byte load per thread over a 2 GiB buffer), "
                "median of 31 launches per occupancy; the highest occupancy's GB/s",
                "read_gb_per_s": gbs,
                "peak": peak,
                "clock": clock,
                "gpu_benches_commit": commit,
                "git_commit": git_commit,
                "timestamp": timestamp,
            },
            indent=2,
        )
        + "\n"
    )
    mb.log(f"  wrote {relative(run_json)}")

    references = reference_profiles(args.gpu, version, base_path)
    comparison = compare(PARAM, value, references)
    summary_path = out / f"{args.gpu}.{version}.gpu-benches.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "suite": "gpu-benches",
                "gpu": args.gpu,
                "arch": arch,
                "base_clock_ghz": base_ghz,
                "base_profile": relative(base_path),
                "run_json": relative(run_json),
                "rows": rows,
                "comparison": {PARAM: comparison},
                "reference_profiles": {k: relative(p) for k, p in references.items()},
                "gpu_benches_commit": commit,
                "git_commit": git_commit,
                "timestamp": timestamp,
            },
            indent=2,
        )
        + "\n"
    )
    refs = "  ".join(
        f"{n} {r['value']:.4g} ({r['diff_pct']:+.1f}%)" if r["diff_pct"] is not None else f"{n} {r['value']}"
        for n, r in comparison.items()
        if n != "gpu-benches"
    )
    mb.log(f"\nComparison: {PARAM} {value:.4g}   {refs or '(no reference profiles)'}")
    mb.log(f"Results: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
