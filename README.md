# GASP - GPU Analytical Speedup Predictor

Work in progress static CUDA kernel analyzer. GASP predicts how fast CUDA kernels run relative to each other, such as a well-implemented kernel against a poor one, without running them.

## Prerequisites
- Python 3.12+
- CUDA Toolkit 12.0+ (`nvcc`, `nvdisasm`, `cuobjdump`, plus `ncu` and a GPU for `--run`)
- GNU Make
- Git
- Docker (optional)

## Installation

### Running locally

```bash
git clone --recurse-submodules git@github.com:jaqarrick/gasp-gpu.git
cd gasp-gpu
python3 -m venv .venv
source .venv/bin/activate
make install
```

`make install` installs the Python dependencies, then runs `scripts/install_faial.sh` to fetch faial (pico) and `scripts/check_requirements.sh` to verify the setup. faial is installed to `~/.local/share/faial` with wrappers in `~/.local/bin` (override with `FAIAL_PREFIX` / `BIN_DIR`); it is supported on Linux x86-64 and macOS arm64, and `libgmp10` must be installed on Linux. Run `scripts/uninstall_faial.sh` to remove it.

### Docker (recommended)

```bash
git clone --recurse-submodules git@github.com:jaqarrick/gasp-gpu.git
cd gasp-gpu
make build
```

The CUDA image is quite large, so this will take some time the first build.

### Dev Container
Open the project in VS Code and select **Dev Containers: Reopen in Container**.
Python, CUDA, and all dependencies configured automatically. [More info here](./docs/contributing.md).

## Usage

Compare two kernels. The first is the reference, and shared flags apply to both:

```bash
make run ARGS="--gpu b200 --problem-size 1000000 --block-size 256 benchmarks/custom/vecadd_good.cu benchmarks/custom/vecadd_poor.cu"
```

Pick kernels out of a file that holds several. Without a kernel name, GASP stops and lists them:

```bash
make run ARGS="--gpu b200 kernels.cu::vecadd_good kernels.cu::vecadd_poor"
```

Give each kernel its own settings with a TOML file (see [`vecadd.toml`](benchmarks/custom/vecadd.toml)):

```bash
make run ARGS="--gpu b200 --config benchmarks/custom/vecadd.toml"
```

```toml
[defaults]
problem_size = 1000000
block_size = 256

[[kernel]]
name = "good"
file = "benchmarks/custom/vecadd_good.cu"

[[kernel]]
name = "poor"
file = "benchmarks/custom/vecadd_poor.cu"
block_size = 128
```

A single kernel prints its predicted time with no speedup:

```bash
make run ARGS="--gpu b200 benchmarks/custom/vecadd.cu"
```

### How kernels are compared

- **Reference:** the first kernel, unless `--reference NAME` is given.
- **Speedup:** `T_reference / T_kernel`, using predicted T_GPU in cycles. Above 1 means faster than the reference.
- **Names:** `name` in a config. Otherwise the kernel name from `file.cu::kernel`, or else the file name without `.cu`.
- **Same work:** compared kernels should do the same work. Different problem sizes or program arguments print a warning.
- **Pico:** the `Pico` column shows `fallback` when Pico gave no usable cost for that kernel. A warning is printed if only some kernels fell back, because their speedups then compare different coalescing assumptions.

### Config files

- A setting comes from the kernel entry, then `[defaults]`, then the built-in default.
- Settings are `file`, `name`, `kernel`, `problem_size`, `block_size`, `grid_size`, `loop_iterations`, `include_dirs`, `nvcc_flags` and `program_args`. `name` is only allowed on a kernel entry.
- `include_dirs`, `nvcc_flags` and `program_args` are arrays of strings, such as `program_args = ["1000000"]`.
- Paths are relative to the directory GASP runs from, not to the config file.
- `--config` can't be combined with input files or per-kernel flags.

### Measuring with `--run`

`--run` builds each program and runs it under `ncu`, then prints predicted against measured cycles and speedup:

```bash
make run ARGS="--gpu rtx2080ti --run benchmarks/custom/vecadd_good.cu benchmarks/custom/vecadd_poor.cu"
```

- Each kernel runs `--repeats N` times (default 3). Cycles are ncu's Elapsed Cycles (`gpc__cycles_elapsed.max`). The table shows the median, with the min-max spread beside it.
- Measured speedup is the ratio of the medians. Its range is the widest the runs allow, so treat a gap inside the range as no measured difference.
- Every run has to launch the requested grid and block size, or GASP stops. See [the benchmark notes](benchmarks/custom/README.md) for how to write a program for `--run`.

### Docker

```bash
make build
make docker-run ARGS="--gpu b200 --config benchmarks/custom/vecadd.toml"
make docker-run DOCKER_FLAGS="--gpus all" ARGS="--gpu rtx2080ti --run --config benchmarks/custom/vecadd.toml"
```

- `docker-run` mounts only `benchmarks/` and `hardware/`. Kernels and configs must be there, or already be in the image from the last `make build`.
- `src/` isn't mounted, so code changes need `make build`.
- Traces are lost when the container exits unless they go to a mounted path, such as `--trace-file benchmarks/traces.log`.
- `--run` needs a GPU in the container, which `DOCKER_FLAGS="--gpus all"` requests. This needs the NVIDIA container runtime, and running `ncu` in a container hasn't been tested yet.
- The Dev Container has no GPU access, so `--run` isn't available there.

### Options

These apply to the whole run:

| Flag                  | Description                                              |
|-----------------------|----------------------------------------------------------|
| `--gpu`               | Target GPU (e.g. `b200`, `h100`, `a100`)                 |
| `--profile`           | Path to hardware profile JSON                            |
| `--config`            | TOML file listing the kernels and their settings         |
| `--reference`         | Kernel the others are compared with (default: the first) |
| `--run`               | Run each program under `ncu` and compare with the prediction |
| `--repeats`           | `ncu` runs per kernel with `--run` (default: 3)          |
| `--verbose`           | Verbose output                                           |
| `--trace`             | Write traces to the trace file; optional comma-separated modules (`config,cuda,cfg,sass,occupancy,pico,dependency,predict,profiler`), default `all` |
| `--trace-file`        | Trace file path (default `./traces.log`); implies `--trace all` |
| `--help`              | Show usage                                               |
| `--version`           | Show version                                             |

These set a kernel's settings, and apply to every input file on the command line:

| Flag                  | Config key        | Description                                             |
|-----------------------|-------------------|---------------------------------------------------------|
| `--problem-size`      | `problem_size`    | Total elements (default: 1000000)                       |
| `--block-size`        | `block_size`      | Threads per block (default: 256)                        |
| `--grid-size`         | `grid_size`       | Number of blocks (derived from problem size if omitted) |
| `--loop-iterations`   | `loop_iterations` | Iterations per loop, applied to every loop. Required if the kernel has loops |
| `-I`, `--include-dir` | `include_dirs`    | Include directory passed to `nvcc` (repeatable)         |
| `--nvcc-flags`        | `nvcc_flags`      | Extra flags passed to `nvcc`                            |
| `--program-args`      | `program_args`    | Arguments passed to the program with `--run`            |
