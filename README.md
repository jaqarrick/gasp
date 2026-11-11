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
git clone git@github.com:jaqarrick/gasp.git
cd gasp
python3 -m venv .venv
source .venv/bin/activate
make install
```

`make install` installs the Python dependencies, then runs `scripts/install_faial.sh` to fetch faial (pico) and `scripts/check_requirements.sh` to verify the setup. faial is installed to `~/.local/share/faial` with wrappers in `~/.local/bin` (override with `FAIAL_PREFIX` / `BIN_DIR`); it is supported on Linux x86-64 and macOS arm64, and `libgmp10` must be installed on Linux. If the wrapper directory is not on `PATH`, the install script offers to add it to your shell's rc file; otherwise set `GASP_FAIAL_COST` to the `faial-cost` wrapper's path (e.g. `export GASP_FAIAL_COST=~/.local/bin/faial-cost`) and gasp uses it instead of looking on `PATH`. Run `scripts/uninstall_faial.sh` to remove it.

The CUDA tools must be on `PATH` before `make install` and `make run`. On a cluster with environment modules, load CUDA first (`module load cuda`; `module avail cuda` lists the versions).

### Docker

```bash
git clone git@github.com:jaqarrick/gasp-gpu.git
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
make run ARGS="--gpu b200 --problem-size 1000000 --block-size 256 benchmarks/vecadd/vecadd_good.cu benchmarks/vecadd/vecadd_poor.cu"
```

Pick kernels out of a file that holds several. Without a kernel name, GASP stops and lists them:

```bash
make run ARGS="--gpu b200 kernels.cu::vecadd_good kernels.cu::vecadd_poor"
```

Give each kernel its own settings with a TOML file (see [`vecadd.toml`](benchmarks/vecadd/vecadd.toml)):

```bash
make run ARGS="--gpu b200 --config benchmarks/vecadd/vecadd.toml"
```

```toml
[defaults]
problem_size = 1000000
block_size = 256

[[kernel]]
name = "good"
file = "benchmarks/vecadd/vecadd_good.cu"

[[kernel]]
name = "poor"
file = "benchmarks/vecadd/vecadd_poor.cu"
block_size = 128
```

A single kernel prints its predicted time with no speedup:

```bash
make run ARGS="--gpu b200 benchmarks/vecadd/vecadd.cu"
```

Use your own hardware profile instead of a built-in GPU. It uses the same format as [`hardware/profiles/*.json`](hardware/profiles/README.md), and the target architecture comes from its `arch` field:

```bash
make run ARGS="--profile hardware/profiles/rtx2080ti.json benchmarks/vecadd/vecadd.cu"
```

### How kernels are compared

- **Reference:** the first kernel, unless `--reference NAME` is given.
- **Speedup:** `T_reference / T_kernel`, using predicted T_GPU in cycles. Above 1 means faster than the reference.
- **Names:** `name` in a config. Otherwise the kernel name from `file.cu::kernel`, or else the file name without `.cu`.
- **Same work:** compared kernels should do the same work. Different problem sizes, program arguments or `params` print a warning.
- **Pico:** the `Pico` column shows `fallback` when Pico gave no usable cost for that kernel. A warning is printed if only some kernels fell back, because their speedups then compare different coalescing assumptions.

### Config files

- A setting comes from the kernel entry, then `[defaults]`, then the built-in default.
- Settings are `file`, `name`, `kernel`, `problem_size`, `footprint`, `reuse`, `block_size`, `grid_size`, `loop_iterations`, `loop_counts`, `include_dirs`, `nvcc_flags`, `program_args` and `params`. `name` and `loop_counts` are only allowed on a kernel entry.
- `include_dirs`, `nvcc_flags` and `program_args` are arrays of strings, such as `program_args = ["1000000"]`.
- `footprint` is the number of distinct bytes of global memory the kernel reads and writes, such as `footprint = 12000000` for vecadd on 1000000 floats (three arrays of 4-byte elements). The cache model takes footprint / bytes accessed as the share of accesses that are first touches and must come from DRAM, so reuse between threads counts. L2 then holds what it can of the rest, by the footprint against the L2 size. Without it, GASP estimates reuse from each thread's own accesses only, which misses kernels where threads share data, such as matrix multiply.
- `reuse` is the share of global accesses, from 0 to 1, that hit L1 on data the same block just loaded, and sets the L1 hit ratio directly. Work it out from the code: in a 16x16 matrix multiply each value is read by 16 threads of the block, so 15 of 16 reads hit (`reuse = 0.9375`); a kernel whose threads read only their own data has `reuse = 0`. Without it, L1 is estimated from the working set, which misses reuse that happens right after a load.
- `block_size` and `grid_size` are a number or an array of up to 3 dimensions, such as `block_size = [16, 16]`. A `block_size` with several dimensions needs `grid_size`, because the grid can't be derived from `problem_size`.
- `params` gives Pico the values of the kernel's arguments, such as `params = { M = 1024, N = 1024, K = 1024 }`. Without them, a cost that depends on an argument makes Pico fall back. Every name must be an argument of the kernel, or Pico fails and falls back for that kernel, so keep `params` in `[defaults]` only for kernels with the same arguments. A kernel's `params` are merged name by name over those in `[defaults]`, so `params = { K = 512 }` changes only `K`.
- Paths are relative to the directory GASP runs from, not to the config file.
- `--config` can't be combined with input files or per-kernel flags.

### Loop counts

GASP multiplies each loop's instructions by its trip count, which it can't work out on its own yet. There are two ways to give it:

- `loop_counts = [64, 0, 0]` on a `[[kernel]]` entry gives each loop its own count, in the order the loops appear in the compiled code.
- `loop_iterations` (`--loop-iterations`) gives every loop the same count. `loop_counts` wins if both are set, and GASP warns.

The count is trips per entry into the loop; nested loops multiply. A count of 0 means the loop doesn't run.

The list follows the loops in the compiled code (SASS), not the source. nvcc unrolls loops, so one source loop can become several: `mm_basic`'s `k` loop is a main loop that does 16 values of `k` per trip, plus 4-step and 1-step loops for what's left. For K = 1024 that's `[64, 0, 0]`. `--verbose` prints the loops so you can work out each count, and a missing or wrong-length `loop_counts` prints them before GASP stops:

```
#  Source line                                              Inside  Insts  Loads/trip  Count
1  mm_basic.cu:13  sum += a[row * K + k] * b[k * N + col];  -       70     32 global   64
2  mm_basic.cu:13  sum += a[row * K + k] * b[k * N + col];  -       24     8 global    0
3  mm_basic.cu:13  sum += a[row * K + k] * b[k * N + col];  -       10     2 global    0
```

- **Source line:** where the loop starts in the source. GASP compiles with `-lineinfo`, which doesn't change the SASS. Several loops on the same line come from one source loop.
- **Loads/trip:** global and shared loads in one trip. Divide by the loads one source iteration does to get how many source iterations a trip covers: here 32 / 2 = 16, 8 / 2 = 4 and 2 / 2 = 1.
- **Inside:** the enclosing loop, for nested loops.

The counts depend on the problem size, and the loops can change with the architecture or nvcc version, so check the table when either changes.

### Measuring with `--run`

`--run` builds each program and runs it under `ncu`, then prints predicted against measured cycles and speedup:

```bash
make run ARGS="--gpu rtx2080ti --run benchmarks/vecadd/vecadd_good.cu benchmarks/vecadd/vecadd_poor.cu"
```

- Each kernel runs `--repeats N` times (default 3). Cycles are ncu's Elapsed Cycles (`gpc__cycles_elapsed.max`). The table shows the median, with the min-max spread beside it.
- Measured speedup is the ratio of the medians. Its range is the widest the runs allow, so treat a gap inside the range as no measured difference.
- Every run has to launch the requested grid and block size, or GASP stops. See [the benchmark notes](benchmarks/README.md) for how to write a program for `--run`.

### Overrides

An override replaces a group of the model's static inputs with what ncu measured, to see how much of the error that input causes. Profile once and save the measurements, then replay them with any override. Replaying needs no GPU:

```bash
make run ARGS="--gpu rtx2080ti --run --config benchmarks/vecadd/vecadd.toml --save-metrics vecadd.metrics.json"
make run ARGS="--gpu rtx2080ti --config benchmarks/vecadd/vecadd.toml --metrics vecadd.metrics.json --override cache=profiled"
```

`--override GROUP=SOURCE` is repeatable. A group not named stays static.

| Group        | Inputs replaced                          | `profiled` (from the metrics file)                     | `crude`       |
|--------------|------------------------------------------|--------------------------------------------------------|---------------|
| `cache`      | L1 and L2 hit ratios                     | ncu's L1 hit rate for global loads; L2 read lookup hits / lookups | -  |
| `coalescing` | Transactions per warp access             | Global-load sectors / requests / 4, at least 1         | 1 (coalesced) |
| `bank`       | Bank conflicts per warp                  | Shared load + store bank conflicts / warps             | 0             |
| `counts`     | Total, global memory and SFU instructions per warp | ncu's executed instruction counts / warps    | -             |

- `all=profiled` sets every group. `crude` is what the model assumes without Pico.
- ILP, MLP and the sync instruction count have no profiled source, so they're always static.
- The metrics file holds ncu's raw values for every run, keyed by kernel name. GASP stops if the GPU, a kernel name, or a kernel's block or grid size differs from the file, or if a metric a `profiled` override needs is missing.
- `--metrics` also prints the predicted vs measured table from the file. It can't be combined with `--run`.

The full ablation, one row per override set, is in [`experiments/ablation`](experiments/ablation/README.md): `make ablation ARGS="..."`.

### Models

Every run predicts each kernel with two models and shows one row per kernel per model:

| Model     | Module                     | What it is |
|-----------|----------------------------|------------|
| `GPUPerf` | `src/gasp/models/gpuperf/` | GPUPerf as published (Sim et al.). One fix: SFU instructions are left out of `#insts` and of the average instruction latency, as in the paper, so they're charged only in O_SFU (Eq. 9). |
| `GASPerf` | `src/gasp/models/gasperf/` | Our model. It starts as an exact copy of GPUPerf, and formula changes go here, never into GPUPerf. |

- Speedups compare kernels under the same model: each kernel's speedup is against the reference kernel's T_GPU under that model.
- Static analysis, the cache model and overrides run before the model, so both models get the same inputs. With `--run`, each kernel is profiled once.
- Both models take `inputs.ModelInputs` (`src/gasp/models/inputs.py`) and have a `run()` that computes everything from ITILP (Eqs. 4–5) to T_exec (Eq. 1). Launch overhead is added after either one. `src/gasp/models/__init__.py` lists the models.

### Docker

```bash
make build
make docker-run ARGS="--gpu b200 --config benchmarks/vecadd/vecadd.toml"
make docker-run DOCKER_FLAGS="--gpus all" ARGS="--gpu rtx2080ti --run --config benchmarks/vecadd/vecadd.toml"
```

- `docker-run` mounts only `benchmarks/` and `hardware/`. Kernels and configs must be there, or already be in the image from the last `make build`.
- A `--profile` file has to be under `hardware/` or `benchmarks/` so the container can see it.
- `src/` isn't mounted, so code changes need `make build`.
- Traces are lost when the container exits unless they go to a mounted path, such as `--trace-file benchmarks/traces.log`.
- `--run` needs a GPU in the container, which `DOCKER_FLAGS="--gpus all"` requests. This needs the NVIDIA container runtime, and running `ncu` in a container hasn't been tested yet.
- The Dev Container has no GPU access, so `--run` isn't available there.
- Replaying a metrics file (`--metrics`, `--override`) needs no GPU, so it works in any container. The file has to be under `benchmarks/` or `hardware/` to be visible to `docker-run`. `--save-metrics` has to write to one of those paths too, or the file is lost with the container. `make ablation` runs locally only.

### Options

These apply to the whole run:

| Flag                  | Description                                              |
|-----------------------|----------------------------------------------------------|
| `--gpu`               | Target GPU: `rtx2080ti` (measured profile), `b200`, `h100`, `a100` (placeholders) |
| `--profile`           | Path to a hardware profile JSON (same format as `hardware/profiles/*.json`); can't be combined with `--gpu` |
| `--config`            | TOML file listing the kernels and their settings         |
| `--reference`         | Kernel the others are compared with (default: the first) |
| `--run`               | Run each program under `ncu` and compare with the prediction |
| `--repeats`           | `ncu` runs per kernel with `--run` (default: 3)          |
| `--save-metrics FILE` | With `--run`: save the ncu measurements to `FILE` (JSON) |
| `--metrics FILE`      | Use the measurements in `FILE` instead of running `ncu`; can't be combined with `--run` |
| `--override GROUP=SOURCE` | Replace a group of static inputs (repeatable), see [Overrides](#overrides) |
| `--verbose`           | Verbose output                                           |
| `--trace`             | Write traces to the trace file; optional comma-separated modules (`config,cuda,cfg,sass,occupancy,pico,dependency,predict,profiler`), default `all` |
| `--trace-file`        | Trace file path (default `./traces.log`); implies `--trace all` |
| `--help`              | Show usage                                               |
| `--version`           | Show version                                             |

These set a kernel's settings, and apply to every input file on the command line:

| Flag                  | Config key        | Description                                             |
|-----------------------|-------------------|---------------------------------------------------------|
| `--problem-size`      | `problem_size`    | Total elements (default: 1000000)                       |
| `--footprint`         | `footprint`       | Distinct global bytes, for the cache model (optional)   |
| `--reuse`             | `reuse`           | Share of global accesses that hit L1, 0 to 1 (optional) |
| `--block-size`        | `block_size`      | Threads per block, such as `256` or `16x16` (default: 256) |
| `--grid-size`         | `grid_size`       | Blocks in the grid, such as `64x64` (derived from problem size if omitted and the block is 1D) |
| `--loop-iterations`   | `loop_iterations` | Iterations per loop, applied to every loop. Required if the kernel has loops, unless its config entry sets `loop_counts` (see [Loop counts](#loop-counts)) |
| `-I`, `--include-dir` | `include_dirs`    | Include directory passed to `nvcc` (repeatable)         |
| `--nvcc-flags`        | `nvcc_flags`      | Extra flags passed to `nvcc`                            |
| `--program-args`      | `program_args`    | Arguments passed to the program with `--run`            |
| `--param NAME=VALUE`  | `params`          | Value of a kernel argument for Pico, such as `M=1024` (repeatable) |
