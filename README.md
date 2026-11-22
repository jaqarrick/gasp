# GASP - GPU Analytical Speedup Predictor

GASP is a fully static CUDA kernel speedup prediction tool. It estimates the speedup of one base kernel with one or more other kernels that perform similar work. Since GASP is fully static, the developer can test changes to an implementation without having to run anything on a GPU.

GASP predicts each kernel's total runtime in clock cycles, $T_k$, and gives the speedup of each kernel over the reference kernel as

```math
S_k = \frac{T_r}{T_k}
```

where $T_r$ is the predicted runtime of the reference kernel (the first kernel in the config, or the one named with `--reference`) and $T_k$ is that of kernel $k$. $S_k > 1$ means kernel $k$ is faster than the reference.

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
make install
source .venv/bin/activate
```

`make install` does the following:
- installs the Python dependencies
- installs the `faial-cost` binary to your local `$PATH` (supported on Linux x86-64 and macOS arm64, and `libgmp10` must be installed on Linux). Run `scripts/uninstall_faial.sh` to remove it.
- checks that the CUDA toolkit is installed

### Docker (optional)

```bash
git clone git@github.com:jaqarrick/gasp.git
cd gasp
make build
```

The CUDA image is quite large, so this will take some time the first build.

### Dev Container
Open the project in VS Code and select **Dev Containers: Reopen in Container**.
Python, CUDA, and all dependencies configured automatically. [More info here](./docs/contributing.md).

## Usage

Compare two kernels, targeting a specific GPU. The first is the reference, and shared flags apply to both:

```bash
make run ARGS="--gpu rtx2080ti --problem-size 1000000 --block-size 256 benchmarks/vecadd/vecadd_good.cu benchmarks/vecadd/vecadd_poor.cu"
```

Pick kernels out of a file that holds several. Without a kernel name, GASP stops and lists them:

```bash
make run ARGS="--gpu rtx2080ti kernels.cu::vecadd_good kernels.cu::vecadd_poor"
```

The cleanest method is to create a launch config for the kernels, providing inputs and settings per kernel or for the entire run.
For example, a config like [this one](benchmarks/vecadd/configs/vecadd.toml):

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

Use the config:

```bash
make run ARGS="--gpu rtx2080ti --config benchmarks/vecadd/configs/vecadd.toml"
```

A single kernel prints its predicted time with no speedup:

```bash
make run ARGS="--gpu rtx2080ti benchmarks/vecadd/vecadd.cu"
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

### GASP Options

These apply to the whole run:

| Flag                  | Description                                              |
|-----------------------|----------------------------------------------------------|
| `--gpu`               | Target GPU: `rtx2080ti`, `titanv`, `rtx4070` or `a100` (measured profiles in `hardware/profiles/`) |
| `--profile`           | Path to a hardware profile JSON (same format as `hardware/profiles/*.json`); can't be combined with `--gpu` |
| `--config`            | TOML file listing the kernels and their settings         |
| `--reference`         | Kernel the others are compared with (default: the first) |
| `--run`               | Run each program under `ncu` and compare with the prediction |
| `--repeats`           | `ncu` runs per kernel with `--run` (default: 3)          |
| `--save-metrics FILE` | With `--run`: save the ncu measurements to `FILE` (JSON) |
| `--metrics FILE`      | Use the measurements in `FILE` instead of running `ncu`; can't be combined with `--run` |
| `--override GROUP=SOURCE` | Replace a group of static inputs (repeatable), see [Overrides](./docs/overrides.md) |
| `--json`              | Write the prediction (and measured results) as JSON to stdout, everything else to stderr |
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
| `--reuse`             | `reuse`           | Share of global accesses (0 to 1) that hit L1; sets the L1 hit ratio directly (optional) |
| `--block-size`        | `block_size`      | Threads per block, such as `256` or `16x16` (default: 256) |
| `--grid-size`         | `grid_size`       | Blocks in the grid, such as `64x64` (derived from problem size if omitted and the block is 1D) |
| `--loop-iterations`   | `loop_iterations` | Iterations per loop, applied to every loop. Required if the kernel has loops, unless its config entry sets `loop_counts` (see [Loop counts](./docs/loops.md)) |
| `-I`, `--include-dir` | `include_dirs`    | Include directory passed to `nvcc` (repeatable)         |
| `--nvcc-flags`        | `nvcc_flags`      | Extra flags passed to `nvcc`                           |
| `--program-args`      | `program_args`    | Arguments passed to the program with `--run`            |
| `--param NAME=VALUE`  | `params`          | Value of a kernel argument for Pico, such as `M=1024` (repeatable) |

### Config files

We recommend using a `.toml` file and key over a CLI flag. More [here](./docs/configs.md).

### Acknowledgments

GASP builds on GPUPerf (Sim et al., PPoPP 2012), its MWP-CWP model (Hong and Kim, ISCA 2009), and Faial (Cogumbreiro et al., FMSD 2024). See [docs/citations.md](./docs/citations.md) for the full references and BibTeX.
