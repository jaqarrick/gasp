# GASP - GPU Analytical Speedup Predictor

Work in progress static CUDA kernel analyzer

## Prerequisites
- Python 3.12+
- CUDA Toolkit 12.0+ (`nvcc`, `nvdisasm`, `cuobjdump`)
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

### Local

```bash
make run ARGS="benchmarks/custom/vecadd.cu --gpu b200 --verbose"
```

### Docker

```bash
make docker-run ARGS="benchmarks/custom/vecadd.cu --gpu b200 --verbose"
```

### Options

| Flag             | Description                                              |
|------------------|----------------------------------------------------------|
| `--gpu`          | Target GPU (e.g. `b200`, `h100`, `a100`)                |
| `--block_size`   | Threads per block (default: 256)                         |
| `--grid_size`    | Number of blocks (derived from problem size if omitted)  |
| `--problem_size` | Total elements                                      |
| `--profile`      | Path to hardware profile JSON                            |
| `--verbose`      | Verbose output                              |
| `--log-traces`   | Write traces to the trace file; optional comma-separated modules (`cuda,cfg,sass,occupancy,pico,dependency,predict,profiler`), default `all` |
| `--trace-file`   | Trace file path (default `./traces.log`); implies `--log-traces all` |
| `--help`         | Show usage                                   |
| `--version`      | Show version                                             |
