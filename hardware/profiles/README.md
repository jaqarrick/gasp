# Profiles

The files within this directory contain the hardware profiles for targeted GPUs against our microbenchmark kernels as well as the official spec from NVIDIA. All `*.placeholder.json` will be replaced once we have finished the runs. The rtx2080 placeholder profile is for running / testing on CIMS.

The `l4` and `v100` placeholders have their own `spec` values, but their `profiled` values are copied from another GPU until the microbenchmarks run on them: `l4` from the `a100` placeholder, `v100` from the `rtx2080ti` placeholder.

## Format

Every profile, built-in or passed with `--profile`, needs:

- top-level `architecture`, `chip`, and `arch` (for example `sm_75`, which is passed to `nvcc` with `--run`)
- a `spec` object with the NVIDIA spec values
- a `profiled` object with the microbenchmark results

The required fields are the ones in `HardwareProfile` (`src/gasp/cli/utils.py`). GASP stops with an error that lists any missing, unknown, or non-numeric fields.

## Shared-memory load throughput

`profiled.shared_mem_load_throughput_cycles` is SM cycles per warp shared load with the shared-memory pipe saturated. No model uses it yet. `rtx2080ti.json` has the measured 2.0001 from `hardware/microbenchmarks/shared_memory_load_throughput.cu`. The placeholders hold values from published bandwidth (bytes per clock per SM; a conflict-free 32-bit warp load is 128 bytes):

| Profile | Value | Basis |
|---|---|---|
| rtx2080ti.placeholder (sm_75) | 2.0 | 64 B/clk, Turing; matches the measured value |
| v100 (sm_70) | 1.0 | 128 B/clk, Volta whitepaper |
| a100 (sm_80) | 1.0 | 128 B/clk, Ampere whitepaper |
| l4 (sm_89) | 1.0 | same SM design as Ampere (derived) |
| h100 (sm_90) | 1.0 | not published; Ampere value assumed |
| b200 (sm_100) | 1.0 | 128 B/clk (derived) |

## Sources

- Source for rtx2080ti L1/shared size and carveout options: https://docs.nvidia.com/cuda/turing-tuning-guide/index.html
- Source for a100 L1/shared size and carveout options: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- Source for h100 L1/shared size and carveout options: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- Source for b200 L1/shared size and carveout options: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- Source for l4 L1/shared size and carveout options: https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
- Source for v100 L1/shared size and carveout options: https://docs.nvidia.com/cuda/volta-tuning-guide/index.html
- l4 L2 size (48 MB) is not from an NVIDIA document; check it against `cudaDeviceProp::l2CacheSize` on the card
- Same values for all of them in one table: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
