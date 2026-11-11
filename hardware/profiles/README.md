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

## L1 throughput

`profiled.l1_cycles_per_request` (fixed cycles per global request) and `profiled.l1_sectors_per_cycle_per_sm` (L1 bandwidth in 32-byte sectors per cycle per SM) feed GASPerf's L1 limit. `rtx2080ti.json` (and `rtx2080ti.placeholder.json`) have values from the `plateaus` of `hardware/microbenchmarks/l1_throughput.cu`: 2.4796 = 1 / 0.4033 requests per cycle (32-bit loads, the same for 2 and 4 sectors per request) and 1.999 (the 64- and 128-bit shapes, 1.9989 and 1.9992). `run_all.py` derives both from the run JSON's `plateaus` (1 / the 32-bit 4-sector request rate, and the mean sector rate of the 64- and 128-bit shapes) and warns about any shape without a plateau.

| Profile | `l1_cycles_per_request` | `l1_sectors_per_cycle_per_sm` |
|---|---|---|
| rtx2080ti, rtx2080ti.placeholder (sm_75) | 2.4796, measured | 1.999, measured |
| v100, a100, l4, b200 | 1.0, assumed | 4.0, 128 B/clk (whitepapers; l4 and b200 derived) |
| h100 | 0.5, assumed | 8.0, 256 B/clk (derived from 2× scaling) |

"Assumed": no published figure, so a 4-sector request is charged only its bandwidth time. On the RTX 2080 Ti the measured fixed cost is higher than that (2.48 against 2.0), so these are likely low.

## L2 bandwidth

`profiled.l2_bandwidth_bytes_per_cycle` is the whole-GPU L2 bandwidth with every load hitting L2, from `hardware/microbenchmarks/l2_bandwidth.cu`: 1164.9505 on the RTX 2080 Ti (`working_sets_agree` and `saturated` true), also copied into `rtx2080ti.placeholder.json`. No model uses it yet (#122). The other placeholders hold published values: v100 2200 (about 2.1 TB/s, clock-adjusted), a100 5120 (A100 whitepaper), l4 5000 (no published B/clk), h100 6912 (ncu peak sustained metric), b200 10000 (21 TB/s local partition).

## Sources

- Source for rtx2080ti L1/shared size and carveout options: https://docs.nvidia.com/cuda/turing-tuning-guide/index.html
- Source for a100 L1/shared size and carveout options: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- Source for h100 L1/shared size and carveout options: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- Source for b200 L1/shared size and carveout options: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- Source for l4 L1/shared size and carveout options: https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
- Source for v100 L1/shared size and carveout options: https://docs.nvidia.com/cuda/volta-tuning-guide/index.html
- l4 L2 size (48 MB) is not from an NVIDIA document; check it against `cudaDeviceProp::l2CacheSize` on the card
- Same values for all of them in one table: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
