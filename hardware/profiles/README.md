# Profiles

Hardware profiles for the GPUs GASP targets: the NVIDIA spec values and the microbenchmark results. `--gpu` accepts `rtx2080ti`, `titanv`, `rtx4070` and `a100`, all measured profiles (`GPU_PROFILES` in `src/gasp/cli/utils.py`). For another GPU, pass a profile with `--profile`.

## Format

Every profile, built-in or passed with `--profile`, needs:

- top-level `architecture`, `chip`, and `arch` (for example `sm_75`, which is passed to `nvcc` with `--run`)
- a `spec` object with the NVIDIA spec values
- a `profiled` object with the microbenchmark results

The required fields are the ones in `HardwareProfile` (`src/gasp/cli/utils.py`). GASP stops with an error that lists any missing, unknown, or non-numeric fields.

## Checking a profile's spec

`scripts/check_gpu_spec.sh <profile.json>` compiles a small CUDA query (`nvcc -arch=native`) and compares the profile's `arch`, `warp_size`, `sms_per_gpu`, `max_warps_per_sm`, `max_blocks_per_sm`, `max_registers_per_sm`, `max_shared_mem_per_sm_bytes` and `l2_cache_size_bytes` with what the GPU reports. It exits 1 on any mismatch. Without an argument it only prints the GPU's values. `simd_width`, `sfu_width`, `l1_shared_array_bytes` and `shared_mem_carveouts_bytes` can't be queried, so check those against the sources below. It prints the max SM clock for reference; `clock_rate_ghz` should be the base clock, which a microbenchmark run reports as "SM clock: median". `--output <path>` writes everything to that file instead of the terminal. `make gpu-spec [PROFILE=<profile.json>] [OUTPUT=<path>]` runs it, for example `make gpu-spec OUTPUT=experiments/results/ablation/a100/batch_<timestamp>/gpu_spec.txt` to keep the GPU's spec next to a batch.

## Shared-memory load throughput

`profiled.shared_mem_load_throughput_cycles` is SM cycles per warp shared load with the shared-memory pipe saturated (a conflict-free 32-bit warp load is 128 bytes). GASPerf uses it for its shared-memory pipe floor.

| Profile | Value |
|---|---|
| rtx2080ti (sm_75) | 2.0021 |
| rtx4070 (sm_89) | 2.004 |
| titanv (sm_70) | 1.0102 |
| a100 (sm_80) | 1.001 |

## L1 throughput

`profiled.l1_cycles_per_request` (fixed cycles per global request) and `profiled.l1_sectors_per_cycle_per_sm` (L1 bandwidth in 32-byte sectors per cycle per SM) feed GASPerf's L1 limit. `run_all.py` derives both from the `plateaus` of `hardware/microbenchmarks/gasp-bm/l1_throughput.cu`: 1 / the 32-bit 4-sector request rate, and the mean sector rate of the 64- and 128-bit shapes. It warns about any shape without a plateau.

| Profile | `l1_cycles_per_request` | `l1_sectors_per_cycle_per_sm` |
|---|---|---|
| rtx2080ti | 2.5233 | 1.9904 |
| rtx4070 | 2.5253 | 3.7917 |
| titanv | 2.2732 | 3.8406 |
| a100 | 2.4963 | 3.8863 |

## L1 miss cost

`profiled.l1_miss_cycles_per_request` is the extra L1 time of a global load request that misses L1 and is served by L2: GASPerf charges `(1 - h1) x` this on top of each request's L1 cost. From `hardware/microbenchmarks/gasp-bm/l1_miss_cost.cu`: 32-bit coalesced loads on 4 SMs, 0, 1, 2, 4 or 8 of every 8 sent past L1 (`ld.global.cg`) to an L2-resident array; the value is the slope of cycles per request against the fraction that miss.

| Profile | Value |
|---|---|
| rtx2080ti | 1.6851 |
| rtx4070 | 1.6219; the cost isn't linear in the miss fraction on this GPU (mixed hits and misses are faster than all hits), so the slope is a rough fit |
| titanv | 1.8764 |
| a100 | 0.2308 |

## L2 bandwidth

`profiled.l2_bandwidth_bytes_per_cycle` is the whole-GPU L2 bandwidth with every load hitting L2. GASPerf uses it for its L2 limit on T_mem.

| Profile | Value |
|---|---|
| rtx2080ti | 1330.6586 (HPMLL; gasp-bm's `l2_bandwidth.cu` gave 1164.9505, and `benchmarks/l2/configs/l2.toml` measured about 1,057 with 32-bit loads, so the HPMLL value may be off by a clock conversion) |
| rtx4070 | 784.0962 |
| titanv | 1773.7093 |
| a100 | 1709.0817 |

## Sources

- RTX 2080 Ti L1/shared size and carveout options: https://docs.nvidia.com/cuda/turing-tuning-guide/index.html
- TITAN V L1/shared size and carveout options: https://docs.nvidia.com/cuda/volta-tuning-guide/index.html
- A100 L1/shared size and carveout options: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- RTX 4070 L1/shared size and carveout options: https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
- Same values for all of them in one table: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
