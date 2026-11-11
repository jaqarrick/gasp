# Profiles

The files within this directory contain the hardware profiles for targeted GPUs against our microbenchmark kernels as well as the official spec from NVIDIA. All `*.placeholder.json` will be replaced once we have finished the runs. The rtx2080 placeholder profile is for running / testing on CIMS.

The `l4` and `v100` placeholders have their own `spec` values, but their `profiled` values are copied from another GPU until the microbenchmarks run on them: `l4` from the `a100` placeholder, `v100` from the `rtx2080ti` placeholder.

## Format

Every profile, built-in or passed with `--profile`, needs:

- top-level `architecture`, `chip`, and `arch` (for example `sm_75`, which is passed to `nvcc` with `--run`)
- a `spec` object with the NVIDIA spec values
- a `profiled` object with the microbenchmark results

The required fields are the ones in `HardwareProfile` (`src/gasp/cli/utils.py`) without a default; the four throughput fields below are optional. GASP stops with an error that lists any missing, unknown, or non-numeric fields.

## Throughput placeholders

Four optional `profiled` fields hold throughput rates. No model uses them yet; they're kept for GASPerf. A profile can leave one out or set it to `null`. Only the L1 rate has a benchmark (`hardware/microbenchmarks/l1_throughput.cu`, not yet run), so the `*.placeholder.json` profiles hold best-effort values from published sources. The measured `rtx2080ti.json` holds none of them: guessed values stay out of it. `run_all.py` adds a measured rate to the profile even when the base profile has no such field.

| Profile | `l1_throughput_sectors_per_cycle_per_sm` | `l2_bandwidth_bytes_per_cycle` | `shared_mem_load_throughput_cycles` |
|---|---|---|---|
| v100 (sm_70) | 4.0, high: 128 B/clk, Volta whitepaper | 2200, medium: about 2.1 TB/s, clock-adjusted | 1.0, high: 128 B/clk = 1 warp load per clock |
| rtx2080ti.placeholder (sm_75) | 2.0, high: 64 B/clk, Turing whitepaper | 1200, low: no published figure | 2.0, medium: 64 B/clk = 2 clocks per warp load |
| a100 (sm_80) | 4.0, high: 128 B/clk, Ampere whitepaper | 5120, high: A100 whitepaper | 1.0, high: 128 B/clk |
| l4 (sm_89) | 4.0, medium: same SM design as Ampere | 5000, low: no published B/clk | 1.0, medium: same SM design |
| h100 (sm_90) | 8.0, medium: 256 B/clk from 2x scaling | 6912, medium: ncu peak sustained metric | 1.0, low: MIO pipe unknown |
| b200 (sm_100) | 4.0, medium: 128 KB L1, Ampere-class | 10000, medium: 21 TB/s local partition | 1.0, medium: 128 B/clk |

`spec.simd_width` is 64 in every profile (since 2026-11-10): the FP32 lanes per SM on Volta and Turing, so Eq. 5's cap is 2 warp instructions per cycle per SM. Ada, Hopper and Blackwell SMs have 128 FP32 lanes, so 64 is half the spec rate for `l4`, `h100` and `b200`.

Confidence: high = whitepaper or measured, medium = derived, low = interpolated. A sector is 32 bytes, so the L1 rate is bytes per clock per SM / 32.

`shared_mem_store_throughput_cycles` is 4.46 on the RTX 2080 Ti, measured by `shared_memory_store_throughput.cu` with a single warp (run `1_0`); it is in both `rtx2080ti.json` and `rtx2080ti.placeholder.json`. The other profiles have no value for it and copy their shared load placeholder.

## Sources

- Source for rtx2080ti L1/shared size and carveout options: https://docs.nvidia.com/cuda/turing-tuning-guide/index.html
- Source for a100 L1/shared size and carveout options: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- Source for h100 L1/shared size and carveout options: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- Source for b200 L1/shared size and carveout options: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- Source for l4 L1/shared size and carveout options: https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
- Source for v100 L1/shared size and carveout options: https://docs.nvidia.com/cuda/volta-tuning-guide/index.html
- l4 L2 size (48 MB) is not from an NVIDIA document; check it against `cudaDeviceProp::l2CacheSize` on the card
- Same values for all of them in one table: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
