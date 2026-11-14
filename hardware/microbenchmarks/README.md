# Microbenchmarks
This are targeted CUDA programs for specific operation measurements

## Building a profile
`make microbenchmarks GPU=<gpu> VERSION=<label>` (`gasp-bm/run_all.py`) runs three suites and merges them into one profile, `hardware/runs/<gpu>/<version>/<gpu>.<version>.json`. Each field takes the first valid result of:

1. gpu-benches' `gpu-stream`, for `dram_bandwidth_bytes_per_cycle` only (see "gpu-benches: DRAM bandwidth").
2. HPMLL (`scripts/run/run_hpmll_bmks.py`, results in `<version>/hpmll/`), for every field it measures: fp32/fp64/int32 instruction latency, L1/L2/DRAM latency, shared load latency and throughput, and L1 and L2 bandwidth. Its `mem_bw` is not run, since `gpu-stream` supplies that field.
3. gasp-bm (`gasp-bm/*.cu`), for every other field, and for an HPMLL field whose HPMLL result was invalid or missing.

The gasp-bm benchmarks that HPMLL overrides still run, as the fallback. `shared_memory_store_cycles` also needs gasp-bm's shared load latency. `sync_cost` gets the same DRAM latency and bandwidth as the profile. `<gpu>.<version>.sources.json` records each field's source file, the gasp-bm value an HPMLL value replaced (`replaces`), and why an HPMLL result was not used (`hpmll_not_used`). Fields nothing measures are copied from the base profile. `ARGS='--only hpmll,dram_latency'` runs a subset; results already in the version directory are reused.

`make microbenchmarks-hpmll` and `make microbenchmarks-gpu-benches` still run one suite on its own.

## Citation: HPMLL NVIDIA Hopper Benchmark
The `hpmll/` submodule is [HPMLL/NVIDIA-Hopper-Benchmark](https://github.com/HPMLL/NVIDIA-Hopper-Benchmark) (GPL-3.0). Its authors ask that the project and these papers be cited:

- Weile Luo, Ruibo Fan, Zeyu Li, Dayou Du, Qiang Wang, and Xiaowen Chu. "[Benchmarking and Dissecting the Nvidia Hopper GPU Architecture.](https://ieeexplore.ieee.org/abstract/document/10579250)" In 2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS), pp. 656-667. IEEE, 2024.
- Weile Luo, Ruibo Fan, Zeyu Li, Dayou Du, Hongyuan Liu, Qiang Wang, and Xiaowen Chu. "[Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis.](https://arxiv.org/abs/2501.12084)" arXiv preprint arXiv:2501.12084 (2025).

```bibtex
@inproceedings{luo2024benchmarking,
  title={Benchmarking and dissecting the nvidia hopper gpu architecture},
  author={Luo, Weile and Fan, Ruibo and Li, Zeyu and Du, Dayou and Wang, Qiang and Chu, Xiaowen},
  booktitle={2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  pages={656--667},
  year={2024},
  organization={IEEE}
}

@article{luo2025dissecting,
  title={Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis},
  author={Luo, Weile and Fan, Ruibo and Li, Zeyu and Du, Dayou and Liu, Hongyuan and Wang, Qiang and Chu, Xiaowen},
  journal={arXiv preprint arXiv:2501.12084},
  year={2025}
}
```

## Citation: gpu-benches
The `gpu-benches/` submodule is [RRZE-HPC/gpu-benches](https://github.com/RRZE-HPC/gpu-benches) (GPL-3.0). It is built and run from the submodule as published, not copied into GASP. Its README lists these papers as using its benchmarks and welcomes citations:

- Dominik Ernst, Markus Holzer, Georg Hager, Matthias Knorr, and Gerhard Wellein. "[Analytical performance estimation during code generation on modern GPUs.](https://doi.org/10.1016/j.jpdc.2022.11.003)" Journal of Parallel and Distributed Computing 173 (2023): 152-167.
- Dominik Ernst, Georg Hager, Jonas Thies, and Gerhard Wellein. "[Performance engineering for real and complex tall & skinny matrix multiplication kernels on GPUs.](https://doi.org/10.1177/1094342020965661)" The International Journal of High Performance Computing Applications 35, no. 1 (2020): 5-19.

```bibtex
@article{ernst2023analytical,
  title={Analytical performance estimation during code generation on modern GPUs},
  author={Ernst, Dominik and Holzer, Markus and Hager, Georg and Knorr, Matthias and Wellein, Gerhard},
  journal={Journal of Parallel and Distributed Computing},
  volume={173},
  pages={152--167},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.jpdc.2022.11.003}
}

@article{ernst2020performance,
  title={Performance engineering for real and complex tall \& skinny matrix multiplication kernels on GPUs},
  author={Ernst, Dominik and Hager, Georg and Thies, Jonas and Wellein, Gerhard},
  journal={The International Journal of High Performance Computing Applications},
  volume={35},
  number={1},
  pages={5--19},
  year={2020},
  publisher={SAGE},
  doi={10.1177/1094342020965661}
}
```

## gpu-benches: DRAM bandwidth

Its `gpu-stream` supplies the profile's `dram_bandwidth_bytes_per_cycle`: the highest read-kernel GB/s over its occupancy sweep, divided by the base profile's `clock_rate_ghz`. It runs without ncu, since it times from the host. This is exact because DRAM-bound GB/s does not depend on the SM clock, so no clock lock is needed.

`make microbenchmarks` runs it through `scripts/run/run_gpu_benches.py`, which writes the run JSON into the version directory, where `sync_cost` also reads it. `make microbenchmarks-gpu-benches GPU=<gpu> VERSION=<label>` runs it on its own and also compares the value with the base profile and with this version's gasp-bm and HPMLL profiles, in `hardware/runs/<gpu>/<version>/gpu-benches/<gpu>.<version>.gpu-benches.json`.

DRAM latency stays with gasp-bm's `dram_latency.cu`. gpu-benches' `gpu-latency` would need its source patched to give base-clock cycles and a usable run time, so it is not used.

## Shared memory load microbenchmark
* Microbenchmark that measures the a sequence of shared-memory loads with a conflict-free warp access pattern
* Launches one warp, where each lane follows their own dependent chain through shared memory. Each lane accesses a different bank on each load.
* Runs 200 samples, computes statistics, and writes to a results JSON.
Build:
* A100: nvcc -O3 -std=c++17 -arch=sm_80 shared_memory_load.cu -o shared_memory_load
* H100: nvcc -O3 -std=c++17 -arch=sm_90 shared_memory_load.cu -o shared_memory_load
* B200: nvcc -O3 -std=c++17 -arch=sm_100 shared_memory_load.cu -o shared_memory_load
Run example:
* ./shared_memory_load --reps 200 --gpu <gpu_name>

## Shared memory load throughput
* Measures `shared_mem_load_throughput_cycles`: SM cycles per warp shared load (`LDS`) when the shared-memory pipe is saturated. `shared_memory_load.cu` measures the latency instead.
* One block per SM; each warp issues 16 independent, conflict-free loads per group. Cycles per load = (span − empty-loop span) / (warps on the SM × loads per warp), per-SM `clock64` span.
* Sweeps warps per SM (1, 2, 4, …, 32) and reports the plateau of the loads-only shape. A second shape (10 loads to 1 store per group, tiled matmul's ratio) shows whether loads and stores share the pipe. It warns if the two highest warp counts differ by more than 2% (no plateau). Every configuration is in the run JSON.
* The value is an upper bound on the pipe's cost per load: each group also has 8 XORs and the loop counter.
* Writes the profile field `shared_mem_load_throughput_cycles` (2.0001 on the RTX 2080 Ti). No model uses it yet.
* Check under ncu: zero shared-load bank conflicts, no global traffic in the timed kernel.
Build:
* `nvcc -O3 -std=c++17 -arch=sm_XX shared_memory_load_throughput.cu -o shared_memory_load_throughput` (sm_XX from the GPU's compute capability)
Run through `make microbenchmarks`:
* `make microbenchmarks GPU=<gpu> VERSION=<label> ARGS='--only shared_memory_load_throughput'`

## L1 throughput
* Measures how many global-load sectors (and requests) per cycle one SM's L1 serves when every load hits.
* One block per SM reads its own slice of a global array, at most half of L1, warmed first, so every timed load is an L1 hit. Each warp keeps 8 independent loads in flight, 4,096 loads per warp; per-SM `clock64` span.
* Sweeps warps per SM (1–32) and load shape: 32-bit with 4 or 2 sectors per request, 64-bit, 128-bit, and 32-bit with 2 global stores per 8 loads (whether stores share the queue). `value` is the plateau sector rate of 32-bit 4-sector loads; `requests_per_cycle_per_sm` and every shape's plateau are in the run JSON.
* `run_all.py` passes `--l1-bytes` (the unified L1/shared array minus the smallest carveout, from the profile spec).
* `run_all.py` writes two profile fields from the `plateaus`: `l1_cycles_per_request` (1 / the 32-bit 4-sector request rate) and `l1_sectors_per_cycle_per_sm` (mean sector rate of the 64- and 128-bit shapes). GASPerf's L1 limit uses both.
* Check under ncu: L1 hit rate near 100% (`run_all.py` runs it with `--cache-control none`).
Build:
* `nvcc -O3 -std=c++17 -arch=sm_XX l1_throughput.cu -o l1_throughput` (sm_XX from the GPU's compute capability)
Run through `make microbenchmarks`:
* `make microbenchmarks GPU=<gpu> VERSION=<label> ARGS='--only l1_throughput'`

## L2 bandwidth
* Measures `l2_bandwidth_bytes_per_cycle`: bytes per SM cycle the whole GPU's L2 serves when every load hits L2. Same unit and timing as `dram_bandwidth.cu`, so the two compare directly.
* The working set is a fraction of L2, warmed first. Loads are `ld.global.cg` (not cached in L1), and every block sweeps the whole working set from its own starting row, so L1 can't serve them. Each thread keeps 4 independent 16-byte loads in flight.
* Sweeps the working set (20, 30, 40, 50% of L2; the rate should be flat) and the number of blocks (a quarter of the SMs up to the occupancy limit; the rate should stop rising). `value` is the median at 40% of L2 with the most blocks; the JSON says whether the neighbouring working sets and the two highest block counts agree within 5%.
* L2 size from the device (`--l2-cache-size KB` overrides it).
* Writes the profile field `l2_bandwidth_bytes_per_cycle`. No model uses it yet.
* Check under ncu: L1 hit rate near 0, L2 hit rate near 100%, DRAM throughput near 0, and a value above the DRAM bandwidth.
Build:
* `nvcc -O3 -std=c++17 -arch=sm_XX l2_bandwidth.cu -o l2_bandwidth` (sm_XX from the GPU's compute capability)
Run through `make microbenchmarks`:
* `make microbenchmarks GPU=<gpu> VERSION=<label> ARGS='--only l2_bandwidth'`

## Constant memory load microbenchmarks
* Measures the dependent, uniform constant-cache hit latency (const_mem_latency_cycles profile field)
* Measured with a dependent load chain over a working set in constant memory.
* Store result in hardware/runs/const_mem_latency_cycles.<gpu>.run.json
* Build a cubin and inspect SASS with `cuobjdump --dump-sass` and verify that the timed loop contains the expected constant-load instruction (such as `LDC`) and that each load depends on the previous result.
Build:
* A100: `nvcc -O3 -std=c++17 -arch=sm_80 constant_memory_load.cu -o constant_memory_load`
* H100: `nvcc -O3 -std=c++17 -arch=sm_90 constant_memory_load.cu -o constant_memory_load`
* B200: `nvcc -O3 -std=c++17 -arch=sm_100 constant_memory_load.cu -o constant_memory_load`

## Kernel Launch Overhead
* Measures the overhead of launching an empty CUDA kernel across different grid and block configurations. The benchmark uses NVIDIA Nsight Compute (ncu) to collect GPU cycle metrics and fits a linear model.
The model reports:
* launch_base_cycles: Estimated baseline launch overhead, in GPU cycles.
* launch_per_thread_cycles: Estimated additional overhead per configured thread, in cycles/thread.

Change the file paths and architecture arguments below as necessary.
To build:
* A100: nvcc -O3 -std=c++17 -arch=sm_80 launch_overhead_bench.cu -o launch_overhead_bench
* H100: nvcc -O3 -std=c++17 -arch=sm_90 launch_overhead_bench.cu -o launch_overhead_bench
* B200: nvcc -O3 -std=c++17 -arch=sm_100 launch_overhead_bench.cu -o launch_overhead_bench
* Then run the script from any directory, e.g. from the repository root: `python3 hardware/microbenchmarks/run_launch_overhead.py --binary hardware/microbenchmarks/launch_overhead_bench`. Results always go to `hardware/runs/` and raw ncu output to `launch_overhead_ncu_raw/` at the repository root, unless overridden with `--output-dir` / `--raw-dir`.

The script:
1. Identifies the GPU and records its hardware metadata.
2. Profiles eight grid/block configurations using Nsight Compute.
3. Measures cold and warm kernel launches over multiple repetitions.
4. Fits a linear model to the warm-launch measurements.
5. Writes one JSON report per fitted parameter.

The configurations used are:
(1, 1)
(1, 32)
(1, 256)
(1, 1024)
(32, 256)
(128, 256)
(256, 256)
(512, 256)

## Sync cost (sync_factor)
* Measures GPUPerf's sync factor (gamma, dimensionless; profile field `sync_factor`): the share of DRAM latency a `bar.sync` exposes per warp.
* Runs each configuration (loads and FFMAs per iteration, warps per block, blocks per SM) twice: with a `bar.sync` per loop iteration and without one. One wave, per-SM `clock64` span. gamma = F_sync / (dram_latency x mem_ratio), with F_sync = (T_bar - T_no_bar) / (iters x warps_per_SM). The result is the median over configurations that are co-resident and not bandwidth-bound.
* Needs the DRAM latency and DRAM bandwidth: pass `--dram-latency CYCLES --dram-bandwidth BYTES_PER_CYCLE`, or it reads `dram_latency_cycles.<gpu>.run.json` and `dram_bandwidth_bytes_per_cycle.<gpu>.run.json` from `--out-dir`.
* Records the SM clock (clock64 / globaltimer) in the run JSON's `clock` block, so `run_all.py` can check it ran at base clock.
Build:
* `nvcc -O3 -std=c++17 -arch=sm_XX sync_cost.cu -o sync_cost` (sm_XX from the GPU's compute capability)
Run through `make microbenchmarks` (runs under ncu at base clock, after dram_latency and dram_bandwidth):
* All three: `make microbenchmarks GPU=<gpu> VERSION=<label> ARGS='--only dram_latency,dram_bandwidth,sync_cost'`
* Only sync_cost, reusing DRAM results already in that version directory: `make microbenchmarks GPU=<gpu> VERSION=<label> ARGS='--only sync_cost --force'`
