# Microbenchmarks
This are targeted CUDA programs for specific operation measurements

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
* Not written to the profile yet (no profile field); the run JSON goes to the version directory.
* Check under ncu: zero shared-load bank conflicts, no global traffic in the timed kernel.
Build:
* `nvcc -O3 -std=c++17 -arch=sm_XX shared_memory_load_throughput.cu -o shared_memory_load_throughput` (sm_XX from the GPU's compute capability)
Run through `make microbenchmarks`:
* `make microbenchmarks GPU=<gpu> VERSION=<label> ARGS='--only shared_memory_load_throughput'`

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
