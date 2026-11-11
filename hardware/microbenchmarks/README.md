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

## L1 throughput
* Measures `l1_throughput_sectors_per_cycle_per_sm`: the plateau rate of global loads that hit L1, in 32-byte sectors per SM cycle. Feeds the `l1-floor` model option. Design: `.local/l1_throughput.md`.
* One block per SM reads its own slice of a global array (the largest power of two up to half of L1, `--l1-bytes`, default 64 KB). A warm pass loads the slice, then each warp issues 4096 independent coalesced loads, 8 in flight. The span per SM comes from `clock64`.
* Sweeps warps per SM (1 to 32) and the load shape: 32-bit with 4 sectors per request, 32-bit with 2 (lanes 16-31 repeat lanes 0-15, as in basic matmul), 64-bit (8 sectors), 128-bit (16 sectors), plus 32-bit with 2 global stores per 8 loads. If requests per cycle stay flat across widths, requests are the limit; if sectors stay flat, sectors are. The mixed variant shows whether stores share the load queue.
* The profile value is the plateau of the 32-bit 4-sector loads, accepted when the two highest warp counts agree within 2% (`plateau_ok`). Every configuration's request and sector rates are in the run JSON.
* `run_all.py` passes `--l1-bytes` from the profile spec (the unified array minus the smallest carveout) and adds the value to the profile.
* Check: the timed loop has only `LDG` of the expected width (and `STG` in the mixed variant) plus address arithmetic; ncu shows an L1 hit rate near 100% and L2 and DRAM throughput near zero.
Build and run:
* `nvcc -O3 -std=c++17 -arch=sm_75 l1_throughput.cu -o l1_throughput`
* `ncu --clock-control base --cache-control none ./l1_throughput --gpu rtx2080ti --out-dir hardware/runs/rtx2080ti/l1`
* or `python hardware/microbenchmarks/run_all.py --gpu rtx2080ti --version 1.1 --only l1_throughput`
