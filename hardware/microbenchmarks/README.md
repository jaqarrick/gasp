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
* Then run the script from repository root: ./run_launch_overhead.py.

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
