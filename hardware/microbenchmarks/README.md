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
Change the file paths and architecture arguments below as necessary.
To build:
* A100: nvcc -O3 -std=c++17 -arch=sm_80 launch_overhead_bench.cu -o launch_overhead_bench
* H100: nvcc -O3 -std=c++17 -arch=sm_90 launch_overhead_bench.cu -o launch_overhead_bench
* B200: nvcc -O3 -std=c++17 -arch=sm_100 launch_overhead_bench.cu -o launch_overhead_bench
* Then run the script ./run_launch_overhead.py. This runs 4 configurations on the GPU and collects metrics then makes a linear model. Results are printed to a JSON file in hardware/runs/.
