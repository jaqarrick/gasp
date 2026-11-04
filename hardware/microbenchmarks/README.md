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
* ./shared_memory_load --reps 200 --gpu titan_v
