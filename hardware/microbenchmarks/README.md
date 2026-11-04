# Microbenchmarks
This are targeted CUDA programs for specific operation measurements

## Shared-memory store latency microbenchmark
* Measures conflict-free warp issuing back-to-back shared-memory stores followed by CTA memory fence.
* A memory fence is a structure that tells the GPU to not let later operations pass earlier shared-memory stores from this thread block. This is to ensure the timer includes store operations completing.

Build:
* A100: nvcc -O3 -std=c++17 -arch=sm_80 shared_memory_store_latency.cu -o shared_memory_store_latency
* H100: nvcc -O3 -std=c++17 -arch=sm_90 shared_memory_store_latency.cu -o shared_memory_store_latency
* B200: nvcc -O3 -std=c++17 -arch=sm_100 shared_memory_store_latency.cu -o shared_memory_store_latency
Run: ./shared_memory_store_latency --reps 200 --gpu <gpu>
