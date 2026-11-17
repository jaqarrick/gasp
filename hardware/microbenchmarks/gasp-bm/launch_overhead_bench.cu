// This microbenchmark measures effective GPU-timeline cycles with an empty kernel launch.
// This does not measure CPU-side launch calls.
// The Python driver (run_launch_overhead.py) uses ncu to read sm__cycles_elapsed / sm__cycles_active for every launch.
//
// Usage: ./launch_overhead_bench GRID BLOCK [WARMUP]
//  - Launches the empty kernel WARMUP times (default 1), then once more.

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "common.h"

#define BENCH_VERSION 2

// Minimal empty kernel used to isolate raw hardware launch overhead.
__global__ void launch_empty_kernel() {}

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s GRID BLOCK [WARMUP]\n       %s --info\n", prog, prog);
}

// Helper function prints hardware and driver metadata as a single JSON line for run_launch_overhead.py's output file
static void print_gpu_info(int device, const cudaDeviceProp& prop) {
    int runtime_version = 0;
    int driver_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    const unsigned char* u = (const unsigned char*)prop.uuid.bytes;

    printf("GPU_INFO {\"bench_version\":%d,\"device\":%d,\"name\":\"%s\"",
           BENCH_VERSION, device, json_escape(prop.name).c_str());
    printf(",\"uuid\":\"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\"",
           u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
           u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    printf(",\"compute_capability\":\"%d.%d\",\"num_sms\":%d,\"l2_cache_bytes\":%d",
           prop.major, prop.minor, prop.multiProcessorCount, (int)prop.l2CacheSize);
    printf(",\"max_warps_per_sm\":%d,\"shared_mem_per_sm_bytes\":%zu",
           prop.maxThreadsPerMultiProcessor / 32,
           (size_t)prop.sharedMemPerMultiprocessor);
    printf(",\"cuda_runtime\":\"%d.%d\",\"cuda_driver\":\"%d.%d\",\"nvcc\":\"%d.%d\"}\n",
           runtime_version / 1000, (runtime_version % 1000) / 10,
           driver_version / 1000, (driver_version % 1000) / 10,
           __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
    fflush(stdout);
}

int main(int argc, char** argv) {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    print_gpu_info(device, prop);

    if (argc == 2 && strcmp(argv[1], "--info") == 0) {
        return 0;
    }
    if (argc != 3 && argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    char* end = nullptr;
    errno = 0;
    long grid = strtol(argv[1], &end, 10);
    if (errno != 0 || end == argv[1] || *end != '\0' || grid < 1 || grid > INT_MAX) {
        print_usage(argv[0]);
        return 1;
    }

    errno = 0;
    long block = strtol(argv[2], &end, 10);
    if (errno != 0 || end == argv[2] || *end != '\0' || block < 1 || block > INT_MAX) {
        print_usage(argv[0]);
        return 1;
    }

    long warmup = 1;
    if (argc == 4) {
        errno = 0;
        warmup = strtol(argv[3], &end, 10);
        if (errno != 0 || end == argv[3] || *end != '\0' || warmup < 0 || warmup > INT_MAX) {
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate grid and block against GPU device limits
    if (block > prop.maxThreadsPerBlock) {
        fprintf(stderr, "BLOCK %ld exceeds this GPU's limit of %d threads per block\n",
                block, prop.maxThreadsPerBlock);
        return 1;
    }
    if (grid > prop.maxGridSize[0]) {
        fprintf(stderr, "GRID %ld exceeds this GPU's limit of %d blocks\n",
                grid, prop.maxGridSize[0]);
        return 1;
    }

    // Warm-up launches, profiled by ncu as the cold-to-warm sequence
    for (int i = 0; i < (int)warmup; ++i) {
        launch_empty_kernel<<<(int)grid, (int)block>>>();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Final measured launch
    launch_empty_kernel<<<(int)grid, (int)block>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
