#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// References https://github.com/accel-sim/gpu-app-collection/blob/dev/src/cuda/GPU_Microbenchmark/ubench/system/kernel_lat/kernel_lat.cu

// Measures effective GPU-timeline cycles per minimal kernel launch.
// Does not measure CPU-side launch-call times: the Python driver
// (run_launch_overhead.py) reads sm__cycles_elapsed / sm__cycles_active from
// ncu for every launch this program makes.
//
// Usage:
//   launch_overhead_bench GRID BLOCK [WARMUP]
//       Launches the empty kernel WARMUP times (default 1), then once more.
//       ncu profiles every launch: the first one is the cold launch and the
//       last one is the warm launch.
//   launch_overhead_bench --info
//       Prints the GPU_INFO line and exits without launching anything.
//
// Every run prints one line "GPU_INFO {json}" to stdout so the driver script
// can check that all runs used the same GPU.

// we use ncu as an external profiler to measure the hardware cycles elapsed during the empty kernel.

#define BENCH_VERSION 2

// minimum empty kernel
__global__ void launch_empty_kernel() {
}

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s GRID BLOCK [WARMUP]\n       %s --info\n", prog, prog);
}

// Parses a base-10 integer in [min_value, INT_MAX]. Returns false on any error.
static bool parse_int(const char* text, int min_value, int* out) {
    char* end = nullptr;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < min_value || value > INT_MAX) {
        return false;
    }
    *out = (int)value;
    return true;
}

static bool cuda_ok(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return true;
    fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    return false;
}

static void print_json_string(const char* s) {
    putchar('"');
    for (; *s; ++s) {
        if (*s == '"' || *s == '\\') putchar('\\');
        putchar(*s);
    }
    putchar('"');
}

// CUDA encodes versions as major*1000 + minor*10 (e.g. 12040 -> 12.4).
static void print_gpu_info(int device, const cudaDeviceProp& prop) {
    int runtime_version = 0;
    int driver_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    const unsigned char* u = (const unsigned char*)prop.uuid.bytes;

    printf("GPU_INFO {\"bench_version\":%d,\"device\":%d,\"name\":", BENCH_VERSION, device);
    print_json_string(prop.name);
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
    if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice") ||
        !cuda_ok(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties")) {
        return 1;
    }
    print_gpu_info(device, prop);

    if (argc == 2 && strcmp(argv[1], "--info") == 0) {
        return 0;
    }
    if (argc != 3 && argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    int grid = 0;
    int block = 0;
    int warmup = 1;
    if (!parse_int(argv[1], 1, &grid) || !parse_int(argv[2], 1, &block) ||
        (argc == 4 && !parse_int(argv[3], 0, &warmup))) {
        print_usage(argv[0]);
        return 1;
    }

    // validate grid + block against the device limits
    if (block > prop.maxThreadsPerBlock) {
        fprintf(stderr, "BLOCK %d exceeds this GPU's limit of %d threads per block\n",
                block, prop.maxThreadsPerBlock);
        return 1;
    }
    if (grid > prop.maxGridSize[0]) {
        fprintf(stderr, "GRID %d exceeds this GPU's limit of %d blocks\n",
                grid, prop.maxGridSize[0]);
        return 1;
    }

    // Warmup launches (same configuration as the measured launch)
    for (int i = 0; i < warmup; ++i) {
        launch_empty_kernel<<<grid, block>>>();
    }
    if (!cuda_ok(cudaGetLastError(), "Warmup kernel launch") ||
        !cuda_ok(cudaDeviceSynchronize(), "Warmup kernel execution")) {
        return 1;
    }

    // Measured launch
    launch_empty_kernel<<<grid, block>>>();
    if (!cuda_ok(cudaGetLastError(), "Kernel launch") ||
        !cuda_ok(cudaDeviceSynchronize(), "Kernel execution")) {
        return 1;
    }
    return 0;
}
