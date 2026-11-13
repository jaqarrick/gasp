#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "common.h"

constexpr int NUM_NODES = 64;
constexpr int WARP_THREADS = 32;
constexpr int CHASE_LOADS = 1024;
constexpr int UNROLL = 16;
constexpr int DEFAULT_REPS = 200;

static_assert(CHASE_LOADS % UNROLL == 0,
              "CHASE_LOADS must be a multiple of UNROLL");

static const char* PARAMETER = "shared_mem_load_latency_cycles";
static const char* SCHEMA_VERSION = "1.1.0";
static const char* BENCH_VERSION = "1.1.0";

__device__ __forceinline__ uint32_t load_shared_address(uint32_t address) {
    uint32_t next;
    asm volatile(
        "ld.shared.u32 %0, [%1];"
        : "=r"(next)
        : "r"(address)
        : "memory");
    return next;
}

#define CHASE_ONE p = load_shared_address(p)

// clock_out[0..1] = lane 0's clock64 cycles and globaltimer ns around the timed region
__global__ void shared_chase_kernel(unsigned long long* out, unsigned long long* clock_out) {
    __shared__ uint32_t ring[NUM_NODES * WARP_THREADS];

    const int lane = threadIdx.x;
    const uint32_t shared_base =
        static_cast<uint32_t>(__cvta_generic_to_shared(ring));

    // interleave node slots by lane: slot (node, lane) is node*32 + lane, so the 32 lanes address 32 different banks.
    for (int node = 0; node < NUM_NODES; ++node) {
        const int next_node = (node + 1) % NUM_NODES;
        const int slot = node * WARP_THREADS + lane;
        const int next_slot = next_node * WARP_THREADS + lane;
        ring[slot] = shared_base +
                     static_cast<uint32_t>(next_slot * sizeof(uint32_t));
    }
    __syncthreads();

    uint32_t p = shared_base +
                 static_cast<uint32_t>(lane * sizeof(uint32_t));

    // Warm up the dependent chains before timing
#pragma unroll 1
    for (int i = 0; i < 2 * NUM_NODES; ++i) {
        CHASE_ONE;
    }

    __syncthreads();

    const uint64_t baseline_start = read_clock64();
    const uint64_t baseline_stop = read_clock64();

    const uint64_t g0 = read_globaltimer();
    const uint64_t start = read_clock64();
#pragma unroll 1
    for (int i = 0; i < CHASE_LOADS; i += UNROLL) {
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
    }
    const uint64_t stop = read_clock64();
    const uint64_t g1 = read_globaltimer();
    if (lane == 0) {
        clock_out[0] = stop - start;
        clock_out[1] = g1 - g0;
    }

    out[lane] = stop - start;
    out[WARP_THREADS + lane] = baseline_stop - baseline_start;
    out[2 * WARP_THREADS + lane] = p;
}

int main(int argc, char** argv) {
    int reps = DEFAULT_REPS;
    std::string gpu_override;
    std::string out_dir = "hardware/runs";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--reps" && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (arg == "--gpu" && i + 1 < argc) {
            gpu_override = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
        } else {
            fprintf(stderr,
                    "Usage: %s [--reps N] [--gpu NAME] [--out-dir DIR]\n",
                    argv[0]);
            return 1;
        }
    }

    if (reps < 100) {
        fprintf(stderr, "--reps must be at least 100\n");
        return 1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const std::string gpu_name = prop.name;
    const std::string gpu = gpu_override.empty()
        ? short_gpu_name(gpu_name)
        : gpu_override;
    const std::string arch =
        "sm_" + std::to_string(prop.major * 10 + prop.minor);
    const std::string architecture = architecture_name(prop.major, prop.minor);

    unsigned long long* device_output = nullptr;
    unsigned long long* device_clock = nullptr;  // lane 0's cycles and ns, for the SM clock
    CUDA_CHECK(cudaMalloc(&device_clock, 2 * sizeof(unsigned long long)));
    double sum_clock_cycles = 0.0, sum_clock_ns = 0.0;
    CUDA_CHECK(cudaMalloc(
        &device_output, 3 * WARP_THREADS * sizeof(unsigned long long)));

    std::vector<double> raw_cycles;
    std::vector<double> clock_overhead_samples;
    raw_cycles.reserve(reps);
    clock_overhead_samples.reserve(reps);

    for (int rep = 0; rep < reps; ++rep) {
        shared_chase_kernel<<<1, WARP_THREADS>>>(device_output, device_clock);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long host_clock[2] = {0, 0};
        CUDA_CHECK(cudaMemcpy(host_clock, device_clock, sizeof(host_clock),
                              cudaMemcpyDeviceToHost));
        if (host_clock[1] > 0) {
            sum_clock_cycles += static_cast<double>(host_clock[0]);
            sum_clock_ns += static_cast<double>(host_clock[1]);
        }

        std::array<unsigned long long, 3 * WARP_THREADS> host_output{};
        CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                              sizeof(host_output), cudaMemcpyDeviceToHost));

        std::vector<double> lane_cycles(WARP_THREADS);
        std::vector<double> lane_overheads(WARP_THREADS);
        for (int lane = 0; lane < WARP_THREADS; ++lane) {
            lane_cycles[lane] = static_cast<double>(host_output[lane]);
            lane_overheads[lane] =
                static_cast<double>(host_output[WARP_THREADS + lane]);
        }

        raw_cycles.push_back(median_of(lane_cycles));
        clock_overhead_samples.push_back(median_of(lane_overheads));
    }

    const double clock_overhead = median_of(clock_overhead_samples);
    std::vector<double> per_load_cycles;
    per_load_cycles.reserve(reps);

    for (double cycles : raw_cycles) {
        per_load_cycles.push_back(
            (cycles - clock_overhead) / CHASE_LOADS);
    }

    const Stats stats = compute_stats(per_load_cycles);
    const double value = stats.median;

    int runtime_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

    int shared_mem_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &shared_mem_per_sm,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor,
        device));

    const int max_warps_per_sm =
        prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::ostringstream json;
    json.setf(std::ios::fixed);
    json.precision(4);

    json << "{\n"
         << "  \"schema_version\": \"" << SCHEMA_VERSION << "\",\n"
         << "  \"benchmark\": \"shared_memory_load.cu\",\n"
         << "  \"benchmark_version\": \"" << BENCH_VERSION << "\",\n"
         << "  \"parameter\": \"" << PARAMETER << "\",\n"
         << "  \"value\": " << value << ",\n"
         << "  \"unit\": \"cycles\",\n"
         << "  \"gpu\": \"" << json_escape(gpu) << "\",\n"
         << "  \"architecture\": \"" << architecture << "\",\n"
         << "  \"arch\": \"" << arch << "\",\n"
         << "  \"device\": {\n"
         << "    \"name\": \"" << json_escape(gpu_name) << "\",\n"
         << "    \"sms\": " << prop.multiProcessorCount << ",\n"
         << "    \"max_warps_per_sm\": " << max_warps_per_sm << ",\n"
         << "    \"l2_cache_bytes\": " << prop.l2CacheSize << ",\n"
         << "    \"shared_mem_per_sm_bytes\": " << shared_mem_per_sm << "\n"
         << "  },\n"
         << json_stats(stats)
         // raw_cycles: clock64 delta per rep (median over the 32 lanes), overhead NOT subtracted.
         << "  \"raw_cycles\": " << json_array(raw_cycles) << ",\n"
         // clock.sm_clock_ghz: total clock64 cycles / total globaltimer ns over all reps
         << "  \"clock\": {\"sm_clock_ghz\": "
         << (sum_clock_ns > 0.0 ? sum_clock_cycles / sum_clock_ns : 0.0) << "},\n"
         << "  \"measurement\": {\n"
         << "    \"method\": \"dependent conflict-free warp shared-memory load chain\",\n"
         << "    \"load_instruction\": \"ld.shared.u32\",\n"
         << "    \"timer\": \"clock64 (cycles), globaltimer (SM clock)\",\n"
         << "    \"clock_overhead_cycles\": " << clock_overhead << ",\n"
         << "    \"value_formula\": \"per_load = (raw_cycles[i] - clock_overhead_cycles) / loads_per_rep; value = median(per_load)\",\n"
         << "    \"nodes_per_lane\": " << NUM_NODES << ",\n"
         << "    \"loads_per_rep\": " << CHASE_LOADS << ",\n"
         << "    \"threads_per_block\": " << WARP_THREADS << ",\n"
         << "    \"reps\": " << reps << "\n"
         << "  },\n"
         << "  \"cuda_version\": \"" << cuda_version_string(runtime_version)
         << "\",\n"
         << "  \"driver_version\": \"" << json_escape(driver_version())
         << "\",\n"
         << "  \"timestamp\": \"" << utc_timestamp() << "\"\n"
         << "}\n";

    const std::string result = json.str();
    const std::string output_path = write_run_json(out_dir, PARAMETER, gpu, result);
    if (output_path.empty()) return 1;

    printf("%s", result.c_str());
    fprintf(stderr, "Saved results to %s\n", output_path.c_str());

    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_clock));
    return 0;
}
