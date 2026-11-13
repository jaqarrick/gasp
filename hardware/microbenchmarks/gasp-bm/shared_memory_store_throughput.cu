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

constexpr int WARP_THREADS = 32;
constexpr int STORE_SLOTS = 16;
constexpr int STORE_INSTRUCTIONS = 1024;
constexpr int UNROLL = 16;
constexpr int DEFAULT_REPS = 200;

static_assert(STORE_INSTRUCTIONS % UNROLL == 0,
              "STORE_INSTRUCTIONS must be a multiple of UNROLL");

static const char* PARAMETER = "shared_mem_store_throughput_cycles";
static const char* SCHEMA_VERSION = "1.1.0";
static const char* BENCH_VERSION = "1.1.0";

__device__ __forceinline__ void shared_memory_fence() {
    asm volatile("membar.cta;" ::: "memory");
}

#define STORE_AT(slot)                                                         \
    store_shared_volatile(                                                     \
        shared_base +                                                          \
            static_cast<uint32_t>(                                              \
                ((slot) * WARP_THREADS + lane) * sizeof(uint32_t)),            \
        static_cast<uint32_t>(lane + (slot)))

#define STORE_16_SLOTS                                                        \
    STORE_AT(0); STORE_AT(1); STORE_AT(2); STORE_AT(3);                       \
    STORE_AT(4); STORE_AT(5); STORE_AT(6); STORE_AT(7);                       \
    STORE_AT(8); STORE_AT(9); STORE_AT(10); STORE_AT(11);                     \
    STORE_AT(12); STORE_AT(13); STORE_AT(14); STORE_AT(15)

// clock_out[0..1] = lane 0's clock64 cycles and globaltimer ns around the timed region
__global__ void shared_store_kernel(unsigned long long* out, unsigned long long* clock_out) {
    __shared__ uint32_t shared_words[STORE_SLOTS * WARP_THREADS];

    const int lane = threadIdx.x;
    const uint32_t shared_base =
        static_cast<uint32_t>(__cvta_generic_to_shared(shared_words));

    // Warm up the shared-memory path before collecting the baseline.
#pragma unroll 1
    for (int i = 0; i < 2 * STORE_SLOTS; ++i) {
        const int slot = i % STORE_SLOTS;
        const uint32_t address =
            shared_base +
            static_cast<uint32_t>(
                (slot * WARP_THREADS + lane) * sizeof(uint32_t));
        store_shared_volatile(
            address, static_cast<uint32_t>(lane + slot));
    }
    shared_memory_fence();
    __syncthreads();

    // Baseline includes the clock reads and a fence, but no timed store batch.
    const uint64_t baseline_start = read_clock64();
    shared_memory_fence();
    const uint64_t baseline_stop = read_clock64();

    const uint64_t g0 = read_globaltimer();
    const uint64_t start = read_clock64();
#pragma unroll 1
    for (int i = 0; i < STORE_INSTRUCTIONS; i += UNROLL) {
        STORE_16_SLOTS;
    }
    shared_memory_fence();
    const uint64_t stop = read_clock64();
    const uint64_t g1 = read_globaltimer();
    if (lane == 0) {
        clock_out[0] = stop - start;
        clock_out[1] = g1 - g0;
    }

    // Store one timing sample per lane for host-side aggregation.
    out[lane] = stop - start;
    out[WARP_THREADS + lane] = baseline_stop - baseline_start;
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
        &device_output, 2 * WARP_THREADS * sizeof(unsigned long long)));

    std::vector<double> raw_cycles;
    std::vector<double> fence_baseline_samples;
    raw_cycles.reserve(reps);
    fence_baseline_samples.reserve(reps);

    for (int rep = 0; rep < reps; ++rep) {
        shared_store_kernel<<<1, WARP_THREADS>>>(device_output, device_clock);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long host_clock[2] = {0, 0};
        CUDA_CHECK(cudaMemcpy(host_clock, device_clock, sizeof(host_clock),
                              cudaMemcpyDeviceToHost));
        if (host_clock[1] > 0) {
            sum_clock_cycles += static_cast<double>(host_clock[0]);
            sum_clock_ns += static_cast<double>(host_clock[1]);
        }

        std::array<unsigned long long, 2 * WARP_THREADS> host_output{};
        CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                              sizeof(host_output), cudaMemcpyDeviceToHost));

        std::vector<double> lane_cycles(WARP_THREADS);
        std::vector<double> lane_baselines(WARP_THREADS);
        for (int lane = 0; lane < WARP_THREADS; ++lane) {
            lane_cycles[lane] = static_cast<double>(host_output[lane]);
            lane_baselines[lane] =
                static_cast<double>(host_output[WARP_THREADS + lane]);
        }

        raw_cycles.push_back(median_of(lane_cycles));
        fence_baseline_samples.push_back(median_of(lane_baselines));
    }

    const double fence_baseline = median_of(fence_baseline_samples);
    std::vector<double> per_store_instruction_cycles;
    per_store_instruction_cycles.reserve(reps);

    for (double cycles : raw_cycles) {
        per_store_instruction_cycles.push_back(
            (cycles - fence_baseline) / STORE_INSTRUCTIONS);
    }

    const Stats stats = compute_stats(per_store_instruction_cycles);
    const double value = stats.median;

    int runtime_version = 0;
    int driver_api_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_api_version));

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
         << "  \"benchmark\": \"shared_memory_store_latency.cu\",\n"
         << "  \"benchmark_version\": \"" << BENCH_VERSION << "\",\n"
         << "  \"parameter\": \"" << PARAMETER << "\",\n"
         << "  \"value\": " << value << ",\n"
         << "  \"unit\": \"cycles_per_warp_store_instruction\",\n"
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
         << "  \"raw_cycles\": " << json_array(raw_cycles) << ",\n"
         << "  \"fence_baseline_samples_cycles\": " << json_array(fence_baseline_samples) << ",\n"
         << "  \"fence_baseline_cycles\": " << fence_baseline << ",\n"
         << "  \"per_store_instruction_cycles\": " << json_array(per_store_instruction_cycles) << ",\n"
         // clock.sm_clock_ghz: total clock64 cycles / total globaltimer ns over all reps
         << "  \"clock\": {\"sm_clock_ghz\": "
         << (sum_clock_ns > 0.0 ? sum_clock_cycles / sum_clock_ns : 0.0) << "},\n"
         << "  \"measurement\": {\n"
         << "    \"method\": \"back-to-back conflict-free warp shared stores with CTA fence\",\n"
         << "    \"store_instruction\": \"st.volatile.shared.u32\",\n"
         << "    \"timer\": \"clock64 (cycles), globaltimer (SM clock)\",\n"
         << "    \"store_instructions_per_rep\": " << STORE_INSTRUCTIONS << ",\n"
         << "    \"store_slots_per_lane\": " << STORE_SLOTS << ",\n"
         << "    \"threads_per_block\": " << WARP_THREADS << ",\n"
         << "    \"reps\": " << reps << "\n"
         << "  },\n"
         << "  \"cuda_version\": \"" << cuda_version_string(runtime_version)
         << "\",\n"
         << "  \"driver_api_version\": \""
         << cuda_version_string(driver_api_version) << "\",\n"
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
