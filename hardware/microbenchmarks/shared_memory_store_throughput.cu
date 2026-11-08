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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

constexpr int WARP_THREADS = 32;
constexpr int STORE_SLOTS = 16;
constexpr int STORE_INSTRUCTIONS = 1024;
constexpr int UNROLL = 16;
constexpr int DEFAULT_REPS = 200;

static_assert(STORE_INSTRUCTIONS % UNROLL == 0,
              "STORE_INSTRUCTIONS must be a multiple of UNROLL");

static const char* PARAMETER = "shared_mem_store_throughput_cycles";
static const char* SCHEMA_VERSION = "1.0.0";
static const char* BENCH_VERSION = "1.0.0";

__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t cycles;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycles) :: "memory");
    return cycles;
}

__device__ __forceinline__ void store_shared_volatile(
    uint32_t address, uint32_t value) {
    asm volatile(
        "st.volatile.shared.u32 [%0], %1;"
        :
        : "r"(address), "r"(value)
        : "memory");
}

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

__global__ void shared_store_kernel(unsigned long long* out) {
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

    const uint64_t start = read_clock64();
#pragma unroll 1
    for (int i = 0; i < STORE_INSTRUCTIONS; i += UNROLL) {
        STORE_16_SLOTS;
    }
    shared_memory_fence();
    const uint64_t stop = read_clock64();

    // Store one timing sample per lane for host-side aggregation.
    out[lane] = stop - start;
    out[WARP_THREADS + lane] = baseline_stop - baseline_start;
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t count = values.size();
    if (count % 2 == 0) {
        return 0.5 * (values[count / 2 - 1] + values[count / 2]);
    }
    return values[count / 2];
}

static std::string json_escape(const std::string& value) {
    std::string escaped;
    for (char c : value) {
        if (c == '"' || c == '\\') {
            escaped += '\\';
            escaped += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            escaped += ' ';
        } else {
            escaped += c;
        }
    }
    return escaped;
}

static std::string short_gpu_name(const std::string& full_name) {
    std::string lower;
    for (char c : full_name) {
        lower += static_cast<char>(
            std::tolower(static_cast<unsigned char>(c)));
    }

    for (const char* name : {"a100", "h100", "b200", "titan v"}) {
        if (lower.find(name) != std::string::npos) {
            std::string short_name = name;
            std::replace(short_name.begin(), short_name.end(), ' ', '_');
            return short_name;
        }
    }

    std::string result;
    for (char c : lower) {
        result += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
    }
    return result;
}

static std::string architecture_name(int major) {
    switch (major) {
        case 7: return "volta";
        case 8: return "ampere";
        case 9: return "hopper";
        case 10: return "blackwell";
        default: return "unknown";
    }
}

static std::string cuda_version_string(int version) {
    return std::to_string(version / 1000) + "." +
           std::to_string((version % 1000) / 10);
}

static std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}

static std::string driver_version() {
    FILE* pipe = popen(
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null",
        "r");
    if (pipe == nullptr) {
        return "unknown";
    }

    char buffer[128];
    std::string result;
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result = buffer;
        while (!result.empty() &&
               std::isspace(static_cast<unsigned char>(result.back()))) {
            result.pop_back();
        }
    }
    pclose(pipe);
    return result.empty() ? "unknown" : result;
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
    const std::string architecture = architecture_name(prop.major);

    unsigned long long* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        &device_output, 2 * WARP_THREADS * sizeof(unsigned long long)));

    std::vector<double> raw_cycles;
    std::vector<double> fence_baseline_samples;
    raw_cycles.reserve(reps);
    fence_baseline_samples.reserve(reps);

    for (int rep = 0; rep < reps; ++rep) {
        shared_store_kernel<<<1, WARP_THREADS>>>(device_output);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

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

        raw_cycles.push_back(median(lane_cycles));
        fence_baseline_samples.push_back(median(lane_baselines));
    }

    const double fence_baseline = median(fence_baseline_samples);
    std::vector<double> per_store_instruction_cycles;
    per_store_instruction_cycles.reserve(reps);

    for (double cycles : raw_cycles) {
        per_store_instruction_cycles.push_back(
            (cycles - fence_baseline) / STORE_INSTRUCTIONS);
    }

    const double value = median(per_store_instruction_cycles);
    const double mean =
        std::accumulate(per_store_instruction_cycles.begin(),
                        per_store_instruction_cycles.end(), 0.0) / reps;

    double variance = 0.0;
    for (double sample : per_store_instruction_cycles) {
        variance += (sample - mean) * (sample - mean);
    }
    const double stddev = std::sqrt(variance / (reps - 1));
    const double min_value =
        *std::min_element(per_store_instruction_cycles.begin(),
                          per_store_instruction_cycles.end());
    const double max_value =
        *std::max_element(per_store_instruction_cycles.begin(),
                          per_store_instruction_cycles.end());

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
         << "  \"stats\": {\n"
         << "    \"median\": " << value << ",\n"
         << "    \"mean\": " << mean << ",\n"
         << "    \"std\": " << stddev << ",\n"
         << "    \"min\": " << min_value << ",\n"
         << "    \"max\": " << max_value << "\n"
         << "  },\n"
         << "  \"raw_cycles\": [";

    for (size_t i = 0; i < raw_cycles.size(); ++i) {
        json << (i == 0 ? "" : ", ") << raw_cycles[i];
    }

    json << "],\n"
         << "  \"fence_baseline_samples_cycles\": [";

    for (size_t i = 0; i < fence_baseline_samples.size(); ++i) {
        json << (i == 0 ? "" : ", ") << fence_baseline_samples[i];
    }

    json << "],\n"
         << "  \"fence_baseline_cycles\": " << fence_baseline << ",\n"
         << "  \"per_store_instruction_cycles\": [";

    for (size_t i = 0; i < per_store_instruction_cycles.size(); ++i) {
        json << (i == 0 ? "" : ", ") << per_store_instruction_cycles[i];
    }

    json << "],\n"
         << "  \"measurement\": {\n"
         << "    \"method\": \"back-to-back conflict-free warp shared stores with CTA fence\",\n"
         << "    \"store_instruction\": \"st.volatile.shared.u32\",\n"
         << "    \"timer\": \"clock64\",\n"
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

    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) {
        fprintf(stderr, "Cannot create output directory: %s\n",
                ec.message().c_str());
        CUDA_CHECK(cudaFree(device_output));
        return 1;
    }

    const fs::path output_path =
        fs::path(out_dir) /
        (std::string(PARAMETER) + "." + gpu + ".run.json");

    FILE* file = std::fopen(output_path.string().c_str(), "w");
    if (file == nullptr) {
        perror("Could not open results JSON");
        CUDA_CHECK(cudaFree(device_output));
        return 1;
    }

    const std::string result = json.str();
    bool write_ok = std::fputs(result.c_str(), file) >= 0;
    if (std::fclose(file) != 0) {
        write_ok = false;
    }

    if (!write_ok) {
        fprintf(stderr, "Failed writing results to %s\n",
                output_path.string().c_str());
        CUDA_CHECK(cudaFree(device_output));
        return 1;
    }

    printf("%s", result.c_str());
    fprintf(stderr, "Saved results to %s\n", output_path.string().c_str());

    CUDA_CHECK(cudaFree(device_output));
    return 0;
}
