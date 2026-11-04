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

constexpr int NUM_NODES = 64;
constexpr int WARP_THREADS = 32;
constexpr int CHASE_LOADS = 1024;
constexpr int UNROLL = 16;
constexpr int DEFAULT_REPS = 200;

static_assert(CHASE_LOADS % UNROLL == 0,
              "CHASE_LOADS must be a multiple of UNROLL");

static const char* PARAMETER = "shared_mem_load_latency_cycles";
static const char* SCHEMA_VERSION = "1.0.0";
static const char* BENCH_VERSION = "1.0.0";

__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t cycles;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycles) :: "memory");
    return cycles;
}

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

__global__ void shared_chase_kernel(unsigned long long* out) {
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

    out[lane] = stop - start;
    out[WARP_THREADS + lane] = baseline_stop - baseline_start;
    out[2 * WARP_THREADS + lane] = p;
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
        &device_output, 3 * WARP_THREADS * sizeof(unsigned long long)));

    std::vector<double> raw_cycles;
    std::vector<double> clock_overhead_samples;
    raw_cycles.reserve(reps);
    clock_overhead_samples.reserve(reps);

    for (int rep = 0; rep < reps; ++rep) {
        shared_chase_kernel<<<1, WARP_THREADS>>>(device_output);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

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

        raw_cycles.push_back(median(lane_cycles));
        clock_overhead_samples.push_back(median(lane_overheads));
    }

    const double clock_overhead = median(clock_overhead_samples);
    std::vector<double> per_load_cycles;
    per_load_cycles.reserve(reps);

    for (double cycles : raw_cycles) {
        per_load_cycles.push_back(
            (cycles - clock_overhead) / CHASE_LOADS);
    }

    const double value = median(per_load_cycles);
    const double mean =
        std::accumulate(per_load_cycles.begin(), per_load_cycles.end(), 0.0)
        / reps;

    double variance = 0.0;
    for (double sample : per_load_cycles) {
        variance += (sample - mean) * (sample - mean);
    }
    const double stddev = std::sqrt(variance / (reps - 1));
    const double min_value =
        *std::min_element(per_load_cycles.begin(), per_load_cycles.end());
    const double max_value =
        *std::max_element(per_load_cycles.begin(), per_load_cycles.end());

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
         << "  \"clock_overhead_samples_cycles\": [";

    for (size_t i = 0; i < clock_overhead_samples.size(); ++i) {
        json << (i == 0 ? "" : ", ") << clock_overhead_samples[i];
    }

    json << "],\n"
         << "  \"clock_overhead_cycles\": " << clock_overhead << ",\n"
         << "  \"per_load_cycles\": [";

    for (size_t i = 0; i < per_load_cycles.size(); ++i) {
        json << (i == 0 ? "" : ", ") << per_load_cycles[i];
    }

    json << "],\n"
         << "  \"measurement\": {\n"
         << "    \"method\": \"dependent conflict-free warp shared-memory load chain\",\n"
         << "    \"load_instruction\": \"ld.shared.u32\",\n"
         << "    \"timer\": \"clock64\",\n"
         << "    \"nodes_per_lane\": " << NUM_NODES << ",\n"
         << "    \"loads_per_rep\": " << CHASE_LOADS << ",\n"
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
