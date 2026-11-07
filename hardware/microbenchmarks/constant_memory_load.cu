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

// ring of addresses stored in constant memory
__constant__ uint64_t c_ring[NUM_NODES];

static const char* PARAMETER = "const_mem_latency_cycles";
static const char* SCHEMA_VERSION = "1.0.0";
static const char* BENCH_VERSION = "1.0.0";

// increment address to next item in ring
#define CHASE_ONE p = load_constant_pointer(p)

__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t cycles;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycles) :: "memory");
    return cycles;
}

__device__ __forceinline__ uint64_t load_constant_pointer(uint64_t address) {
    uint64_t next;
    asm volatile(
        "ld.const.u64 %0, [%1];"
        : "=l"(next)
        : "l"(address)
        : "memory");
    return next;
}

__global__ void constant_chase_kernel(unsigned long long* out) {
    const int lane = threadIdx.x;
    uint64_t p = reinterpret_cast<uint64_t>(c_ring);

// warm up the constant-memory ring before timing
#pragma unroll 1
    for (int i = 0; i < 2 * NUM_NODES; ++i) {
        CHASE_ONE;
    }

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

    // reads the timing and final index of the first lane in warp
    if (lane == 0) {
        out[0] = stop - start;
        out[1] = baseline_stop - baseline_start;
        out[2] = p;
    }
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 == 0
        ? 0.5 * (values[n / 2 - 1] + values[n / 2])
        : values[n / 2];
}

//helper function to make a string safe to insert into JSON
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

// converts gpu compute-capability major number into the broader architecture label for use in JSON
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

// runs nvidia-smi to read installed NVIDIA driver version for results JSON
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
        if (std::string(argv[i]) == "--reps" && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--gpu" && i + 1 < argc) {
            gpu_override = argv[++i];
        } else if (std::string(argv[i]) == "--out-dir" && i + 1 < argc) {
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

    void* ring_address = nullptr;
    CUDA_CHECK(cudaGetSymbolAddress(&ring_address, c_ring));

    std::array<uint64_t, NUM_NODES> host_ring{};
    const uint64_t base_address =
        reinterpret_cast<uint64_t>(ring_address);

    for (int i = 0; i < NUM_NODES; ++i) {
        const int next = (i + 1) % NUM_NODES;
        host_ring[i] = base_address +
                       static_cast<uint64_t>(next) * sizeof(uint64_t);
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_ring, host_ring.data(),
                                  sizeof(host_ring)));

    unsigned long long* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(&device_output, 3 * sizeof(unsigned long long)));

    std::vector<double> raw_cycles;
    std::vector<double> clock_overhead_cycles;
    raw_cycles.reserve(reps);
    clock_overhead_cycles.reserve(reps);

    //run the kernel for X reps
    for (int rep = 0; rep < reps; ++rep) {
        constant_chase_kernel<<<1, WARP_THREADS>>>(device_output);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned long long output[3]{};
        CUDA_CHECK(cudaMemcpy(output, device_output, sizeof(output),
                              cudaMemcpyDeviceToHost));

        raw_cycles.push_back(static_cast<double>(output[0]));
        clock_overhead_cycles.push_back(static_cast<double>(output[1]));
    }
    // statistics collection
    const double clock_overhead = median(clock_overhead_cycles);
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

    // json file processing
    std::ostringstream json;
    json.setf(std::ios::fixed);
    json.precision(4);

    json << "{\n"
         << "  \"schema_version\": \"" << SCHEMA_VERSION << "\",\n"
         << "  \"benchmark\": \"constant_memory_load.cu\",\n"
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

    for (size_t i = 0; i < clock_overhead_cycles.size(); ++i) {
        json << (i == 0 ? "" : ", ") << clock_overhead_cycles[i];
    }

    json << "],\n"
         << "  \"clock_overhead_cycles\": " << clock_overhead << ",\n"
         << "  \"per_load_cycles\": [";

    for (size_t i = 0; i < per_load_cycles.size(); ++i) {
        json << (i == 0 ? "" : ", ") << per_load_cycles[i];
    }

    json << "],\n"
         << "  \"measurement\": {\n"
         << "    \"method\": \"dependent uniform constant-cache-hit pointer chase\",\n"
         << "    \"load_instruction\": \"ld.const.u64\",\n"
         << "    \"timer\": \"clock64\",\n"
         << "    \"nodes\": " << NUM_NODES << ",\n"
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
    const bool write_ok =
        std::fputs(result.c_str(), file) >= 0 && std::fclose(file) == 0;

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
