// Shared-memory store latency microbenchmark.
//
// Measures shared_mem_store_latency_cycles: the cycles a conflict-free warp
// shared-memory store adds before a dependent instruction can proceed.
//
// A store has no register result, so it cannot be chained on its own. Each
// step of the timed chain stores p into its own slot and then loads it back:
//     st.volatile.shared.u32 [p], p;
//     p = ld.volatile.shared.u32 [p];
// The load must see the store, and the next store's address and data depend
// on the load, so the critical path per step is store + load. The same kernel
// also times a load-only chain over the same slots:
//     p = ld.volatile.shared.u32 [p];
// and the store latency is the difference per step. Both chains share the
// addresses, the unroll and the loop shape, so clock and loop overhead cancel.
//
// The load-only value is cross-checked against the shared_memory_load run
// (<out-dir>/shared_mem_load_latency_cycles.<gpu>.run.json) if it exists.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_XX shared_memory_store.cu -o shared_memory_store
// Run:   ./shared_memory_store [--reps N] [--gpu NAME] [--out-dir DIR]
//        Writes <out-dir>/shared_mem_store_latency_cycles.<gpu>.run.json

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
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
constexpr int CHAIN_STEPS = 1024;
constexpr int UNROLL = 16;
constexpr int PASSES = 2;  // pass 0 warms the instruction cache; last pass is kept
constexpr int WARMUP_STEPS = 64;
constexpr int DEFAULT_REPS = 200;
constexpr double REFERENCE_TOLERANCE_CYCLES = 2.0;

static_assert(CHAIN_STEPS % UNROLL == 0,
              "CHAIN_STEPS must be a multiple of UNROLL");
static_assert(UNROLL == 16, "the unrolled bodies below have 16 steps");

static const char* PARAMETER = "shared_mem_store_latency_cycles";
static const char* LOAD_PARAMETER = "shared_mem_load_latency_cycles";
static const char* SCHEMA_VERSION = "1.1.0";
static const char* BENCH_VERSION = "1.1.0";

__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t cycles;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycles) :: "memory");
    return cycles;
}

// Nanoseconds since an arbitrary epoch, independent of the SM clock. Read next to
// clock64 so the run records the SM clock it ran at (cycles / ns).
__device__ __forceinline__ uint64_t read_globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

__device__ __forceinline__ uint32_t load_shared(uint32_t address) {
    uint32_t value;
    asm volatile(
        "ld.volatile.shared.u32 %0, [%1];"
        : "=r"(value)
        : "r"(address)
        : "memory");
    return value;
}

__device__ __forceinline__ void store_shared(uint32_t address, uint32_t value) {
    asm volatile(
        "st.volatile.shared.u32 [%0], %1;"
        :
        : "r"(address), "r"(value)
        : "memory");
}

// Store p into slot p, then load it back. The slot holds its own address, so
// the load returns p and the chain stays on the same conflict-free slot.
#define STORE_STEP store_shared(p, p); p = load_shared(p)
#define LOAD_STEP p = load_shared(p)

#define REPEAT_16(step)                                                        \
    step; step; step; step; step; step; step; step;                           \
    step; step; step; step; step; step; step; step

// out[0*32 + lane] = cycles for CHAIN_STEPS store+load steps (clock overhead included)
// out[1*32 + lane] = cycles for CHAIN_STEPS load-only steps (clock overhead included)
// out[2*32 + lane] = clock overhead (two back-to-back clock reads)
// out[3*32 + lane] = final p, so every step has an observable result
// clock_out[0..1] = lane 0's clock64 cycles and globaltimer ns around the timed region
__global__ void shared_store_latency_kernel(unsigned long long* out, unsigned long long* clock_out) {
    __shared__ uint32_t slots[WARP_THREADS];

    const int lane = threadIdx.x;
    // Lane l uses slot l: the 32 lanes hit 32 different banks.
    const uint32_t self = static_cast<uint32_t>(
        __cvta_generic_to_shared(&slots[lane]));
    slots[lane] = self;
    __syncthreads();

    uint32_t p = self;

    // Warm up both paths before timing.
#pragma unroll 1
    for (int i = 0; i < WARMUP_STEPS; ++i) {
        STORE_STEP;
        LOAD_STEP;
    }
    __syncthreads();

    uint64_t store_start = 0, store_stop = 0;
    uint64_t g0 = 0, g1 = 0;
    uint64_t load_start = 0, load_stop = 0;

#pragma unroll 1
    for (int pass = 0; pass < PASSES; ++pass) {
        g0 = read_globaltimer();
        store_start = read_clock64();
#pragma unroll 1
        for (int i = 0; i < CHAIN_STEPS; i += UNROLL) {
            REPEAT_16(STORE_STEP);
        }
        store_stop = read_clock64();
        g1 = read_globaltimer();

        load_start = read_clock64();
#pragma unroll 1
        for (int i = 0; i < CHAIN_STEPS; i += UNROLL) {
            REPEAT_16(LOAD_STEP);
        }
        load_stop = read_clock64();
    }

    const uint64_t baseline_start = read_clock64();
    const uint64_t baseline_stop = read_clock64();

    if (lane == 0) {
        clock_out[0] = store_stop - store_start;
        clock_out[1] = g1 - g0;
    }
    out[lane] = store_stop - store_start;
    out[WARP_THREADS + lane] = load_stop - load_start;
    out[2 * WARP_THREADS + lane] = baseline_stop - baseline_start;
    out[3 * WARP_THREADS + lane] = p;
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

// Reads "value" from a run JSON written by another microbenchmark. Returns
// a value <= 0 if the file or the key is missing.
static double read_run_value(const std::string& path) {
    std::ifstream f(path);
    if (!f) return -1.0;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    const size_t k = s.find("\"value\":");
    if (k == std::string::npos) return -1.0;
    return atof(s.c_str() + k + 8);
}

static void append_array(std::ostringstream& json,
                         const std::vector<double>& values) {
    json << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        json << (i == 0 ? "" : ", ") << values[i];
    }
    json << "]";
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

    constexpr int OUT_WORDS = 4 * WARP_THREADS;
    unsigned long long* device_output = nullptr;
    unsigned long long* device_clock = nullptr;  // lane 0's cycles and ns, for the SM clock
    CUDA_CHECK(cudaMalloc(&device_clock, 2 * sizeof(unsigned long long)));
    double sum_clock_cycles = 0.0, sum_clock_ns = 0.0;
    CUDA_CHECK(cudaMalloc(&device_output,
                          OUT_WORDS * sizeof(unsigned long long)));

    std::vector<double> raw_store_chain_cycles;
    std::vector<double> raw_load_only_cycles;
    std::vector<double> clock_overhead_samples;
    raw_store_chain_cycles.reserve(reps);
    raw_load_only_cycles.reserve(reps);
    clock_overhead_samples.reserve(reps);

    for (int rep = 0; rep < reps; ++rep) {
        shared_store_latency_kernel<<<1, WARP_THREADS>>>(device_output, device_clock);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long host_clock[2] = {0, 0};
        CUDA_CHECK(cudaMemcpy(host_clock, device_clock, sizeof(host_clock),
                              cudaMemcpyDeviceToHost));
        if (host_clock[1] > 0) {
            sum_clock_cycles += static_cast<double>(host_clock[0]);
            sum_clock_ns += static_cast<double>(host_clock[1]);
        }

        std::array<unsigned long long, OUT_WORDS> host_output{};
        CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                              sizeof(host_output), cudaMemcpyDeviceToHost));

        std::vector<double> lane_store(WARP_THREADS);
        std::vector<double> lane_load(WARP_THREADS);
        std::vector<double> lane_overhead(WARP_THREADS);
        for (int lane = 0; lane < WARP_THREADS; ++lane) {
            lane_store[lane] = static_cast<double>(host_output[lane]);
            lane_load[lane] =
                static_cast<double>(host_output[WARP_THREADS + lane]);
            lane_overhead[lane] =
                static_cast<double>(host_output[2 * WARP_THREADS + lane]);
        }

        raw_store_chain_cycles.push_back(median(lane_store));
        raw_load_only_cycles.push_back(median(lane_load));
        clock_overhead_samples.push_back(median(lane_overhead));
    }

    const double clock_overhead = median(clock_overhead_samples);

    std::vector<double> per_store_cycles;
    std::vector<double> per_step_store_plus_load;
    std::vector<double> per_step_load_only;
    per_store_cycles.reserve(reps);
    per_step_store_plus_load.reserve(reps);
    per_step_load_only.reserve(reps);

    for (int rep = 0; rep < reps; ++rep) {
        const double store_plus_load =
            (raw_store_chain_cycles[rep] - clock_overhead) / CHAIN_STEPS;
        const double load_only =
            (raw_load_only_cycles[rep] - clock_overhead) / CHAIN_STEPS;
        per_step_store_plus_load.push_back(store_plus_load);
        per_step_load_only.push_back(load_only);
        per_store_cycles.push_back(store_plus_load - load_only);
    }

    const double value = median(per_store_cycles);
    const double store_plus_load_cycles = median(per_step_store_plus_load);
    const double load_only_cycles = median(per_step_load_only);
    const double mean =
        std::accumulate(per_store_cycles.begin(), per_store_cycles.end(), 0.0)
        / reps;

    double variance = 0.0;
    for (double sample : per_store_cycles) {
        variance += (sample - mean) * (sample - mean);
    }
    const double stddev = std::sqrt(variance / (reps - 1));
    const double min_value =
        *std::min_element(per_store_cycles.begin(), per_store_cycles.end());
    const double max_value =
        *std::max_element(per_store_cycles.begin(), per_store_cycles.end());

    // Cross-check the in-kernel load-only chain against the load benchmark.
    namespace fs = std::filesystem;
    const fs::path reference_path =
        fs::path(out_dir) /
        (std::string(LOAD_PARAMETER) + "." + gpu + ".run.json");
    const double load_reference = read_run_value(reference_path.string());
    const bool has_reference = load_reference > 0.0;
    if (!has_reference) {
        fprintf(stderr,
                "No load-latency reference at %s (run shared_memory_load "
                "first to cross-check); using the in-kernel baseline only\n",
                reference_path.string().c_str());
    } else if (std::fabs(load_only_cycles - load_reference) >
               REFERENCE_TOLERANCE_CYCLES) {
        fprintf(stderr,
                "Warning: in-kernel load-only chain is %.2f cycles/step but "
                "%s reports %.2f; the subtraction baseline may be off\n",
                load_only_cycles, reference_path.string().c_str(),
                load_reference);
    }

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
         << "  \"benchmark\": \"shared_memory_store.cu\",\n"
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
         << "  \"store_plus_load_cycles\": " << store_plus_load_cycles << ",\n"
         << "  \"load_only_cycles\": " << load_only_cycles << ",\n"
         << "  \"load_latency_reference_cycles\": ";
    if (has_reference) {
        json << load_reference;
    } else {
        json << "null";
    }
    json << ",\n";

    // Raw clock64 deltas per rep (median over the 32 lanes), overhead NOT subtracted.
    json << "  \"raw_store_chain_cycles\": ";
    append_array(json, raw_store_chain_cycles);
    json << ",\n  \"raw_load_only_cycles\": ";
    append_array(json, raw_load_only_cycles);
    json << ",\n  \"per_store_cycles\": ";
    append_array(json, per_store_cycles);

    json << ",\n"
         // clock.sm_clock_ghz: total clock64 cycles / total globaltimer ns over all reps
         << "  \"clock\": {\"sm_clock_ghz\": "
         << (sum_clock_ns > 0.0 ? sum_clock_cycles / sum_clock_ns : 0.0) << "},\n"
         << "  \"measurement\": {\n"
         << "    \"method\": \"dependent st.shared->ld.shared chain minus ld.shared-only chain\",\n"
         << "    \"store_instruction\": \"st.volatile.shared.u32\",\n"
         << "    \"load_instruction\": \"ld.volatile.shared.u32\",\n"
         << "    \"timer\": \"clock64 (cycles), globaltimer (SM clock)\",\n"
         << "    \"clock_overhead_cycles\": " << clock_overhead << ",\n"
         << "    \"value_formula\": \"per_store = (raw_store_chain_cycles[i] - raw_load_only_cycles[i]) / steps_per_rep; value = median(per_store)\",\n"
         << "    \"load_reference_file\": \""
         << json_escape(reference_path.string()) << "\",\n"
         << "    \"steps_per_rep\": " << CHAIN_STEPS << ",\n"
         << "    \"unroll\": " << UNROLL << ",\n"
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
    CUDA_CHECK(cudaFree(device_clock));
    return 0;
}

// ---------------------------------------------------------------------------
// SASS inspection checklist:
//   cuobjdump --dump-sass shared_memory_store
// In shared_store_latency_kernel:
//   * Store-chain timed loop: 16 pairs of `STS [Rp], Rp` followed by
//     `LDS Rp, [Rp]`, the same register for address, data and result, with
//     no ALU instructions between the pairs.
//   * Load-only timed loop: 16 dependent `LDS Rp, [Rp]`.
//   * Both loops have the same loop overhead (counter IADD3, ISETP, BRA), so
//     it cancels in the difference.
//   * CS2R (clock) right before and right after each timed loop, and a
//     globaltimer read (SR_GLOBALTIMER) just outside the store loop's pair.
// If ptxas drops the LDS after the STS (store-to-load forwarding in the
// compiler) or adds instructions to only one loop, the result is not a store
// latency.
// ---------------------------------------------------------------------------
