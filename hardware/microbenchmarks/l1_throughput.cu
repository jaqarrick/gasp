// l1_throughput.cu: how many global-load sectors per cycle one SM's L1 serves.
//
// Each block reads its own slice of a global array, at most half of L1, so
// after a warm pass every timed load is an L1 hit. Warps issue UNROLL
// independent loads per group (addresses come from the loop counter, never
// from loaded data). The span per SM is taken with clock64 around the timed
// loop; rate = load requests issued on the SM / span.
//
// Sweep: warps per SM (1..32, one block per SM), load shape (32-bit with 4 or
// 2 sectors per request, 64-bit, 128-bit), and, for the 32-bit 4-sector shape,
// a variant with STORES global stores per group mixed in, to see whether
// stores share the queue. Design: .local/l1_throughput.md (handoff I44).
//
// The profile value is the plateau sector rate of 32-bit, 4-sector loads with
// no stores. Every configuration's request and sector rates are in the JSON.
//
// Usage: l1_throughput [--reps N] [--gpu NAME] [--out-dir DIR] [--l1-bytes B]
// Run under ncu with --cache-control none (run_all.py does), or ncu flushes
// L1 between launches; the warm pass is inside the kernel, so this only adds
// noise, but check the ncu L1 hit rate (near 100%) either way.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <map>
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
constexpr int UNROLL = 8;         // independent loads in flight per warp
constexpr int ITERATIONS = 512;   // timed groups per warp: 4096 loads
constexpr int STORES = 2;         // stores per group in the mixed variant
constexpr int WARMUP_LAUNCHES = 3;
constexpr int DEFAULT_REPS = 100;
constexpr int DEFAULT_L1_BYTES = 64 * 1024;
constexpr double PLATEAU_TOLERANCE = 0.02;
constexpr int TIMING_FIELDS = 4;  // start, end, smid, warps

static const char* PARAMETER = "l1_throughput_sectors_per_cycle_per_sm";
static const char* SCHEMA_VERSION = "1.1.0";
static const char* BENCH_VERSION = "1.0.0";

__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t cycles;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycles) :: "memory");
    return cycles;
}

__device__ __forceinline__ uint64_t read_globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

__device__ __forceinline__ uint32_t read_smid() {
    uint32_t id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

// Default cache policy (ld.global, no .cs/.cg), so loads allocate in L1 as a
// normal kernel's do. Returns the loaded words XORed together.
template <int WIDTH>
__device__ __forceinline__ uint32_t load_global(const char* p);

template <>
__device__ __forceinline__ uint32_t load_global<4>(const char* p) {
    uint32_t a;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(a) : "l"(p));
    return a;
}

template <>
__device__ __forceinline__ uint32_t load_global<8>(const char* p) {
    uint32_t a, b;
    asm volatile("ld.global.v2.u32 {%0, %1}, [%2];" : "=r"(a), "=r"(b) : "l"(p));
    return a ^ b;
}

template <>
__device__ __forceinline__ uint32_t load_global<16>(const char* p) {
    uint32_t a, b, c, d;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                 : "l"(p));
    return a ^ b ^ c ^ d;
}

__device__ __forceinline__ void store_global(uint32_t* p, uint32_t v) {
    asm volatile("st.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// One block per SM. Lane l reads (l % distinct_lanes) * WIDTH bytes into a
// row, so a request touches distinct_lanes * WIDTH / 32 sectors. Rows are
// visited round-robin across warps; rows is a power of two.
template <int WIDTH, int N_STORES>
__global__ void l1_kernel(const char* __restrict__ data, uint32_t slice_bytes,
                          uint32_t distinct_lanes, uint32_t* scratch,
                          uint32_t* sink, unsigned long long* timing) {
    const uint32_t lane = threadIdx.x % WARP_THREADS;
    const uint32_t warp = threadIdx.x / WARP_THREADS;
    const uint32_t warps = blockDim.x / WARP_THREADS;
    const char* slice = data + static_cast<size_t>(blockIdx.x) * slice_bytes;
    const uint32_t row_bytes = distinct_lanes * WIDTH;
    const uint32_t mask = slice_bytes / row_bytes - 1;
    const uint32_t lane_offset = (lane % distinct_lanes) * WIDTH;
    uint32_t* store_row =
        scratch + ((blockIdx.x * 32 + warp) * (N_STORES > 0 ? N_STORES : 1)) *
                      WARP_THREADS + lane;

    uint32_t acc = 0;
    // warm pass: the whole slice into L1, untimed
    for (uint32_t r = warp; r <= mask; r += warps) {
        acc ^= load_global<WIDTH>(slice + r * row_bytes + lane_offset);
    }
    __syncthreads();

    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

    uint32_t row = warp;
#pragma unroll 1
    for (int it = 0; it < ITERATIONS; ++it) {
        uint32_t v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const uint32_t r = (row + u * warps) & mask;
            v[u] = load_global<WIDTH>(slice + r * row_bytes + lane_offset);
        }
#pragma unroll
        for (int s = 0; s < N_STORES; ++s) {
            // the accumulator from earlier groups, so stores don't wait on these loads
            store_global(store_row + s * WARP_THREADS, acc + s);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) acc ^= v[u];
        row = (row + UNROLL * warps) & mask;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const uint64_t end = read_clock64();
        unsigned long long* t = timing + blockIdx.x * TIMING_FIELDS;
        t[0] = start;
        t[1] = end;
        t[2] = read_smid();
        t[3] = warps;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;  // keeps acc live
}

// SM clock for the JSON: clock64 cycles per globaltimer ns over a spin
__global__ void clock_kernel(unsigned long long* out) {
    const uint64_t g0 = read_globaltimer();
    const uint64_t c0 = read_clock64();
    uint64_t c = c0;
    while (c - c0 < 2000000) c = read_clock64();
    const uint64_t g1 = read_globaltimer();
    out[0] = c - c0;
    out[1] = g1 - g0;
}

struct Shape {
    int width;           // bytes per lane
    int distinct_lanes;  // lanes reading different words
    int stores;          // stores per group
    std::string name() const {
        return std::to_string(width * 8) + "bit_" +
               std::to_string(distinct_lanes * width / 32) + "sectors" +
               (stores ? "_" + std::to_string(stores) + "stores" : "");
    }
    double sectors_per_request() const { return distinct_lanes * width / 32.0; }
};

struct Result {
    Shape shape;
    int warps_per_sm;
    double requests_per_cycle;   // median over reps, of the median over SMs
    double sectors_per_cycle;
    double accesses_per_cycle;   // loads + stores
    double spread;               // (max - min) / median of the per-rep values
    int sms_measured;
};

template <int WIDTH, int N_STORES>
static void launch(int grid, int threads, const char* data, uint32_t slice,
                   uint32_t distinct, uint32_t* scratch, uint32_t* sink,
                   unsigned long long* timing) {
    static bool configured = false;
    if (!configured) {
        // all of the unified array as L1: the kernel uses no shared memory
        CUDA_CHECK(cudaFuncSetAttribute(
            l1_kernel<WIDTH, N_STORES>,
            cudaFuncAttributePreferredSharedMemoryCarveout, 0));
        configured = true;
    }
    l1_kernel<WIDTH, N_STORES><<<grid, threads>>>(data, slice, distinct, scratch,
                                                   sink, timing);
}

static void launch_shape(const Shape& s, int grid, int threads, const char* data,
                         uint32_t slice, uint32_t* scratch, uint32_t* sink,
                         unsigned long long* timing) {
    const uint32_t d = s.distinct_lanes;
    if (s.stores == 0) {
        switch (s.width) {
            case 4: launch<4, 0>(grid, threads, data, slice, d, scratch, sink, timing); break;
            case 8: launch<8, 0>(grid, threads, data, slice, d, scratch, sink, timing); break;
            case 16: launch<16, 0>(grid, threads, data, slice, d, scratch, sink, timing); break;
        }
    } else {
        launch<4, STORES>(grid, threads, data, slice, d, scratch, sink, timing);
    }
    CUDA_CHECK(cudaGetLastError());
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    if (n == 0) return 0.0;
    return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
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
        lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    std::string result;
    for (char c : lower) {
        result += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
    }
    return result;
}

static std::string architecture_name(int major, int minor) {
    switch (major) {
        case 7: return minor >= 5 ? "turing" : "volta";
        case 8: return minor == 9 ? "ada" : "ampere";
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

int main(int argc, char** argv) {
    int reps = DEFAULT_REPS;
    int l1_bytes = DEFAULT_L1_BYTES;
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
        } else if (arg == "--l1-bytes" && i + 1 < argc) {
            l1_bytes = std::atoi(argv[++i]);
        } else {
            fprintf(stderr,
                    "Usage: %s [--reps N] [--gpu NAME] [--out-dir DIR] "
                    "[--l1-bytes B]\n", argv[0]);
            return 1;
        }
    }
    if (reps < 10) {
        fprintf(stderr, "--reps must be at least 10\n");
        return 1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const int sms = prop.multiProcessorCount;
    const int max_warps = std::min(prop.maxThreadsPerBlock,
                                   prop.maxThreadsPerMultiProcessor) / WARP_THREADS;

    // slice per SM: the largest power of two up to half of L1
    uint32_t slice = 1;
    while (slice * 2 <= static_cast<uint32_t>(l1_bytes / 2)) slice *= 2;
    const int max_row_bytes = 16 * WARP_THREADS;
    if (slice < static_cast<uint32_t>(max_row_bytes)) {
        fprintf(stderr, "--l1-bytes %d is too small\n", l1_bytes);
        return 1;
    }

    const std::vector<Shape> shapes = {
        {4, 32, 0}, {4, 16, 0}, {8, 32, 0}, {16, 32, 0}, {4, 32, STORES}};
    std::vector<int> warp_counts;
    for (int w = 1; w <= max_warps; w *= 2) warp_counts.push_back(w);

    char* d_data = nullptr;
    uint32_t* d_scratch = nullptr;
    uint32_t* d_sink = nullptr;
    unsigned long long* d_timing = nullptr;
    unsigned long long* d_clock = nullptr;
    const size_t data_bytes = static_cast<size_t>(sms) * slice;
    const size_t scratch_words =
        static_cast<size_t>(sms) * 32 * STORES * WARP_THREADS;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_scratch, scratch_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_timing, sms * TIMING_FIELDS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_clock, 2 * sizeof(unsigned long long)));
    {
        std::vector<uint32_t> fill(data_bytes / sizeof(uint32_t));
        uint32_t x = 2463534242u;
        for (auto& w : fill) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            w = x;
        }
        CUDA_CHECK(cudaMemcpy(d_data, fill.data(), data_bytes, cudaMemcpyHostToDevice));
    }

    std::vector<Result> results;
    std::vector<unsigned long long> timing(sms * TIMING_FIELDS);
    for (const Shape& shape : shapes) {
        for (int warps : warp_counts) {
            const int threads = warps * WARP_THREADS;
            for (int i = 0; i < WARMUP_LAUNCHES; ++i) {
                launch_shape(shape, sms, threads, d_data, slice, d_scratch, d_sink, d_timing);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> rates;
            int sms_measured = 0;
            for (int rep = 0; rep < reps; ++rep) {
                launch_shape(shape, sms, threads, d_data, slice, d_scratch, d_sink, d_timing);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(timing.data(), d_timing,
                                      timing.size() * sizeof(unsigned long long),
                                      cudaMemcpyDeviceToHost));
                // per SM: requests issued by its blocks / (latest end - earliest start)
                std::map<unsigned long long, std::vector<unsigned long long>> per_sm;
                for (int b = 0; b < sms; ++b) {
                    const unsigned long long* t = &timing[b * TIMING_FIELDS];
                    auto& e = per_sm[t[2]];
                    if (e.empty()) e = {t[0], t[1], 0};
                    e[0] = std::min(e[0], t[0]);
                    e[1] = std::max(e[1], t[1]);
                    e[2] += t[3];
                }
                std::vector<double> sm_rates;
                for (const auto& kv : per_sm) {
                    const auto& e = kv.second;
                    const double span = static_cast<double>(e[1] - e[0]);
                    const double requests =
                        static_cast<double>(e[2]) * ITERATIONS * UNROLL;
                    if (span > 0) sm_rates.push_back(requests / span);
                }
                sms_measured = static_cast<int>(sm_rates.size());
                rates.push_back(median(sm_rates));
            }
            const double m = median(rates);
            const double lo = *std::min_element(rates.begin(), rates.end());
            const double hi = *std::max_element(rates.begin(), rates.end());
            const double stores_per_load =
                static_cast<double>(shape.stores) / UNROLL;
            results.push_back({shape, warps, m, m * shape.sectors_per_request(),
                               m * (1.0 + stores_per_load),
                               m > 0 ? (hi - lo) / m : 0.0, sms_measured});
            fprintf(stderr, "%-24s warps/SM %2d: %.4f requests/cycle, %.4f sectors/cycle\n",
                    shape.name().c_str(), warps, m, m * shape.sectors_per_request());
        }
    }

    // plateau per shape: the largest rate; accepted when the two highest warp
    // counts agree within PLATEAU_TOLERANCE
    struct Plateau { double requests; double sectors; bool ok; };
    std::map<std::string, Plateau> plateaus;
    for (const Shape& shape : shapes) {
        std::vector<const Result*> rs;
        for (const auto& r : results) {
            if (r.shape.name() == shape.name()) rs.push_back(&r);
        }
        const Result* best = *std::max_element(
            rs.begin(), rs.end(), [](const Result* a, const Result* b) {
                return a->requests_per_cycle < b->requests_per_cycle;
            });
        bool ok = false;
        if (rs.size() >= 2) {
            const double a = rs[rs.size() - 1]->requests_per_cycle;
            const double b = rs[rs.size() - 2]->requests_per_cycle;
            ok = std::fabs(a - b) <= PLATEAU_TOLERANCE * std::max(a, b);
        }
        plateaus[shape.name()] = {best->requests_per_cycle, best->sectors_per_cycle, ok};
        if (!ok) {
            fprintf(stderr, "WARNING: %s has no plateau: the two highest warp counts "
                    "differ by more than %.0f%%\n", shape.name().c_str(),
                    PLATEAU_TOLERANCE * 100);
        }
    }
    const Plateau& main_plateau = plateaus[shapes[0].name()];
    const double value = main_plateau.sectors;

    double sm_clock_ghz = 0.0;
    {
        clock_kernel<<<1, 1>>>(d_clock);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long c[2];
        CUDA_CHECK(cudaMemcpy(c, d_clock, sizeof(c), cudaMemcpyDeviceToHost));
        if (c[1] > 0) sm_clock_ghz = static_cast<double>(c[0]) / c[1];
    }

    int runtime_version = 0, driver_api_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_api_version));

    const std::string gpu_name = prop.name;
    const std::string gpu = gpu_override.empty() ? short_gpu_name(gpu_name) : gpu_override;
    const std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);

    std::ostringstream json;
    json.setf(std::ios::fixed);
    json.precision(4);
    json << "{\n"
         << "  \"schema_version\": \"" << SCHEMA_VERSION << "\",\n"
         << "  \"benchmark\": \"l1_throughput.cu\",\n"
         << "  \"benchmark_version\": \"" << BENCH_VERSION << "\",\n"
         << "  \"parameter\": \"" << PARAMETER << "\",\n"
         << "  \"value\": " << value << ",\n"
         << "  \"unit\": \"32-byte sectors per SM cycle (global loads, L1 hits)\",\n"
         << "  \"plateau_ok\": " << (main_plateau.ok ? "true" : "false") << ",\n"
         << "  \"requests_per_cycle_per_sm\": " << main_plateau.requests << ",\n"
         << "  \"gpu\": \"" << json_escape(gpu) << "\",\n"
         << "  \"architecture\": \"" << architecture_name(prop.major, prop.minor) << "\",\n"
         << "  \"arch\": \"" << arch << "\",\n"
         << "  \"device\": {\n"
         << "    \"name\": \"" << json_escape(gpu_name) << "\",\n"
         << "    \"sms\": " << sms << ",\n"
         << "    \"l2_cache_bytes\": " << prop.l2CacheSize << "\n"
         << "  },\n"
         << "  \"plateaus\": {";
    bool first = true;
    for (const Shape& shape : shapes) {
        const Plateau& p = plateaus[shape.name()];
        json << (first ? "\n" : ",\n") << "    \"" << shape.name() << "\": {"
             << "\"requests_per_cycle\": " << p.requests
             << ", \"sectors_per_cycle\": " << p.sectors
             << ", \"plateau_ok\": " << (p.ok ? "true" : "false") << "}";
        first = false;
    }
    json << "\n  },\n  \"configs\": [";
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        json << (i ? ",\n" : "\n") << "    {\"shape\": \"" << r.shape.name() << "\""
             << ", \"load_width_bytes\": " << r.shape.width
             << ", \"sectors_per_request\": " << r.shape.sectors_per_request()
             << ", \"stores_per_group\": " << r.shape.stores
             << ", \"warps_per_sm\": " << r.warps_per_sm
             << ", \"requests_per_cycle\": " << r.requests_per_cycle
             << ", \"sectors_per_cycle\": " << r.sectors_per_cycle
             << ", \"accesses_per_cycle\": " << r.accesses_per_cycle
             << ", \"rep_spread\": " << r.spread
             << ", \"sms_measured\": " << r.sms_measured << "}";
    }
    json << "\n  ],\n"
         << "  \"clock\": {\"sm_clock_ghz\": " << sm_clock_ghz << "},\n"
         << "  \"measurement\": {\n"
         << "    \"method\": \"per-block slice in L1, warm pass, independent coalesced loads; per-SM clock64 span\",\n"
         << "    \"load_instruction\": \"ld.global.u32 / .v2.u32 / .v4.u32 (default cache policy)\",\n"
         << "    \"slice_bytes_per_sm\": " << slice << ",\n"
         << "    \"l1_bytes\": " << l1_bytes << ",\n"
         << "    \"loads_per_warp\": " << ITERATIONS * UNROLL << ",\n"
         << "    \"unroll\": " << UNROLL << ",\n"
         << "    \"stores_per_group_mixed\": " << STORES << ",\n"
         << "    \"blocks\": " << sms << ",\n"
         << "    \"reps\": " << reps << "\n"
         << "  },\n"
         << "  \"cuda_version\": \"" << cuda_version_string(runtime_version) << "\",\n"
         << "  \"driver_api_version\": \"" << cuda_version_string(driver_api_version) << "\",\n"
         << "  \"timestamp\": \"" << utc_timestamp() << "\"\n"
         << "}\n";

    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) {
        fprintf(stderr, "Cannot create output directory: %s\n", ec.message().c_str());
        return 1;
    }
    const fs::path output_path =
        fs::path(out_dir) / (std::string(PARAMETER) + "." + gpu + ".run.json");
    FILE* file = std::fopen(output_path.string().c_str(), "w");
    if (file == nullptr) {
        perror("Could not open results JSON");
        return 1;
    }
    const std::string result = json.str();
    bool write_ok = std::fputs(result.c_str(), file) >= 0;
    if (std::fclose(file) != 0) write_ok = false;
    if (!write_ok) {
        fprintf(stderr, "Failed writing results to %s\n", output_path.string().c_str());
        return 1;
    }
    printf("%s", result.c_str());
    fprintf(stderr, "Saved results to %s\n", output_path.string().c_str());

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaFree(d_timing));
    CUDA_CHECK(cudaFree(d_clock));
    return 0;
}
