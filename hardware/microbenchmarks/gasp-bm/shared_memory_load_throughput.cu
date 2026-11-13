// shared_memory_load_throughput.cu: SM cycles per warp shared-memory load when
// the shared-memory pipe is saturated.
//
// One block per SM. Every warp issues LOADS independent, conflict-free shared
// loads per group: lane l reads word slot * 32 + l, so the 32 lanes hit 32
// banks, and no address depends on a loaded value. (The dependent chain, which
// gives the latency, is shared_memory_load.cu.) The span per SM is taken with
// clock64 around the timed loop:
//
//     cycles per load = (span - overhead) / (warps on the SM x loads per warp)
//
// where overhead is the span of the same kernel with an empty loop.
//
// Sweep: warps per SM (1..32), since one warp may be limited by its own issue
// rate and the pipe's limit shows as a plateau; and a mixed shape with
// MIX_LOADS loads to MIX_STORES store per group (tiled matmul's ratio), to see
// whether loads and stores share the pipe. Design:
// .local/shared_memory_load_throughput.md (handoff I46, I66).
//
// The profile value is the plateau of the loads-only shape. A group is 16 LDS,
// 8 XORs and the loop counter (sm_75 SASS), so the value is an upper bound on
// the pipe's own cost per load. Every configuration is in the JSON.
//
// Usage: shared_memory_load_throughput [--reps N] [--gpu NAME] [--out-dir DIR]
// Check under ncu: shared-load bank conflicts 0, no global traffic in the timed
// kernel, and with 1 warp a value at or below the shared load latency.

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
#include "common.h"

constexpr int WARP_THREADS = 32;
constexpr int LOAD_SLOTS = 16;    // rows of 32 words every warp reads
constexpr int MAX_WARPS = 32;     // store rows, one per warp
constexpr int LOADS = 16;         // loads per group, loads-only shape
constexpr int MIX_LOADS = 10;     // loads per group, mixed shape
constexpr int MIX_STORES = 1;     // stores per group, mixed shape
constexpr int ITERATIONS = 256;   // timed groups per warp
constexpr int ACCS = 4;           // XOR accumulators the loaded values go into
constexpr int WARMUP_LAUNCHES = 3;
constexpr int DEFAULT_REPS = 200;
constexpr double PLATEAU_TOLERANCE = 0.02;
constexpr int TIMING_FIELDS = 4;  // start, end, smid, warps

static_assert(LOADS <= LOAD_SLOTS && MIX_LOADS <= LOAD_SLOTS,
              "each load of a group reads its own slot");

static const char* PARAMETER = "shared_mem_load_throughput_cycles";
static const char* SCHEMA_VERSION = "1.1.0";
static const char* BENCH_VERSION = "1.0.0";

// volatile: the compiler may not merge repeated loads of one address
__device__ __forceinline__ uint32_t load_shared_volatile(uint32_t address) {
    uint32_t value;
    asm volatile("ld.volatile.shared.u32 %0, [%1];"
                 : "=r"(value)
                 : "r"(address)
                 : "memory");
    return value;
}

// N_LOADS loads and N_STORES stores per group; N_LOADS = 0 is the empty loop
// that gives the overhead (clock reads and barriers).
template <int N_LOADS, int N_STORES>
__global__ void shared_load_kernel(uint32_t* sink, unsigned long long* timing) {
    __shared__ uint32_t load_words[LOAD_SLOTS * WARP_THREADS];
    __shared__ uint32_t store_words[MAX_WARPS * WARP_THREADS];

    const uint32_t lane = threadIdx.x % WARP_THREADS;
    const uint32_t warp = threadIdx.x / WARP_THREADS;
    const uint32_t warps = blockDim.x / WARP_THREADS;
    const uint32_t load_base =
        static_cast<uint32_t>(__cvta_generic_to_shared(load_words)) +
        lane * sizeof(uint32_t);
    const uint32_t store_address =
        static_cast<uint32_t>(__cvta_generic_to_shared(store_words)) +
        (warp * WARP_THREADS + lane) * sizeof(uint32_t);

    // fill: every warp writes the same value to a word, so the race is benign
    for (int slot = 0; slot < LOAD_SLOTS; ++slot) {
        load_words[slot * WARP_THREADS + lane] =
            (slot * WARP_THREADS + lane) * 2654435761u;
    }
    store_words[warp * WARP_THREADS + lane] = lane;
    __syncthreads();

    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

    // ACCS accumulators, so the XORs are short chains and not one per group
    uint32_t acc[ACCS] = {};
#pragma unroll 1
    for (int it = 0; it < ITERATIONS; ++it) {
        const uint32_t earlier = acc[0];
#pragma unroll
        for (int u = 0; u < N_LOADS; ++u) {
            acc[u % ACCS] ^= load_shared_volatile(
                load_base + u * WARP_THREADS * sizeof(uint32_t));
        }
#pragma unroll
        for (int s = 0; s < N_STORES; ++s) {
            // the accumulator from earlier groups, so stores don't wait on these loads
            store_shared_volatile(store_address, earlier + s);
        }
    }
    uint32_t total = 0;
#pragma unroll
    for (int a = 0; a < ACCS; ++a) total ^= acc[a];
    __syncthreads();

    if (threadIdx.x == 0) {
        const uint64_t end = read_clock64();
        unsigned long long* t = timing + blockIdx.x * TIMING_FIELDS;
        t[0] = start;
        t[1] = end;
        t[2] = read_smid();
        t[3] = warps;
    }
    if (total == 0x9e3779b9u) sink[0] = total;  // keeps the loads live
}

struct Shape {
    int loads;   // per group
    int stores;  // per group
    std::string name() const {
        if (loads == 0) return "empty";
        return stores ? std::to_string(loads) + "loads_" +
                            std::to_string(stores) + "store"
                      : "loads_only";
    }
};

struct Result {
    Shape shape;
    int warps_per_sm;
    double span_cycles;         // median over reps, of the median over SMs
    double overhead_cycles;     // the empty loop's span at this warp count
    double cycles_per_load;     // (span - overhead) / loads on the SM
    double cycles_per_access;   // the same over loads + stores
    double spread;              // (max - min) / median of the per-rep spans
    int sms_measured;
};

static void launch_shape(const Shape& s, int grid, int threads, uint32_t* sink,
                         unsigned long long* timing) {
    if (s.loads == 0) {
        shared_load_kernel<0, 0><<<grid, threads>>>(sink, timing);
    } else if (s.stores == 0) {
        shared_load_kernel<LOADS, 0><<<grid, threads>>>(sink, timing);
    } else {
        shared_load_kernel<MIX_LOADS, MIX_STORES><<<grid, threads>>>(sink, timing);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Median over reps of the median per-SM span (latest end - earliest start of
// the SM's blocks), with the warps that ran on a typical SM.
struct Span {
    double cycles;
    double warps_on_sm;
    double spread;
    int sms_measured;
};

static Span measure(const Shape& shape, int sms, int threads, int reps,
                    uint32_t* d_sink, unsigned long long* d_timing) {
    std::vector<unsigned long long> timing(sms * TIMING_FIELDS);
    for (int i = 0; i < WARMUP_LAUNCHES; ++i) {
        launch_shape(shape, sms, threads, d_sink, d_timing);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> spans, warps_seen;
    int sms_measured = 0;
    for (int rep = 0; rep < reps; ++rep) {
        launch_shape(shape, sms, threads, d_sink, d_timing);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(timing.data(), d_timing,
                              timing.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        std::map<unsigned long long, std::vector<unsigned long long>> per_sm;
        for (int b = 0; b < sms; ++b) {
            const unsigned long long* t = &timing[b * TIMING_FIELDS];
            auto& e = per_sm[t[2]];
            if (e.empty()) e = {t[0], t[1], 0};
            e[0] = std::min(e[0], t[0]);
            e[1] = std::max(e[1], t[1]);
            e[2] += t[3];
        }
        // cycles per warp on each SM, so SMs that got two blocks compare
        std::vector<double> sm_spans, sm_warps;
        for (const auto& kv : per_sm) {
            const auto& e = kv.second;
            if (e[1] > e[0] && e[2] > 0) {
                sm_spans.push_back(static_cast<double>(e[1] - e[0]));
                sm_warps.push_back(static_cast<double>(e[2]));
            }
        }
        sms_measured = static_cast<int>(sm_spans.size());
        spans.push_back(median_of(sm_spans));
        warps_seen.push_back(median_of(sm_warps));
    }
    const double m = median_of(spans);
    const double lo = *std::min_element(spans.begin(), spans.end());
    const double hi = *std::max_element(spans.begin(), spans.end());
    return {m, median_of(warps_seen), m > 0 ? (hi - lo) / m : 0.0, sms_measured};
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
            fprintf(stderr, "Usage: %s [--reps N] [--gpu NAME] [--out-dir DIR]\n",
                    argv[0]);
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
    const int max_warps =
        std::min({prop.maxThreadsPerBlock / WARP_THREADS,
                  prop.maxThreadsPerMultiProcessor / WARP_THREADS, MAX_WARPS});

    const Shape empty{0, 0};
    const std::vector<Shape> shapes = {{LOADS, 0}, {MIX_LOADS, MIX_STORES}};
    std::vector<int> warp_counts;
    for (int w = 1; w <= max_warps; w *= 2) warp_counts.push_back(w);

    uint32_t* d_sink = nullptr;
    unsigned long long* d_timing = nullptr;
    unsigned long long* d_clock = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_timing, sms * TIMING_FIELDS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_clock, 2 * sizeof(unsigned long long)));

    std::map<int, double> overhead;  // warps per SM -> the empty loop's span
    for (int warps : warp_counts) {
        overhead[warps] =
            measure(empty, sms, warps * WARP_THREADS, reps, d_sink, d_timing).cycles;
    }

    std::vector<Result> results;
    for (const Shape& shape : shapes) {
        for (int warps : warp_counts) {
            const Span span =
                measure(shape, sms, warps * WARP_THREADS, reps, d_sink, d_timing);
            const double net = span.cycles - overhead[warps];
            const double loads = span.warps_on_sm * ITERATIONS * shape.loads;
            const double accesses =
                span.warps_on_sm * ITERATIONS * (shape.loads + shape.stores);
            results.push_back({shape, warps, span.cycles, overhead[warps],
                               loads > 0 ? net / loads : 0.0,
                               accesses > 0 ? net / accesses : 0.0, span.spread,
                               span.sms_measured});
            fprintf(stderr, "%-16s warps/SM %2d: %.4f cycles/load, %.4f cycles/access\n",
                    shape.name().c_str(), warps, results.back().cycles_per_load,
                    results.back().cycles_per_access);
        }
    }

    // plateau per shape: the lowest cycles per access; accepted when the two
    // highest warp counts agree within PLATEAU_TOLERANCE
    struct Plateau { double per_load; double per_access; bool ok; };
    std::map<std::string, Plateau> plateaus;
    for (const Shape& shape : shapes) {
        std::vector<const Result*> rs;
        for (const auto& r : results) {
            if (r.shape.name() == shape.name()) rs.push_back(&r);
        }
        const Result* best = *std::min_element(
            rs.begin(), rs.end(), [](const Result* a, const Result* b) {
                return a->cycles_per_access < b->cycles_per_access;
            });
        bool ok = false;
        if (rs.size() >= 2) {
            const double a = rs[rs.size() - 1]->cycles_per_access;
            const double b = rs[rs.size() - 2]->cycles_per_access;
            ok = std::fabs(a - b) <= PLATEAU_TOLERANCE * std::max(a, b);
        }
        plateaus[shape.name()] = {best->cycles_per_load, best->cycles_per_access, ok};
        if (!ok) {
            fprintf(stderr, "WARNING: %s has no plateau: the two highest warp counts "
                    "differ by more than %.0f%%\n", shape.name().c_str(),
                    PLATEAU_TOLERANCE * 100);
        }
    }
    const Plateau& main_plateau = plateaus[shapes[0].name()];
    const double value = main_plateau.per_load;

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
         << "  \"benchmark\": \"shared_memory_load_throughput.cu\",\n"
         << "  \"benchmark_version\": \"" << BENCH_VERSION << "\",\n"
         << "  \"parameter\": \"" << PARAMETER << "\",\n"
         << "  \"value\": " << value << ",\n"
         << "  \"unit\": \"SM cycles per warp shared load instruction, at the plateau\",\n"
         << "  \"plateau_ok\": " << (main_plateau.ok ? "true" : "false") << ",\n"
         << "  \"gpu\": \"" << json_escape(gpu) << "\",\n"
         << "  \"architecture\": \"" << architecture_name(prop.major, prop.minor) << "\",\n"
         << "  \"arch\": \"" << arch << "\",\n"
         << "  \"device\": {\n"
         << "    \"name\": \"" << json_escape(gpu_name) << "\",\n"
         << "    \"sms\": " << sms << ",\n"
         << "    \"max_warps_per_sm\": "
         << prop.maxThreadsPerMultiProcessor / WARP_THREADS << "\n"
         << "  },\n"
         << "  \"plateaus\": {";
    bool first = true;
    for (const Shape& shape : shapes) {
        const Plateau& p = plateaus[shape.name()];
        json << (first ? "\n" : ",\n") << "    \"" << shape.name() << "\": {"
             << "\"cycles_per_load\": " << p.per_load
             << ", \"cycles_per_access\": " << p.per_access
             << ", \"plateau_ok\": " << (p.ok ? "true" : "false") << "}";
        first = false;
    }
    json << "\n  },\n  \"configs\": [";
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        json << (i ? ",\n" : "\n") << "    {\"shape\": \"" << r.shape.name() << "\""
             << ", \"loads_per_group\": " << r.shape.loads
             << ", \"stores_per_group\": " << r.shape.stores
             << ", \"warps_per_sm\": " << r.warps_per_sm
             << ", \"span_cycles\": " << r.span_cycles
             << ", \"overhead_cycles\": " << r.overhead_cycles
             << ", \"cycles_per_load\": " << r.cycles_per_load
             << ", \"cycles_per_access\": " << r.cycles_per_access
             << ", \"rep_spread\": " << r.spread
             << ", \"sms_measured\": " << r.sms_measured << "}";
    }
    json << "\n  ],\n"
         << "  \"clock\": {\"sm_clock_ghz\": " << sm_clock_ghz << "},\n"
         << "  \"measurement\": {\n"
         << "    \"method\": \"independent conflict-free warp shared loads, one block per SM; per-SM clock64 span minus the empty loop's\",\n"
         << "    \"load_instruction\": \"ld.volatile.shared.u32\",\n"
         << "    \"store_instruction\": \"st.volatile.shared.u32\",\n"
         << "    \"value_formula\": \"(span - overhead) / (warps on the SM x groups x loads per group), lowest over warps per SM, loads_only shape\",\n"
         << "    \"groups_per_warp\": " << ITERATIONS << ",\n"
         << "    \"loads_per_group\": " << LOADS << ",\n"
         << "    \"mixed_loads_per_group\": " << MIX_LOADS << ",\n"
         << "    \"mixed_stores_per_group\": " << MIX_STORES << ",\n"
         << "    \"blocks\": " << sms << ",\n"
         << "    \"reps\": " << reps << "\n"
         << "  },\n"
         << "  \"cuda_version\": \"" << cuda_version_string(runtime_version) << "\",\n"
         << "  \"driver_api_version\": \"" << cuda_version_string(driver_api_version) << "\",\n"
         << "  \"timestamp\": \"" << utc_timestamp() << "\"\n"
         << "}\n";

    const std::string result = json.str();
    const std::string output_path = write_run_json(out_dir, PARAMETER, gpu, result);
    if (output_path.empty()) return 1;

    printf("%s", result.c_str());
    fprintf(stderr, "Saved results to %s\n", output_path.c_str());

    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaFree(d_timing));
    CUDA_CHECK(cudaFree(d_clock));
    return 0;
}
