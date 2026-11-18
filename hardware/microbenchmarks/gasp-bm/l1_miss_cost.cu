// this microbenchmark measures how many extra L1 cycles a global load request costs when it misses L1 and is served by L2.
//
// Each SM reads a slice of a global array, at most half of L1, so after a warm
// pass those loads become L1 hits. Some loads in every group are instead sent to
// a 1 MB array that stays in L2, using ld.global.cg so they skip L1 and are
// always L1 misses. Warps issue UNROLL independent loads per group (addresses
// come from the loop counter, never from loaded data). The span per SM is
// taken with clock64 around the timed loop; cost = span / load requests issued
// on the SM.
//
// For sweep: misses per group of UNROLL loads (0, 1, 2, 4, 8) and warps per SM
// (1..max warps per block). For each miss count the lowest cost over the warp
// counts is its plateau. A line is then fitted through (miss fraction, plateau
// cost): cycles per request = intercept + slope * miss fraction.
//
// The profile value is the slope, which represents the extra cycles one request costs when it
// misses L1 instead of hitting. The intercept is the cost of one L1 hit. Every
// configuration's cost is in the JSON.
//
// Usage: l1_miss_cost [--reps N] [--gpu NAME] [--out-dir DIR] [--l1-bytes B] [--sms N]

#include <algorithm>
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
constexpr int MAX_BLOCK_WARPS = 32; // hardware limit of 1024 threads per block
constexpr int UNROLL = 8; // independent loads in flight per warp
constexpr int ITERATIONS = 512; // timed groups per warp: 4096 loads
constexpr int WARMUP_LAUNCHES = 3;
constexpr int DEFAULT_REPS = 100;
constexpr int DEFAULT_L1_BYTES = 64 * 1024;
constexpr int DEFAULT_SMS = 4; // blocks launched, one per SM
constexpr uint32_t MISS_BYTES = 1u << 20;  // stays in L2 on every target GPU
constexpr double PLATEAU_TOLERANCE = 0.02;
constexpr int TIMING_FIELDS = 4; // start, end, smid, warps
constexpr int MISS_COUNTS[] = {0, 1, 2, 4, 8}; // L2 loads per group of UNROLL

static const char* PARAMETER = "l1_miss_cycles_per_request";
static const char* SCHEMA_VERSION = "1.0.0";
static const char* BENCH_VERSION = "1.0.0";

// Default cache policy (ld.global), so the load hits in L1. Used for warm pass and hit loads
__device__ __forceinline__ uint32_t load_l1(const char* p) {
    uint32_t a;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(a) : "l"(p));
    return a;
}

// ld.global.cg caches in L2 only, so this load misses L1 and is served by L2
__device__ __forceinline__ uint32_t load_l2(const char* p) {
    uint32_t a; // .cg - cache global
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(a) : "l"(p));
    return a;
}

// One block per SM. slice is indexed by smid, so each SM warms and reads its
// own part of hits. In every group of "unroll" loads, the first MISSES go to the
// L2 array and the rest to L1. A warp load is 32 lanes * 4 bytes = one 128 byte row.
template <int MISSES>
__global__ void miss_kernel(const char* __restrict__ hits, uint32_t slice_bytes,
                            const char* __restrict__ misses, uint32_t* sink,
                            unsigned long long* timing) {
    const uint32_t lane = threadIdx.x % WARP_THREADS;
    const uint32_t warp = threadIdx.x / WARP_THREADS;
    const uint32_t warps = blockDim.x / WARP_THREADS;
    const uint32_t smid = read_smid();
    const char* slice = hits + static_cast<size_t>(smid) * slice_bytes;
    const uint32_t row_bytes = WARP_THREADS * 4;
    const uint32_t hit_mask = slice_bytes / row_bytes - 1; // rows in slice - 1
    const uint32_t miss_mask = MISS_BYTES / row_bytes - 1; // rows in L2 array - 1
    const uint32_t lane_offset = lane * 4;

    uint32_t acc = 0;
    // warm pass, loading the whole slice into L1
    for (uint32_t r = warp; r <= hit_mask; r += warps) {
        acc ^= load_l1(slice + r * row_bytes + lane_offset);
    }
    __syncthreads();

    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

    uint32_t row = warp;
    // start the L2 rows at a different place for each SM and warp
    uint32_t miss_row = (smid * 997u + warp) & miss_mask;
#pragma unroll 1
    for (int it = 0; it < ITERATIONS; ++it) {
        uint32_t v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            if (u < MISSES) {
                // L1 miss, L2 hit
                const uint32_t r = (miss_row + u * warps) & miss_mask;
                v[u] = load_l2(misses + r * row_bytes + lane_offset);
            } else {
                // L1 hit
                const uint32_t r = (row + u * warps) & hit_mask;
                v[u] = load_l1(slice + r * row_bytes + lane_offset);
            }
        }
        // use all loaded values only after every load in the group is issued
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) acc ^= v[u];
        row = (row + UNROLL * warps) & hit_mask;
        miss_row = (miss_row + UNROLL * warps) & miss_mask;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const uint64_t end = read_clock64();
        unsigned long long* t = timing + blockIdx.x * TIMING_FIELDS;
        t[0] = start;
        t[1] = end;
        t[2] = smid;
        t[3] = warps;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;  // keeps acc live
}

template <int MISSES>
static void launch(int grid, int threads, const char* hits, uint32_t slice,
                   const char* misses, uint32_t* sink, unsigned long long* timing) {
    static bool configured = false;
    if (!configured) {
        // all of the unified array as L1, because the kernel uses no shared memory
        CUDA_CHECK(cudaFuncSetAttribute(
            miss_kernel<MISSES>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
        configured = true;
    }
    miss_kernel<MISSES><<<grid, threads>>>(hits, slice, misses, sink, timing);
}

// Pick the instantiation for the runtime miss count
static void launch_misses(int m, int grid, int threads, const char* hits, uint32_t slice,
                          const char* misses, uint32_t* sink, unsigned long long* timing) {
    switch (m) {
        case 0: launch<0>(grid, threads, hits, slice, misses, sink, timing); break;
        case 1: launch<1>(grid, threads, hits, slice, misses, sink, timing); break;
        case 2: launch<2>(grid, threads, hits, slice, misses, sink, timing); break;
        case 4: launch<4>(grid, threads, hits, slice, misses, sink, timing); break;
        case 8: launch<8>(grid, threads, hits, slice, misses, sink, timing); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

struct Result {
    int misses; // L2 loads per group of UNROLL
    int warps_per_sm;
    double cycles_per_request;  // median over reps, of the median over SMs
    double spread; // (max - min) / median of the per-rep values
    int sms_measured;
};

int main(int argc, char** argv) {
    int reps = DEFAULT_REPS;
    int l1_bytes = DEFAULT_L1_BYTES;
    int sms_used = DEFAULT_SMS;
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
        } else if (arg == "--sms" && i + 1 < argc) {
            sms_used = std::atoi(argv[++i]);
        } else {
            fprintf(stderr,
                    "Usage: %s [--reps N] [--gpu NAME] [--out-dir DIR] "
                    "[--l1-bytes B] [--sms N]\n", argv[0]);
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
    if (sms_used < 1 || sms_used > sms) {
        fprintf(stderr, "--sms must be between 1 and %d\n", sms);
        return 1;
    }
    // one block per SM, so the sweep stops at the block limit of 32 warps
    const int max_warps = std::min(prop.maxThreadsPerMultiProcessor / WARP_THREADS,
                                   MAX_BLOCK_WARPS);

    // slice per SM is the largest power of two up to half of L1
    uint32_t slice = 1;
    while (slice * 2 <= static_cast<uint32_t>(l1_bytes / 2)) slice *= 2;
    if (slice < static_cast<uint32_t>(WARP_THREADS * 4)) {
        fprintf(stderr, "--l1-bytes %d is too small\n", l1_bytes);
        return 1;
    }

    // warps per SM: powers of two up to max_warps, plus max_warps itself
    std::vector<int> warp_counts;
    for (int w = 1; w <= max_warps; w *= 2) warp_counts.push_back(w);
    if (warp_counts.back() < max_warps) warp_counts.push_back(max_warps);

    char* d_hits = nullptr;
    char* d_misses = nullptr;
    uint32_t* d_sink = nullptr;
    unsigned long long* d_timing = nullptr;
    unsigned long long* d_clock = nullptr;
    // hits holds one slice for every SM on the GPU, because smid picks the slice
    const size_t hit_bytes = static_cast<size_t>(sms) * slice;
    CUDA_CHECK(cudaMalloc(&d_hits, hit_bytes));
    CUDA_CHECK(cudaMalloc(&d_misses, MISS_BYTES));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_timing, sms_used * TIMING_FIELDS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_clock, 2 * sizeof(unsigned long long)));
    {
        // fill arrays with xorshift random words
        std::vector<uint32_t> fill(std::max(hit_bytes, static_cast<size_t>(MISS_BYTES)) /
                                   sizeof(uint32_t));
        uint32_t x = 2463534242u;
        for (auto& w : fill) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            w = x;
        }
        CUDA_CHECK(cudaMemcpy(d_hits, fill.data(), hit_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_misses, fill.data(), MISS_BYTES, cudaMemcpyHostToDevice));
    }

    std::vector<Result> results;
    std::vector<unsigned long long> timing(sms_used * TIMING_FIELDS);
    for (int m : MISS_COUNTS) {
        for (int warps : warp_counts) {
            const int grid = sms_used;
            const int threads = warps * WARP_THREADS;
            for (int i = 0; i < WARMUP_LAUNCHES; ++i) {
                launch_misses(m, grid, threads, d_hits, slice, d_misses, d_sink, d_timing);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> costs;
            int sms_measured = 0;
            for (int rep = 0; rep < reps; ++rep) {
                launch_misses(m, grid, threads, d_hits, slice, d_misses, d_sink, d_timing);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(timing.data(), d_timing,
                                      grid * TIMING_FIELDS * sizeof(unsigned long long),
                                      cudaMemcpyDeviceToHost));
                // per SM: (latest end - earliest start) / requests issued by its blocks
                std::map<unsigned long long, std::vector<unsigned long long>> per_sm;
                for (int b = 0; b < grid; ++b) {
                    const unsigned long long* t = &timing[b * TIMING_FIELDS];
                    auto& e = per_sm[t[2]];
                    if (e.empty()) e = {t[0], t[1], 0};
                    e[0] = std::min(e[0], t[0]);
                    e[1] = std::max(e[1], t[1]);
                    e[2] += t[3];
                }
                std::vector<double> sm_costs;
                for (const auto& kv : per_sm) {
                    const auto& e = kv.second;
                    const double span = static_cast<double>(e[1] - e[0]);
                    const double requests = static_cast<double>(e[2]) * ITERATIONS * UNROLL;
                    if (requests > 0) sm_costs.push_back(span / requests);
                }
                sms_measured = static_cast<int>(sm_costs.size());
                costs.push_back(median_of(sm_costs));
            }
            const double c = median_of(costs);
            const double lo = *std::min_element(costs.begin(), costs.end());
            const double hi = *std::max_element(costs.begin(), costs.end());
            results.push_back({m, warps, c, c > 0 ? (hi - lo) / c : 0.0, sms_measured});
            fprintf(stderr, "misses %d/%d  warps/SM %2d: %.4f cycles/request\n",
                    m, UNROLL, warps, c);
        }
    }

    // plateau per miss count: the lowest cost; accepted when the two highest
    // warp counts agree within PLATEAU_TOLERANCE
    struct Plateau { double fraction; double cycles; bool ok; };
    std::vector<Plateau> plateaus;
    bool all_ok = true;
    for (int m : MISS_COUNTS) {
        std::vector<const Result*> rs;
        for (const auto& r : results) {
            if (r.misses == m) rs.push_back(&r);
        }
        const Result* best = *std::min_element(
            rs.begin(), rs.end(), [](const Result* a, const Result* b) {
                return a->cycles_per_request < b->cycles_per_request;
            });
        bool ok = false;
        if (rs.size() >= 2) {
            const double a = rs[rs.size() - 1]->cycles_per_request;
            const double b = rs[rs.size() - 2]->cycles_per_request;
            ok = std::fabs(a - b) <= PLATEAU_TOLERANCE * std::max(a, b);
        }
        if (!ok) {
            all_ok = false;
            fprintf(stderr, "WARNING: misses %d/%d has no plateau: the two highest warp "
                    "counts differ by more than %.0f%%\n", m, UNROLL,
                    PLATEAU_TOLERANCE * 100);
        }
        plateaus.push_back({static_cast<double>(m) / UNROLL, best->cycles_per_request, ok});
    }

    // least-squares line through the plateaus
    // cycles per request = intercept + slope times miss fraction
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const auto& p : plateaus) {
        sx += p.fraction; sy += p.cycles;
        sxx += p.fraction * p.fraction; sxy += p.fraction * p.cycles;
    }
    const double n = static_cast<double>(plateaus.size());
    const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const double intercept = (sy - slope * sx) / n;
    // worst gap between a plateau and the line: how linear the cost really is
    double max_residual = 0;
    for (const auto& p : plateaus) {
        max_residual = std::max(max_residual,
                                std::fabs(p.cycles - (intercept + slope * p.fraction)));
    }
    const double value = slope;

    // SM clock, so cycles can be converted to time
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
         << "  \"benchmark\": \"l1_miss_cost.cu\",\n"
         << "  \"benchmark_version\": \"" << BENCH_VERSION << "\",\n"
         << "  \"parameter\": \"" << PARAMETER << "\",\n"
         << "  \"value\": " << value << ",\n"
         << "  \"unit\": \"extra SM cycles per global load request that misses L1 (L2 hit)\",\n"
         << "  \"plateau_ok\": " << (all_ok ? "true" : "false") << ",\n"
         << "  \"intercept_cycles_per_request\": " << intercept << ",\n"
         << "  \"max_fit_residual_cycles\": " << max_residual << ",\n"
         << "  \"gpu\": \"" << json_escape(gpu) << "\",\n"
         << "  \"architecture\": \"" << architecture_name(prop.major, prop.minor) << "\",\n"
         << "  \"arch\": \"" << arch << "\",\n"
         << "  \"device\": {\n"
         << "    \"name\": \"" << json_escape(gpu_name) << "\",\n"
         << "    \"sms\": " << sms << ",\n"
         << "    \"l2_cache_bytes\": " << prop.l2CacheSize << "\n"
         << "  },\n"
         << "  \"plateaus\": [";
    for (size_t i = 0; i < plateaus.size(); ++i) {
        const Plateau& p = plateaus[i];
        json << (i ? ",\n" : "\n") << "    {\"miss_fraction\": " << p.fraction
             << ", \"cycles_per_request\": " << p.cycles
             << ", \"plateau_ok\": " << (p.ok ? "true" : "false") << "}";
    }
    json << "\n  ],\n  \"configs\": [";
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        json << (i ? ",\n" : "\n") << "    {\"misses_per_group\": " << r.misses
             << ", \"warps_per_sm\": " << r.warps_per_sm
             << ", \"cycles_per_request\": " << r.cycles_per_request
             << ", \"rep_spread\": " << r.spread
             << ", \"sms_measured\": " << r.sms_measured << "}";
    }
    json << "\n  ],\n"
         << "  \"clock\": {\"sm_clock_ghz\": " << sm_clock_ghz << "},\n"
         << "  \"measurement\": {\n"
         << "    \"method\": \"per-SM slice in L1 (hits) and a 1 MB L2-resident array via ld.global.cg (misses), independent coalesced 32-bit loads; per-SM clock64 span; slope of cycles per request against miss fraction\",\n"
         << "    \"slice_bytes_per_sm\": " << slice << ",\n"
         << "    \"miss_bytes\": " << MISS_BYTES << ",\n"
         << "    \"l1_bytes\": " << l1_bytes << ",\n"
         << "    \"sms_used\": " << sms_used << ",\n"
         << "    \"loads_per_warp\": " << ITERATIONS * UNROLL << ",\n"
         << "    \"unroll\": " << UNROLL << ",\n"
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

    CUDA_CHECK(cudaFree(d_hits));
    CUDA_CHECK(cudaFree(d_misses));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaFree(d_timing));
    CUDA_CHECK(cudaFree(d_clock));
    return 0;
}
