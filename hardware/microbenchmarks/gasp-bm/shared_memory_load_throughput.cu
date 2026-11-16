// shared_memory_load_throughput.cu
// Measures shared_mem_load_throughput_cycles: SM cycles per warp shared-memory
// load when the shared-memory pipe is busy.
//
// Every warp issues independent, conflict-free loads (lane L reads word
// slot*32 + L, so the 32 lanes use 32 banks, and no address depends on a loaded
// value). The latency chain is in shared_memory_load.cu. Here we time the whole
// SM with clock64 around the loop:
//     cycles per load = (span - overhead) / (warps on the SM x loads per warp)
// The overhead is the span of the same kernel with an empty loop.
//
// Sweep: warps per SM from 1 up to the maximum. One warp may be limited by its
// own issue rate, and the pipe limit shows up as a plateau. A second shape mixes
// 10 loads with 1 store per group (the tiled-matmul ratio) to see whether loads
// and stores share the pipe.
//
// The result is the plateau of the loads-only shape. A group is 16 loads, 8 XORs
// and the loop counter, so the value is an upper bound on the pipe's own cost
// per load. Every configuration is in the JSON.
//
// Run: ./shared_memory_load_throughput [--reps N] [--gpu NAME] [--out-dir DIR]

#include <map>

#include <cuda_runtime.h>
#include "common.h"

constexpr int WARP_THREADS = 32;
constexpr int LOAD_SLOTS = 16;         // rows of 32 words that every warp reads
constexpr int MAX_BLOCK_WARPS = 32;    // warps per block, at most (1024 threads)
constexpr int LOADS = 16;              // loads per group, loads-only shape
constexpr int MIX_LOADS = 10;          // loads per group, mixed shape
constexpr int MIX_STORES = 1;          // stores per group, mixed shape
constexpr int ITERATIONS = 256;        // timed groups per warp
constexpr int ACCS = 4;                // XOR accumulators the loaded values go into
constexpr int WARMUP_LAUNCHES = 3;
constexpr int DEFAULT_REPS = 200;
constexpr double PLATEAU_TOLERANCE = 0.02;
constexpr int TIMING_FIELDS = 4;       // start, end, SM id, warps

static_assert(LOADS <= LOAD_SLOTS && MIX_LOADS <= LOAD_SLOTS, "each load of a group reads its own slot");

static const char* PARAMETER = "shared_mem_load_throughput_cycles";

// "volatile" stops the compiler from merging repeated loads of one address.
__device__ __forceinline__ uint32_t load_shared_volatile(uint32_t address) {
    uint32_t value;
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(value) : "r"(address) : "memory");
    return value;
}

// N_LOADS loads and N_STORES stores per group. N_LOADS = 0 is the empty loop,
// which gives the overhead (clock reads and barriers).
template <int N_LOADS, int N_STORES>
__global__ void shared_load_kernel(uint32_t* sink, unsigned long long* timing) {
    __shared__ uint32_t load_words[LOAD_SLOTS * WARP_THREADS];
    __shared__ uint32_t store_words[MAX_BLOCK_WARPS * WARP_THREADS];

    const uint32_t lane = threadIdx.x % WARP_THREADS;
    const uint32_t warp = threadIdx.x / WARP_THREADS;
    const uint32_t warps = blockDim.x / WARP_THREADS;
    const uint32_t load_base = (uint32_t)__cvta_generic_to_shared(load_words) + lane * 4;
    const uint32_t store_address =
        (uint32_t)__cvta_generic_to_shared(store_words) + (warp * WARP_THREADS + lane) * 4;

    // Fill the words. Every warp writes the same value to a word, so the race is harmless.
    for (int slot = 0; slot < LOAD_SLOTS; ++slot) {
        load_words[slot * WARP_THREADS + lane] = (slot * WARP_THREADS + lane) * 2654435761u;
    }
    store_words[warp * WARP_THREADS + lane] = lane;
    __syncthreads();

    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

    // Several accumulators, so the XORs form short chains and not one long one.
    uint32_t acc[ACCS] = {};
#pragma unroll 1
    for (int it = 0; it < ITERATIONS; ++it) {
        const uint32_t earlier = acc[0];
#pragma unroll
        for (int u = 0; u < N_LOADS; ++u) {
            acc[u % ACCS] ^= load_shared_volatile(load_base + u * WARP_THREADS * 4);
        }
#pragma unroll
        for (int s = 0; s < N_STORES; ++s) {
            // Store a value from an earlier group, so the stores don't wait on these loads.
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
    if (total == 0x9e3779b9u) sink[0] = total;   // use the result so the loads are not removed
}

struct Shape {
    int loads;    // per group
    int stores;   // per group
    std::string name() const {
        if (loads == 0) return "empty";
        return stores ? std::to_string(loads) + "loads_" + std::to_string(stores) + "store" : "loads_only";
    }
};

struct Result {
    Shape shape;
    int warps_per_sm;
    double span_cycles;         // median over reps of the median over SMs
    double overhead_cycles;     // the empty loop's span at this warp count
    double cycles_per_load;     // (span - overhead) / loads on the SM
    double cycles_per_access;   // the same, over loads + stores
    double spread;              // (max - min) / median of the per-rep spans
    int sms_measured;
};

struct Span {
    double cycles;
    double warps_on_sm;
    double spread;
    int sms_measured;
};

static void launch_shape(const Shape& s, int grid, int threads, uint32_t* sink, unsigned long long* timing) {
    if (s.loads == 0) shared_load_kernel<0, 0><<<grid, threads>>>(sink, timing);
    else if (s.stores == 0) shared_load_kernel<LOADS, 0><<<grid, threads>>>(sink, timing);
    else shared_load_kernel<MIX_LOADS, MIX_STORES><<<grid, threads>>>(sink, timing);
    CUDA_CHECK(cudaGetLastError());
}

// Run one shape at one warp count. Returns the median over reps of the median
// per-SM span (latest block end - earliest block start on that SM).
static Span measure(const Shape& shape, int sms, int warps_per_sm, int reps,
                    uint32_t* d_sink, unsigned long long* d_timing) {
    // Above 32 warps per SM, use two blocks per SM.
    const int blocks_per_sm = (warps_per_sm + MAX_BLOCK_WARPS - 1) / MAX_BLOCK_WARPS;
    const int warps_per_block = warps_per_sm / blocks_per_sm;
    const int grid = sms * blocks_per_sm;
    const int threads = warps_per_block * WARP_THREADS;

    std::vector<unsigned long long> timing(grid * TIMING_FIELDS);
    for (int i = 0; i < WARMUP_LAUNCHES; ++i) launch_shape(shape, grid, threads, d_sink, d_timing);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> spans, warps_seen;
    int sms_measured = 0;
    for (int rep = 0; rep < reps; ++rep) {
        launch_shape(shape, grid, threads, d_sink, d_timing);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(timing.data(), d_timing, timing.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));

        // Per SM: {earliest start, latest end, total warps}.
        std::map<unsigned long long, std::vector<unsigned long long>> per_sm;
        for (int b = 0; b < grid; ++b) {
            const unsigned long long* t = &timing[b * TIMING_FIELDS];
            auto& e = per_sm[t[2]];
            if (e.empty()) e = {t[0], t[1], 0};
            e[0] = std::min(e[0], t[0]);
            e[1] = std::max(e[1], t[1]);
            e[2] += t[3];
        }
        std::vector<double> sm_spans, sm_warps;
        for (const auto& kv : per_sm) {
            const auto& e = kv.second;
            if (e[1] > e[0] && e[2] > 0) {
                sm_spans.push_back((double)(e[1] - e[0]));
                sm_warps.push_back((double)e[2]);
            }
        }
        sms_measured = (int)sm_spans.size();
        spans.push_back(median_of(sm_spans));
        warps_seen.push_back(median_of(sm_warps));
    }
    const double m = median_of(spans);
    const double lo = *std::min_element(spans.begin(), spans.end());
    const double hi = *std::max_element(spans.begin(), spans.end());
    return {m, median_of(warps_seen), m > 0 ? (hi - lo) / m : 0.0, sms_measured};
}

struct Plateau {
    double per_load;
    double per_access;
    bool ok;
};

int main(int argc, char** argv) {
    // ---- Command line ----
    int reps = DEFAULT_REPS;
    std::string gpu_override;
    std::string out_dir = "hardware/runs";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (arg == "--gpu" && i + 1 < argc) gpu_override = argv[++i];
        else if (arg == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else {
            fprintf(stderr, "Usage: %s [--reps N] [--gpu NAME] [--out-dir DIR]\n", argv[0]);
            return 1;
        }
    }
    if (reps < 10) { fprintf(stderr, "--reps must be at least 10\n"); return 1; }

    // ---- GPU info ----
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const int sms = prop.multiProcessorCount;
    const int max_warps = prop.maxThreadsPerMultiProcessor / WARP_THREADS;
    const int max_blocks_per_sm = (max_warps + MAX_BLOCK_WARPS - 1) / MAX_BLOCK_WARPS;
    const int max_grid = sms * max_blocks_per_sm;

    const std::string gpu_name = prop.name;
    const std::string gpu = gpu_override.empty() ? short_gpu_name(gpu_name) : gpu_override;
    const std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);

    // Warps per SM to test: 1, 2, 4, ... and the maximum.
    std::vector<int> warp_counts;
    for (int w = 1; w <= max_warps; w *= 2) warp_counts.push_back(w);
    if (warp_counts.back() < max_warps) warp_counts.push_back(max_warps);

    const Shape empty{0, 0};
    const std::vector<Shape> shapes = {{LOADS, 0}, {MIX_LOADS, MIX_STORES}};

    // ---- Memory ----
    uint32_t* d_sink = nullptr;
    unsigned long long* d_timing = nullptr;
    unsigned long long* d_clock = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_timing, max_grid * TIMING_FIELDS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_clock, 2 * sizeof(unsigned long long)));

    // ---- Overhead: the empty loop at each warp count ----
    std::map<int, double> overhead;
    for (int warps : warp_counts) overhead[warps] = measure(empty, sms, warps, reps, d_sink, d_timing).cycles;

    // ---- Sweep: every shape at every warp count ----
    std::vector<Result> results;
    for (const Shape& shape : shapes) {
        for (int warps : warp_counts) {
            const Span span = measure(shape, sms, warps, reps, d_sink, d_timing);
            const double net = span.cycles - overhead[warps];
            const double loads = span.warps_on_sm * ITERATIONS * shape.loads;
            const double accesses = span.warps_on_sm * ITERATIONS * (shape.loads + shape.stores);
            results.push_back({shape, warps, span.cycles, overhead[warps],
                               loads > 0 ? net / loads : 0.0,
                               accesses > 0 ? net / accesses : 0.0,
                               span.spread, span.sms_measured});
            fprintf(stderr, "%-16s warps/SM %2d: %.4f cycles/load, %.4f cycles/access\n",
                    shape.name().c_str(), warps, results.back().cycles_per_load,
                    results.back().cycles_per_access);
        }
    }

    // ---- Plateau of each shape ----
    // The plateau is the lowest cycles per access. It is accepted when the two
    // highest warp counts agree within PLATEAU_TOLERANCE. The results of one
    // shape are next to each other, one per warp count.
    const size_t n = warp_counts.size();
    std::vector<Plateau> plateaus;
    for (size_t s = 0; s < shapes.size(); ++s) {
        const Result* best = &results[s * n];
        for (size_t i = s * n; i < (s + 1) * n; ++i)
            if (results[i].cycles_per_access < best->cycles_per_access) best = &results[i];

        bool ok = false;
        if (n >= 2) {
            const double a = results[(s + 1) * n - 1].cycles_per_access;
            const double b = results[(s + 1) * n - 2].cycles_per_access;
            ok = std::fabs(a - b) <= PLATEAU_TOLERANCE * std::max(a, b);
        }
        plateaus.push_back({best->cycles_per_load, best->cycles_per_access, ok});
        if (!ok)
            fprintf(stderr, "WARNING: %s has no plateau: the two highest warp counts differ by more than %.0f%%\n",
                    shapes[s].name().c_str(), PLATEAU_TOLERANCE * 100);
    }
    const Plateau& main_plateau = plateaus[0];   // the loads-only shape
    const double value = main_plateau.per_load;

    // ---- SM clock speed ----
    double sm_clock_ghz = 0.0;
    clock_kernel<<<1, 1>>>(d_clock);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long c[2];
    CUDA_CHECK(cudaMemcpy(c, d_clock, sizeof(c), cudaMemcpyDeviceToHost));
    if (c[1] > 0) sm_clock_ghz = (double)c[0] / c[1];

    int cuda_runtime = 0, driver_api = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver_api));

    // ---- Save JSON ----
    std::ostringstream json;
    json.setf(std::ios::fixed);
    json.precision(4);
    json << "{\n"
         << "  \"benchmark\": \"shared_memory_load_throughput.cu\",\n"
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
         << "    \"max_warps_per_sm\": " << max_warps << "\n"
         << "  },\n"
         << "  \"groups_per_warp\": " << ITERATIONS << ",\n"
         << "  \"loads_per_group\": " << LOADS << ",\n"
         << "  \"mixed_loads_per_group\": " << MIX_LOADS << ",\n"
         << "  \"mixed_stores_per_group\": " << MIX_STORES << ",\n"
         << "  \"max_blocks\": " << max_grid << ",\n"
         << "  \"reps\": " << reps << ",\n"
         << "  \"plateaus\": {";
    for (size_t s = 0; s < shapes.size(); ++s) {
        const Plateau& p = plateaus[s];
        json << (s ? ",\n" : "\n") << "    \"" << shapes[s].name() << "\": {"
             << "\"cycles_per_load\": " << p.per_load
             << ", \"cycles_per_access\": " << p.per_access
             << ", \"plateau_ok\": " << (p.ok ? "true" : "false") << "}";
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
         << "  \"cuda_version\": \"" << cuda_version_string(cuda_runtime) << "\",\n"
         << "  \"driver_api_version\": \"" << cuda_version_string(driver_api) << "\",\n"
         << "  \"timestamp\": \"" << utc_timestamp() << "\"\n"
         << "}\n";

    const std::string output_path = write_run_json(out_dir, PARAMETER, gpu, json.str());
    if (output_path.empty()) return 1;

    printf("%s", json.str().c_str());
    fprintf(stderr, "Saved: %s\n", output_path.c_str());

    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaFree(d_timing));
    CUDA_CHECK(cudaFree(d_clock));
    return 0;
}

// SASS check: cuobjdump -sass shared_memory_load_throughput.cubin
// In the loads-only kernel the loop should have 16 LDS loads per group with a few
// XORs and the loop counter. Under ncu, shared-load bank conflicts should be 0
// and there should be no global traffic in the timed kernel.
