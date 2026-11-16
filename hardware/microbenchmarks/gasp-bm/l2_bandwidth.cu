// l2_bandwidth.cu
// Measures l2_bandwidth_bytes_per_cycle: bytes per SM cycle the whole GPU
// reads from L2. Same unit as dram_bandwidth.cu.
//
// The buffer is a fraction of L2, so after warm-up every load is an L2 hit.
// Loads use ld.global.cg (skip L1), and every block reads the whole buffer.
// value = bytes read / (kernel time in ns x SM clock in GHz)
//
// Sweep: buffer of 20-50% of L2 (rate should stay flat) and block counts up
// to the occupancy limit (rate should stop rising). The JSON records both
// checks as working_sets_agree and saturated.
//
// Run: ./l2_bandwidth [--gpu NAME] [--out-dir DIR] [--reps N] [--passes N]
//                     [--l2-cache-size KB]
// Under ncu, use --cache-control none, or ncu flushes L2 between launches.

#include <chrono>

#include <cuda_runtime.h>
#include "common.h"

static const char* PARAMETER = "l2_bandwidth_bytes_per_cycle";

constexpr int THREADS = 256;                    // threads per block
constexpr int UNROLL = 4;                       // independent 16-byte loads in flight per thread
constexpr int VEC_BYTES = 16;                   // one uint4 per load
constexpr int ROW_BYTES = THREADS * VEC_BYTES;  // bytes a block reads per row
constexpr int WARMUP_LAUNCHES = 5;              // untimed launches, at least
constexpr double WARMUP_MS = 2000.0;            // keep the GPU busy so the SM clock settles
constexpr int CONFIG_WARMUP_LAUNCHES = 3;       // untimed launches per config (fills L2)
constexpr int DEFAULT_REPS = 100;
constexpr int DEFAULT_PASSES = 4;               // times each block reads the whole buffer
constexpr int MIN_TOTAL_PASSES = 100;           // total reads per launch, in buffers
constexpr int PROFILE_PCT = 40;                 // buffer size (% of L2) used for the result
constexpr double AGREE_TOLERANCE = 0.05;
constexpr uint64_t DATA_SEED = 0x2545F4914F6CDD1DULL;

// ts[4*block + 0..3] = {start ns, end ns, start clock, end clock}
// "rows" and every block's start row are multiples of UNROLL, so a group of
// UNROLL rows never wraps around the end of the buffer.
__global__ void __launch_bounds__(THREADS)
l2_read_kernel(const uint4* buf, uint32_t rows, uint32_t passes, uint64_t* ts, uint32_t* sink) {
    // Each block starts at a different row.
    const uint32_t start = (uint64_t)blockIdx.x * rows / gridDim.x / UNROLL * UNROLL;

    uint64_t c0 = 0, t0 = 0;
    if (threadIdx.x == 0) { c0 = read_clock64(); t0 = read_globaltimer(); }

    uint32_t sum = 0;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        uint32_t row = start;
        for (uint32_t done = 0; done < rows; done += UNROLL) {
            uint4 v[UNROLL];

            // Send UNROLL independent loads. "asm volatile" stops the compiler
            // from removing or merging them.
#pragma unroll
            for (int k = 0; k < UNROLL; ++k) {
                const uint4* addr = buf + (size_t)(row + k) * THREADS + threadIdx.x;
                asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(v[k].x), "=r"(v[k].y), "=r"(v[k].z), "=r"(v[k].w)
                             : "l"(addr));
            }
            // Use the results so the loads are not dead code.
#pragma unroll
            for (int k = 0; k < UNROLL; ++k) sum ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;

            row += UNROLL;
            if (row >= rows) row = 0;
        }
    }

    __syncthreads();   // the block ends when its slowest thread ends
    if (threadIdx.x == 0) {
        ts[4 * blockIdx.x + 0] = t0;
        ts[4 * blockIdx.x + 1] = read_globaltimer();
        ts[4 * blockIdx.x + 2] = c0;
        ts[4 * blockIdx.x + 3] = read_clock64();
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

struct Result {
    int pct;                 // buffer size, % of L2
    size_t buffer_bytes;
    int blocks;
    uint32_t passes;
    double bytes_per_cycle;  // median over reps
    double gb_per_s;
    double sm_clock_ghz;
    double spread;           // (max - min) / median over reps
};

// True if a and b are within AGREE_TOLERANCE of each other.
static bool agree(double a, double b) {
    return std::fabs(a - b) <= AGREE_TOLERANCE * std::max(a, b);
}

static const Result* find(const std::vector<Result>& results, int pct, int blocks) {
    for (const Result& r : results)
        if (r.pct == pct && r.blocks == blocks) return &r;
    return nullptr;
}

int main(int argc, char** argv) {
    // ---- Command line ----
    std::string out_dir = "./hardware/runs";
    std::string gpu_override;
    int reps = DEFAULT_REPS;
    int passes_arg = DEFAULT_PASSES;
    long l2_cache_kb = 0;   // 0 = use the size the GPU reports
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--gpu") && i + 1 < argc) gpu_override = argv[++i];
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--passes") && i + 1 < argc) passes_arg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--l2-cache-size") && i + 1 < argc) l2_cache_kb = atol(argv[++i]);
        else {
            fprintf(stderr, "Usage: %s [--gpu NAME] [--out-dir DIR] [--reps N] "
                            "[--passes N] [--l2-cache-size KB]\n", argv[0]);
            return 1;
        }
    }
    if (reps < 10) { fprintf(stderr, "--reps must be at least 10\n"); return 1; }
    if (passes_arg < 1) { fprintf(stderr, "--passes must be at least 1\n"); return 1; }
    if (l2_cache_kb < 0) { fprintf(stderr, "--l2-cache-size must be positive (KB)\n"); return 1; }

    // ---- GPU info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const std::string gpu_name = prop.name;
    const std::string gpu = gpu_override.empty() ? short_gpu_name(gpu_name) : gpu_override;
    const std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);
    const int sms = prop.multiProcessorCount;

    // L2 size to use: the one the user gave, or the one the GPU reports.
    const size_t l2_bytes = l2_cache_kb ? (size_t)l2_cache_kb * 1024 : (size_t)prop.l2CacheSize;
    if (l2_bytes == 0) {
        fprintf(stderr, "the GPU reports no L2 size; pass --l2-cache-size KB\n");
        return 1;
    }

    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, l2_read_kernel, THREADS, 0));
    if (blocks_per_sm < 1) { fprintf(stderr, "kernel cannot launch on this GPU\n"); return 1; }

    // Block counts to test: a quarter and a half of the SMs, then 1, 2, 4...
    // blocks per SM up to the occupancy limit.
    std::vector<int> block_counts;
    if (sms / 4 >= 1) block_counts.push_back(sms / 4);
    if (sms / 2 >= 1) block_counts.push_back(sms / 2);
    for (int b = 1; b <= blocks_per_sm; b *= 2) block_counts.push_back(b * sms);
    const int max_blocks = block_counts.back();

    // Buffer sizes to test, in % of L2, rounded down to a whole number of rows groups.
    const std::vector<int> pcts = {20, 30, 40, 50};
    const size_t granule = (size_t)ROW_BYTES * UNROLL;
    auto buffer_size = [&](int pct) { return (l2_bytes * pct / 100) / granule * granule; };
    if (buffer_size(pcts.front()) == 0) {
        fprintf(stderr, "L2 (%zu bytes) is too small for the sweep\n", l2_bytes);
        return 1;
    }

    fprintf(stderr, "GPU: %s (%s), SMs: %d, L2: %.1f MB (%s), up to %d blocks of %d threads\n",
            gpu_name.c_str(), arch.c_str(), sms, l2_bytes / (1024.0 * 1024.0),
            l2_cache_kb ? "user-supplied" : "device-reported", max_blocks, THREADS);

    // ---- Memory ----
    const size_t max_buffer_bytes = buffer_size(pcts.back());
    uint8_t* d_buf = nullptr;
    uint64_t* d_ts = nullptr;
    uint32_t* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, max_buffer_bytes));
    CUDA_CHECK(cudaMalloc(&d_ts, (size_t)max_blocks * 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sink, (size_t)max_blocks * THREADS * sizeof(uint32_t)));

    // Fill with random data (the memory system may compress repeated values).
    fill_random_kernel<<<sms, THREADS>>>((uint64_t*)d_buf, max_buffer_bytes / 8, DATA_SEED);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Passes per block: enough that every launch reads at least MIN_TOTAL_PASSES buffers.
    auto passes_for = [&](int blocks) {
        const int needed = (MIN_TOTAL_PASSES + blocks - 1) / blocks;
        return (uint32_t)std::max(passes_arg, needed);
    };
    auto launch = [&](int pct, int blocks) {
        const uint32_t rows = buffer_size(pct) / ROW_BYTES;
        l2_read_kernel<<<blocks, THREADS>>>((const uint4*)d_buf, rows, passes_for(blocks), d_ts, d_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // ---- Warm-up ----
    // The SM clock needs a few hundred ms to ramp up from idle, so keep the GPU busy.
    const auto warmup_start = std::chrono::steady_clock::now();
    int warmup_done = 0;
    double elapsed_ms = 0.0;
    do {
        launch(PROFILE_PCT, max_blocks);
        ++warmup_done;
        elapsed_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - warmup_start).count();
    } while (warmup_done < WARMUP_LAUNCHES || elapsed_ms < WARMUP_MS);

    // ---- Measure one config (buffer size and block count) ----
    auto measure = [&](int pct, int blocks) -> Result {
        for (int i = 0; i < CONFIG_WARMUP_LAUNCHES; ++i) launch(pct, blocks);

        const size_t buffer_bytes = buffer_size(pct);
        const uint32_t passes = passes_for(blocks);
        const double bytes_read = (double)buffer_bytes * passes * blocks;

        std::vector<uint64_t> h_ts((size_t)blocks * 4);
        std::vector<double> rates, gb_rates, clocks;
        for (int r = 0; r < reps; ++r) {
            launch(pct, blocks);
            CUDA_CHECK(cudaMemcpy(h_ts.data(), d_ts, h_ts.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

            // Kernel time = latest block end - earliest block start.
            // SM clock = clock64 cycles / globaltimer ns, taken from each block.
            uint64_t first_start = UINT64_MAX, last_end = 0;
            std::vector<double> block_ghz;
            for (int b = 0; b < blocks; ++b) {
                const uint64_t ns_start = h_ts[4 * b + 0], ns_end = h_ts[4 * b + 1];
                const uint64_t cyc_start = h_ts[4 * b + 2], cyc_end = h_ts[4 * b + 3];
                first_start = std::min(first_start, ns_start);
                last_end = std::max(last_end, ns_end);
                if (ns_end > ns_start) block_ghz.push_back((double)(cyc_end - cyc_start) / (double)(ns_end - ns_start));
            }
            if (last_end <= first_start || block_ghz.empty()) {
                fprintf(stderr, "bad timestamps (buffer %d%% of L2, %d blocks, rep %d)\n", pct, blocks, r);
                exit(EXIT_FAILURE);
            }

            const double span_ns = (double)(last_end - first_start);
            const double ghz = median_of(block_ghz);
            rates.push_back(bytes_read / (span_ns * ghz));
            gb_rates.push_back(bytes_read / span_ns);
            clocks.push_back(ghz);
        }

        const double median = median_of(rates);
        const double lo = *std::min_element(rates.begin(), rates.end());
        const double hi = *std::max_element(rates.begin(), rates.end());
        fprintf(stderr, "buffer %2d%% of L2, %4d blocks: %.2f bytes/cycle (%.1f GB/s)\n",
                pct, blocks, median, median_of(gb_rates));
        return {pct, buffer_bytes, blocks, passes, median, median_of(gb_rates),
                median_of(clocks), median > 0 ? (hi - lo) / median : 0.0};
    };

    // ---- Run the sweep ----
    // Every buffer size with the most blocks, then the other block counts at PROFILE_PCT.
    std::vector<Result> results;
    for (int pct : pcts) results.push_back(measure(pct, max_blocks));
    for (int blocks : block_counts)
        if (blocks != max_blocks) results.push_back(measure(PROFILE_PCT, blocks));

    // ---- Result and checks ----
    const Result* main_result = find(results, PROFILE_PCT, max_blocks);
    const double value = main_result->bytes_per_cycle;

    // Does the rate stay the same at 30% and 50% of L2?
    const bool working_sets_agree = agree(value, find(results, 30, max_blocks)->bytes_per_cycle) &&
                                    agree(value, find(results, 50, max_blocks)->bytes_per_cycle);

    // Does the rate stay the same with the next smaller block count?
    bool saturated = false;
    if (block_counts.size() >= 2) {
        const int previous = block_counts[block_counts.size() - 2];
        saturated = agree(value, find(results, PROFILE_PCT, previous)->bytes_per_cycle);
    }

    if (!working_sets_agree)
        fprintf(stderr, "WARNING: the rate at 30%% or 50%% of L2 differs from the rate at %d%% by more "
                        "than %.0f%%. The data may be getting evicted from L2.\n",
                PROFILE_PCT, AGREE_TOLERANCE * 100);
    if (!saturated)
        fprintf(stderr, "WARNING: the rate is still rising with the number of blocks, so this is not a "
                        "whole-GPU limit (it may be per SM).\n");

    int cuda_runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime));

    // ---- Save JSON ----
    std::ostringstream j;
    j.setf(std::ios::fixed);
    j.precision(4);
    j << "{\n"
      << "  \"benchmark\": \"l2_bandwidth.cu\",\n"
      << "  \"parameter\": \"" << PARAMETER << "\",\n"
      << "  \"value\": " << value << ",\n"
      << "  \"unit\": \"bytes/cycle\",\n"
      << "  \"working_sets_agree\": " << (working_sets_agree ? "true" : "false") << ",\n"
      << "  \"saturated\": " << (saturated ? "true" : "false") << ",\n"
      << "  \"gpu\": \"" << json_escape(gpu) << "\",\n"
      << "  \"gpu_name\": \"" << json_escape(gpu_name) << "\",\n"
      << "  \"arch\": \"" << arch << "\",\n"
      << "  \"num_sms\": " << sms << ",\n"
      << "  \"l2_cache_bytes\": " << prop.l2CacheSize << ",\n"
      << "  \"l2_cache_kb_assumed\": " << l2_bytes / 1024 << ",\n"
      << "  \"reps\": " << reps << ",\n"
      << "  \"configs\": [";
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        j << (i ? ",\n" : "\n") << "    {\"working_set_pct_of_l2\": " << r.pct
          << ", \"working_set_bytes\": " << r.buffer_bytes
          << ", \"blocks\": " << r.blocks
          << ", \"passes_per_block\": " << r.passes
          << ", \"bytes_per_cycle\": " << r.bytes_per_cycle
          << ", \"bytes_per_cycle_per_block\": " << r.bytes_per_cycle / r.blocks
          << ", \"gb_per_s\": " << r.gb_per_s
          << ", \"sm_clock_ghz\": " << r.sm_clock_ghz
          << ", \"rep_spread\": " << r.spread << "}";
    }
    j << "\n  ],\n"
      << "  \"clock\": {\"sm_clock_ghz\": " << main_result->sm_clock_ghz << "},\n"
      << "  \"cuda_version\": \"" << cuda_version_string(cuda_runtime) << "\",\n"
      << "  \"timestamp\": \"" << utc_timestamp() << "\"\n"
      << "}\n";

    const std::string out_path = write_run_json(out_dir, PARAMETER, gpu, j.str());
    if (out_path.empty()) return 1;

    printf("%s", j.str().c_str());
    fprintf(stderr, "\nL2 bandwidth: %.2f bytes/cycle (%.1f GB/s at %.3f GHz)\nSaved: %s\n",
            value, main_result->gb_per_s, main_result->sm_clock_ghz, out_path.c_str());

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_ts));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}

