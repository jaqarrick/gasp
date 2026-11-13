// l2_bandwidth.cu: bytes per SM cycle the L2 serves when every load is an L2
// hit, whole GPU. Same unit and timing as dram_bandwidth.cu, so the two values
// compare directly.
//
// The working set is a fraction of L2, so after a warm pass nothing comes from
// DRAM. Two guards keep L1 from serving the loads (the L1s together are close
// to L2 in size):
//   * loads are ld.global.cg (cache at L2 and below, not L1);
//   * every block sweeps the whole working set, from its own starting row, so
//     one SM's traffic is far larger than its L1 even if .cg were ignored.
// Each thread issues UNROLL independent 16-byte loads per group; a warp reads
// 512 contiguous bytes. Addresses never depend on loaded data.
//
//     value = bytes_read / (kernel_span_ns x sm_clock_GHz)
//
// kernel_span_ns = latest block end - earliest block start (globaltimer), and
// the SM clock is clock64 / globaltimer in the same kernel.
//
// Sweep: working set 20, 30, 40, 50% of L2 (the rate should be flat; a drop at
// the top means evictions), and the number of blocks from a quarter of the SMs
// up to the occupancy limit (the rate should stop rising; if it keeps scaling
// with blocks the limit is per SM, not whole-GPU). Design:
// .local/l2_bandwidth.md (handoff I48).
//
// The profile value is the median at 40% of L2 with the most blocks. The JSON
// says whether the neighbouring working sets agree within 5% and whether the
// two highest block counts do.
//
// Usage: l2_bandwidth [--gpu NAME] [--out-dir DIR] [--reps N] [--passes N]
//                     [--l2-cache-size KB]
// Run under ncu with --cache-control none (run_all.py does), or ncu flushes L2
// between launches. Check under ncu: L1 hit rate near 0, L2 hit rate near 100%,
// DRAM throughput near zero, and a value above the DRAM bandwidth.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "common.h"

static const char* PARAMETER = "l2_bandwidth_bytes_per_cycle";
static const char* SCHEMA_VERSION = "1.1.0";
static const char* BENCH_VERSION = "1.0.0";

constexpr int THREADS = 256;            // threads per block
constexpr int UNROLL = 4;               // independent 16-byte loads in flight per thread
constexpr int VEC_BYTES = 16;           // one uint4 per load
constexpr int ROW_BYTES = THREADS * VEC_BYTES;  // what a block reads per row
constexpr int WARMUP_LAUNCHES = 5;      // minimum untimed launches
constexpr double WARMUP_MS = 2000.0;    // keep the GPU busy so the SM clock settles
constexpr int CONFIG_WARMUP_LAUNCHES = 3;  // untimed, per configuration: fills L2
constexpr int DEFAULT_REPS = 100;
constexpr int DEFAULT_PASSES = 4;       // sweeps of the working set per block
constexpr int MIN_TOTAL_PASSES = 100;   // bytes per launch >= this many working sets
constexpr int PROFILE_PCT = 40;         // the working set the profile value is read at
constexpr int MAX_PCT = 60;             // above this, evictions are likely
constexpr double AGREE_TOLERANCE = 0.05;
constexpr uint64_t DATA_SEED = 0x2545F4914F6CDD1DULL;

// ts[4*block + {0,1,2,3}] = {timer_start_ns, timer_end_ns, clock_start, clock_end}
// rows is a multiple of UNROLL, and so is every block's starting row, so a
// group of UNROLL rows never wraps.
__global__ void __launch_bounds__(THREADS)
l2_read_kernel(const uint4* __restrict__ buf, uint32_t rows, uint32_t passes,
               uint64_t* ts, uint32_t* sink) {
    const uint32_t start = static_cast<uint32_t>(
        static_cast<uint64_t>(blockIdx.x) * rows / gridDim.x / UNROLL * UNROLL);

    uint64_t c0 = 0, t0 = 0;
    if (threadIdx.x == 0) { c0 = read_clock64(); t0 = read_globaltimer(); }

    uint32_t acc = 0;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        uint32_t row = start;
        for (uint32_t done = 0; done < rows; done += UNROLL) {
            uint4 v[UNROLL];
#pragma unroll
            for (int k = 0; k < UNROLL; ++k) {
                const uint4* addr =
                    buf + static_cast<size_t>(row + k) * THREADS + threadIdx.x;
                // volatile PTX: the compiler cannot delete or merge these loads
                asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(v[k].x), "=r"(v[k].y), "=r"(v[k].z), "=r"(v[k].w)
                             : "l"(addr));
            }
#pragma unroll
            for (int k = 0; k < UNROLL; ++k) acc ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
            row += UNROLL;
            if (row >= rows) row = 0;
        }
    }

    __syncthreads();  // the block ends when its slowest thread ends
    if (threadIdx.x == 0) {
        ts[4 * blockIdx.x + 0] = t0;
        ts[4 * blockIdx.x + 1] = read_globaltimer();
        ts[4 * blockIdx.x + 2] = c0;
        ts[4 * blockIdx.x + 3] = read_clock64();
    }
    sink[static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = acc;
}

struct Result {
    int pct;                 // working set, % of L2
    size_t working_set_bytes;
    int blocks;
    uint32_t passes;
    double bytes_per_cycle;  // median over reps
    double gb_per_s;
    double sm_clock_ghz;
    double spread;           // (max - min) / median of the per-rep values
};

static bool agree(double a, double b) {
    return std::fabs(a - b) <= AGREE_TOLERANCE * std::max(a, b);
}

int main(int argc, char** argv) {
    std::string out_dir = "./hardware/runs";
    std::string gpu_override;
    int reps = DEFAULT_REPS;
    int passes_arg = DEFAULT_PASSES;
    long l2_cache_kb = 0;  // 0 = the size the GPU reports
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

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const std::string gpu_name = prop.name;
    const std::string gpu = gpu_override.empty() ? short_gpu_name(gpu_name) : gpu_override;
    const std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);
    const int sms = prop.multiProcessorCount;
    const size_t l2_bytes = l2_cache_kb ? static_cast<size_t>(l2_cache_kb) * 1024
                                        : static_cast<size_t>(prop.l2CacheSize);
    if (l2_bytes == 0) {
        fprintf(stderr, "the GPU reports no L2 size; pass --l2-cache-size KB\n");
        return 1;
    }

    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, l2_read_kernel, THREADS, 0));
    if (blocks_per_sm < 1) { fprintf(stderr, "kernel cannot launch on this GPU\n"); return 1; }

    // blocks: a quarter and a half of the SMs, then 1, 2, 4... blocks per SM
    std::vector<int> block_counts;
    for (int divisor : {4, 2}) {
        if (sms / divisor >= 1) block_counts.push_back(sms / divisor);
    }
    for (int b = 1; b <= blocks_per_sm; b *= 2) block_counts.push_back(b * sms);
    const int max_blocks = block_counts.back();

    const std::vector<int> pcts = {20, 30, 40, 50};
    const size_t granule = static_cast<size_t>(ROW_BYTES) * UNROLL;
    auto working_set = [&](int pct) {
        return (l2_bytes * pct / 100) / granule * granule;
    };
    const size_t buffer_bytes = working_set(pcts.back());
    if (working_set(pcts.front()) == 0 || pcts.back() > MAX_PCT) {
        fprintf(stderr, "L2 (%zu bytes) is too small for the sweep\n", l2_bytes);
        return 1;
    }

    fprintf(stderr, "GPU: %s (%s), SMs: %d, L2: %.1f MB (%s), blocks up to %d x %d threads\n",
            gpu_name.c_str(), arch.c_str(), sms, l2_bytes / (1024.0 * 1024.0),
            l2_cache_kb ? "user-supplied" : "device-reported", max_blocks, THREADS);

    uint8_t* d_buf = nullptr;
    uint64_t* d_ts = nullptr;
    uint32_t* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, buffer_bytes));
    CUDA_CHECK(cudaMalloc(&d_ts, static_cast<size_t>(max_blocks) * 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(max_blocks) * THREADS * sizeof(uint32_t)));
    fill_random_kernel<<<sms, THREADS>>>(reinterpret_cast<uint64_t*>(d_buf),
                                         buffer_bytes / 8, DATA_SEED);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const uint4* d_vec = reinterpret_cast<const uint4*>(d_buf);

    auto passes_for = [&](int blocks) {
        const int needed = (MIN_TOTAL_PASSES + blocks - 1) / blocks;
        return static_cast<uint32_t>(std::max(passes_arg, needed));
    };
    auto launch = [&](int pct, int blocks) {
        const uint32_t rows = static_cast<uint32_t>(working_set(pct) / ROW_BYTES);
        l2_read_kernel<<<blocks, THREADS>>>(d_vec, rows, passes_for(blocks), d_ts, d_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // Warm up by time: the SM clock ramps up from idle over several hundred ms.
    int warmup_launches_done = 0;
    {
        const auto t_start = std::chrono::steady_clock::now();
        double elapsed_ms = 0.0;
        do {
            launch(PROFILE_PCT, max_blocks);
            ++warmup_launches_done;
            elapsed_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t_start).count();
        } while (warmup_launches_done < WARMUP_LAUNCHES || elapsed_ms < WARMUP_MS);
    }

    auto measure = [&](int pct, int blocks) -> Result {
        for (int i = 0; i < CONFIG_WARMUP_LAUNCHES; ++i) launch(pct, blocks);
        const size_t bytes = working_set(pct);
        const uint32_t passes = passes_for(blocks);
        const double bytes_read = static_cast<double>(bytes) * passes * blocks;
        std::vector<uint64_t> h_ts(static_cast<size_t>(blocks) * 4);
        std::vector<double> rates, gbs, clocks;
        for (int r = 0; r < reps; ++r) {
            launch(pct, blocks);
            CUDA_CHECK(cudaMemcpy(h_ts.data(), d_ts, h_ts.size() * sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost));
            uint64_t t_first = UINT64_MAX, t_last = 0;
            std::vector<double> ghz_per_block;
            for (int b = 0; b < blocks; ++b) {
                const uint64_t ts0 = h_ts[4 * b + 0], ts1 = h_ts[4 * b + 1];
                const uint64_t cs0 = h_ts[4 * b + 2], cs1 = h_ts[4 * b + 3];
                t_first = std::min(t_first, ts0);
                t_last = std::max(t_last, ts1);
                if (ts1 > ts0) {
                    ghz_per_block.push_back(static_cast<double>(cs1 - cs0) /
                                            static_cast<double>(ts1 - ts0));
                }
            }
            if (t_last <= t_first || ghz_per_block.empty()) {
                fprintf(stderr, "bad timestamps (working set %d%%, %d blocks, rep %d)\n",
                        pct, blocks, r);
                exit(EXIT_FAILURE);
            }
            const double span_ns = static_cast<double>(t_last - t_first);
            const double ghz = median_of(ghz_per_block);
            rates.push_back(bytes_read / (span_ns * ghz));
            gbs.push_back(bytes_read / span_ns);
            clocks.push_back(ghz);
        }
        const double m = median_of(rates);
        const double lo = *std::min_element(rates.begin(), rates.end());
        const double hi = *std::max_element(rates.begin(), rates.end());
        fprintf(stderr, "working set %2d%% of L2, %4d blocks: %.2f bytes/cycle (%.1f GB/s)\n",
                pct, blocks, m, median_of(gbs));
        return {pct, bytes, blocks, passes, m, median_of(gbs), median_of(clocks),
                m > 0 ? (hi - lo) / m : 0.0};
    };

    // every working set at the most blocks, then the other block counts at 40%
    std::vector<Result> results;
    for (int pct : pcts) results.push_back(measure(pct, max_blocks));
    for (int blocks : block_counts) {
        if (blocks != max_blocks) results.push_back(measure(PROFILE_PCT, blocks));
    }

    auto find = [&](int pct, int blocks) -> const Result* {
        for (const auto& r : results) {
            if (r.pct == pct && r.blocks == blocks) return &r;
        }
        return nullptr;
    };
    const Result* main_result = find(PROFILE_PCT, max_blocks);
    const double value = main_result->bytes_per_cycle;
    const bool working_sets_agree =
        agree(value, find(30, max_blocks)->bytes_per_cycle) &&
        agree(value, find(50, max_blocks)->bytes_per_cycle);
    bool saturated = false;
    if (block_counts.size() >= 2) {
        const int previous = block_counts[block_counts.size() - 2];
        saturated = agree(value, find(PROFILE_PCT, previous)->bytes_per_cycle);
    }
    if (!working_sets_agree) {
        fprintf(stderr, "WARNING: the rate at 30%% or 50%% of L2 differs from the rate at "
                        "%d%% by more than %.0f%%; the working set may be evicted.\n",
                PROFILE_PCT, AGREE_TOLERANCE * 100);
    }
    if (!saturated) {
        fprintf(stderr, "WARNING: the rate is still rising with the number of blocks, so "
                        "this is not a whole-GPU plateau (the limit may be per SM).\n");
    }

    int rt_ver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt_ver));

    std::ostringstream j;
    j.setf(std::ios::fixed);
    j.precision(4);
    j << "{\n"
      << "  \"schema_version\": \"" << SCHEMA_VERSION << "\",\n"
      << "  \"benchmark\": \"l2_bandwidth.cu\",\n"
      << "  \"benchmark_version\": \"" << BENCH_VERSION << "\",\n"
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
      << "  \"configs\": [";
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        j << (i ? ",\n" : "\n") << "    {\"working_set_pct_of_l2\": " << r.pct
          << ", \"working_set_bytes\": " << r.working_set_bytes
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
      << "  \"measurement\": {\n"
      << "    \"method\": \"L2-resident working set, every block sweeps all of it from its own row, loads bypass L1\",\n"
      << "    \"load_instruction\": \"ld.global.cg.v4.u32\",\n"
      << "    \"timer\": \"globaltimer (kernel span) and clock64 (SM clock)\",\n"
      << "    \"value_formula\": \"bytes_read / (kernel_span_ns x sm_clock_ghz) at " << PROFILE_PCT
      << "% of L2 and the most blocks\",\n"
      << "    \"l2_cache_bytes_assumed\": " << l2_bytes << ",\n"
      << "    \"l2_cache_size_source\": \"" << (l2_cache_kb ? "user" : "device") << "\",\n"
      << "    \"data_pattern\": \"splitmix64 pseudo-random 64-bit words (fixed seed)\",\n"
      << "    \"bytes_per_load\": " << VEC_BYTES << ",\n"
      << "    \"loads_in_flight_per_thread\": " << UNROLL << ",\n"
      << "    \"threads_per_block\": " << THREADS << ",\n"
      << "    \"max_blocks_per_sm\": " << blocks_per_sm << ",\n"
      << "    \"warmup_launches\": " << warmup_launches_done << ",\n"
      << "    \"reps\": " << reps << "\n"
      << "  },\n"
      << "  \"cuda_version\": \"" << cuda_version_string(rt_ver) << "\",\n"
      << "  \"timestamp\": \"" << utc_timestamp() << "\"\n"
      << "}\n";

    const std::string out_path = write_run_json(out_dir, PARAMETER, gpu, j.str());
    if (out_path.empty()) return 1;

    printf("%s", j.str().c_str());
    fprintf(stderr, "\nL2 bandwidth: %.2f bytes/cycle (%.1f GB/s at %.3f GHz)\nSaved: %s\n",
            value, main_result->gb_per_s, main_result->sm_clock_ghz,
            out_path.c_str());

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_ts));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}
