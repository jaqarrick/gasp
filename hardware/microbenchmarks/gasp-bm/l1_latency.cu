// l1_latency.cu
// ---------------------------------------------------------------------------
// Measures : l1_cache_hit_latency_cycles
// Method   : Pointer-chasing (dependent load chain), working set fits in L1.
//            Clock-read overhead is measured in the same kernel and subtracted.
//
// Build    : nvcc -O3 -std=c++17 -arch=sm_80  l1_latency.cu -o l1_latency   (A100)
//            nvcc -O3 -std=c++17 -arch=sm_90  l1_latency.cu -o l1_latency   (H100)
//            nvcc -O3 -std=c++17 -arch=sm_100 l1_latency.cu -o l1_latency   (B200)
//
// Run      : ./l1_latency [--gpu NAME] [--out-dir DIR] [--reps N] [--l1-cache-size KB]
//            Writes  <out-dir>/l1_cache_hit_latency_cycles.<gpu>.run.json
//            Default out-dir = ./hardware/runs   (created if missing)
//            --l1-cache-size KB = the L1 size to assume (default 64). The working
//            set is 1/4 of it (default 16 KB), so it always fits in L1. Use it for
//            hardware with a different L1; CUDA does not report the L1 size.
//            --gpu overrides the short GPU name (default: derived from the
//            device name, e.g. "NVIDIA A100-SXM4-80GB" -> "a100").
//
// Inspect  : nvcc -O3 -arch=sm_80 -cubin l1_latency.cu -o l1_latency.cubin
//            cuobjdump -sass l1_latency.cubin     (checklist at bottom)
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "common.h"

// ---- Identity of this benchmark -------------------------------------------
static const char* PARAMETER      = "l1_cache_hit_latency_cycles";
static const char* SCHEMA_VERSION = "1.2.0";   // bump if the JSON layout changes
static const char* BENCH_VERSION  = "1.3.0";   // bump if the measurement method changes

// ---- Experiment parameters -------------------------------------------------
// The working set is 1/4 of the assumed L1 size, so after warm-up every access
// in the timed region should be an L1 hit. The default assumed size (64 KB) is
// deliberately on the small side: that keeps the 16 KB working set inside L1
// even on GPUs whose L1 is partly given over to shared memory.
constexpr int DEFAULT_L1_CACHE_KB = 64;
constexpr int L1_FRACTION_DIV     = 4;      // working set = L1 size / 4
constexpr int LINE_BYTES          = 128;
constexpr int CHASE_LOADS       = 1024;   // dependent loads in the timed region
constexpr int UNROLL            = 16;     // loads per loop iteration
constexpr int DEFAULT_REPS      = 200;    // independent kernel launches (>=100)

static_assert(CHASE_LOADS % UNROLL == 0, "CHASE_LOADS must be a multiple of UNROLL");

// ---- Device ----------------------------------------------------------------

// Dependent load: the same register is address and destination, so there is
// no address arithmetic between loads. ".ca" = allocate in L1.
#define CHASE_ONE asm volatile("ld.global.ca.u64 %0, [%0];" : "+l"(p)::"memory");

// out[0] = gross cycles of the timed chain (clock overhead included)
// out[1] = clock overhead (two back-to-back clock reads, nothing between)
// out[2] = globaltimer ns around the timed chain (read outside the clock64 pair)
__global__ void l1_chase_kernel(uint64_t start, int num_nodes, long long* out, uint64_t* sink) {
    uint64_t p = start;

    // 1) Warm-up: two laps so every line is resident in L1.
#pragma unroll 1
    for (int i = 0; i < 2 * num_nodes; ++i) {
        CHASE_ONE
    }

    // Depends on p, so the clock reads below cannot issue before the last
    // warm-up load has returned.
    if (p == 0) {
        *sink = 0;
        return;
    }

    // 2) Baseline: cost of reading the clock itself.
    uint64_t b0 = read_clock64();
    uint64_t b1 = read_clock64();

    // 3) Timed region: CHASE_LOADS strictly dependent L1-hit loads.
    uint64_t g0 = read_globaltimer();
    uint64_t t0 = read_clock64();
#pragma unroll 1
    for (int i = 0; i < CHASE_LOADS; i += UNROLL) {
        CHASE_ONE CHASE_ONE CHASE_ONE CHASE_ONE
        CHASE_ONE CHASE_ONE CHASE_ONE CHASE_ONE
        CHASE_ONE CHASE_ONE CHASE_ONE CHASE_ONE
        CHASE_ONE CHASE_ONE CHASE_ONE CHASE_ONE
    }
    uint64_t t1 = read_clock64();
    uint64_t g1 = read_globaltimer();

    out[0] = static_cast<long long>(t1 - t0);
    out[1] = static_cast<long long>(b1 - b0);
    out[2] = static_cast<long long>(g1 - g0);
    *sink  = p;  // keep the chain observable
}

// ---- Main ------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string out_dir = "./hardware/runs";
    std::string gpu_override;
    int reps = DEFAULT_REPS;
    long l1_cache_kb = DEFAULT_L1_CACHE_KB;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--gpu") && i + 1 < argc) gpu_override = argv[++i];
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--l1-cache-size") && i + 1 < argc) l1_cache_kb = atol(argv[++i]);
        else {
            fprintf(stderr, "Usage: %s [--gpu NAME] [--out-dir DIR] [--reps N] [--l1-cache-size KB]\n", argv[0]);
            return 1;
        }
    }
    if (reps < 100) { fprintf(stderr, "reps must be >= 100 (issue #3)\n"); return 1; }
    if (l1_cache_kb < 8) { fprintf(stderr, "--l1-cache-size must be >= 8 (KB)\n"); return 1; }

    // Working set = 1/4 of the assumed L1 size (default 64 KB -> 16 KB).
    const size_t working_set_bytes = static_cast<size_t>(l1_cache_kb) * 1024 / L1_FRACTION_DIV;
    const int num_nodes = static_cast<int>(working_set_bytes / LINE_BYTES);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const std::string gpu_name = prop.name;
    const std::string gpu = gpu_override.empty() ? short_gpu_name(gpu_name) : gpu_override;
    const std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);
    fprintf(stderr, "GPU: %s (%s), SMs: %d, assumed L1: %ld KB, working set: %zu KB\n",
            gpu_name.c_str(), arch.c_str(), prop.multiProcessorCount, l1_cache_kb,
            working_set_bytes / 1024);

    int rt_ver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt_ver));

    // Pointer ring: node i's first 8 bytes hold the device address of the next
    // node. Sattolo's shuffle gives ONE cycle covering all nodes in random order.
    uint8_t* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, working_set_bytes));

    std::vector<int> next(num_nodes);
    {
        std::vector<int> perm(num_nodes);
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937 rng(12345);
        for (int i = num_nodes - 1; i > 0; --i) {
            std::uniform_int_distribution<int> d(0, i - 1);
            std::swap(perm[i], perm[d(rng)]);
        }
        for (int i = 0; i < num_nodes; ++i) next[perm[i]] = perm[(i + 1) % num_nodes];
    }
    std::vector<uint8_t> h_buf(working_set_bytes, 0);
    for (int i = 0; i < num_nodes; ++i) {
        uint64_t a = reinterpret_cast<uint64_t>(d_buf) + static_cast<uint64_t>(next[i]) * LINE_BYTES;
        *reinterpret_cast<uint64_t*>(&h_buf[static_cast<size_t>(i) * LINE_BYTES]) = a;
    }
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf.data(), working_set_bytes, cudaMemcpyHostToDevice));

    long long* d_out;
    uint64_t*  d_sink;
    CUDA_CHECK(cudaMalloc(&d_out, 3 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));

    std::vector<long long> raw_cycles, overheads;
    double sum_clock_cycles = 0.0, sum_clock_ns = 0.0;  // for the SM clock
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaMemset(d_out, 0, 3 * sizeof(long long)));
        l1_chase_kernel<<<1, 1>>>(reinterpret_cast<uint64_t>(d_buf), num_nodes, d_out, d_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        long long h[3];
        CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
        raw_cycles.push_back(h[0]);
        overheads.push_back(h[1]);
        if (h[2] > 0) {
            sum_clock_cycles += static_cast<double>(h[0]);
            sum_clock_ns += static_cast<double>(h[2]);
        }
    }

    // Overhead = median of the baseline measurements; subtract from every rep.
    std::vector<double> ov(overheads.begin(), overheads.end());
    const double overhead = median_of(ov);

    // Per-run breakdown: gross (clock overhead included) -> net (overhead removed) -> per load.
    std::vector<double> net_cycles, per_load, gross_per_load;
    for (long long c : raw_cycles) {
        net_cycles.push_back(static_cast<double>(c) - overhead);
        per_load.push_back((static_cast<double>(c) - overhead) / CHASE_LOADS);
        gross_per_load.push_back(static_cast<double>(c) / CHASE_LOADS);
    }
    const double uncorrected_value = median_of(gross_per_load);  // no subtraction

    const Stats stats = compute_stats(per_load);
    const double median = stats.median;
    if (sum_clock_ns <= 0.0) { fprintf(stderr, "ERROR: globaltimer did not advance\n"); return 1; }
    // SM clock over all reps: total cycles / total ns (globaltimer updates every ~1 us).
    const double clock_ghz_all = sum_clock_cycles / sum_clock_ns;

    if (median > 80.0)
        fprintf(stderr, "WARNING: median %.1f cycles is too high for an L1 hit. "
                        "Loads may be bypassing L1 (check SASS).\n", median);

    // ---- Build JSON --------------------------------------------------------
    std::string driver = driver_version();
    std::string commit = git_commit();

    std::ostringstream j;
    j.setf(std::ios::fixed);
    j.precision(4);
    j << "{\n";
    j << "  \"schema_version\": \"" << SCHEMA_VERSION << "\",\n";
    j << "  \"benchmark\": \"l1_latency.cu\",\n";
    j << "  \"benchmark_version\": \"" << BENCH_VERSION << "\",\n";
    j << "  \"parameter\": \"" << PARAMETER << "\",\n";
    j << "  \"value\": " << median << ",\n";
    j << "  \"unit\": \"cycles\",\n";
    j << "  \"gpu\": \"" << json_escape(gpu) << "\",\n";
    j << "  \"gpu_name\": \"" << json_escape(gpu_name) << "\",\n";
    j << "  \"arch\": \"" << arch << "\",\n";
    j << "  \"num_sms\": " << prop.multiProcessorCount << ",\n";
    j << json_stats(stats);
    // raw_cycles           : per run, clock64 delta around the chain, clock-read cost INCLUDED
    // overhead_samples     : per run, clock64 delta of two back-to-back reads (cost of the clock)
    // overhead_cycles      : median of overhead_samples; the value subtracted from every run
    // net_cycles           : raw_cycles - overhead_cycles (per run)
    // per_load_cycles      : net_cycles / loads_per_rep (per run); stats are computed over these
    j << "  \"raw_cycles\": " << json_array(raw_cycles) << ",\n";
    j << "  \"overhead_samples_cycles\": " << json_array(overheads) << ",\n";
    j << "  \"overhead_cycles\": " << overhead << ",\n";
    j << "  \"overhead_subtracted\": true,\n";
    j << "  \"net_cycles\": " << json_array(net_cycles) << ",\n";
    j << "  \"per_load_cycles\": " << json_array(per_load) << ",\n";
    j << "  \"uncorrected_value\": " << uncorrected_value << ",\n";
    // clock.sm_clock_ghz: total clock64 cycles / total globaltimer ns over all reps
    j << "  \"clock\": {\"sm_clock_ghz\": " << clock_ghz_all << "},\n";
    j << "  \"measurement\": {\n";
    j << "    \"method\": \"pointer-chase, working set fits in L1\",\n";
    j << "    \"load_instruction\": \"ld.global.ca.u64\",\n";
    j << "    \"timer\": \"clock64 (cycles), globaltimer (SM clock)\",\n";
    j << "    \"working_set_bytes\": " << working_set_bytes << ",\n";
    j << "    \"l1_cache_size_kb_assumed\": " << l1_cache_kb << ",\n";
    j << "    \"line_bytes\": " << LINE_BYTES << ",\n";
    j << "    \"loads_per_rep\": " << CHASE_LOADS << ",\n";
    j << "    \"reps\": " << reps << ",\n";
    j << "    \"blocks\": 1,\n";
    j << "    \"threads_per_block\": 1\n";
    j << "  },\n";
    j << "  \"cuda_version\": \"" << cuda_version_string(rt_ver) << "\",\n";
    j << "  \"nvcc_version\": \"" << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "\",\n";
    j << "  \"driver_version\": \"" << json_escape(driver) << "\",\n";
    j << "  \"git_commit\": \"" << json_escape(commit) << "\",\n";
    j << "  \"timestamp\": \"" << utc_timestamp() << "\"\n";
    j << "}\n";

    // ---- Save to <out_dir>/{parameter}.{gpu}.run.json -----------------------
    const std::string out_path = write_run_json(out_dir, PARAMETER, gpu, j.str());
    if (out_path.empty()) return 1;

    printf("%s", j.str().c_str());
    fprintf(stderr, "\nL1 hit latency: %.2f cycles (std %.3f, overhead %.1f cycles subtracted)\nSaved: %s\n",
            median, stats.std, overhead, out_path.c_str());

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}

// ---------------------------------------------------------------------------
// SASS inspection checklist (required by issue #11):
//   cuobjdump -sass l1_latency.cubin
// In l1_chase_kernel you should see:
//   * clock reads (CS2R) back-to-back for the baseline, then one before and
//     one after the timed loop, with a globaltimer read (SR_GLOBALTIMER)
//     just outside each of the timed pair
//   * inside the timed loop, 16 consecutive 64-bit LDG instructions where each
//     address register is the previous instruction's destination
//   * no IADD/IMAD/LEA between those loads
// If loads were removed, merged or reordered, the result is invalid.
// ---------------------------------------------------------------------------
