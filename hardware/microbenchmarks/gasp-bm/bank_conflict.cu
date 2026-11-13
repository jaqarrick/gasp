// bank_conflict.cu
// ---------------------------------------------------------------------------
// Measures : bank_penalty_cycles
// Method   : Shared memory access with forced conflicts.
//
// Definition: the extra cycles a warp-wide shared-memory load takes for each
// additional serialized bank access. A load where N lanes hit the same bank
// (with different addresses) needs N passes through the shared-memory pipeline,
// so its latency grows linearly with the conflict degree N:
//
//     latency(N) = base_latency + bank_penalty * (N - 1)
//
// One warp runs a dependent pointer chase through shared memory. All 32 lanes
// chase at once, and lane L starts at word L*stride, so on a 32-bank memory
// the conflict degree is N = gcd(stride, 32): strides 1, 2, 4, 8, 16, 32 give
// N = 1 (no conflict), 2, 4, 8, 16, 32. Every step advances all lanes by 32
// words, which keeps each lane in the same bank, so the conflict pattern is
// identical at every load. Because each load needs the previous result, the
// time per load is the full load latency for that conflict degree. The penalty
// is the slope of a straight-line fit of latency against (N - 1).
//
// The result `value` is that slope. The fit is done for every repetition, so
// `stats` describes the spread of the slope itself; the intercept is saved as
// `base_latency_cycles` (the conflict-free shared load latency).
//
// Build    : nvcc -O3 -std=c++17 -arch=sm_80  bank_conflict.cu -o bank_conflict   (A100)
//            nvcc -O3 -std=c++17 -arch=sm_90  bank_conflict.cu -o bank_conflict   (H100)
//            nvcc -O3 -std=c++17 -arch=sm_100 bank_conflict.cu -o bank_conflict   (B200)
//            (-arch must match the GPU you run on, e.g. sm_75 for an RTX 2080 Ti)
//
// Run      : ./bank_conflict [--gpu NAME] [--out-dir DIR] [--reps N]
//            Writes  <out-dir>/bank_penalty_cycles.<gpu>.run.json
//            Default out-dir = ./hardware/runs   (created if missing)
//
// Inspect  : nvcc -O3 -arch=sm_80 -cubin bank_conflict.cu -o bank_conflict.cubin
//            cuobjdump -sass bank_conflict.cubin     (checklist at bottom)
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
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "common.h"

static const char* PARAMETER = "bank_penalty_cycles";

// ---- Experiment parameters -------------------------------------------------
constexpr int SMEM_WORDS   = 8192;    // 32 KB of 32-bit words (a multiple of 32)
constexpr int CHASE_LOADS  = 1024;    // dependent loads in the timed region
constexpr int UNROLL       = 16;      // loads per loop iteration
constexpr int WARMUP_LOADS = 512;     // dependent loads before timing
constexpr int DEFAULT_REPS = 200;     // launches per conflict degree (>=100)
// Strides in words; with 32 banks the conflict degree is gcd(stride, 32).
constexpr int STRIDES[]    = {1, 2, 4, 8, 16, 32};
constexpr int NUM_STRIDES  = sizeof(STRIDES) / sizeof(STRIDES[0]);

static_assert(CHASE_LOADS % UNROLL == 0, "CHASE_LOADS must be a multiple of UNROLL");
static_assert(SMEM_WORDS % 32 == 0, "SMEM_WORDS must be a multiple of 32");

// ---- Device ----------------------------------------------------------------

// Dependent shared-memory load: the register holds a shared-memory address and
// receives the address of the next word, so there is no arithmetic between loads.
#define CHASE_ONE asm volatile("ld.shared.u32 %0, [%0];" : "+r"(p)::"memory");

// out[0] = gross cycles of the timed chain (clock overhead included)
// out[1] = clock overhead (two back-to-back clock reads, nothing between)
// out[2] = globaltimer ns around the timed chain (for the SM clock)
__global__ void bank_kernel(int stride_words, long long* out, unsigned* sink) {
    __shared__ unsigned smem[SMEM_WORDS];
    const int lane = threadIdx.x;   // launched with exactly one warp

    // Shared-window address of smem[0]. Every word holds the address of the word
    // 32 positions later (wrapping), so each lane stays in the same bank.
    const unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    for (int i = lane; i < SMEM_WORDS; i += 32) smem[i] = base + 4u * ((i + 32) % SMEM_WORDS);
    __syncwarp();

    // Lane L starts at word L*stride: bank (L*stride) mod 32, so the conflict
    // degree is gcd(stride, 32).
    unsigned p = base + 4u * static_cast<unsigned>(lane * stride_words);

#pragma unroll 1
    for (int i = 0; i < WARMUP_LOADS; ++i) {
        CHASE_ONE
    }

    // Depends on p, so the clock reads below cannot issue before the last
    // warm-up load has returned. p is always a multiple of 4, never 1.
    if (p == 1u) {
        *sink = 0;
        return;
    }
    __syncwarp();

    // Baseline: cost of reading the clock itself.
    uint64_t b0 = read_clock64();
    uint64_t b1 = read_clock64();

    // Timed region: CHASE_LOADS strictly dependent shared loads.
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

    if (lane == 0) {
        out[0] = static_cast<long long>(t1 - t0);
        out[1] = static_cast<long long>(b1 - b0);
        out[2] = static_cast<long long>(g1 - g0);
    }
    sink[lane] = p;   // keep the chain observable
}

// ---- Host helpers ----------------------------------------------------------

static int gcd_int(int a, int b) { return b == 0 ? a : gcd_int(b, a % b); }

// Least-squares line y = intercept + slope * x.
struct Fit { double slope, intercept, r2; };
static Fit fit_line(const std::vector<double>& x, const std::vector<double>& y) {
    const double n = static_cast<double>(x.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < x.size(); ++i) { sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i]; }
    Fit f;
    f.slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    f.intercept = (sy - f.slope * sx) / n;
    double ss_res = 0, ss_tot = 0;
    const double ybar = sy / n;
    for (size_t i = 0; i < x.size(); ++i) {
        const double e = y[i] - (f.intercept + f.slope * x[i]);
        ss_res += e * e;
        ss_tot += (y[i] - ybar) * (y[i] - ybar);
    }
    f.r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 1.0;
    return f;
}

// ---- Main ------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string out_dir = "./hardware/runs";
    std::string gpu_override;
    int reps = DEFAULT_REPS;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--gpu") && i + 1 < argc) gpu_override = argv[++i];
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else { fprintf(stderr, "Usage: %s [--gpu NAME] [--out-dir DIR] [--reps N]\n", argv[0]); return 1; }
    }
    if (reps < 100) { fprintf(stderr, "reps must be >= 100 (issue #3)\n"); return 1; }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const std::string gpu_name = prop.name;
    const std::string gpu = gpu_override.empty() ? short_gpu_name(gpu_name) : gpu_override;
    const std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);
    fprintf(stderr, "GPU: %s (%s), SMs: %d, one warp, %d conflict degrees\n",
            gpu_name.c_str(), arch.c_str(), prop.multiProcessorCount, NUM_STRIDES);

    int rt_ver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt_ver));

    long long* d_out = nullptr;
    unsigned* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, 3 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sink, 32 * sizeof(unsigned)));

    // raw[s][r] = gross cycles, ovh[s][r] = clock overhead, for stride s, repetition r.
    std::vector<std::vector<long long>> raw(NUM_STRIDES), ovh(NUM_STRIDES);
    // Totals over every timed run, for the SM clock (cycles / ns).
    double sum_clock_cycles = 0.0, sum_clock_ns = 0.0;
    for (int s = 0; s < NUM_STRIDES; ++s) {
        for (int w = 0; w < 3; ++w) {                 // untimed warm-up launches
            bank_kernel<<<1, 32>>>(STRIDES[s], d_out, d_sink);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int r = 0; r < reps; ++r) {
            CUDA_CHECK(cudaMemset(d_out, 0, 3 * sizeof(long long)));
            bank_kernel<<<1, 32>>>(STRIDES[s], d_out, d_sink);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            long long h[3];
            CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
            raw[s].push_back(h[0]);
            ovh[s].push_back(h[1]);
            if (h[2] > 0) {
                sum_clock_cycles += static_cast<double>(h[0]);
                sum_clock_ns += static_cast<double>(h[2]);
            }
        }
    }

    // Overhead = median of ALL baseline measurements; subtract from every run.
    std::vector<double> all_ovh;
    for (int s = 0; s < NUM_STRIDES; ++s) for (long long v : ovh[s]) all_ovh.push_back(static_cast<double>(v));
    const double overhead = median_of(all_ovh);

    // per_load[s][r] = latency of one warp-wide shared load, in cycles.
    std::vector<std::vector<double>> per_load(NUM_STRIDES);
    std::vector<double> conflict_degree(NUM_STRIDES), x(NUM_STRIDES);
    for (int s = 0; s < NUM_STRIDES; ++s) {
        conflict_degree[s] = gcd_int(STRIDES[s], 32);
        x[s] = conflict_degree[s] - 1.0;
        for (long long c : raw[s]) per_load[s].push_back((static_cast<double>(c) - overhead) / CHASE_LOADS);
    }

    // One straight-line fit per repetition: latency(N) = base + penalty * (N - 1).
    std::vector<double> penalty, base;
    for (int r = 0; r < reps; ++r) {
        std::vector<double> y(NUM_STRIDES);
        for (int s = 0; s < NUM_STRIDES; ++s) y[s] = per_load[s][r];
        const Fit f = fit_line(x, y);
        penalty.push_back(f.slope);
        base.push_back(f.intercept);
    }

    // Fit through the median latency at each conflict degree (for the summary and R^2).
    std::vector<double> med_lat(NUM_STRIDES);
    for (int s = 0; s < NUM_STRIDES; ++s) med_lat[s] = median_of(per_load[s]);
    const Fit mfit = fit_line(x, med_lat);

    const Stats stats = compute_stats(penalty);
    const double median = stats.median;
    const double base_median = median_of(base);

    fprintf(stderr, "\nMedian shared-load latency per conflict degree (cycles):\n");
    for (int s = 0; s < NUM_STRIDES; ++s)
        fprintf(stderr, "  stride %2d  (%2d-way): %8.3f\n", STRIDES[s], static_cast<int>(conflict_degree[s]), med_lat[s]);
    fprintf(stderr, "Fit through the medians: latency = %.3f + %.4f * (N - 1)   (R^2 = %.5f)\n",
            mfit.intercept, mfit.slope, mfit.r2);

    if (mfit.r2 < 0.99)
        fprintf(stderr, "WARNING: latency is not a straight line in the conflict degree (R^2 %.4f); "
                        "the slope may not mean 'cycles per extra conflict'. Check the SASS.\n", mfit.r2);
    if (median <= 0.0)
        fprintf(stderr, "WARNING: non-positive penalty; the conflicts were not forced. Check the SASS "
                        "(loads removed or bank mapping differs).\n");
    if (median > 8.0)
        fprintf(stderr, "WARNING: penalty %.2f cycles per extra conflict is unusually high; the GPU may "
                        "be shared. Check nvidia-smi and rerun.\n", median);
    if (base_median < 8.0 || base_median > 60.0)
        fprintf(stderr, "WARNING: conflict-free latency %.1f cycles is outside the usual 8 to 60 range for "
                        "a shared load.\n", base_median);
    if (stats.std > 0.05 * std::fabs(median) && stats.std > 0.01)
        fprintf(stderr, "WARNING: the penalty varies between runs (std %.4f); another process may be "
                        "sharing the GPU. Check nvidia-smi and rerun.\n", stats.std);

    // ---- Build JSON --------------------------------------------------------
    std::string driver = driver_version();

    std::ostringstream j;
    j.setf(std::ios::fixed);
    j.precision(4);
    j << "{\n";
    j << "  \"benchmark\": \"bank_conflict.cu\",\n";
    j << "  \"parameter\": \"" << PARAMETER << "\",\n";
    j << "  \"value\": " << median << ",\n";
    j << "  \"unit\": \"cycles\",\n";
    j << "  \"gpu\": \"" << json_escape(gpu) << "\",\n";
    j << "  \"gpu_name\": \"" << json_escape(gpu_name) << "\",\n";
    j << "  \"arch\": \"" << arch << "\",\n";
    j << "  \"num_sms\": " << prop.multiProcessorCount << ",\n";
    j << json_stats(stats);
    // stats/value are over per_run_penalty_cycles: the slope of latency vs (conflict degree - 1),
    // fitted separately for each repetition. Per-stride arrays below hold one entry per repetition:
    //   raw_cycles_by_conflict_degree : clock64 delta around the chain, clock-read cost INCLUDED
    //   overhead_cycles               : median clock-read cost (all runs); subtracted from every run
    //   per_load_cycles_by_conflict_degree : (raw - overhead) / loads_per_rep
    j << "  \"base_latency_cycles\": " << base_median << ",\n";
    j << "  \"loads_per_rep\": " << CHASE_LOADS << ",\n";
    j << "  \"per_run_penalty_cycles\": " << json_array(penalty) << ",\n";
    j << "  \"overhead_cycles\": " << overhead << ",\n";
    j << "  \"overhead_subtracted\": true,\n";
    j << "  \"sweep\": [\n";
    for (int s = 0; s < NUM_STRIDES; ++s) {
        j << "    {\"conflict_degree\": " << static_cast<int>(conflict_degree[s])
          << ", \"stride_words\": " << STRIDES[s]
          << ", \"median_cycles_per_load\": " << med_lat[s]
          << ", \"raw_cycles\": " << json_array(raw[s]);
        j << ", \"per_load_cycles\": " << json_array(per_load[s]);
        j << "}" << (s + 1 < NUM_STRIDES ? "," : "") << "\n";
    }
    j << "  ],\n";
    j << "  \"fit_through_medians\": {\"slope\": " << mfit.slope << ", \"intercept\": " << mfit.intercept
      << ", \"r_squared\": " << mfit.r2 << "},\n";
    // clock.sm_clock_ghz: total clock64 cycles / total globaltimer ns over all timed runs
    j << "  \"clock\": {\"sm_clock_ghz\": "
      << (sum_clock_ns > 0.0 ? sum_clock_cycles / sum_clock_ns : 0.0) << "},\n";
    j << "  \"cuda_version\": \"" << cuda_version_string(rt_ver) << "\",\n";
    j << "  \"nvcc_version\": \"" << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "\",\n";
    j << "  \"driver_version\": \"" << json_escape(driver) << "\",\n";
    j << "  \"timestamp\": \"" << utc_timestamp() << "\"\n";
    j << "}\n";

    // ---- Save to <out_dir>/{parameter}.{gpu}.run.json -----------------------
    const std::string out_path = write_run_json(out_dir, PARAMETER, gpu, j.str());
    if (out_path.empty()) return 1;

    printf("%s", j.str().c_str());
    fprintf(stderr, "\nBank penalty: %.4f cycles per extra conflict (std %.4f), conflict-free latency %.2f cycles\nSaved: %s\n",
            median, stats.std, base_median, out_path.c_str());

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}

// ---------------------------------------------------------------------------
// SASS inspection checklist (required by issue #11):
//   cuobjdump -sass bank_conflict.cubin
// In bank_kernel you should see:
//   * a clock read (CS2R) pair for the baseline, then one clock read before and
//     one after the timed loop, with a globaltimer read just outside each of
//     those two (the SM clock record; not inside the timed region)
//   * inside the timed loop, 16 consecutive shared-memory loads (LDS or LDS.U)
//     where each address register is the previous instruction's destination
//   * no IADD/IMAD/LEA between those loads
//   * a single kernel for all strides: the stride only appears in the starting
//     address, so the conflict degree must NOT be baked into the loop
// If loads were removed, merged or reordered, the result is invalid.
// ---------------------------------------------------------------------------
