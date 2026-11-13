// departure_delay.cu
// ---------------------------------------------------------------------------
// Measures : departure_delay_cycles
// Method   : Back-to-back warp memory requests (Jarmusch Table VII).
//
// Definition used here (Hong & Kim): the minimum number of SM cycles between
// two consecutive warp-level memory requests leaving one SM for the memory
// system. We measure it as the steady-state throughput of the SM's outbound
// request path:
//
//     departure_delay = cycles / (number of warp-level requests)
//
// One block (so exactly one SM is active and nothing else competes for L2)
// runs many warps. Every warp issues back-to-back, independent, fully
// coalesced loads: one warp instruction = 32 lanes x 4 bytes = one 128-byte
// line = one request. The loads use ld.global.cg (bypass L1, served by L2) and
// the working set (2 MB) fits in L2, so every request leaves the SM and hits
// L2. With enough warps and loads in flight (Little's law), the time per
// request is set by how fast requests can depart, not by L2 latency.
//
// A sweep over the warp count checks that: the reported value is the smallest
// per-request time across the sweep. If it kept falling as warps were added,
// the benchmark would still be latency-bound (see the stderr summary).
// Start-up and drain (about one L2 latency) are spread over thousands of
// requests, so they add well under 1%.
//
// Build    : nvcc -O3 -std=c++17 -arch=sm_80  departure_delay.cu -o departure_delay   (A100)
//            nvcc -O3 -std=c++17 -arch=sm_90  departure_delay.cu -o departure_delay   (H100)
//            nvcc -O3 -std=c++17 -arch=sm_100 departure_delay.cu -o departure_delay   (B200)
//            (-arch must match the GPU you run on, e.g. sm_75 for an RTX 2080 Ti)
//
// Run      : ./departure_delay [--gpu NAME] [--out-dir DIR] [--reps N]
//            Writes  <out-dir>/departure_delay_cycles.<gpu>.run.json
//            Default out-dir = ./hardware/runs   (created if missing)
//
// Inspect  : nvcc -O3 -arch=sm_80 -cubin departure_delay.cu -o departure_delay.cubin
//            cuobjdump -sass departure_delay.cubin     (checklist at bottom)
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

static const char* PARAMETER = "departure_delay_cycles";

// ---- Experiment parameters -------------------------------------------------
constexpr int LINE_BYTES      = 128;   // one warp-wide 4-byte load = one 128-byte line
constexpr int MAX_WARPS       = 32;    // one block of up to 1024 threads
constexpr int UNROLL          = 8;     // independent loads in flight per warp
constexpr int ITERS_PER_WARP  = 512;   // requests per warp (multiple of UNROLL)
constexpr int WARMUP_LAUNCHES = 3;     // per warp count; also brings the data into L2
constexpr int DEFAULT_REPS    = 200;   // timed launches per warp count (>=100)
constexpr int WARP_SWEEP[]    = {8, 16, 32};

static_assert(ITERS_PER_WARP % UNROLL == 0, "ITERS_PER_WARP must be a multiple of UNROLL");

// ---- Device ----------------------------------------------------------------

// out[0] = gross cycles of the timed loop (clock + barrier overhead included)
// out[1] = clock overhead (two back-to-back clock reads, nothing between)
// out[2] = globaltimer ns around the timed loop (read outside the clock64 pair)
__global__ void departure_kernel(const unsigned* __restrict__ buf, long long* out, unsigned* sink) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    // This warp owns ITERS_PER_WARP consecutive 128-byte lines; lane reads word `lane` of each.
    const unsigned* base = buf + static_cast<size_t>(warp) * ITERS_PER_WARP * 32 + lane;

    uint64_t b0 = 0, b1 = 0, t0 = 0, g0 = 0;
    unsigned acc = 0;

    __syncthreads();
    if (threadIdx.x == 0) { b0 = read_clock64(); b1 = read_clock64(); }   // baseline
    __syncthreads();
    if (threadIdx.x == 0) { g0 = read_globaltimer(); t0 = read_clock64(); }

    for (int i = 0; i < ITERS_PER_WARP; i += UNROLL) {
        unsigned v[UNROLL];
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            const unsigned* addr = base + static_cast<size_t>(i + k) * 32;
            // Volatile PTX so the loads cannot be removed or merged. .cg = L2 only,
            // so every request leaves the SM. Lanes read consecutive words, so one
            // warp instruction is exactly one coalesced 128-byte request.
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v[k]) : "l"(addr));
        }
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) acc ^= v[k];
    }

    __syncthreads();   // the block ends when its slowest warp ends
    if (threadIdx.x == 0) {
        out[0] = static_cast<long long>(read_clock64() - t0);
        const uint64_t g1 = read_globaltimer();
        out[1] = static_cast<long long>(b1 - b0);
        out[2] = static_cast<long long>(g1 - g0);
    }
    sink[threadIdx.x] = acc;   // keeps the loaded data observable
}

// ---- Host helpers ----------------------------------------------------------

struct SweepResult {
    int warps = 0;
    std::vector<long long> raw_cycles, overheads;   // per rep
    double overhead = 0.0;                          // median overhead
    std::vector<double> net_cycles, per_request;    // per rep
    double median = 0.0;
};

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

    const size_t buf_bytes = static_cast<size_t>(MAX_WARPS) * ITERS_PER_WARP * LINE_BYTES;   // 2 MB
    fprintf(stderr, "GPU: %s (%s), SMs: %d, L2: %.1f MB, working set: %.1f MB, one block on one SM\n",
            gpu_name.c_str(), arch.c_str(), prop.multiProcessorCount,
            prop.l2CacheSize / (1024.0 * 1024.0), buf_bytes / (1024.0 * 1024.0));
    if (buf_bytes * 2 > static_cast<size_t>(prop.l2CacheSize)) {
        fprintf(stderr, "ERROR: working set (%zu bytes) is too large to stay in L2 (%d bytes)\n",
                buf_bytes, prop.l2CacheSize);
        return 1;
    }

    int rt_ver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt_ver));

    unsigned* d_buf = nullptr;
    long long* d_out = nullptr;
    unsigned* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, buf_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, 3 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sink, MAX_WARPS * 32 * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(d_buf, 0x01, buf_bytes));

    std::vector<SweepResult> sweep;
    // SM clock over every timed launch: total clock64 cycles / total globaltimer ns.
    // One launch is only tens of us, so per-launch values are coarse (~1 us updates).
    double sum_clock_cycles = 0.0, sum_clock_ns = 0.0;
    for (int warps : WARP_SWEEP) {
        SweepResult sr;
        sr.warps = warps;
        const int threads = warps * 32;
        const double requests = static_cast<double>(warps) * ITERS_PER_WARP;

        for (int w = 0; w < WARMUP_LAUNCHES; ++w) {      // loads data into L2, settles clocks
            departure_kernel<<<1, threads>>>(d_buf, d_out, d_sink);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int r = 0; r < reps; ++r) {
            CUDA_CHECK(cudaMemset(d_out, 0, 3 * sizeof(long long)));
            departure_kernel<<<1, threads>>>(d_buf, d_out, d_sink);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            long long h[3];
            CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
            sr.raw_cycles.push_back(h[0]);
            sr.overheads.push_back(h[1]);
            if (h[2] > 0) {
                sum_clock_cycles += static_cast<double>(h[0]);
                sum_clock_ns += static_cast<double>(h[2]);
            }
        }
        std::vector<double> ov(sr.overheads.begin(), sr.overheads.end());
        sr.overhead = median_of(ov);
        for (long long c : sr.raw_cycles) {
            sr.net_cycles.push_back(static_cast<double>(c) - sr.overhead);
            sr.per_request.push_back((static_cast<double>(c) - sr.overhead) / requests);
        }
        sr.median = median_of(sr.per_request);
        sweep.push_back(std::move(sr));
    }

    if (sum_clock_ns <= 0.0) { fprintf(stderr, "ERROR: globaltimer did not advance\n"); return 1; }
    // Departure delay is SM-side, but it is measured against L2, which runs on its own
    // clock; the profile needs runs at the clock the validation runs at (README "Base clock").
    const double clock_ghz_all = sum_clock_cycles / sum_clock_ns;

    // Report the saturated (smallest) per-request time.
    size_t best = 0;
    for (size_t i = 1; i < sweep.size(); ++i)
        if (sweep[i].median < sweep[best].median) best = i;
    const SweepResult& S = sweep[best];
    const double requests_per_rep = static_cast<double>(S.warps) * ITERS_PER_WARP;

    std::vector<double> gross_per_request;
    for (long long c : S.raw_cycles) gross_per_request.push_back(static_cast<double>(c) / requests_per_rep);
    const double uncorrected_value = median_of(gross_per_request);

    const Stats stats = compute_stats(S.per_request);
    const double median = S.median;

    fprintf(stderr, "\nSweep (median cycles per warp request):\n");
    for (const SweepResult& sr : sweep)
        fprintf(stderr, "  %2d warps: %.4f%s\n", sr.warps, sr.median, &sr == &S ? "   <- reported" : "");
    if (best == 0 && sweep.size() > 1)
        fprintf(stderr, "WARNING: the fewest warps gave the smallest value; the sweep is not behaving "
                        "as expected. Check the SASS and nvidia-smi.\n");
    if (sweep.size() > 1 && sweep[sweep.size() - 1].median > 1.10 * median)
        fprintf(stderr, "NOTE: more warps made it slower; the load path may be saturating, which is "
                        "fine, but check the sweep above.\n");
    if (median > 20.0)
        fprintf(stderr, "WARNING: median %.1f cycles per request is high; the run may be latency-bound "
                        "or the GPU is shared. Check nvidia-smi and rerun.\n", median);
    if (median < 0.4)
        fprintf(stderr, "WARNING: median %.2f cycles per request is implausibly low; check the SASS "
                        "(loads may have been removed).\n", median);
    if (median > 1.05 * stats.min)
        fprintf(stderr, "WARNING: median %.3f is >5%% above the fastest run (%.3f). Another process may "
                        "be sharing the GPU; check nvidia-smi and rerun.\n", median, stats.min);

    // ---- Build JSON --------------------------------------------------------
    std::string driver = driver_version();

    std::ostringstream j;
    j.setf(std::ios::fixed);
    j.precision(4);
    j << "{\n";
    j << "  \"benchmark\": \"departure_delay.cu\",\n";
    j << "  \"parameter\": \"" << PARAMETER << "\",\n";
    j << "  \"value\": " << median << ",\n";
    j << "  \"unit\": \"cycles\",\n";
    j << "  \"gpu\": \"" << json_escape(gpu) << "\",\n";
    j << "  \"gpu_name\": \"" << json_escape(gpu_name) << "\",\n";
    j << "  \"arch\": \"" << arch << "\",\n";
    j << "  \"num_sms\": " << prop.multiProcessorCount << ",\n";
    j << json_stats(stats);
    // Arrays below are for the reported warp count (warps_per_block), one entry per timed launch:
    //   raw_cycles          : clock64 delta around the loop, clock-read cost INCLUDED
    //   overhead_samples    : clock64 delta of two back-to-back reads (cost of the clock)
    //   overhead_cycles     : median of overhead_samples; the value subtracted from every run
    //   net_cycles          : raw_cycles - overhead_cycles
    //   per_request_cycles  : net_cycles / requests_per_rep; stats are computed over these
    j << "  \"warps_per_block\": " << S.warps << ",\n";
    j << "  \"requests_per_rep\": " << static_cast<long long>(requests_per_rep) << ",\n";
    j << "  \"raw_cycles\": " << json_array(S.raw_cycles) << ",\n";
    j << "  \"overhead_samples_cycles\": " << json_array(S.overheads) << ",\n";
    j << "  \"overhead_cycles\": " << S.overhead << ",\n";
    j << "  \"overhead_subtracted\": true,\n";
    j << "  \"net_cycles\": " << json_array(S.net_cycles) << ",\n";
    j << "  \"per_request_cycles\": " << json_array(S.per_request) << ",\n";
    j << "  \"uncorrected_value\": " << uncorrected_value << ",\n";
    j << "  \"sweep_median_cycles_per_request\": {";
    for (size_t i = 0; i < sweep.size(); ++i)
        j << (i ? ", " : "") << "\"" << sweep[i].warps << "_warps\": " << sweep[i].median;
    j << "},\n";
    // sm_clock_ghz: total clock64 cycles / total globaltimer ns over every timed launch
    j << "  \"clock\": {\"sm_clock_ghz\": " << clock_ghz_all << "},\n";
    j << "  \"cuda_version\": \"" << cuda_version_string(rt_ver) << "\",\n";
    j << "  \"nvcc_version\": \"" << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "\",\n";
    j << "  \"driver_version\": \"" << json_escape(driver) << "\",\n";
    j << "  \"timestamp\": \"" << utc_timestamp() << "\"\n";
    j << "}\n";

    // ---- Save to <out_dir>/{parameter}.{gpu}.run.json -----------------------
    const std::string out_path = write_run_json(out_dir, PARAMETER, gpu, j.str());
    if (out_path.empty()) return 1;

    printf("%s", j.str().c_str());
    fprintf(stderr, "\nDeparture delay: %.3f cycles per warp request at %.3f GHz SM clock (std %.4f, "
                    "overhead %.1f cycles subtracted)\nSaved: %s\n",
            median, clock_ghz_all, stats.std, S.overhead, out_path.c_str());

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}

// ---------------------------------------------------------------------------
// SASS inspection checklist (required by issue #11):
//   cuobjdump -sass departure_delay.cubin
// In departure_kernel you should see:
//   * a clock read (CS2R) pair for the baseline, then one clock read before and
//     one after the timed loop, only for thread 0, with a globaltimer read
//     (SR_GLOBALTIMER) just outside each of the timed pair
//   * inside the loop, 8 consecutive 32-bit global loads (LDG.E, carrying the
//     same cache-operator modifier as l2_latency, e.g. .STRONG.GPU on Turing)
//     issued before any of their results is used
//   * no extra loads or stores in the loop, and the loop must still exist: if
//     the loads were removed the result will look impossibly small
// ---------------------------------------------------------------------------
