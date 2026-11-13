#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "common.h"

static const char* PARAMETER      = "sync_factor_cycles";
static const char* SCHEMA_VERSION = "1.2.0";
static const char* BENCH_VERSION  = "1.2.0";

constexpr int WARP                 = 32;
constexpr int LINE_BYTES           = WARP * 4;
constexpr int WORKING_SET_L2_MULT  = 4;           // ring >= 4 x L2
constexpr int DEFAULT_REPS         = 100;         // launches per variant per config (>=100)
constexpr int DEFAULT_ITERS        = 256;         // barriers per warp per launch
constexpr int DEFAULT_LOOP_OVERHEAD = 5;          // pointer IADD3 + IADD3.X, counter IADD3, ISETP, BRA
constexpr double CV_WARN           = 0.25;        // warn if Γ varies more than this across configs
// A config whose no-barrier variant streams at >= this fraction of the measured DRAM
// bandwidth is bandwidth-bound: the barrier hides behind DRAM queueing, which is not
// the latency-exposure effect GPUPerf's sync term models, so it is excluded from Γ.
constexpr double DEFAULT_BW_FRAC   = 0.85;

constexpr int LOADS[]        = {1, 4};
constexpr int FFMAS[]        = {0, 8, 32, 128};
constexpr int WARPS_PER_BLOCK[] = {4, 8, 16};
constexpr int BLOCKS_PER_SM[]   = {1, 2, 4};

// M - global loads per iter
// C - number of FFMAs per iteration
// BAR should include barrier?
template <int M, int C, bool BAR>
__global__ void sync_kernel(const float* __restrict__ buf, int iters, float k,
                            long long* t_start, long long* t_end,
                            unsigned long long* g_start, unsigned long long* g_end,
                            int* smid, float* sink) {
    extern __shared__ float smem[];
    const int tid  = threadIdx.x;
    const int lane = tid % WARP;
    const int wid  = blockIdx.x * (blockDim.x / WARP) + tid / WARP;

    const char* p = reinterpret_cast<const char*>(buf)
                  + static_cast<size_t>(wid) * iters * M * LINE_BYTES
                  + lane * 4;

    // Conflict-free shared slots: store m of thread tid goes to word m x blockDim + tid.
    uint32_t saddr[M];
    const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
#pragma unroll
    for (int m = 0; m < M; ++m) saddr[m] = sbase + (m * blockDim.x + tid) * 4;

    float acc[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    // Same start for every warp of the block, in both variants.
    asm volatile("bar.sync 0;" ::: "memory");
    // globaltimer outside the clock64 pair, so it does not add to the cycle span.
    if (tid == 0) {
        g_start[blockIdx.x] = read_globaltimer();
        t_start[blockIdx.x] = read_clock64();
    }

#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        float v[M];
#pragma unroll
        for (int m = 0; m < M; ++m)
            asm volatile("ld.global.cg.f32 %0, [%1];"
                         : "=f"(v[m]) : "l"(p + m * LINE_BYTES) : "memory");
#pragma unroll
        for (int c = 0; c < C; ++c)
            asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(acc[c % 4]) : "f"(k));
#pragma unroll
        for (int m = 0; m < M; ++m)
            asm volatile("st.shared.f32 [%0], %1;" :: "r"(saddr[m]), "f"(v[m]) : "memory");
        if (BAR) asm volatile("bar.sync 0;" ::: "memory");
        p += M * LINE_BYTES;
    }

    asm volatile("bar.sync 0;" ::: "memory");
    if (tid == 0) {
        t_end[blockIdx.x] = read_clock64();
        g_end[blockIdx.x] = read_globaltimer();
        smid[blockIdx.x]  = read_smid();
    }
    // Keep the FFMA results observable.
    const float s = acc[0] + acc[1] + acc[2] + acc[3];
    if (s == 12345.0f) *sink = s;
}

using KernelFn = void (*)(const float*, int, float, long long*, long long*,
                          unsigned long long*, unsigned long long*, int*, float*);

template <int M, int C>
static void kernels_for(KernelFn& no_bar, KernelFn& bar) {
    no_bar = sync_kernel<M, C, false>;
    bar    = sync_kernel<M, C, true>;
}

static bool lookup_kernels(int m, int c, KernelFn& no_bar, KernelFn& bar) {
#define CASE(M_, C_) if (m == M_ && c == C_) { kernels_for<M_, C_>(no_bar, bar); return true; }
    CASE(1, 0) CASE(1, 8) CASE(1, 32) CASE(1, 128)
    CASE(4, 0) CASE(4, 8) CASE(4, 32) CASE(4, 128)
#undef CASE
    return false;
}

static double stddev_of(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double var = 0.0;
    for (double x : v) var += (x - mean) * (x - mean);
    return std::sqrt(var / (v.size() - 1));
}

struct LaunchResult {
    double span;         // median over SMs of (last block end - first block start)
    bool   co_resident;  // every SM ran exactly blocks_per_sm blocks
    double clock_cycles; // sum over blocks of the block's clock64 span
    double clock_ns;     // sum over blocks of the block's globaltimer span
};

static LaunchResult run_once(KernelFn fn, int grid, int threads, size_t smem,
                             const float* slice, int iters, int num_sms, int blocks_per_sm,
                             long long* d_start, long long* d_end,
                             unsigned long long* d_gstart, unsigned long long* d_gend,
                             int* d_smid, float* d_sink,
                             std::vector<long long>& h_start, std::vector<long long>& h_end,
                             std::vector<unsigned long long>& h_gstart,
                             std::vector<unsigned long long>& h_gend,
                             std::vector<int>& h_smid) {
    fn<<<grid, threads, smem>>>(slice, iters, 0.0f, d_start, d_end, d_gstart, d_gend, d_smid, d_sink);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_start.data(), d_start, grid * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_end.data(), d_end, grid * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gstart.data(), d_gstart, grid * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gend.data(), d_gend, grid * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_smid.data(), d_smid, grid * sizeof(int), cudaMemcpyDeviceToHost));

    // Each block's clock64 and globaltimer spans come from one SM, so their ratio is
    // that SM's clock. Summed over blocks, globaltimer's ~1 us updates average out.
    double clock_cycles = 0.0, clock_ns = 0.0;
    for (int b = 0; b < grid; ++b) {
        clock_cycles += static_cast<double>(h_end[b] - h_start[b]);
        clock_ns     += static_cast<double>(h_gend[b] - h_gstart[b]);
    }

    std::map<int, long long> first, last;
    std::map<int, int> count;
    for (int b = 0; b < grid; ++b) {
        const int s = h_smid[b];
        if (!count[s]++) { first[s] = h_start[b]; last[s] = h_end[b]; }
        first[s] = std::min(first[s], h_start[b]);
        last[s]  = std::max(last[s], h_end[b]);
    }
    bool co_resident = static_cast<int>(count.size()) == num_sms;
    std::vector<double> spans;
    for (const auto& [s, n] : count) {
        if (n != blocks_per_sm) co_resident = false;
        spans.push_back(static_cast<double>(last[s] - first[s]));
    }
    return {median_of(spans), co_resident, clock_cycles, clock_ns};
}

struct ConfigResult {
    int loads, ffmas, warps_per_block, blocks_per_sm, warps_per_sm;
    double mem_ratio;
    double t_no_bar, t_bar, t_no_bar_std, t_bar_std;
    double delta, f_sync, gamma;
    double delta_per_iter;   // delta / iters: SM cycles each barrier round adds
    double achieved_bw;      // no-barrier variant, bytes per SM cycle across the GPU
    bool co_resident;
    bool bw_bound;           // achieved_bw >= bw_frac x measured DRAM bandwidth
};

int main(int argc, char** argv) {
    std::string out_dir = "./hardware/runs";
    std::string gpu_override;
    int reps = DEFAULT_REPS;
    int iters = DEFAULT_ITERS;
    int loop_overhead = DEFAULT_LOOP_OVERHEAD;
    double dram_latency = 0.0;   // 0 = read from the dram_latency run
    double dram_bw = 0.0;        // 0 = read from the dram_bandwidth run
    double bw_frac = DEFAULT_BW_FRAC;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--gpu") && i + 1 < argc) gpu_override = argv[++i];
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dram-latency") && i + 1 < argc) dram_latency = atof(argv[++i]);
        else if (!strcmp(argv[i], "--dram-bandwidth") && i + 1 < argc) dram_bw = atof(argv[++i]);
        else if (!strcmp(argv[i], "--bw-frac") && i + 1 < argc) bw_frac = atof(argv[++i]);
        else if (!strcmp(argv[i], "--loop-overhead") && i + 1 < argc) loop_overhead = atoi(argv[++i]);
        else {
            fprintf(stderr, "Usage: %s [--gpu NAME] [--out-dir DIR] [--reps N] [--iters N] "
                            "[--dram-latency CYCLES] [--dram-bandwidth BYTES_PER_CYCLE] "
                            "[--bw-frac F] [--loop-overhead N]\n", argv[0]);
            return 1;
        }
    }
    if (reps < 100) { fprintf(stderr, "reps must be >= 100 (issue #3)\n"); return 1; }
    if (iters < 1) { fprintf(stderr, "--iters must be >= 1\n"); return 1; }
    if (loop_overhead < 0) { fprintf(stderr, "--loop-overhead must be >= 0\n"); return 1; }
    if (bw_frac <= 0.0) { fprintf(stderr, "--bw-frac must be > 0\n"); return 1; }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.warpSize != WARP) { fprintf(stderr, "expected warp size %d, got %d\n", WARP, prop.warpSize); return 1; }
    const std::string gpu_name = prop.name;
    const std::string gpu = gpu_override.empty() ? short_gpu_name(gpu_name) : gpu_override;
    const std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);
    const int num_sms = prop.multiProcessorCount;

    namespace fs = std::filesystem;
    std::string dram_source = "user";
    if (dram_latency <= 0.0) {
        const fs::path p = fs::path(out_dir) / ("dram_latency_cycles." + gpu + ".run.json");
        dram_latency = read_run_value(p.string(), "dram_latency_cycles");
        dram_source = p.string();
        if (dram_latency <= 0.0) {
            fprintf(stderr, "No DRAM latency: pass --dram-latency CYCLES or run dram_latency first "
                            "(looked for %s)\n", p.string().c_str());
            return 1;
        }
    }
    std::string bw_source = "user";
    if (dram_bw <= 0.0) {
        const fs::path p = fs::path(out_dir) / ("dram_bandwidth_bytes_per_cycle." + gpu + ".run.json");
        dram_bw = read_run_value(p.string(), "dram_bandwidth_bytes_per_cycle");
        bw_source = p.string();
        if (dram_bw <= 0.0) {
            fprintf(stderr, "No DRAM bandwidth: pass --dram-bandwidth BYTES_PER_CYCLE or run "
                            "dram_bandwidth first (looked for %s)\n", p.string().c_str());
            return 1;
        }
    }

    // Ring: at least 4x L2. Preferred: the largest launch twice over, so consecutive
    // launches read disjoint slices. If that does not fit in free memory, the largest
    // launch plus one L2 is enough: a launch that wraps to the start of the ring finds
    // at most the previous launch's last L2's worth of lines cached, and it streams far
    // more than L2 before reaching them, so its loads still miss L2.
    const int max_m = *std::max_element(std::begin(LOADS), std::end(LOADS));
    const int max_w = *std::max_element(std::begin(WARPS_PER_BLOCK), std::end(WARPS_PER_BLOCK));
    const int max_b = *std::max_element(std::begin(BLOCKS_PER_SM), std::end(BLOCKS_PER_SM));
    const size_t max_launch_bytes =
        static_cast<size_t>(num_sms) * max_b * max_w * iters * max_m * LINE_BYTES;
    const size_t l2_ring = static_cast<size_t>(WORKING_SET_L2_MULT) * prop.l2CacheSize;
    const size_t preferred_ring = std::max(l2_ring, 2 * max_launch_bytes);
    const size_t minimum_ring = std::max(l2_ring, max_launch_bytes + prop.l2CacheSize);
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const size_t usable_b = free_b / 10 * 8;
    if (minimum_ring > usable_b) {
        fprintf(stderr, "ERROR: ring (at least %zu bytes) does not fit in free device memory "
                        "(%zu bytes); lower --iters or free the GPU\n", minimum_ring, free_b);
        return 1;
    }
    const size_t ring_bytes = std::min(preferred_ring, usable_b);
    if (ring_bytes < preferred_ring)
        fprintf(stderr, "Note: ring shrunk from %.1f MB to %.1f MB to fit free device memory "
                        "(%.1f MB); consecutive large launches share lines past one L2\n",
                preferred_ring / 1048576.0, ring_bytes / 1048576.0, free_b / 1048576.0);

    fprintf(stderr, "GPU: %s (%s), SMs: %d, L2: %.1f MB, ring: %.1f MB, DRAM latency: %.1f (%s), "
                    "DRAM bandwidth: %.1f B/cycle (%s)\n",
            gpu_name.c_str(), arch.c_str(), num_sms, prop.l2CacheSize / 1048576.0,
            ring_bytes / 1048576.0, dram_latency, dram_source.c_str(), dram_bw, bw_source.c_str());

    int rt_ver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt_ver));

    float* d_ring = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ring, ring_bytes));
    CUDA_CHECK(cudaMemset(d_ring, 0, ring_bytes));

    const int max_grid = num_sms * max_b;
    long long *d_start, *d_end;
    unsigned long long *d_gstart, *d_gend;
    int* d_smid;
    float* d_sink;
    CUDA_CHECK(cudaMalloc(&d_start, max_grid * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_end, max_grid * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_gstart, max_grid * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_gend, max_grid * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_smid, max_grid * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));
    std::vector<long long> h_start(max_grid), h_end(max_grid);
    std::vector<unsigned long long> h_gstart(max_grid), h_gend(max_grid);
    std::vector<int> h_smid(max_grid);

    size_t cursor = 0;   // byte offset of the next launch's slice in the ring
    auto next_slice = [&](size_t bytes) {
        if (cursor + bytes > ring_bytes) cursor = 0;
        const float* s = reinterpret_cast<const float*>(reinterpret_cast<char*>(d_ring) + cursor);
        cursor += bytes;
        return s;
    };

    std::vector<ConfigResult> results;
    std::vector<std::string> skipped;
    // SM clock over the recorded launches: per-launch cycles / ns, and the totals.
    std::vector<double> clock_ghz;
    double sum_clock_cycles = 0.0, sum_clock_ns = 0.0;
    for (int m : LOADS)
    for (int c : FFMAS)
    for (int w : WARPS_PER_BLOCK)
    for (int b : BLOCKS_PER_SM) {
        KernelFn k_no_bar, k_bar;
        if (!lookup_kernels(m, c, k_no_bar, k_bar)) { fprintf(stderr, "no kernel for M=%d C=%d\n", m, c); return 1; }
        const int threads = w * WARP;
        const size_t smem = static_cast<size_t>(m) * threads * sizeof(float);
        const int grid = num_sms * b;
        char label[96];
        snprintf(label, sizeof(label), "loads=%d ffmas=%d warps/block=%d blocks/SM=%d", m, c, w, b);

        int fit_no_bar = 0, fit_bar = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit_no_bar, k_no_bar, threads, smem));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit_bar, k_bar, threads, smem));
        if (std::min(fit_no_bar, fit_bar) < b) {
            skipped.push_back(std::string(label) + " (only " + std::to_string(std::min(fit_no_bar, fit_bar)) + " blocks fit)");
            continue;
        }

        const size_t launch_bytes = static_cast<size_t>(grid) * w * iters * m * LINE_BYTES;
        // Warm-up (icache, clocks), not recorded.
        for (KernelFn fn : {k_no_bar, k_bar})
            run_once(fn, grid, threads, smem, next_slice(launch_bytes), iters, num_sms, b,
                     d_start, d_end, d_gstart, d_gend, d_smid, d_sink,
                     h_start, h_end, h_gstart, h_gend, h_smid);

        std::vector<double> t_no_bar, t_bar;
        bool co_resident = true;
        for (int r = 0; r < reps; ++r) {
            // Alternate the order so drift affects both variants alike.
            const bool bar_first = r % 2;
            for (int v = 0; v < 2; ++v) {
                const bool is_bar = (v == 0) == bar_first;
                const LaunchResult lr = run_once(is_bar ? k_bar : k_no_bar, grid, threads, smem,
                                                 next_slice(launch_bytes), iters, num_sms, b,
                                                 d_start, d_end, d_gstart, d_gend, d_smid, d_sink,
                                                 h_start, h_end, h_gstart, h_gend, h_smid);
                co_resident = co_resident && lr.co_resident;
                if (lr.clock_ns > 0.0) {
                    clock_ghz.push_back(lr.clock_cycles / lr.clock_ns);
                    sum_clock_cycles += lr.clock_cycles;
                    sum_clock_ns += lr.clock_ns;
                }
                (is_bar ? t_bar : t_no_bar).push_back(lr.span);
            }
        }

        ConfigResult cr{};
        cr.loads = m; cr.ffmas = c; cr.warps_per_block = w; cr.blocks_per_sm = b;
        cr.warps_per_sm = w * b;
        cr.mem_ratio = static_cast<double>(m) / (2 * m + c + 1 + loop_overhead);
        cr.t_no_bar = median_of(t_no_bar);
        cr.t_bar = median_of(t_bar);
        cr.t_no_bar_std = stddev_of(t_no_bar);
        cr.t_bar_std = stddev_of(t_bar);
        cr.delta = cr.t_bar - cr.t_no_bar;
        cr.f_sync = cr.delta / (static_cast<double>(iters) * cr.warps_per_sm);
        cr.gamma = cr.f_sync / (dram_latency * cr.mem_ratio);
        cr.delta_per_iter = cr.delta / iters;
        // The span is per SM, but every SM streams at once, so launch bytes over the
        // span is the GPU-wide rate in bytes per SM cycle, the same unit as dram_bw.
        cr.achieved_bw = static_cast<double>(launch_bytes) / cr.t_no_bar;
        cr.co_resident = co_resident;
        cr.bw_bound = cr.achieved_bw >= bw_frac * dram_bw;
        results.push_back(cr);

        const char* note = !co_resident ? "  (not co-resident, excluded)"
                         : cr.bw_bound  ? "  (bandwidth-bound, excluded)" : "";
        fprintf(stderr, "%-48s ratio %.4f  T %10.0f -> %10.0f  BW %6.1f  F_sync %8.2f  gamma %8.3f%s\n",
                label, cr.mem_ratio, cr.t_no_bar, cr.t_bar, cr.achieved_bw, cr.f_sync, cr.gamma, note);
    }

    // ---- Γ over the usable configurations ----------------------------------
    // Usable = every block co-resident and not bandwidth-bound.
    std::vector<double> gammas, deltas_per_iter;
    int bw_bound_count = 0;
    double sxy = 0.0, sxx = 0.0;   // least squares through the origin: F_sync = Γ x (lat x ratio)
    for (const auto& r : results) {
        if (!r.co_resident) continue;
        if (r.bw_bound) { ++bw_bound_count; continue; }
        gammas.push_back(r.gamma);
        deltas_per_iter.push_back(r.delta_per_iter);
        const double x = dram_latency * r.mem_ratio;
        sxy += x * r.f_sync;
        sxx += x * x;
    }
    if (gammas.empty()) {
        fprintf(stderr, "ERROR: no configuration was both co-resident and below %.0f%% of DRAM "
                        "bandwidth\n", 100 * bw_frac);
        return 1;
    }
    const double median_delta_per_iter = median_of(deltas_per_iter);
    const double median = median_of(gammas);
    const double mean = std::accumulate(gammas.begin(), gammas.end(), 0.0) / gammas.size();
    const double stddev = stddev_of(gammas);
    const double mn = *std::min_element(gammas.begin(), gammas.end());
    const double mx = *std::max_element(gammas.begin(), gammas.end());
    const double gamma_fit = sxy / sxx;
    const double cv = std::fabs(mean) > 0 ? stddev / std::fabs(mean) : INFINITY;

    // SM clock the cycles were counted at. The cycle spans depend on it, so the profile
    // needs runs at the clock the validation runs at: ncu locks base clock (README).
    if (clock_ghz.empty()) { fprintf(stderr, "ERROR: globaltimer did not advance\n"); return 1; }
    const double clock_ghz_all = sum_clock_cycles / sum_clock_ns;
    const double clock_min = *std::min_element(clock_ghz.begin(), clock_ghz.end());
    const double clock_max = *std::max_element(clock_ghz.begin(), clock_ghz.end());
    if (clock_max > 1.02 * clock_min)
        fprintf(stderr, "WARNING: SM clock varied %.3f-%.3f GHz across launches; the cycle counts mix "
                        "clocks. Run under ncu to lock base clock (README).\n", clock_min, clock_max);

    for (const auto& s : skipped) fprintf(stderr, "skipped: %s\n", s.c_str());
    if (cv > CV_WARN)
        fprintf(stderr, "WARNING: gamma varies by %.0f%% (std/mean) across configurations, so one "
                        "gamma does not fit them all. Check the per-configuration values against "
                        "memory ratio and warps per SM before using the median.\n", 100 * cv);

    // ---- Build JSON --------------------------------------------------------
    std::string driver = driver_version();
    std::string commit = git_commit();

    std::ostringstream j;
    j.setf(std::ios::fixed);
    j.precision(4);
    j << "{\n";
    j << "  \"schema_version\": \"" << SCHEMA_VERSION << "\",\n";
    j << "  \"benchmark\": \"sync_cost.cu\",\n";
    j << "  \"benchmark_version\": \"" << BENCH_VERSION << "\",\n";
    j << "  \"parameter\": \"" << PARAMETER << "\",\n";
    j << "  \"value\": " << median << ",\n";
    j << "  \"unit\": \"dimensionless\",\n";
    j << "  \"gpu\": \"" << json_escape(gpu) << "\",\n";
    j << "  \"gpu_name\": \"" << json_escape(gpu_name) << "\",\n";
    j << "  \"arch\": \"" << arch << "\",\n";
    j << "  \"num_sms\": " << num_sms << ",\n";
    j << "  \"l2_cache_bytes\": " << prop.l2CacheSize << ",\n";
    // stats are over the usable (co-resident, not bandwidth-bound) configurations' Γ, not over reps
    j << "  \"stats\": {\n";
    j << "    \"median\": " << median << ",\n";
    j << "    \"mean\": " << mean << ",\n";
    j << "    \"std\": " << stddev << ",\n";
    j << "    \"min\": " << mn << ",\n";
    j << "    \"max\": " << mx << ",\n";
    j << "    \"cv\": " << cv << ",\n";
    j << "    \"least_squares_fit\": " << gamma_fit << ",\n";
    j << "    \"configs_used\": " << gammas.size() << ",\n";
    j << "    \"configs_bandwidth_bound\": " << bw_bound_count << ",\n";
    j << "    \"median_delta_per_iter_cycles\": " << median_delta_per_iter << "\n";
    j << "  },\n";
    // per configuration:
    //   t_no_bar_cycles, t_bar_cycles : median over reps of the per-SM span (median over SMs)
    //   delta_cycles                  : t_bar - t_no_bar, measured O_sync per SM
    //   delta_per_iter_cycles         : delta / iters, SM cycles each barrier round adds
    //   f_sync_cycles                 : delta / (iters x warps_per_sm)
    //   gamma                         : f_sync / (dram_latency x mem_ratio)
    //   achieved_bw_bytes_per_cycle   : no-barrier launch bytes / t_no_bar
    //   bandwidth_bound               : achieved_bw >= bw_frac x dram_bandwidth (excluded)
    j << "  \"configs\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        j << "    {\"loads_per_iter\": " << r.loads
          << ", \"ffmas_per_iter\": " << r.ffmas
          << ", \"warps_per_block\": " << r.warps_per_block
          << ", \"blocks_per_sm\": " << r.blocks_per_sm
          << ", \"warps_per_sm\": " << r.warps_per_sm
          << ", \"mem_ratio\": " << r.mem_ratio
          << ", \"t_no_bar_cycles\": " << r.t_no_bar
          << ", \"t_no_bar_std\": " << r.t_no_bar_std
          << ", \"t_bar_cycles\": " << r.t_bar
          << ", \"t_bar_std\": " << r.t_bar_std
          << ", \"delta_cycles\": " << r.delta
          << ", \"delta_per_iter_cycles\": " << r.delta_per_iter
          << ", \"f_sync_cycles\": " << r.f_sync
          << ", \"gamma\": " << r.gamma
          << ", \"achieved_bw_bytes_per_cycle\": " << r.achieved_bw
          << ", \"co_resident\": " << (r.co_resident ? "true" : "false")
          << ", \"bandwidth_bound\": " << (r.bw_bound ? "true" : "false") << "}"
          << (i + 1 < results.size() ? "," : "") << "\n";
    }
    j << "  ],\n";
    // clock: SM clock during the recorded launches. sm_clock_ghz = total clock64 cycles /
    // total globaltimer ns over every block of every launch; min/max are per launch
    j << "  \"clock\": {\n";
    j << "    \"sm_clock_ghz\": " << clock_ghz_all << ",\n";
    j << "    \"sm_clock_ghz_min\": " << clock_min << ",\n";
    j << "    \"sm_clock_ghz_max\": " << clock_max << "\n";
    j << "  },\n";
    j << "  \"skipped\": [";
    for (size_t i = 0; i < skipped.size(); ++i) j << (i ? ", " : "") << "\"" << json_escape(skipped[i]) << "\"";
    j << "],\n";
    j << "  \"measurement\": {\n";
    j << "    \"method\": \"bar.sync vs no bar.sync, one wave, per-SM clock64 span; GPUPerf Eqs. 7-8 solved for gamma\",\n";
    j << "    \"load_instruction\": \"ld.global.cg.f32\",\n";
    j << "    \"timer\": \"clock64 (cycles), globaltimer (SM clock)\",\n";
    j << "    \"dram_latency_cycles\": " << dram_latency << ",\n";
    j << "    \"dram_latency_source\": \"" << json_escape(dram_source) << "\",\n";
    j << "    \"dram_bandwidth_bytes_per_cycle\": " << dram_bw << ",\n";
    j << "    \"dram_bandwidth_source\": \"" << json_escape(bw_source) << "\",\n";
    j << "    \"bw_frac\": " << bw_frac << ",\n";
    j << "    \"loop_overhead_insts\": " << loop_overhead << ",\n";
    j << "    \"iters\": " << iters << ",\n";
    j << "    \"reps_per_variant\": " << reps << ",\n";
    j << "    \"ring_bytes\": " << ring_bytes << "\n";
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
    fprintf(stderr, "\nGamma: %.3f median over %zu configurations (std %.3f, least-squares fit %.3f); "
                    "%d bandwidth-bound excluded\nMedian barrier cost: %.1f SM cycles per barrier round\n"
                    "Saved: %s\n",
            median, gammas.size(), stddev, gamma_fit, bw_bound_count, median_delta_per_iter,
            out_path.c_str());

    CUDA_CHECK(cudaFree(d_ring));
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_end));
    CUDA_CHECK(cudaFree(d_gstart));
    CUDA_CHECK(cudaFree(d_gend));
    CUDA_CHECK(cudaFree(d_smid));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}

// ---------------------------------------------------------------------------
// SASS inspection checklist:
//   cuobjdump -sass sync_cost.cubin
// For each sync_kernel<M, C, BAR> instantiation, inside the timed loop:
//   * M LDG.E (with .STRONG.GPU on Turing, from .cg) off one address register,
//     at offsets 0x0, 0x80, 0x100, 0x180 (no IADD/LEA between them)
//   * C FFMA, none of them reading an LDG destination
//   * M STS, each storing an LDG destination
//   * BAR.SYNC only in the BAR = true instances, after the STS
//   * the loop overhead: count the instructions in the loop that are none of
//     the above (expected 5: IADD3 + IADD3.X on the pointer, IADD3 on the
//     counter, ISETP, BRA). If the count differs, rerun with --loop-overhead N,
//     or the memory ratio, and so Γ, will be off.
// Outside the loop, a BAR.SYNC and CS2R (clock) before the loop and after it.
// ---------------------------------------------------------------------------
