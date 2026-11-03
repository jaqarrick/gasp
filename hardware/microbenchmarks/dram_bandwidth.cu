// dram_bandwidth.cu
// ---------------------------------------------------------------------------
// Measures : dram_bandwidth_bytes_per_cycle
// Method   : Streaming, fully coalesced reads of a buffer much larger than L2,
//            with every SM busy. Each thread issues independent 16-byte loads
//            (4 in flight per thread, grid-stride), so the memory system is
//            saturated and DRAM bandwidth, not latency, is the limit.
//
// Units    : bytes per SM clock cycle for the WHOLE GPU:
//              bytes_read / (kernel_span_ns * sm_clock_GHz)
//            kernel_span_ns = latest block end - earliest block start
//            (in-kernel %globaltimer, so launch overhead is excluded) and
//            sm_clock_GHz = clock64 cycles / globaltimer ns, measured inside
//            the same kernel (so it is the clock the SMs actually ran at).
//            A memory-bound kernel's GB/s does not depend on the SM clock, but
//            bytes/cycle does. So the benchmark first keeps the GPU busy for
//            WARMUP_MS, letting the SM clock reach its sustained value (it ramps
//            up from idle over several hundred ms), and then times the runs.
//
// Build    : nvcc -O3 -std=c++17 -arch=sm_80  dram_bandwidth.cu -o dram_bandwidth   (A100)
//            nvcc -O3 -std=c++17 -arch=sm_90  dram_bandwidth.cu -o dram_bandwidth   (H100)
//            nvcc -O3 -std=c++17 -arch=sm_100 dram_bandwidth.cu -o dram_bandwidth   (B200)
//            (-arch must match the GPU you run on, e.g. sm_75 for an RTX 2080 Ti)
//
// Run      : ./dram_bandwidth [--gpu NAME] [--out-dir DIR] [--reps N]
//                             [--working-set-kb N] [--l2-cache-size KB]
//            Writes  <out-dir>/dram_bandwidth_bytes_per_cycle.<gpu>.run.json
//            Default out-dir = ./hardware/runs   (created if missing)
//            --l2-cache-size KB = the L2 size to assume (default: the size the
//            GPU reports). Default working set = min(1 GiB, 1/4 of free GPU
//            memory), but never less than 4x L2. An explicit --working-set-kb
//            must be at least 2x the L2 size so reads cannot be served by L2.
//
// Inspect  : nvcc -O3 -arch=sm_80 -cubin dram_bandwidth.cu -o dram_bandwidth.cubin
//            cuobjdump -sass dram_bandwidth.cubin     (checklist at bottom)
// ---------------------------------------------------------------------------
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
#include <numeric>
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

// ---- Identity of this benchmark -------------------------------------------
static const char* PARAMETER      = "dram_bandwidth_bytes_per_cycle";
static const char* SCHEMA_VERSION = "1.1.0";   // bump if the JSON layout changes
static const char* BENCH_VERSION  = "1.1.0";   // bump if the measurement method changes

// ---- Experiment parameters -------------------------------------------------
constexpr int  THREADS          = 256;     // threads per block
constexpr int  UNROLL           = 4;       // independent 16-byte loads in flight per thread
constexpr int  VEC_BYTES        = 16;      // one uint4 per load
constexpr int  WARMUP_LAUNCHES  = 5;       // minimum untimed launches
constexpr double WARMUP_MS      = 2000.0;  // keep the GPU busy this long so the SM clock settles
constexpr int  DEFAULT_REPS     = 200;     // timed launches (>=100)
constexpr int  DEFAULT_MULT_L2  = 4;       // default working set is at least 4 x L2
constexpr int  MIN_L2_MULT      = 2;       // explicit working set must be >= 2 x L2
constexpr long DEFAULT_WS_KB    = 1024L * 1024L;   // 1 GiB

// ---- Device ----------------------------------------------------------------
__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t c;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)::"memory");
    return c;
}

__device__ __forceinline__ uint64_t read_globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
    return t;
}

// ts[4*block + {0,1,2,3}] = {timer_start_ns, timer_end_ns, clock_start, clock_end}
// n_vec is a multiple of (gridDim.x * blockDim.x * UNROLL), so every load in the
// loop is in range without a tail case.
__global__ void __launch_bounds__(THREADS)
dram_read_kernel(const uint4* __restrict__ buf, size_t n_vec, uint64_t* ts, uint32_t* sink) {
    const size_t tid    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    uint64_t c0 = 0, t0 = 0;
    if (threadIdx.x == 0) { c0 = read_clock64(); t0 = read_globaltimer(); }

    uint32_t acc = 0;
    for (size_t idx = tid; idx < n_vec; idx += stride * UNROLL) {
        uint4 v[UNROLL];
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            const uint4* addr = buf + idx + static_cast<size_t>(k) * stride;
            // Volatile PTX: the compiler cannot delete or merge these loads.
            // .cs = cache-streaming (evict-first); consecutive threads read
            // consecutive 16-byte words, so each warp reads 512 contiguous bytes.
            asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(v[k].x), "=r"(v[k].y), "=r"(v[k].z), "=r"(v[k].w)
                         : "l"(addr));
        }
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) acc ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
    }

    __syncthreads();   // the block ends when its slowest thread ends
    if (threadIdx.x == 0) {
        ts[4 * blockIdx.x + 0] = t0;
        ts[4 * blockIdx.x + 1] = read_globaltimer();
        ts[4 * blockIdx.x + 2] = c0;
        ts[4 * blockIdx.x + 3] = read_clock64();
    }
    sink[tid] = acc;   // keeps the loaded data observable
}

// ---- Host helpers ----------------------------------------------------------
static std::string run_cmd(const char* cmd) {
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "";
    char buf[256];
    std::string s;
    while (fgets(buf, sizeof(buf), pipe)) s += buf;
    pclose(pipe);
    while (!s.empty() && isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

static std::string json_escape(const std::string& in) {
    std::string o;
    for (char c : in) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
    return o;
}

static std::string cuda_ver_string(int v) {   // 12080 -> "12.8"
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

static std::string short_gpu_name(const std::string& full) {
    std::string l;
    for (char c : full) l += static_cast<char>(tolower(static_cast<unsigned char>(c)));
    for (const char* k : {"a100", "h100", "h200", "b200", "b100"})
        if (l.find(k) != std::string::npos) return k;
    std::string s;
    for (char c : l) s += isalnum(static_cast<unsigned char>(c)) ? c : '_';
    return s;
}

static std::string utc_timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char b[32];
    std::strftime(b, sizeof(b), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return b;
}

static double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct Stats { double median, mean, stddev, mn, mx; };
static Stats compute_stats(const std::vector<double>& v) {
    Stats s;
    s.median = median_of(v);
    s.mean   = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double var = 0.0;
    for (double x : v) var += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(var / (v.size() - 1));
    s.mn = *std::min_element(v.begin(), v.end());
    s.mx = *std::max_element(v.begin(), v.end());
    return s;
}

static void json_array(std::ostringstream& j, const char* key, const std::vector<double>& v) {
    j << "  \"" << key << "\": [";
    for (size_t i = 0; i < v.size(); ++i) j << (i ? ", " : "") << v[i];
    j << "],\n";
}

// ---- Main ------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string out_dir = "./hardware/runs";
    std::string gpu_override;
    int reps = DEFAULT_REPS;
    long working_set_kb = 0;   // 0 = choose automatically
    long l2_cache_kb = 0;      // 0 = use the size the GPU reports
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--gpu") && i + 1 < argc) gpu_override = argv[++i];
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--working-set-kb") && i + 1 < argc) working_set_kb = atol(argv[++i]);
        else if (!strcmp(argv[i], "--l2-cache-size") && i + 1 < argc) l2_cache_kb = atol(argv[++i]);
        else {
            fprintf(stderr, "Usage: %s [--gpu NAME] [--out-dir DIR] [--reps N] "
                            "[--working-set-kb N] [--l2-cache-size KB]\n", argv[0]);
            return 1;
        }
    }
    if (reps < 100) { fprintf(stderr, "reps must be >= 100 (issue #3)\n"); return 1; }
    if (l2_cache_kb < 0) { fprintf(stderr, "--l2-cache-size must be positive (KB)\n"); return 1; }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const std::string gpu_name = prop.name;
    const std::string gpu = gpu_override.empty() ? short_gpu_name(gpu_name) : gpu_override;
    const std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);
    const long l2_kb_used = l2_cache_kb ? l2_cache_kb : static_cast<long>(prop.l2CacheSize) / 1024;

    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    // Working set: at least 4x L2, normally 1 GiB (capped at 1/4 of free memory).
    if (working_set_kb == 0) {
        long cap_kb = static_cast<long>(free_b / 1024 / 4);
        working_set_kb = std::min(DEFAULT_WS_KB, cap_kb);
        working_set_kb = std::max(working_set_kb, static_cast<long>(DEFAULT_MULT_L2) * l2_kb_used);
    }
    if (working_set_kb < MIN_L2_MULT * l2_kb_used) {
        fprintf(stderr, "working set (%ld KB) must be >= %dx the L2 size (%ld KB) so reads cannot be "
                        "served by L2\n", working_set_kb, MIN_L2_MULT, l2_kb_used);
        return 1;
    }
    if (static_cast<size_t>(working_set_kb) * 1024 > free_b / 10 * 8) {
        fprintf(stderr, "ERROR: working set (%ld KB) does not fit in free device memory (%zu bytes)\n",
                working_set_kb, free_b);
        return 1;
    }

    // Launch shape: fill every SM to its occupancy limit, in a single wave.
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, dram_read_kernel, THREADS, 0));
    if (blocks_per_sm < 1) { fprintf(stderr, "kernel cannot launch on this GPU\n"); return 1; }
    const int grid = blocks_per_sm * prop.multiProcessorCount;
    const size_t grid_threads = static_cast<size_t>(grid) * THREADS;

    // Round the buffer down to a whole number of grid-wide, unrolled passes.
    const size_t granule = grid_threads * UNROLL * VEC_BYTES;
    const size_t requested_bytes = static_cast<size_t>(working_set_kb) * 1024;
    const size_t bytes_used = (requested_bytes / granule) * granule;
    if (bytes_used == 0) { fprintf(stderr, "working set smaller than one pass (%zu bytes)\n", granule); return 1; }
    const size_t n_vec = bytes_used / VEC_BYTES;

    fprintf(stderr, "GPU: %s (%s), SMs: %d, L2: %.1f MB (%s), working set: %.1f MB, grid: %d x %d\n",
            gpu_name.c_str(), arch.c_str(), prop.multiProcessorCount, l2_kb_used / 1024.0,
            l2_cache_kb ? "user-supplied" : "device-reported", bytes_used / (1024.0 * 1024.0),
            grid, THREADS);

    int rt_ver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt_ver));

    uint8_t*  d_buf = nullptr;
    uint64_t* d_ts = nullptr;
    uint32_t* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, bytes_used));
    CUDA_CHECK(cudaMalloc(&d_ts, static_cast<size_t>(grid) * 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sink, grid_threads * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_buf, 0x5A, bytes_used));   // non-zero data

    const uint4* d_vec = reinterpret_cast<const uint4*>(d_buf);
    std::vector<uint64_t> h_ts(static_cast<size_t>(grid) * 4);

    // Warm-up by TIME, not by launch count: the SM clock ramps up from idle over
    // several hundred ms, and bytes/cycle depends on it.
    int warmup_launches_done = 0;
    {
        const auto t_start = std::chrono::steady_clock::now();
        double elapsed_ms = 0.0;
        do {
            dram_read_kernel<<<grid, THREADS>>>(d_vec, n_vec, d_ts, d_sink);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            ++warmup_launches_done;
            elapsed_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t_start).count();
        } while (warmup_launches_done < WARMUP_LAUNCHES || elapsed_ms < WARMUP_MS);
    }

    std::vector<double> kernel_ns, clock_ghz, gb_per_s, bytes_per_cycle;
    for (int r = 0; r < reps; ++r) {
        dram_read_kernel<<<grid, THREADS>>>(d_vec, n_vec, d_ts, d_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_ts.data(), d_ts, h_ts.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        // Kernel span = latest block end - earliest block start.
        uint64_t t_first = UINT64_MAX, t_last = 0;
        std::vector<double> ghz_per_block;
        ghz_per_block.reserve(grid);
        for (int b = 0; b < grid; ++b) {
            const uint64_t ts0 = h_ts[4 * b + 0], ts1 = h_ts[4 * b + 1];
            const uint64_t cs0 = h_ts[4 * b + 2], cs1 = h_ts[4 * b + 3];
            t_first = std::min(t_first, ts0);
            t_last  = std::max(t_last, ts1);
            if (ts1 > ts0) ghz_per_block.push_back(static_cast<double>(cs1 - cs0) / static_cast<double>(ts1 - ts0));
        }
        if (t_last <= t_first || ghz_per_block.empty()) { fprintf(stderr, "bad timestamps in rep %d\n", r); return 1; }

        const double span_ns = static_cast<double>(t_last - t_first);
        const double ghz     = median_of(ghz_per_block);        // cycles per ns
        kernel_ns.push_back(span_ns);
        clock_ghz.push_back(ghz);
        gb_per_s.push_back(static_cast<double>(bytes_used) / span_ns);              // bytes/ns = GB/s
        bytes_per_cycle.push_back(static_cast<double>(bytes_used) / (span_ns * ghz));
    }

    const Stats st  = compute_stats(bytes_per_cycle);
    const double med_gbs = median_of(gb_per_s);
    const double med_ghz = median_of(clock_ghz);

    // Theoretical peak, for a sanity check only (these properties are removed in CUDA 13).
    double peak_gbs = 0.0;
#if defined(CUDART_VERSION) && CUDART_VERSION < 13000
    peak_gbs = 2.0 * (static_cast<double>(prop.memoryClockRate) * 1e3) * (prop.memoryBusWidth / 8.0) / 1e9;
#endif
    const double pct_peak = peak_gbs > 0.0 ? 100.0 * med_gbs / peak_gbs : 0.0;

    if (peak_gbs > 0.0 && pct_peak > 105.0)
        fprintf(stderr, "WARNING: %.0f GB/s is above the theoretical peak (%.0f GB/s); reads may be "
                        "served by L2 (try a larger --working-set-kb).\n", med_gbs, peak_gbs);
    if (peak_gbs > 0.0 && pct_peak < 50.0)
        fprintf(stderr, "WARNING: only %.0f%% of theoretical peak; another process may be sharing "
                        "the GPU, or the GPU is in a low-power state. Check nvidia-smi and rerun.\n", pct_peak);
    // Contamination check on GB/s (the physical quantity, independent of the SM clock).
    const Stats gst = compute_stats(gb_per_s);
    if (gst.median < 0.95 * gst.mx)
        fprintf(stderr, "WARNING: median %.1f GB/s is >5%% below the best run (%.1f GB/s). Another "
                        "process may be sharing the GPU; check nvidia-smi and rerun.\n", gst.median, gst.mx);

    // Clock-drift check: bytes/cycle is only stable if the SM clock is.
    {
        const size_t n = clock_ghz.size();
        const size_t w = std::max<size_t>(1, std::min<size_t>(20, n / 4));
        const double first = median_of(std::vector<double>(clock_ghz.begin(), clock_ghz.begin() + w));
        const double last  = median_of(std::vector<double>(clock_ghz.end() - w, clock_ghz.end()));
        if (std::fabs(last - first) / last > 0.02)
            fprintf(stderr, "WARNING: SM clock drifted from %.3f to %.3f GHz during timing, so "
                            "bytes/cycle is not stable. Rerun (the GPU may still be ramping or "
                            "throttling).\n", first, last);
    }

    // ---- Build JSON --------------------------------------------------------
    std::string driver = run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1");
    std::string commit = run_cmd("git rev-parse --short HEAD 2>/dev/null");
    if (driver.empty()) driver = "unknown";
    if (commit.empty()) commit = "unknown";

    std::ostringstream j;
    j.setf(std::ios::fixed);
    j.precision(4);
    j << "{\n";
    j << "  \"schema_version\": \"" << SCHEMA_VERSION << "\",\n";
    j << "  \"benchmark\": \"dram_bandwidth.cu\",\n";
    j << "  \"benchmark_version\": \"" << BENCH_VERSION << "\",\n";
    j << "  \"parameter\": \"" << PARAMETER << "\",\n";
    j << "  \"value\": " << st.median << ",\n";
    j << "  \"unit\": \"bytes/cycle\",\n";
    j << "  \"gpu\": \"" << json_escape(gpu) << "\",\n";
    j << "  \"gpu_name\": \"" << json_escape(gpu_name) << "\",\n";
    j << "  \"arch\": \"" << arch << "\",\n";
    j << "  \"num_sms\": " << prop.multiProcessorCount << ",\n";
    j << "  \"l2_cache_bytes\": " << prop.l2CacheSize << ",\n";
    j << "  \"stats\": {\n";
    j << "    \"median\": " << st.median << ",\n";
    j << "    \"mean\": " << st.mean << ",\n";
    j << "    \"std\": " << st.stddev << ",\n";
    j << "    \"min\": " << st.mn << ",\n";
    j << "    \"max\": " << st.mx << "\n";
    j << "  },\n";
    // Per-run arrays, one entry per timed launch:
    //   raw_kernel_ns          : latest block end - earliest block start (in-kernel %globaltimer)
    //   sm_clock_ghz_samples   : median over blocks of (clock64 delta / globaltimer ns delta)
    //   gb_per_s_samples       : bytes_read / raw_kernel_ns
    //   per_run_bytes_per_cycle: bytes_read / (raw_kernel_ns * sm_clock_ghz); stats are computed over these
    json_array(j, "raw_kernel_ns", kernel_ns);
    json_array(j, "sm_clock_ghz_samples", clock_ghz);
    json_array(j, "gb_per_s_samples", gb_per_s);
    json_array(j, "per_run_bytes_per_cycle", bytes_per_cycle);
    j << "  \"derived\": {\n";
    j << "    \"bandwidth_gb_per_s_median\": " << med_gbs << ",\n";
    j << "    \"sm_clock_ghz_median\": " << med_ghz << ",\n";
    if (peak_gbs > 0.0) {
        j << "    \"theoretical_peak_gb_per_s\": " << peak_gbs << ",\n";
        j << "    \"percent_of_theoretical_peak\": " << pct_peak << "\n";
    } else {
        j << "    \"theoretical_peak_gb_per_s\": null,\n";
        j << "    \"percent_of_theoretical_peak\": null\n";
    }
    j << "  },\n";
    j << "  \"measurement\": {\n";
    j << "    \"method\": \"streaming coalesced reads, all SMs active, working set exceeds L2\",\n";
    j << "    \"load_instruction\": \"ld.global.cs.v4.u32\",\n";
    j << "    \"timer\": \"globaltimer (kernel span) and clock64 (SM clock)\",\n";
    j << "    \"working_set_bytes\": " << bytes_used << ",\n";
    j << "    \"working_set_requested_bytes\": " << requested_bytes << ",\n";
    j << "    \"l2_cache_size_kb_assumed\": " << l2_kb_used << ",\n";
    j << "    \"l2_cache_size_source\": \"" << (l2_cache_kb ? "user" : "device") << "\",\n";
    j << "    \"bytes_per_load\": " << VEC_BYTES << ",\n";
    j << "    \"loads_in_flight_per_thread\": " << UNROLL << ",\n";
    j << "    \"blocks\": " << grid << ",\n";
    j << "    \"blocks_per_sm\": " << blocks_per_sm << ",\n";
    j << "    \"threads_per_block\": " << THREADS << ",\n";
    j << "    \"warmup_launches\": " << warmup_launches_done << ",\n";
    j << "    \"warmup_ms_min\": " << WARMUP_MS << ",\n";
    j << "    \"reps\": " << reps << "\n";
    j << "  },\n";
    j << "  \"cuda_version\": \"" << cuda_ver_string(rt_ver) << "\",\n";
    j << "  \"nvcc_version\": \"" << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "\",\n";
    j << "  \"driver_version\": \"" << json_escape(driver) << "\",\n";
    j << "  \"git_commit\": \"" << json_escape(commit) << "\",\n";
    j << "  \"timestamp\": \"" << utc_timestamp() << "\"\n";
    j << "}\n";

    // ---- Save to <out_dir>/{parameter}.{gpu}.run.json -----------------------
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) { fprintf(stderr, "Cannot create %s: %s\n", out_dir.c_str(), ec.message().c_str()); return 1; }
    fs::path out_path = fs::path(out_dir) / (std::string(PARAMETER) + "." + gpu + ".run.json");

    FILE* f = fopen(out_path.string().c_str(), "w");
    if (!f) { perror("fopen"); return 1; }
    fputs(j.str().c_str(), f);
    fclose(f);

    printf("%s", j.str().c_str());
    fprintf(stderr, "\nDRAM bandwidth: %.2f bytes/cycle (%.1f GB/s at %.3f GHz, std %.3f)\nSaved: %s\n",
            st.median, med_gbs, med_ghz, st.stddev, out_path.string().c_str());

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_ts));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}

// ---------------------------------------------------------------------------
// SASS inspection checklist (required by issue #11):
//   cuobjdump -sass dram_bandwidth.cubin
// In dram_read_kernel you should see:
//   * 4 back-to-back 128-bit global loads (LDG.E.128, carrying a streaming /
//     evict-first modifier) issued before their results are used, so several
//     loads per thread are in flight at once
//   * clock reads (CS2R) and globaltimer reads (S2UR / CS2R) at the start and
//     end of the kernel, only for thread 0 of each block
//   * no extra loads or stores inside the loop, and the loop must not have been
//     removed: if the loads are missing the bandwidth will look impossibly high
// ---------------------------------------------------------------------------
