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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

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
__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t c;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)::"memory");
    return c;
}

// out[0] = gross cycles of the timed loop (clock + barrier overhead included)
// out[1] = clock overhead (two back-to-back clock reads, nothing between)
__global__ void departure_kernel(const unsigned* __restrict__ buf, long long* out, unsigned* sink) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    // This warp owns ITERS_PER_WARP consecutive 128-byte lines; lane reads word `lane` of each.
    const unsigned* base = buf + static_cast<size_t>(warp) * ITERS_PER_WARP * 32 + lane;

    uint64_t b0 = 0, b1 = 0, t0 = 0;
    unsigned acc = 0;

    __syncthreads();
    if (threadIdx.x == 0) { b0 = read_clock64(); b1 = read_clock64(); }   // baseline
    __syncthreads();
    if (threadIdx.x == 0) t0 = read_clock64();

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
        out[1] = static_cast<long long>(b1 - b0);
    }
    sink[threadIdx.x] = acc;   // keeps the loaded data observable
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
    CUDA_CHECK(cudaMalloc(&d_out, 2 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sink, MAX_WARPS * 32 * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(d_buf, 0x01, buf_bytes));

    std::vector<SweepResult> sweep;
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
            CUDA_CHECK(cudaMemset(d_out, 0, 2 * sizeof(long long)));
            departure_kernel<<<1, threads>>>(d_buf, d_out, d_sink);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            long long h[2];
            CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
            sr.raw_cycles.push_back(h[0]);
            sr.overheads.push_back(h[1]);
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

    // Report the saturated (smallest) per-request time.
    size_t best = 0;
    for (size_t i = 1; i < sweep.size(); ++i)
        if (sweep[i].median < sweep[best].median) best = i;
    const SweepResult& S = sweep[best];
    const double requests_per_rep = static_cast<double>(S.warps) * ITERS_PER_WARP;

    std::vector<double> gross_per_request;
    for (long long c : S.raw_cycles) gross_per_request.push_back(static_cast<double>(c) / requests_per_rep);
    const double uncorrected_value = median_of(gross_per_request);

    const double median = S.median;
    const double mean = std::accumulate(S.per_request.begin(), S.per_request.end(), 0.0) / reps;
    double var = 0.0;
    for (double s : S.per_request) var += (s - mean) * (s - mean);
    const double stddev = std::sqrt(var / (reps - 1));
    const double mn = *std::min_element(S.per_request.begin(), S.per_request.end());
    const double mx = *std::max_element(S.per_request.begin(), S.per_request.end());

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
    if (median > 1.05 * mn)
        fprintf(stderr, "WARNING: median %.3f is >5%% above the fastest run (%.3f). Another process may "
                        "be sharing the GPU; check nvidia-smi and rerun.\n", median, mn);

    // ---- Build JSON --------------------------------------------------------
    std::string driver = run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1");
    if (driver.empty()) driver = "unknown";

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
    j << "  \"stats\": {\n";
    j << "    \"median\": " << median << ",\n";
    j << "    \"mean\": " << mean << ",\n";
    j << "    \"std\": " << stddev << ",\n";
    j << "    \"min\": " << mn << ",\n";
    j << "    \"max\": " << mx << "\n";
    j << "  },\n";
    // Arrays below are for the reported warp count (warps_per_block), one entry per timed launch:
    //   raw_cycles          : clock64 delta around the loop, clock-read cost INCLUDED
    //   overhead_samples    : clock64 delta of two back-to-back reads (cost of the clock)
    //   overhead_cycles     : median of overhead_samples; the value subtracted from every run
    //   net_cycles          : raw_cycles - overhead_cycles
    //   per_request_cycles  : net_cycles / requests_per_rep; stats are computed over these
    j << "  \"warps_per_block\": " << S.warps << ",\n";
    j << "  \"requests_per_rep\": " << static_cast<long long>(requests_per_rep) << ",\n";
    j << "  \"raw_cycles\": [";
    for (size_t i = 0; i < S.raw_cycles.size(); ++i) j << (i ? ", " : "") << S.raw_cycles[i];
    j << "],\n";
    j << "  \"overhead_samples_cycles\": [";
    for (size_t i = 0; i < S.overheads.size(); ++i) j << (i ? ", " : "") << S.overheads[i];
    j << "],\n";
    j << "  \"overhead_cycles\": " << S.overhead << ",\n";
    j << "  \"overhead_subtracted\": true,\n";
    j << "  \"net_cycles\": [";
    for (size_t i = 0; i < S.net_cycles.size(); ++i) j << (i ? ", " : "") << S.net_cycles[i];
    j << "],\n";
    j << "  \"per_request_cycles\": [";
    for (size_t i = 0; i < S.per_request.size(); ++i) j << (i ? ", " : "") << S.per_request[i];
    j << "],\n";
    j << "  \"uncorrected_value\": " << uncorrected_value << ",\n";
    j << "  \"sweep_median_cycles_per_request\": {";
    for (size_t i = 0; i < sweep.size(); ++i)
        j << (i ? ", " : "") << "\"" << sweep[i].warps << "_warps\": " << sweep[i].median;
    j << "},\n";
    j << "  \"cuda_version\": \"" << cuda_ver_string(rt_ver) << "\",\n";
    j << "  \"nvcc_version\": \"" << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "\",\n";
    j << "  \"driver_version\": \"" << json_escape(driver) << "\",\n";
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
    fprintf(stderr, "\nDeparture delay: %.3f cycles per warp request (std %.4f, overhead %.1f cycles subtracted)\nSaved: %s\n",
            median, stddev, S.overhead, out_path.string().c_str());

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
//     one after the timed loop, only for thread 0
//   * inside the loop, 8 consecutive 32-bit global loads (LDG.E, carrying the
//     same cache-operator modifier as l2_latency, e.g. .STRONG.GPU on Turing)
//     issued before any of their results is used
//   * no extra loads or stores in the loop, and the loop must still exist: if
//     the loads were removed the result will look impossibly small
// ---------------------------------------------------------------------------
