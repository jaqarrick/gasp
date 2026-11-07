// l2_latency.cu
// ---------------------------------------------------------------------------
// Measures : l2_cache_hit_latency_cycles
// Method   : Pointer-chasing (dependent load chain). The working set is larger
//            than L1 but fits in L2, and the loads use ld.global.cg (cache at
//            L2, bypass L1), so every timed load is an L2 hit.
//            Clock-read overhead is measured in the same kernel and subtracted.
//
// Build    : nvcc -O3 -std=c++17 -arch=sm_80  l2_latency.cu -o l2_latency   (A100)
//            nvcc -O3 -std=c++17 -arch=sm_90  l2_latency.cu -o l2_latency   (H100)
//            nvcc -O3 -std=c++17 -arch=sm_100 l2_latency.cu -o l2_latency   (B200)
//            (-arch must match the GPU you run on, e.g. sm_75 for an RTX 2080 Ti)
//
// Run      : ./l2_latency [--gpu NAME] [--out-dir DIR] [--reps N]
//                         [--working-set-kb N] [--l1-cache-size KB] [--l2-cache-size KB]
//            --l2-cache-size KB = the L2 size to assume when checking that the
//            working set fits in L2. Default: the size the GPU reports.
//            Writes  <out-dir>/l2_cache_hit_latency_cycles.<gpu>.run.json
//            Default out-dir = ./hardware/runs   (created if missing)
//            --l1-cache-size KB = the L1 size to assume (default 256, the largest
//            L1 among the targets, so the working set surely exceeds it). Unless
//            --working-set-kb is given, the working set is 8x this (default
//            2048 KB = 2 MB), which is inside every target L2 (>= 5.5 MB on
//            Turing, 40 MB on A100). An explicit --working-set-kb must be at
//            least 2x the assumed L1 size.
//            A larger --working-set-kb (e.g. 30000) probes the far L2 partition
//            on dual-partition parts (A100, B200); see issue #3.
//
// Inspect  : nvcc -O3 -arch=sm_80 -cubin l2_latency.cu -o l2_latency.cubin
//            cuobjdump -sass l2_latency.cubin     (checklist at bottom)
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
static const char* PARAMETER      = "l2_cache_hit_latency_cycles";
static const char* SCHEMA_VERSION = "1.1.0";   // bump if the JSON layout changes
static const char* BENCH_VERSION  = "1.2.0";   // bump if the measurement method changes

// ---- Experiment parameters -------------------------------------------------
// Default assumed L1 is the LARGEST among the targets (256 KB), so the working
// set (8x L1 = 2 MB) is guaranteed to exceed L1 and still fit in L2.
constexpr int DEFAULT_L1_CACHE_KB = 256;
constexpr int WORKING_SET_L1_MULT = 8;         // default working set = 8 x L1
constexpr int MIN_L1_MULT         = 2;         // explicit working set must be >= 2 x L1
constexpr int LINE_BYTES   = 128;
constexpr int CHASE_LOADS  = 2048;   // dependent loads in the timed region
constexpr int UNROLL       = 16;     // loads per loop iteration
constexpr int DEFAULT_REPS = 200;    // independent kernel launches (>=100)

static_assert(CHASE_LOADS % UNROLL == 0, "CHASE_LOADS must be a multiple of UNROLL");

// ---- Device ----------------------------------------------------------------
__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t c;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)::"memory");
    return c;
}

// Dependent load: the same register is address and destination, so there is no
// address arithmetic between loads. ".cg" = cache at L2 only, bypass L1, so the
// access is served by L2 even if the line would otherwise fit in L1.
#define CHASE_ONE asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p)::"memory");

// out[0] = gross cycles of the timed chain (clock overhead included)
// out[1] = clock overhead (two back-to-back clock reads, nothing between)
__global__ void l2_chase_kernel(uint64_t start, int num_nodes, long long* out, uint64_t* sink) {
    uint64_t p = start;

    // 1) Warm-up: two full laps of the ring. Lap 1 pulls every line from DRAM
    //    into L2; lap 2 makes sure nothing is still in flight.
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

    // 3) Timed region: CHASE_LOADS strictly dependent L2-hit loads.
    uint64_t t0 = read_clock64();
#pragma unroll 1
    for (int i = 0; i < CHASE_LOADS; i += UNROLL) {
        CHASE_ONE CHASE_ONE CHASE_ONE CHASE_ONE
        CHASE_ONE CHASE_ONE CHASE_ONE CHASE_ONE
        CHASE_ONE CHASE_ONE CHASE_ONE CHASE_ONE
        CHASE_ONE CHASE_ONE CHASE_ONE CHASE_ONE
    }
    uint64_t t1 = read_clock64();

    out[0] = static_cast<long long>(t1 - t0);
    out[1] = static_cast<long long>(b1 - b0);
    *sink  = p;  // keep the chain observable
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

// ---- Main ------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string out_dir = "./hardware/runs";
    std::string gpu_override;
    int reps = DEFAULT_REPS;
    long working_set_kb = 0;   // 0 = derive from the assumed L1 size
    long l1_cache_kb = DEFAULT_L1_CACHE_KB;
    long l2_cache_kb = 0;      // 0 = use the size the GPU reports
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--gpu") && i + 1 < argc) gpu_override = argv[++i];
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--working-set-kb") && i + 1 < argc) working_set_kb = atol(argv[++i]);
        else if (!strcmp(argv[i], "--l1-cache-size") && i + 1 < argc) l1_cache_kb = atol(argv[++i]);
        else if (!strcmp(argv[i], "--l2-cache-size") && i + 1 < argc) l2_cache_kb = atol(argv[++i]);
        else {
            fprintf(stderr, "Usage: %s [--gpu NAME] [--out-dir DIR] [--reps N] "
                            "[--working-set-kb N] [--l1-cache-size KB] [--l2-cache-size KB]\n", argv[0]);
            return 1;
        }
    }
    if (reps < 100) { fprintf(stderr, "reps must be >= 100 (issue #3)\n"); return 1; }
    if (l1_cache_kb < 8) { fprintf(stderr, "--l1-cache-size must be >= 8 (KB)\n"); return 1; }
    if (l2_cache_kb < 0) { fprintf(stderr, "--l2-cache-size must be positive (KB)\n"); return 1; }
    if (working_set_kb == 0) working_set_kb = WORKING_SET_L1_MULT * l1_cache_kb;
    if (working_set_kb < MIN_L1_MULT * l1_cache_kb) {
        fprintf(stderr, "working set (%ld KB) must be >= %dx the assumed L1 size (%ld KB) "
                        "so it exceeds L1\n", working_set_kb, MIN_L1_MULT, l1_cache_kb);
        return 1;
    }

    const size_t working_set_bytes = static_cast<size_t>(working_set_kb) * 1024;
    const int num_nodes = static_cast<int>(working_set_bytes / LINE_BYTES);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const std::string gpu_name = prop.name;
    const std::string gpu = gpu_override.empty() ? short_gpu_name(gpu_name) : gpu_override;
    const std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);
    // L2 size used for the "fits in L2" check: the user's value if given, else
    // what the GPU reports.
    const long l2_kb_used = l2_cache_kb ? l2_cache_kb : static_cast<long>(prop.l2CacheSize) / 1024;
    fprintf(stderr, "GPU: %s (%s), SMs: %d, L2: %.1f MB (%s), assumed L1: %ld KB, working set: %ld KB\n",
            gpu_name.c_str(), arch.c_str(), prop.multiProcessorCount,
            l2_kb_used / 1024.0, l2_cache_kb ? "user-supplied" : "device-reported",
            l1_cache_kb, working_set_kb);

    if (working_set_bytes >= static_cast<size_t>(l2_kb_used) * 1024) {
        fprintf(stderr, "ERROR: working set (%zu bytes) does not fit in L2 (%ld KB); "
                        "this would measure DRAM, not L2.\n",
                working_set_bytes, l2_kb_used);
        return 1;
    }

    int rt_ver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt_ver));

    // Pointer ring: node i's first 8 bytes hold the device address of the next
    // node. Sattolo's shuffle gives ONE cycle covering all nodes in random order,
    // so no prefetcher or stride logic can predict the next address.
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
    CUDA_CHECK(cudaMalloc(&d_out, 2 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));

    std::vector<long long> raw_cycles, overheads;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaMemset(d_out, 0, 2 * sizeof(long long)));
        l2_chase_kernel<<<1, 1>>>(reinterpret_cast<uint64_t>(d_buf), num_nodes, d_out, d_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        long long h[2];
        CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
        raw_cycles.push_back(h[0]);
        overheads.push_back(h[1]);
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

    const double median = median_of(per_load);
    const double mean   = std::accumulate(per_load.begin(), per_load.end(), 0.0) / reps;
    double var = 0.0;
    for (double s : per_load) var += (s - mean) * (s - mean);
    const double stddev = std::sqrt(var / (reps - 1));
    const double mn = *std::min_element(per_load.begin(), per_load.end());
    const double mx = *std::max_element(per_load.begin(), per_load.end());

    if (median < 100.0)
        fprintf(stderr, "WARNING: median %.1f cycles is low for an L2 hit; loads may be hitting L1 "
                        "(check SASS for the .cg cache operator).\n", median);
    if (median > 600.0)
        fprintf(stderr, "WARNING: median %.1f cycles is high for an L2 hit; the working set may be "
                        "spilling to DRAM.\n", median);

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
    j << "  \"benchmark\": \"l2_latency.cu\",\n";
    j << "  \"benchmark_version\": \"" << BENCH_VERSION << "\",\n";
    j << "  \"parameter\": \"" << PARAMETER << "\",\n";
    j << "  \"value\": " << median << ",\n";
    j << "  \"unit\": \"cycles\",\n";
    j << "  \"gpu\": \"" << json_escape(gpu) << "\",\n";
    j << "  \"gpu_name\": \"" << json_escape(gpu_name) << "\",\n";
    j << "  \"arch\": \"" << arch << "\",\n";
    j << "  \"num_sms\": " << prop.multiProcessorCount << ",\n";
    j << "  \"l2_cache_bytes\": " << prop.l2CacheSize << ",\n";
    j << "  \"stats\": {\n";
    j << "    \"median\": " << median << ",\n";
    j << "    \"mean\": " << mean << ",\n";
    j << "    \"std\": " << stddev << ",\n";
    j << "    \"min\": " << mn << ",\n";
    j << "    \"max\": " << mx << "\n";
    j << "  },\n";
    // raw_cycles           : per run, clock64 delta around the chain, clock-read cost INCLUDED
    // overhead_samples     : per run, clock64 delta of two back-to-back reads (cost of the clock)
    // overhead_cycles      : median of overhead_samples; the value subtracted from every run
    // net_cycles           : raw_cycles - overhead_cycles (per run)
    // per_load_cycles      : net_cycles / loads_per_rep (per run); stats are computed over these
    j << "  \"raw_cycles\": [";
    for (size_t i = 0; i < raw_cycles.size(); ++i) j << (i ? ", " : "") << raw_cycles[i];
    j << "],\n";
    j << "  \"overhead_samples_cycles\": [";
    for (size_t i = 0; i < overheads.size(); ++i) j << (i ? ", " : "") << overheads[i];
    j << "],\n";
    j << "  \"overhead_cycles\": " << overhead << ",\n";
    j << "  \"overhead_subtracted\": true,\n";
    j << "  \"net_cycles\": [";
    for (size_t i = 0; i < net_cycles.size(); ++i) j << (i ? ", " : "") << net_cycles[i];
    j << "],\n";
    j << "  \"per_load_cycles\": [";
    for (size_t i = 0; i < per_load.size(); ++i) j << (i ? ", " : "") << per_load[i];
    j << "],\n";
    j << "  \"uncorrected_value\": " << uncorrected_value << ",\n";
    j << "  \"measurement\": {\n";
    j << "    \"method\": \"pointer-chase, working set exceeds L1 and fits in L2\",\n";
    j << "    \"load_instruction\": \"ld.global.cg.u64\",\n";
    j << "    \"timer\": \"clock64\",\n";
    j << "    \"working_set_bytes\": " << working_set_bytes << ",\n";
    j << "    \"l1_cache_size_kb_assumed\": " << l1_cache_kb << ",\n";
    j << "    \"l2_cache_size_kb_assumed\": " << l2_kb_used << ",\n";
    j << "    \"l2_cache_size_source\": \"" << (l2_cache_kb ? "user" : "device") << "\",\n";
    j << "    \"line_bytes\": " << LINE_BYTES << ",\n";
    j << "    \"loads_per_rep\": " << CHASE_LOADS << ",\n";
    j << "    \"reps\": " << reps << ",\n";
    j << "    \"blocks\": 1,\n";
    j << "    \"threads_per_block\": 1\n";
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
    fprintf(stderr, "\nL2 hit latency: %.2f cycles (std %.3f, overhead %.1f cycles subtracted)\nSaved: %s\n",
            median, stddev, overhead, out_path.string().c_str());

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}

// ---------------------------------------------------------------------------
// SASS inspection checklist (required by issue #11):
//   cuobjdump -sass l2_latency.cubin
// In l2_chase_kernel you should see:
//   * clock reads (CS2R) back-to-back for the baseline, then one before and
//     one after the timed loop
//   * inside the timed loop, 16 consecutive 64-bit LDG instructions carrying a
//     cache-operator modifier (the .cg load; the exact text differs from the
//     plain LDG in l1_latency and varies by architecture), each address
//     register being the previous instruction's destination
//   * no IADD/IMAD/LEA between those loads
// If loads were removed, merged or reordered, the result is invalid.
// ---------------------------------------------------------------------------
