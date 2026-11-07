// dram_latency.cu
// ---------------------------------------------------------------------------
// Measures : dram_latency_cycles
// Method   : Pointer-chasing (dependent load chain). The working set is several
//            times larger than L2, so the timed loads miss L2 and are served by
//            DRAM. Loads use ld.global.cg (cache at L2 only, bypass L1).
//            Clock-read overhead is measured in the same kernel and subtracted.
//
// Build    : nvcc -O3 -std=c++17 -arch=sm_80  dram_latency.cu -o dram_latency   (A100)
//            nvcc -O3 -std=c++17 -arch=sm_90  dram_latency.cu -o dram_latency   (H100)
//            nvcc -O3 -std=c++17 -arch=sm_100 dram_latency.cu -o dram_latency   (B200)
//            (-arch must match the GPU you run on, e.g. sm_75 for an RTX 2080 Ti)
//
// Run      : ./dram_latency [--gpu NAME] [--out-dir DIR] [--reps N]
//                           [--working-set-kb N] [--l2-cache-size KB]
//            Writes  <out-dir>/dram_latency_cycles.<gpu>.run.json
//            Default out-dir = ./hardware/runs   (created if missing)
//            --l2-cache-size KB = the L2 size to assume (default: the size the
//            GPU reports). Unless --working-set-kb is given, the working set is
//            4x the L2 size (22 MB on a 2080 Ti, 160 MB on A100, 200 MB on H100).
//            An explicit --working-set-kb must be at least 2x the L2 size.
//
// Notes    : * Every rep starts at a different point on the ring and walks on
//              from where the previous rep stopped, so a line is not revisited
//              until the whole ring (>= 4x L2) has been walked. That keeps lines
//              from one rep out of L2 in the next.
//            * A working set this large also touches many pages, so the result
//              includes some address-translation (TLB) cost, as in most
//              published DRAM-latency measurements.
//
// Inspect  : nvcc -O3 -arch=sm_80 -cubin dram_latency.cu -o dram_latency.cubin
//            cuobjdump -sass dram_latency.cubin     (checklist at bottom)
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
static const char* PARAMETER      = "dram_latency_cycles";
static const char* SCHEMA_VERSION = "1.1.0";   // bump if the JSON layout changes
static const char* BENCH_VERSION  = "1.0.0";   // bump if the measurement method changes

// ---- Experiment parameters -------------------------------------------------
constexpr int WORKING_SET_L2_MULT = 4;     // default working set = 4 x L2
constexpr int MIN_L2_MULT         = 2;     // explicit working set must be >= 2 x L2
constexpr int LINE_BYTES    = 128;
constexpr int WARMUP_LOADS  = 256;   // dependent loads before timing (TLB / icache / clocks)
constexpr int CHASE_LOADS   = 1024;  // dependent loads in the timed region
constexpr int UNROLL        = 16;    // loads per loop iteration
constexpr int DEFAULT_REPS  = 200;   // independent kernel launches (>=100)
constexpr int STEPS_PER_REP = WARMUP_LOADS + CHASE_LOADS;

static_assert(CHASE_LOADS % UNROLL == 0, "CHASE_LOADS must be a multiple of UNROLL");

// ---- Device ----------------------------------------------------------------
__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t c;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)::"memory");
    return c;
}

// Dependent load: the same register is address and destination, so there is no
// address arithmetic between loads. ".cg" = cache at L2 only, bypass L1.
#define CHASE_ONE asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p)::"memory");

// out[0] = gross cycles of the timed chain (clock overhead included)
// out[1] = clock overhead (two back-to-back clock reads, nothing between)
__global__ void dram_chase_kernel(uint64_t start, long long* out, uint64_t* sink) {
    uint64_t p = start;

    // 1) Short warm-up. (No full lap of the ring: it is far too large, and we
    //    WANT the timed loads to miss L2.)
#pragma unroll 1
    for (int i = 0; i < WARMUP_LOADS; ++i) {
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

    // 3) Timed region: CHASE_LOADS strictly dependent DRAM loads.
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
    long working_set_kb = 0;   // 0 = derive from the L2 size
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

    // L2 size used to size the working set: the user's value if given, else
    // what the GPU reports.
    const long l2_kb_used = l2_cache_kb ? l2_cache_kb : static_cast<long>(prop.l2CacheSize) / 1024;
    if (working_set_kb == 0) working_set_kb = WORKING_SET_L2_MULT * l2_kb_used;
    if (working_set_kb < MIN_L2_MULT * l2_kb_used) {
        fprintf(stderr, "working set (%ld KB) must be >= %dx the L2 size (%ld KB) "
                        "so it clearly exceeds L2\n", working_set_kb, MIN_L2_MULT, l2_kb_used);
        return 1;
    }

    const size_t working_set_bytes = static_cast<size_t>(working_set_kb) * 1024;
    const size_t num_nodes_sz = working_set_bytes / LINE_BYTES;
    if (num_nodes_sz > 0x7fffffffULL) { fprintf(stderr, "working set too large\n"); return 1; }
    const int num_nodes = static_cast<int>(num_nodes_sz);

    fprintf(stderr, "GPU: %s (%s), SMs: %d, L2: %.1f MB (%s), working set: %.1f MB\n",
            gpu_name.c_str(), arch.c_str(), prop.multiProcessorCount,
            l2_kb_used / 1024.0, l2_cache_kb ? "user-supplied" : "device-reported",
            working_set_kb / 1024.0);

    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (working_set_bytes > free_b / 10 * 8) {
        fprintf(stderr, "ERROR: working set (%zu bytes) does not fit in free device memory (%zu bytes)\n",
                working_set_bytes, free_b);
        return 1;
    }

    int rt_ver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt_ver));

    // Pointer ring: a random permutation `order` defines the walk order; the
    // first 8 bytes of node order[i] hold the device address of node
    // order[i+1] (wrapping). One cycle covers every node.
    uint8_t* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, working_set_bytes));

    std::vector<int> order(num_nodes);
    std::iota(order.begin(), order.end(), 0);
    {
        std::mt19937 rng(12345);
        std::shuffle(order.begin(), order.end(), rng);
    }
    std::vector<uint8_t> h_buf(working_set_bytes, 0);
    for (int i = 0; i < num_nodes; ++i) {
        const int cur = order[i];
        const int nxt = order[(i + 1) % num_nodes];
        uint64_t a = reinterpret_cast<uint64_t>(d_buf) + static_cast<uint64_t>(nxt) * LINE_BYTES;
        *reinterpret_cast<uint64_t*>(&h_buf[static_cast<size_t>(cur) * LINE_BYTES]) = a;
    }
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf.data(), working_set_bytes, cudaMemcpyHostToDevice));
    h_buf.clear();
    h_buf.shrink_to_fit();

    long long* d_out;
    uint64_t*  d_sink;
    CUDA_CHECK(cudaMalloc(&d_out, 2 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));

    std::vector<long long> raw_cycles, overheads;
    for (int r = 0; r < reps; ++r) {
        // Each rep resumes the walk where the previous rep ended, so it touches
        // lines that have not been used since a full ring ago (>= 4x L2).
        const size_t idx = (static_cast<size_t>(r) * STEPS_PER_REP) % static_cast<size_t>(num_nodes);
        const uint64_t start =
            reinterpret_cast<uint64_t>(d_buf) + static_cast<uint64_t>(order[idx]) * LINE_BYTES;

        CUDA_CHECK(cudaMemset(d_out, 0, 2 * sizeof(long long)));
        dram_chase_kernel<<<1, 1>>>(start, d_out, d_sink);
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

    if (median < 150.0)
        fprintf(stderr, "WARNING: median %.1f cycles is low for DRAM; loads may still be hitting L2 "
                        "(try a larger --working-set-kb).\n", median);
    if (median > 1.05 * mn)
        fprintf(stderr, "WARNING: median %.1f is >5%% above the fastest run (%.1f). Another process may "
                        "be sharing the GPU; check nvidia-smi and rerun.\n", median, mn);

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
    j << "  \"benchmark\": \"dram_latency.cu\",\n";
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
    j << "    \"method\": \"pointer-chase, working set exceeds L2 (includes some TLB cost)\",\n";
    j << "    \"load_instruction\": \"ld.global.cg.u64\",\n";
    j << "    \"timer\": \"clock64\",\n";
    j << "    \"working_set_bytes\": " << working_set_bytes << ",\n";
    j << "    \"l2_cache_size_kb_assumed\": " << l2_kb_used << ",\n";
    j << "    \"l2_cache_size_source\": \"" << (l2_cache_kb ? "user" : "device") << "\",\n";
    j << "    \"line_bytes\": " << LINE_BYTES << ",\n";
    j << "    \"warmup_loads\": " << WARMUP_LOADS << ",\n";
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
    fprintf(stderr, "\nDRAM latency: %.2f cycles (std %.3f, overhead %.1f cycles subtracted)\nSaved: %s\n",
            median, stddev, overhead, out_path.string().c_str());

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}

// ---------------------------------------------------------------------------
// SASS inspection checklist (required by issue #11):
//   cuobjdump -sass dram_latency.cubin
// In dram_chase_kernel you should see:
//   * clock reads (CS2R) back-to-back for the baseline, then one before and
//     one after the timed loop
//   * inside the timed loop, 16 consecutive 64-bit LDG instructions carrying
//     the same cache-operator modifier as l2_latency (.STRONG.GPU on Turing),
//     each address register being the previous instruction's destination
//   * no IADD/IMAD/LEA between those loads
// If loads were removed, merged or reordered, the result is invalid.
// ---------------------------------------------------------------------------
