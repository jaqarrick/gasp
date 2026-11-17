// dram_latency.cu
// Measures dram_latency_cycles: how many cycles one load takes when the
// data comes from DRAM.
//
// One thread follows a chain of pointers over a buffer much bigger than L2
// (4x by default), so the loads miss L2 and go to DRAM. Each load needs the
// address from the previous one, so we time one load at a time. ld.global.cg
// skips L1. The cost of reading the clock is measured in the same kernel and
// subtracted.
//
// Run: ./dram_latency [--gpu NAME] [--out-dir DIR] [--reps N]
//                     [--working-set-kb N] [--l2-cache-size KB]

#include <random>

#include <cuda_runtime.h>
#include "common.h"

static const char* PARAMETER = "dram_latency_cycles";

constexpr int WORKING_SET_L2_MULT = 4;     // default buffer = 4 x L2
constexpr int MIN_L2_MULT         = 2;     // a custom buffer must be at least 2 x L2
constexpr int LINE_BYTES          = 128;   // one pointer per 128-byte line
constexpr int WARMUP_LOADS        = 256;   // loads before timing (TLB, clocks)
constexpr int CHASE_LOADS         = 1024;  // loads in the timed region
constexpr int UNROLL              = 16;    // loads per loop iteration
constexpr int DEFAULT_REPS        = 200;   // kernel launches (>= 100)
constexpr int LOADS_PER_REP       = WARMUP_LOADS + CHASE_LOADS;

static_assert(CHASE_LOADS % UNROLL == 0, "CHASE_LOADS must be a multiple of UNROLL");


//"asm volatile" stops the compiler from removing or merging the loads.
#define CHASE_ONE asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p)::"memory");

// out[0] = cycles for the timed loop (includes the cost of reading the clock)
// out[1] = cost of reading the clock (two clock reads back to back)
// out[2] = nanoseconds for the timed loop (used to get the SM clock speed)
__global__ void dram_chase_kernel(uint64_t start, long long* out, uint64_t* sink) {
    uint64_t p = start;

    // Warm-up.
#pragma unroll 1
    for (int i = 0; i < WARMUP_LOADS; ++i) {
        CHASE_ONE
    }

    // This uses p, so the clock reads below cannot start before the last warm-up
    // load is done. p is never 0, so this never runs.
    if (p == 0) {
        *sink = 0;
        return;
    }

    // Measure the cost of reading the clock.
    uint64_t b0 = read_clock64();
    uint64_t b1 = read_clock64();

    // Timed loop: CHASE_LOADS loads, one after another.
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

    out[0] = t1 - t0;
    out[1] = b1 - b0;
    out[2] = g1 - g0;
    *sink = p;   // use the result so the chain is not removed
}

int main(int argc, char** argv) {
    // ---- Command line ----
    std::string out_dir = "./hardware/runs";
    std::string gpu_override;
    int reps = DEFAULT_REPS;
    long working_set_kb = 0;   // 0 = work it out from the L2 size
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

    // ---- GPU info and buffer size ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const std::string gpu_name = prop.name;
    const std::string gpu = gpu_override.empty() ? short_gpu_name(gpu_name) : gpu_override;
    const std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);

    // L2 size to use: the one the user gave, or the one the GPU reports.
    const long l2_kb = l2_cache_kb ? l2_cache_kb : (long)prop.l2CacheSize / 1024;
    if (working_set_kb == 0) working_set_kb = WORKING_SET_L2_MULT * l2_kb;
    if (working_set_kb < MIN_L2_MULT * l2_kb) {
        fprintf(stderr, "working set (%ld KB) must be at least %dx the L2 size (%ld KB)\n",
                working_set_kb, MIN_L2_MULT, l2_kb);
        return 1;
    }

    const size_t buf_bytes = (size_t)working_set_kb * 1024;
    const size_t num_lines = buf_bytes / LINE_BYTES;
    if (num_lines > 0x7fffffffULL) { fprintf(stderr, "working set too large\n"); return 1; }

    fprintf(stderr, "GPU: %s (%s), SMs: %d, L2: %.1f MB (%s), buffer: %.1f MB\n",
            gpu_name.c_str(), arch.c_str(), prop.multiProcessorCount, l2_kb / 1024.0,
            l2_cache_kb ? "user-supplied" : "device-reported", working_set_kb / 1024.0);

    // Make sure the buffer fits in free GPU memory.
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (buf_bytes > free_bytes / 10 * 8) {
        fprintf(stderr, "ERROR: buffer (%zu bytes) does not fit in free GPU memory (%zu bytes)\n",
                buf_bytes, free_bytes);
        return 1;
    }

    int cuda_runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime));

    // ---- Build the pointer chain ----

    uint8_t* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, buf_bytes));

    std::vector<int> order(num_lines);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(12345);
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<uint8_t> h_buf(buf_bytes, 0);
    for (size_t i = 0; i < num_lines; ++i) {
        const int this_line = order[i];
        const int next_line = order[(i + 1) % num_lines];
        const uint64_t next_addr = (uint64_t)d_buf + (uint64_t)next_line * LINE_BYTES;
        *(uint64_t*)&h_buf[(size_t)this_line * LINE_BYTES] = next_addr;
    }
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf.data(), buf_bytes, cudaMemcpyHostToDevice));
    h_buf.clear();
    h_buf.shrink_to_fit();

    long long* d_out;
    uint64_t* d_sink;
    CUDA_CHECK(cudaMalloc(&d_out, 3 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));

    // ---- Timed runs ----
    std::vector<long long> raw_cycles, overhead_samples;
    double total_cycles = 0.0, total_ns = 0.0;   // for the SM clock speed

    for (int r = 0; r < reps; ++r) {
        // Each rep continues where the last one stopped
        const size_t pos = ((size_t)r * LOADS_PER_REP) % num_lines;
        const uint64_t start = (uint64_t)d_buf + (uint64_t)order[pos] * LINE_BYTES;

        CUDA_CHECK(cudaMemset(d_out, 0, 3 * sizeof(long long)));
        dram_chase_kernel<<<1, 1>>>(start, d_out, d_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        long long h[3];
        CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
        raw_cycles.push_back(h[0]);
        overhead_samples.push_back(h[1]);
        total_cycles += (double)h[0];
        total_ns += (double)h[2];
    }

    if (total_ns <= 0.0) { fprintf(stderr, "ERROR: globaltimer did not advance\n"); return 1; }

    const double sm_clock_ghz = total_cycles / total_ns;

    // ---- Results ----
    // Subtract the clock overhead, then divide by the number of loads.
    const double overhead = median_of(std::vector<double>(overhead_samples.begin(), overhead_samples.end()));
    std::vector<double> net_cycles, per_load, uncorrected;
    for (long long c : raw_cycles) {
        net_cycles.push_back(c - overhead);
        per_load.push_back((c - overhead) / CHASE_LOADS);
        uncorrected.push_back((double)c / CHASE_LOADS);
    }

    const Stats stats = compute_stats(per_load);
    const double median = stats.median;

    fprintf(stderr, "\nDRAM latency: %.2f cycles = %.1f ns at %.3f GHz (std %.3f)\n",
            median, median / sm_clock_ghz, sm_clock_ghz, stats.std);
    if (median < 150.0)
        fprintf(stderr, "WARNING: %.1f cycles is low for DRAM. The loads may still hit L2. "
                        "Try a larger --working-set-kb.\n", median);
    if (median > 1.05 * stats.min)
        fprintf(stderr, "WARNING: median is more than 5%% above the fastest run. Another process may "
                        "be using the GPU. Check nvidia-smi and rerun.\n");

    // ---- Save JSON ----
    std::ostringstream j;
    j.setf(std::ios::fixed);
    j.precision(4);
    j << "{\n";
    j << "  \"benchmark\": \"dram_latency.cu\",\n";
    j << "  \"parameter\": \"" << PARAMETER << "\",\n";
    j << "  \"value\": " << median << ",\n";
    j << "  \"unit\": \"cycles\",\n";
    j << "  \"gpu\": \"" << json_escape(gpu) << "\",\n";
    j << "  \"gpu_name\": \"" << json_escape(gpu_name) << "\",\n";
    j << "  \"arch\": \"" << arch << "\",\n";
    j << "  \"num_sms\": " << prop.multiProcessorCount << ",\n";
    j << "  \"l2_cache_bytes\": " << prop.l2CacheSize << ",\n";
    j << "  \"l2_cache_kb_assumed\": " << l2_kb << ",\n";
    j << "  \"working_set_bytes\": " << buf_bytes << ",\n";
    j << "  \"loads_per_rep\": " << CHASE_LOADS << ",\n";
    j << json_stats(stats);
    j << "  \"raw_cycles\": " << json_array(raw_cycles) << ",\n";
    j << "  \"overhead_samples_cycles\": " << json_array(overhead_samples) << ",\n";
    j << "  \"overhead_cycles\": " << overhead << ",\n";
    j << "  \"overhead_subtracted\": true,\n";
    j << "  \"net_cycles\": " << json_array(net_cycles) << ",\n";
    j << "  \"per_load_cycles\": " << json_array(per_load) << ",\n";
    j << "  \"uncorrected_value\": " << median_of(uncorrected) << ",\n";
    j << "  \"clock\": {\n";
    j << "    \"sm_clock_ghz\": " << sm_clock_ghz << ",\n";
    j << "    \"value_ns\": " << median / sm_clock_ghz << "\n";
    j << "  },\n";
    j << "  \"cuda_version\": \"" << cuda_version_string(cuda_runtime) << "\",\n";
    j << "  \"nvcc_version\": \"" << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "\",\n";
    j << "  \"driver_version\": \"" << json_escape(driver_version()) << "\",\n";
    j << "  \"timestamp\": \"" << utc_timestamp() << "\"\n";
    j << "}\n";

    const std::string out_path = write_run_json(out_dir, PARAMETER, gpu, j.str());
    if (out_path.empty()) return 1;

    printf("%s", j.str().c_str());
    fprintf(stderr, "Saved: %s\n", out_path.c_str());

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}
