// departure_delay.cu
// Measures departure_delay_cycles: how many SM cycles pass between two
// memory requests from different warps leaving the SM.
//
// How it works:
//   - One block with 32 warps runs on one SM.
//   - Each warp reads its own lines of a 2 MB buffer, 8 loads at a time.
//     The loads do not depend on each other, so many are in flight at once.
//   - One warp load = 32 threads x 4 bytes = one 128-byte line = one request.
//   - ld.global.cg skips L1, so every request leaves the SM and is served by L2.
//   - delay = (total cycles - clock overhead) / number of requests

#include <cuda_runtime.h>
#include "common.h"

static const char* PARAMETER = "departure_delay_cycles";

constexpr int WARPS           = 32;    // warps in the block
constexpr int LINE_BYTES      = 128;   // one warp load = one 128-byte line
constexpr int LOADS_PER_WARP  = 512;   // requests each warp sends
constexpr int UNROLL          = 8;     // loads each warp has in flight at once
constexpr int WARMUP_LAUNCHES = 3;     // untimed launches (also loads the buffer into L2)
constexpr int DEFAULT_REPS    = 200;   // timed launches

static_assert(LOADS_PER_WARP % UNROLL == 0, "LOADS_PER_WARP must be a multiple of UNROLL");

// out[0] = cycles for the timed loop (includes the cost of reading the clock)
// out[1] = cost of reading the clock (two clock reads back to back)
// out[2] = nanoseconds for the timed loop (used to get the SM clock speed)
__global__ void departure_kernel(const unsigned* buf, long long* out, unsigned* sink) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // Each warp starts at its own block of lines. Each lane reads one word of a line.
    const unsigned* base = buf + warp * LOADS_PER_WARP * 32 + lane;

    unsigned sum = 0;
    uint64_t b0 = 0, b1 = 0, t0 = 0, g0 = 0;

    // Measure the cost of reading the clock.
    __syncthreads();
    if (threadIdx.x == 0) { b0 = read_clock64(); b1 = read_clock64(); }
    __syncthreads();

    // Start the timer.
    if (threadIdx.x == 0) { g0 = read_globaltimer(); t0 = read_clock64(); }

    // "unroll 1" stops the compiler from unrolling this loop.
#pragma unroll 1
    for (int i = 0; i < LOADS_PER_WARP; i += UNROLL) {
        unsigned v[UNROLL];

        // Send UNROLL loads. "asm volatile" stops the compiler from removing them.
        for (int k = 0; k < UNROLL; k++) {
            const unsigned* addr = base + (i + k) * 32;
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v[k]) : "l"(addr));
        }
        // Use the results so the loads are not dead code.
        for (int k = 0; k < UNROLL; k++) sum ^= v[k];
    }

    // Wait for the slowest warp, then stop the timer.
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint64_t t1 = read_clock64();
        const uint64_t g1 = read_globaltimer();
        out[0] = t1 - t0;
        out[1] = b1 - b0;
        out[2] = g1 - g0;
    }
    sink[threadIdx.x] = sum;
}

int main(int argc, char** argv) {
    // ---- Command line ----
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

    // ---- GPU info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const std::string gpu_name = prop.name;
    const std::string gpu = gpu_override.empty() ? short_gpu_name(gpu_name) : gpu_override;
    const std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);

    const size_t buf_bytes = (size_t)WARPS * LOADS_PER_WARP * LINE_BYTES;   // 2 MB
    fprintf(stderr, "GPU: %s (%s), SMs: %d, L2: %.1f MB, buffer: %.1f MB\n",
            gpu_name.c_str(), arch.c_str(), prop.multiProcessorCount,
            prop.l2CacheSize / (1024.0 * 1024.0), buf_bytes / (1024.0 * 1024.0));
    if (buf_bytes * 2 > (size_t)prop.l2CacheSize) {
        fprintf(stderr, "ERROR: buffer is too big to stay in L2\n");
        return 1;
    }

    int cuda_runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime));

    // ---- Memory ----
    unsigned* d_buf = nullptr;
    long long* d_out = nullptr;
    unsigned* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, buf_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, 3 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sink, WARPS * 32 * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(d_buf, 0x01, buf_bytes));

    const int threads = WARPS * 32;

    // ---- Warm-up (not timed) ----
    for (int i = 0; i < WARMUP_LAUNCHES; ++i) {
        departure_kernel<<<1, threads>>>(d_buf, d_out, d_sink);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Timed runs ----
    std::vector<long long> raw_cycles, overhead_samples;
    double total_cycles = 0.0, total_ns = 0.0;   // for the SM clock speed

    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaMemset(d_out, 0, 3 * sizeof(long long)));
        departure_kernel<<<1, threads>>>(d_buf, d_out, d_sink);
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
    const double requests = (double)WARPS * LOADS_PER_WARP;   // 16384 requests per launch

    // Subtract the clock overhead, then divide by the number of requests.
    const double overhead = median_of(std::vector<double>(overhead_samples.begin(), overhead_samples.end()));
    std::vector<double> net_cycles, per_request, uncorrected;
    for (long long c : raw_cycles) {
        net_cycles.push_back(c - overhead);
        per_request.push_back((c - overhead) / requests);
        uncorrected.push_back(c / requests);
    }

    const Stats stats = compute_stats(per_request);
    const double median = stats.median;

    fprintf(stderr, "\nDeparture delay: %.4f cycles per request (std %.4f), SM clock %.3f GHz\n",
            median, stats.std, sm_clock_ghz);
    if (median > 1.05 * stats.min)
        fprintf(stderr, "WARNING: median is more than 5%% above the fastest run. Another process may be "
                        "using the GPU. Check nvidia-smi and rerun.\n");

    // ---- Save JSON ----
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
    j << "  \"warps_per_block\": " << WARPS << ",\n";
    j << "  \"requests_per_rep\": " << (long long)requests << ",\n";
    j << "  \"raw_cycles\": " << json_array(raw_cycles) << ",\n";
    j << "  \"overhead_samples_cycles\": " << json_array(overhead_samples) << ",\n";
    j << "  \"overhead_cycles\": " << overhead << ",\n";
    j << "  \"overhead_subtracted\": true,\n";
    j << "  \"net_cycles\": " << json_array(net_cycles) << ",\n";
    j << "  \"per_request_cycles\": " << json_array(per_request) << ",\n";
    j << "  \"uncorrected_value\": " << median_of(uncorrected) << ",\n";
    j << "  \"clock\": {\"sm_clock_ghz\": " << sm_clock_ghz << "},\n";
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


