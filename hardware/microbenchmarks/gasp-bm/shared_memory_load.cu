// shared_memory_load.cu
// Measures shared_mem_load_latency_cycles: how many cycles one shared-memory
// load takes when there are no bank conflicts.
//
// One warp follows a chain of pointers in shared memory. Each load needs the
// address from the previous one, so we time one load at a time. The 32 lanes use
// different banks, so there are no conflicts. The cost of reading the clock is
// measured in the same kernel and subtracted.
//
// Run: ./shared_memory_load [--gpu NAME] [--out-dir DIR] [--reps N]

#include <cuda_runtime.h>
#include "common.h"

static const char* PARAMETER = "shared_mem_load_latency_cycles";

constexpr int NUM_NODES = 64;       // nodes in each lane's chain
constexpr int WARP_THREADS = 32;
constexpr int CHASE_LOADS = 1024;   // loads in the timed region
constexpr int UNROLL = 16;          // loads per loop iteration
constexpr int DEFAULT_REPS = 200;

static_assert(CHASE_LOADS % UNROLL == 0, "CHASE_LOADS must be a multiple of UNROLL");

// One dependent load. The register holds the address and gets the next address,
// so there is no math between loads. "asm volatile" stops the compiler from
// removing or merging the loads.
#define CHASE_ONE asm volatile("ld.shared.u32 %0, [%0];" : "+r"(p)::"memory");

// out[0] = cycles for the timed loop (includes the cost of reading the clock)
// out[1] = cost of reading the clock (two clock reads back to back)
// out[2] = nanoseconds for the timed loop (used to get the SM clock speed)
__global__ void shared_chase_kernel(unsigned long long* out, uint32_t* sink) {
    __shared__ uint32_t ring[NUM_NODES * WARP_THREADS];

    const int lane = threadIdx.x;
    const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(ring);

    // Node n of lane L is at slot n*32 + L, so the 32 lanes are always in 32
    // different banks. Each slot holds the address of the next node.
    for (int node = 0; node < NUM_NODES; ++node) {
        const int next_node = (node + 1) % NUM_NODES;
        ring[node * WARP_THREADS + lane] = shared_base + (next_node * WARP_THREADS + lane) * 4;
    }
    __syncthreads();

    uint32_t p = shared_base + lane * 4;

    // Warm-up, not timed.
#pragma unroll 1
    for (int i = 0; i < 2 * NUM_NODES; ++i) {
        CHASE_ONE
    }
    __syncthreads();

    // Measure the cost of reading the clock.
    const uint64_t b0 = read_clock64();
    const uint64_t b1 = read_clock64();

    // Timed loop: CHASE_LOADS loads, one after another.
    const uint64_t g0 = read_globaltimer();
    const uint64_t t0 = read_clock64();
#pragma unroll 1
    for (int i = 0; i < CHASE_LOADS; i += UNROLL) {
        CHASE_ONE CHASE_ONE CHASE_ONE CHASE_ONE
        CHASE_ONE CHASE_ONE CHASE_ONE CHASE_ONE
        CHASE_ONE CHASE_ONE CHASE_ONE CHASE_ONE
        CHASE_ONE CHASE_ONE CHASE_ONE CHASE_ONE
    }
    const uint64_t t1 = read_clock64();
    const uint64_t g1 = read_globaltimer();

    if (lane == 0) {
        out[0] = t1 - t0;
        out[1] = b1 - b0;
        out[2] = g1 - g0;
    }
    sink[lane] = p;   // use the result so the chain is not removed
}

int main(int argc, char** argv) {
    // ---- Command line ----
    int reps = DEFAULT_REPS;
    std::string gpu_override;
    std::string out_dir = "hardware/runs";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (arg == "--gpu" && i + 1 < argc) gpu_override = argv[++i];
        else if (arg == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else {
            fprintf(stderr, "Usage: %s [--reps N] [--gpu NAME] [--out-dir DIR]\n", argv[0]);
            return 1;
        }
    }
    if (reps < 100) { fprintf(stderr, "--reps must be at least 100\n"); return 1; }

    // ---- GPU info ----
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const std::string gpu_name = prop.name;
    const std::string gpu = gpu_override.empty() ? short_gpu_name(gpu_name) : gpu_override;
    const std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);
    const std::string architecture = architecture_name(prop.major, prop.minor);

    int shared_mem_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&shared_mem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    const int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    int cuda_runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime));

    // ---- Memory ----
    unsigned long long* d_out = nullptr;
    uint32_t* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, 3 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_sink, WARP_THREADS * sizeof(uint32_t)));

    // ---- Timed runs ----
    std::vector<long long> raw_cycles, overhead_samples;
    double total_cycles = 0.0, total_ns = 0.0;   // for the SM clock speed

    for (int r = 0; r < reps; ++r) {
        shared_chase_kernel<<<1, WARP_THREADS>>>(d_out, d_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned long long h[3] = {0, 0, 0};
        CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
        raw_cycles.push_back((long long)h[0]);
        overhead_samples.push_back((long long)h[1]);
        if (h[2] > 0) {
            total_cycles += (double)h[0];
            total_ns += (double)h[2];
        }
    }

    // ---- Results ----
    // Subtract the clock overhead, then divide by the number of loads.
    const double overhead = median_of(std::vector<double>(overhead_samples.begin(), overhead_samples.end()));
    std::vector<double> net_cycles, per_load;
    for (long long c : raw_cycles) {
        net_cycles.push_back(c - overhead);
        per_load.push_back((c - overhead) / CHASE_LOADS);
    }
    const Stats stats = compute_stats(per_load);
    const double value = stats.median;

    fprintf(stderr, "Shared-memory load latency: %.3f cycles (std %.4f)\n", value, stats.std);

    // ---- Save JSON ----
    std::ostringstream json;
    json.setf(std::ios::fixed);
    json.precision(4);
    json << "{\n"
         << "  \"benchmark\": \"shared_memory_load.cu\",\n"
         << "  \"parameter\": \"" << PARAMETER << "\",\n"
         << "  \"value\": " << value << ",\n"
         << "  \"unit\": \"cycles\",\n"
         << "  \"gpu\": \"" << json_escape(gpu) << "\",\n"
         << "  \"architecture\": \"" << architecture << "\",\n"
         << "  \"arch\": \"" << arch << "\",\n"
         << "  \"device\": {\n"
         << "    \"name\": \"" << json_escape(gpu_name) << "\",\n"
         << "    \"sms\": " << prop.multiProcessorCount << ",\n"
         << "    \"max_warps_per_sm\": " << max_warps_per_sm << ",\n"
         << "    \"l2_cache_bytes\": " << prop.l2CacheSize << ",\n"
         << "    \"shared_mem_per_sm_bytes\": " << shared_mem_per_sm << "\n"
         << "  },\n"
         << json_stats(stats)
         << "  \"loads_per_rep\": " << CHASE_LOADS << ",\n"
         << "  \"raw_cycles\": " << json_array(raw_cycles) << ",\n"
         << "  \"overhead_samples_cycles\": " << json_array(overhead_samples) << ",\n"
         << "  \"overhead_cycles\": " << overhead << ",\n"
         << "  \"overhead_subtracted\": true,\n"
         << "  \"net_cycles\": " << json_array(net_cycles) << ",\n"
         << "  \"per_load_cycles\": " << json_array(per_load) << ",\n"
         << "  \"clock\": {\"sm_clock_ghz\": " << (total_ns > 0.0 ? total_cycles / total_ns : 0.0) << "},\n"
         << "  \"cuda_version\": \"" << cuda_version_string(cuda_runtime) << "\",\n"
         << "  \"driver_version\": \"" << json_escape(driver_version()) << "\",\n"
         << "  \"timestamp\": \"" << utc_timestamp() << "\"\n"
         << "}\n";

    const std::string output_path = write_run_json(out_dir, PARAMETER, gpu, json.str());
    if (output_path.empty()) return 1;

    printf("%s", json.str().c_str());
    fprintf(stderr, "Saved: %s\n", output_path.c_str());

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}
