// shared_memory_store.cu
// Measures shared_mem_store_latency_cycles: how many cycles a conflict-free
// warp store to shared memory adds before a dependent instruction can go on.
//
// A store has no result register, so it cannot form a chain on its own. Each
// step stores p into its own slot and loads it back:
//     st.volatile.shared.u32 [p], p;   p = ld.volatile.shared.u32 [p];
// The load must see the store, and the next step needs the loaded value, so a
// step costs store + load. The same kernel also times a load-only chain over the
// same slots. Store latency = (store+load time) - (load-only time), per step.
// Both chains use the same addresses and loop shape, so the clock and loop
// overhead cancel.
//
// The load-only time is checked against the shared_memory_load run, if its
// JSON is in the out folder.
//
// Run: ./shared_memory_store [--reps N] [--gpu NAME] [--out-dir DIR]

#include <cuda_runtime.h>
#include "common.h"

constexpr int WARP_THREADS = 32;
constexpr int CHAIN_STEPS = 1024;   // steps in each timed chain
constexpr int UNROLL = 16;          // steps per loop iteration
constexpr int PASSES = 2;           // the first pass warms the instruction cache; the last one is kept
constexpr int WARMUP_STEPS = 64;
constexpr int DEFAULT_REPS = 200;
constexpr double REFERENCE_TOLERANCE_CYCLES = 2.0;

static_assert(CHAIN_STEPS % UNROLL == 0, "CHAIN_STEPS must be a multiple of UNROLL");
static_assert(UNROLL == 16, "the unrolled bodies below have 16 steps");

static const char* PARAMETER = "shared_mem_store_latency_cycles";
static const char* LOAD_PARAMETER = "shared_mem_load_latency_cycles";

// "asm volatile" stops the compiler from removing, merging or reordering these.
__device__ __forceinline__ uint32_t load_shared(uint32_t address) {
    uint32_t value;
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(value) : "r"(address) : "memory");
    return value;
}

__device__ __forceinline__ void store_shared(uint32_t address, uint32_t value) {
    asm volatile("st.volatile.shared.u32 [%0], %1;" : : "r"(address), "r"(value) : "memory");
}

// Store p into slot p, then load it back. The slot holds its own address, so
// the load returns p and the chain stays on the same slot.
#define STORE_STEP store_shared(p, p); p = load_shared(p)
#define LOAD_STEP p = load_shared(p)

#define REPEAT_16(step)                                  \
    step; step; step; step; step; step; step; step;      \
    step; step; step; step; step; step; step; step

// out[0] = cycles for the store+load chain (includes the cost of reading the clock)
// out[1] = cycles for the load-only chain (includes the cost of reading the clock)
// out[2] = cost of reading the clock (two clock reads back to back)
// out[3] = nanoseconds for the store+load chain (used to get the SM clock speed)
__global__ void shared_store_latency_kernel(unsigned long long* out, uint32_t* sink) {
    __shared__ uint32_t slots[WARP_THREADS];

    const int lane = threadIdx.x;
    // Lane L uses slot L, so the 32 lanes are in 32 different banks.
    const uint32_t self = (uint32_t)__cvta_generic_to_shared(&slots[lane]);
    slots[lane] = self;
    __syncthreads();

    uint32_t p = self;

    // Warm-up, not timed.
#pragma unroll 1
    for (int i = 0; i < WARMUP_STEPS; ++i) {
        STORE_STEP;
        LOAD_STEP;
    }
    __syncthreads();

    uint64_t store_start = 0, store_stop = 0, load_start = 0, load_stop = 0;
    uint64_t g0 = 0, g1 = 0;

#pragma unroll 1
    for (int pass = 0; pass < PASSES; ++pass) {
        // Store+load chain.
        g0 = read_globaltimer();
        store_start = read_clock64();
#pragma unroll 1
        for (int i = 0; i < CHAIN_STEPS; i += UNROLL) {
            REPEAT_16(STORE_STEP);
        }
        store_stop = read_clock64();
        g1 = read_globaltimer();

        // Load-only chain.
        load_start = read_clock64();
#pragma unroll 1
        for (int i = 0; i < CHAIN_STEPS; i += UNROLL) {
            REPEAT_16(LOAD_STEP);
        }
        load_stop = read_clock64();
    }

    // Measure the cost of reading the clock.
    const uint64_t b0 = read_clock64();
    const uint64_t b1 = read_clock64();

    if (lane == 0) {
        out[0] = store_stop - store_start;
        out[1] = load_stop - load_start;
        out[2] = b1 - b0;
        out[3] = g1 - g0;
    }
    sink[lane] = p;   // use the result so the chains are not removed
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

    int cuda_runtime = 0, driver_api = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver_api));

    // ---- Memory ----
    unsigned long long* d_out = nullptr;
    uint32_t* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_sink, WARP_THREADS * sizeof(uint32_t)));

    // ---- Timed runs ----
    std::vector<long long> raw_store_chain, raw_load_only, overhead_samples;
    double total_cycles = 0.0, total_ns = 0.0;   // for the SM clock speed

    for (int r = 0; r < reps; ++r) {
        shared_store_latency_kernel<<<1, WARP_THREADS>>>(d_out, d_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned long long h[4] = {0, 0, 0, 0};
        CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
        raw_store_chain.push_back((long long)h[0]);
        raw_load_only.push_back((long long)h[1]);
        overhead_samples.push_back((long long)h[2]);
        if (h[3] > 0) {
            total_cycles += (double)h[0];
            total_ns += (double)h[3];
        }
    }

    // ---- Results ----
    // Subtract the clock overhead from each chain, divide by the number of steps,
    // then store latency = (store+load per step) - (load-only per step).
    const double overhead = median_of(std::vector<double>(overhead_samples.begin(), overhead_samples.end()));
    std::vector<double> per_store_plus_load, per_load_only, per_store;
    for (int r = 0; r < reps; ++r) {
        const double both = (raw_store_chain[r] - overhead) / CHAIN_STEPS;
        const double load = (raw_load_only[r] - overhead) / CHAIN_STEPS;
        per_store_plus_load.push_back(both);
        per_load_only.push_back(load);
        per_store.push_back(both - load);
    }

    const Stats stats = compute_stats(per_store);
    const double value = stats.median;
    const double store_plus_load_cycles = median_of(per_store_plus_load);
    const double load_only_cycles = median_of(per_load_only);

    fprintf(stderr, "Shared-memory store latency: %.3f cycles (std %.4f)\n", value, stats.std);

    // Check the load-only chain against the shared_memory_load run, if it exists.
    const std::string reference_path =
        (std::filesystem::path(out_dir) / (std::string(LOAD_PARAMETER) + "." + gpu + ".run.json")).string();
    const double load_reference = read_run_value(reference_path);
    const bool has_reference = load_reference > 0.0;
    if (!has_reference) {
        fprintf(stderr, "No load-latency run at %s, so no cross-check "
                        "(run shared_memory_load first)\n", reference_path.c_str());
    } else if (std::fabs(load_only_cycles - load_reference) > REFERENCE_TOLERANCE_CYCLES) {
        fprintf(stderr, "WARNING: the load-only chain gives %.2f cycles per step but %s says %.2f. "
                        "The subtraction baseline may be off.\n",
                load_only_cycles, reference_path.c_str(), load_reference);
    }

    // ---- Save JSON ----
    std::ostringstream json;
    json.setf(std::ios::fixed);
    json.precision(4);
    json << "{\n"
         << "  \"benchmark\": \"shared_memory_store.cu\",\n"
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
         << "  \"steps_per_rep\": " << CHAIN_STEPS << ",\n"
         << "  \"store_plus_load_cycles\": " << store_plus_load_cycles << ",\n"
         << "  \"load_only_cycles\": " << load_only_cycles << ",\n"
         << "  \"load_latency_reference_cycles\": ";
    if (has_reference) json << load_reference;
    else json << "null";
    json << ",\n"
         << "  \"raw_store_chain_cycles\": " << json_array(raw_store_chain) << ",\n"
         << "  \"raw_load_only_cycles\": " << json_array(raw_load_only) << ",\n"
         << "  \"overhead_samples_cycles\": " << json_array(overhead_samples) << ",\n"
         << "  \"overhead_cycles\": " << overhead << ",\n"
         << "  \"overhead_subtracted\": true,\n"
         << "  \"per_store_cycles\": " << json_array(per_store) << ",\n"
         << "  \"clock\": {\"sm_clock_ghz\": " << (total_ns > 0.0 ? total_cycles / total_ns : 0.0) << "},\n"
         << "  \"cuda_version\": \"" << cuda_version_string(cuda_runtime) << "\",\n"
         << "  \"driver_api_version\": \"" << cuda_version_string(driver_api) << "\",\n"
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

// SASS check: cuobjdump -sass shared_memory_store.cubin
// In shared_store_latency_kernel:
//   - Store+load loop: 16 pairs of "STS [Rp], Rp" then "LDS Rp, [Rp]", with no
//     other instructions between the pairs.
//   - Load-only loop: 16 chained "LDS Rp, [Rp]".
//   - Both loops have the same loop overhead (counter, compare, branch), so it
//     cancels in the difference.
// If the compiler dropped the LDS after an STS, or added instructions to only
// one loop, the result is not a store latency.
