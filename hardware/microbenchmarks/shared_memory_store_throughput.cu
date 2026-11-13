#include <array>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include "benchmark_formatting.h"

// References the store layout and shmembench fence used in https://github.com/HengWang/gpumembench
// However, we only measure stores. the reference uses both loads and stores. We also only check cycles
// across 1 warp. The reference measures across blocks.

constexpr int WARP_THREADS = 32;
constexpr int STORE_SLOTS = 16;
constexpr int STORE_INSTRUCTIONS = 1024;
constexpr int UNROLL = 16;
constexpr int DEFAULT_REPS = 200;

static_assert(STORE_INSTRUCTIONS % UNROLL == 0,
              "STORE_INSTRUCTIONS must be a multiple of UNROLL");

static const char* PARAMETER = "shared_mem_store_throughput_cycles";

// writes 32 bit value to shared memory at a specific address
// volatile so the compiler does not optimize it away
__device__ __forceinline__ void store_shared_volatile(
    uint32_t address, uint32_t value) {
    asm volatile(
        "st.volatile.shared.u32 [%0], %1;"
        :
        : "r"(address), "r"(value)
        : "memory");
}

// executes membar.cta, which ensures prior shared-memory operations are visible before
// continuing to create a clean timing boundary around the store batch.
// without it, memory reads/writes would probably be reordered inconsistently and create noise
__device__ __forceinline__ void shared_memory_fence() {
    asm volatile("membar.cta;" ::: "memory");
}

//writes every store at a unique address to avoid lane-to-lane conflicts
#define STORE_AT(slot)                                                         \
    store_shared_volatile(                                                     \
        shared_base +                                                          \
            static_cast<uint32_t>(                                              \
                ((slot) * WARP_THREADS + lane) * sizeof(uint32_t)),            \
        static_cast<uint32_t>(lane + (slot)))

#define STORE_16_SLOTS                                                        \
    STORE_AT(0); STORE_AT(1); STORE_AT(2); STORE_AT(3);                       \
    STORE_AT(4); STORE_AT(5); STORE_AT(6); STORE_AT(7);                       \
    STORE_AT(8); STORE_AT(9); STORE_AT(10); STORE_AT(11);                     \
    STORE_AT(12); STORE_AT(13); STORE_AT(14); STORE_AT(15)

// clock_out[0..1] = lane 0's clock64 cycles and globaltimer ns around the timed region
__global__ void shared_store_kernel(unsigned long long* out, unsigned long long* clock_out) {
    __shared__ uint32_t shared_words[STORE_SLOTS * WARP_THREADS];

    const int lane = threadIdx.x;
    const uint32_t shared_base =
        static_cast<uint32_t>(__cvta_generic_to_shared(shared_words));

    // Warm up the shared-memory path before collecting the baseline.
#pragma unroll 1
    for (int i = 0; i < 2 * STORE_SLOTS; ++i) {
        const int slot = i % STORE_SLOTS;
        const uint32_t address =
            shared_base +
            static_cast<uint32_t>(
                (slot * WARP_THREADS + lane) * sizeof(uint32_t));
        store_shared_volatile(
            address, static_cast<uint32_t>(lane + slot));
    }
    shared_memory_fence();
    __syncthreads();

    // Baseline includes the clock reads and a fence, but no timed store batch.
    const uint64_t baseline_start = read_clock64();
    shared_memory_fence();
    const uint64_t baseline_stop = read_clock64();

    const uint64_t g0 = read_globaltimer();
    const uint64_t start = read_clock64();
#pragma unroll 1
    for (int i = 0; i < STORE_INSTRUCTIONS; i += UNROLL) {
        STORE_16_SLOTS;
    }
    shared_memory_fence();
    const uint64_t stop = read_clock64();
    const uint64_t g1 = read_globaltimer();
    if (lane == 0) {
        clock_out[0] = stop - start;
        clock_out[1] = g1 - g0;
    }

    // Store one timing sample per lane for host-side aggregation.
    out[lane] = stop - start;
    out[WARP_THREADS + lane] = baseline_stop - baseline_start;
}

int main(int argc, char** argv) {
    const BenchArgs args = parse_bench_args(argc, argv, DEFAULT_REPS);
    const int reps = args.reps;

    unsigned long long* device_output = nullptr;
    unsigned long long* device_clock = nullptr;  // lane 0's cycles and ns, for the SM clock
    CUDA_CHECK(cudaMalloc(&device_clock, 2 * sizeof(unsigned long long)));
    double sum_clock_cycles = 0.0, sum_clock_ns = 0.0;
    CUDA_CHECK(cudaMalloc(
        &device_output, 2 * WARP_THREADS * sizeof(unsigned long long)));

    std::vector<double> raw_cycles;
    std::vector<double> fence_baseline_samples;
    raw_cycles.reserve(reps);
    fence_baseline_samples.reserve(reps);

    //launch kernel, synchronize, read device_clock, copy output back from device
    for (int rep = 0; rep < reps; ++rep) {
        shared_store_kernel<<<1, WARP_THREADS>>>(device_output, device_clock);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long host_clock[2] = {0, 0};
        CUDA_CHECK(cudaMemcpy(host_clock, device_clock, sizeof(host_clock),
                              cudaMemcpyDeviceToHost));
        if (host_clock[1] > 0) {
            sum_clock_cycles += static_cast<double>(host_clock[0]);
            sum_clock_ns += static_cast<double>(host_clock[1]);
        }

        std::array<unsigned long long, 2 * WARP_THREADS> host_output{};
        CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                              sizeof(host_output), cudaMemcpyDeviceToHost));

        // collects per-lane raw cycles and fence baselines
        std::vector<double> lane_cycles(WARP_THREADS);
        std::vector<double> lane_baselines(WARP_THREADS);
        for (int lane = 0; lane < WARP_THREADS; ++lane) {
            lane_cycles[lane] = static_cast<double>(host_output[lane]);
            lane_baselines[lane] =
                static_cast<double>(host_output[WARP_THREADS + lane]);
        }
        //save each rep's median samples
        raw_cycles.push_back(median(lane_cycles));
        fence_baseline_samples.push_back(median(lane_baselines));
    }

    //we estimate cost by measuring total time for 1024 stores, plus cost of CTA fence
    // then removes fence baseline to isolate true cost of the store instruction itself
    // (to remove the cost of the synchronization boundary)
    const double fence_baseline = median(fence_baseline_samples);
    std::vector<double> per_store_instruction_cycles;
    per_store_instruction_cycles.reserve(reps);

    for (double cycles : raw_cycles) {
        per_store_instruction_cycles.push_back(
            (cycles - fence_baseline) / STORE_INSTRUCTIONS);
    }

    BenchmarkReport report;
    report.benchmark = "shared_memory_store_latency.cu";
    report.parameter = PARAMETER;
    report.unit = "cycles_per_warp_store_instruction";
    report.stats = compute_stats(per_store_instruction_cycles);
    // clock.sm_clock_ghz: total clock64 cycles / total globaltimer ns over all reps
    report.sm_clock_ghz = sum_clock_ns > 0.0 ? sum_clock_cycles / sum_clock_ns : 0.0;

    report.arrays = {
        {"raw_cycles", raw_cycles},
        {"fence_baseline_samples_cycles", fence_baseline_samples},
        {"per_store_instruction_cycles", per_store_instruction_cycles}
    };
    report.scalars = {
        {"fence_baseline_cycles", fence_baseline}
    };
    report.measurement_str = {
        {"method", "back-to-back conflict-free warp shared stores with CTA fence"},
        {"store_instruction", "st.volatile.shared.u32"},
        {"timer", "clock64 (cycles), globaltimer (SM clock)"}
    };
    report.measurement_num = {
        {"store_instructions_per_rep", STORE_INSTRUCTIONS},
        {"store_slots_per_lane", STORE_SLOTS},
        {"threads_per_block", WARP_THREADS},
        {"reps", reps}
    };

    write_report(args, report);

    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_clock));
    return 0;
}
