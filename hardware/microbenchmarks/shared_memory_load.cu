#include <array>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include "benchmark_formatting.h"

// References https://github.com/shen203/GPU_Microbenchmark/blob/master/shared_lat/shared_lat.cu
// and https://github.com/HPMLL/NVIDIA-Hopper-Benchmark/blob/main/RegularUnits/shared_lat/shared_lat.cu
// both these papers provided the striding technique to avoid bank conflicts.

// This benchmark operates with a pointer-chase technique, where ld.shared.u32 must rely on the previous load
// operation to access the next address. This strict dependency chain ensures the GPU has to wait for the memory
// loads to wait before proceeding.

constexpr int NUM_NODES = 64;
constexpr int WARP_THREADS = 32;
constexpr int CHASE_LOADS = 1024;
constexpr int UNROLL = 16;
constexpr int DEFAULT_REPS = 200;

static_assert(CHASE_LOADS % UNROLL == 0,
              "CHASE_LOADS must be a multiple of UNROLL");

static const char* PARAMETER = "shared_mem_load_latency_cycles";

// ld.shared.u32 must wait for the previous node's address.
__device__ __forceinline__ uint32_t load_shared_address(uint32_t address) {
    uint32_t next;
    asm volatile(
        "ld.shared.u32 %0, [%1];"
        : "=r"(next)
        : "r"(address)
        : "memory");
    return next;
}

#define CHASE_ONE p = load_shared_address(p)

// clock_out[0..1] = lane 0's clock64 cycles and globaltimer ns around the timed region
__global__ void shared_chase_kernel(unsigned long long* out, unsigned long long* clock_out) {
    __shared__ uint32_t ring[NUM_NODES * WARP_THREADS];

    const int lane = threadIdx.x;
    const uint32_t shared_base =
        static_cast<uint32_t>(__cvta_generic_to_shared(ring));

    // interleave node slots by lane: slot (node, lane) is node*32 + lane, so the 32 lanes address 32 different banks.
    for (int node = 0; node < NUM_NODES; ++node) {
        const int next_node = (node + 1) % NUM_NODES;
        const int slot = node * WARP_THREADS + lane;
        const int next_slot = next_node * WARP_THREADS + lane;
        ring[slot] = shared_base +
                     static_cast<uint32_t>(next_slot * sizeof(uint32_t));
    }
    __syncthreads();

    uint32_t p = shared_base +
                 static_cast<uint32_t>(lane * sizeof(uint32_t));

    // Warm up the dependent chains before timing
#pragma unroll 1
    for (int i = 0; i < 2 * NUM_NODES; ++i) {
        CHASE_ONE;
    }

    __syncthreads();

    const uint64_t baseline_start = read_clock64();
    const uint64_t baseline_stop = read_clock64();

    const uint64_t g0 = read_globaltimer();
    const uint64_t start = read_clock64();
// runs 1024 dependent shared loads under timer
#pragma unroll 1
    for (int i = 0; i < CHASE_LOADS; i += UNROLL) {
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
        CHASE_ONE;
    }
    const uint64_t stop = read_clock64();
    const uint64_t g1 = read_globaltimer();
    if (lane == 0) {
        clock_out[0] = stop - start;
        clock_out[1] = g1 - g0;
    }

    out[lane] = stop - start;
    out[WARP_THREADS + lane] = baseline_stop - baseline_start;
    out[2 * WARP_THREADS + lane] = p;
}

int main(int argc, char** argv) {
    const BenchArgs args = parse_bench_args(argc, argv, DEFAULT_REPS);
    const int reps = args.reps;

    unsigned long long* device_output = nullptr;
    unsigned long long* device_clock = nullptr;  // lane 0's cycles and ns, for the SM clock
    CUDA_CHECK(cudaMalloc(&device_clock, 2 * sizeof(unsigned long long)));
    double sum_clock_cycles = 0.0, sum_clock_ns = 0.0;
    CUDA_CHECK(cudaMalloc(
        &device_output, 3 * WARP_THREADS * sizeof(unsigned long long)));

    std::vector<double> raw_cycles;
    std::vector<double> clock_overhead_samples;
    raw_cycles.reserve(reps);
    clock_overhead_samples.reserve(reps);

    for (int rep = 0; rep < reps; ++rep) {
        shared_chase_kernel<<<1, WARP_THREADS>>>(device_output, device_clock);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long host_clock[2] = {0, 0};
        CUDA_CHECK(cudaMemcpy(host_clock, device_clock, sizeof(host_clock),
                              cudaMemcpyDeviceToHost));
        if (host_clock[1] > 0) {
            sum_clock_cycles += static_cast<double>(host_clock[0]);
            sum_clock_ns += static_cast<double>(host_clock[1]);
        }

        std::array<unsigned long long, 3 * WARP_THREADS> host_output{};
        CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                              sizeof(host_output), cudaMemcpyDeviceToHost));

        std::vector<double> lane_cycles(WARP_THREADS);
        std::vector<double> lane_overheads(WARP_THREADS);
        for (int lane = 0; lane < WARP_THREADS; ++lane) {
            lane_cycles[lane] = static_cast<double>(host_output[lane]);
            lane_overheads[lane] =
                static_cast<double>(host_output[WARP_THREADS + lane]);
        }

        raw_cycles.push_back(median(lane_cycles));
        clock_overhead_samples.push_back(median(lane_overheads));
    }

    const double clock_overhead = median(clock_overhead_samples);
    std::vector<double> per_load_cycles;
    per_load_cycles.reserve(reps);

    for (double cycles : raw_cycles) {
        per_load_cycles.push_back(
            (cycles - clock_overhead) / CHASE_LOADS);
    }

    BenchmarkReport report;
    report.benchmark = "shared_memory_load.cu";
    report.parameter = PARAMETER;
    report.unit = "cycles";
    report.stats = compute_stats(per_load_cycles);
    // clock.sm_clock_ghz: total clock64 cycles / total globaltimer ns over all reps
    report.sm_clock_ghz = sum_clock_ns > 0.0 ? sum_clock_cycles / sum_clock_ns : 0.0;

    // Raw clock64 delta per rep (median over the 32 lanes), overhead NOT subtracted.
    report.arrays = {
        {"raw_cycles", raw_cycles}
    };
    report.measurement_str = {
        {"method", "dependent conflict-free warp shared-memory load chain"},
        {"load_instruction", "ld.shared.u32"},
        {"timer", "clock64 (cycles), globaltimer (SM clock)"},
        {"value_formula", "per_load = (raw_cycles[i] - clock_overhead_cycles) / loads_per_rep; value = median(per_load)"}
    };
    report.measurement_num = {
        {"clock_overhead_cycles", clock_overhead},
        {"nodes_per_lane", NUM_NODES},
        {"loads_per_rep", CHASE_LOADS},
        {"threads_per_block", WARP_THREADS},
        {"reps", reps}
    };

    write_report(args, report);

    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_clock));
    return 0;
}
