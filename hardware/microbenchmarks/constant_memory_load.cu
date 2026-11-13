#include <array>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include "benchmark_formatting.h"

// References https://github.com/sjfeng1999/gpu-arch-microbenchmark/blob/master/memory/memory_latency.cu
// This benchmark measures the load operations from constant memory using a ring and pointer-chasing.
// Since constant memory is read-only and hardware-broadcasted, every thread from our warp reads the same address simultaneously

constexpr int NUM_NODES = 64;
constexpr int WARP_THREADS = 32;
constexpr int CHASE_LOADS = 1024;
constexpr int UNROLL = 16;
constexpr int DEFAULT_REPS = 200;

static_assert(CHASE_LOADS % UNROLL == 0,
              "CHASE_LOADS must be a multiple of UNROLL");

// ring of addresses stored in constant memory
__constant__ uint64_t c_ring[NUM_NODES];

static const char* PARAMETER = "const_mem_latency_cycles";

// increment address to next item in ring
#define CHASE_ONE p = load_constant_pointer(p)

__device__ __forceinline__ uint64_t load_constant_pointer(uint64_t address) {
    uint64_t next;
    asm volatile(
        "ld.const.u64 %0, [%1];"
        : "=l"(next)
        : "l"(address)
        : "memory");
    return next;
}

// clock_out[0..1] = lane 0's clock64 cycles and globaltimer ns around the timed region
__global__ void constant_chase_kernel(unsigned long long* out, unsigned long long* clock_out) {
    const int lane = threadIdx.x;
    uint64_t p = reinterpret_cast<uint64_t>(c_ring);

// warm up the constant-memory ring before timing
#pragma unroll 1
    for (int i = 0; i < 2 * NUM_NODES; ++i) {
        CHASE_ONE;
    }

    const uint64_t baseline_start = read_clock64();
    const uint64_t baseline_stop = read_clock64();

    const uint64_t g0 = read_globaltimer();
    const uint64_t start = read_clock64();
//runs 1024 times thru the constant memory ring
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

    // reads the timing and final index of the first lane in warp
    if (lane == 0) {
        out[0] = stop - start;
        out[1] = baseline_stop - baseline_start;
        out[2] = p;
    }
}

int main(int argc, char** argv) {
    const BenchArgs args = parse_bench_args(argc, argv, DEFAULT_REPS);
    const int reps = args.reps;

    void* ring_address = nullptr;
    CUDA_CHECK(cudaGetSymbolAddress(&ring_address, c_ring));

    std::array<uint64_t, NUM_NODES> host_ring{};
    const uint64_t base_address =
        reinterpret_cast<uint64_t>(ring_address);

    for (int i = 0; i < NUM_NODES; ++i) {
        const int next = (i + 1) % NUM_NODES;
        host_ring[i] = base_address +
                       static_cast<uint64_t>(next) * sizeof(uint64_t);
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_ring, host_ring.data(),
                                  sizeof(host_ring)));

    unsigned long long* device_output = nullptr;
    unsigned long long* device_clock = nullptr;  // lane 0's cycles and ns, for the SM clock
    CUDA_CHECK(cudaMalloc(&device_clock, 2 * sizeof(unsigned long long)));
    double sum_clock_cycles = 0.0, sum_clock_ns = 0.0;
    CUDA_CHECK(cudaMalloc(&device_output, 3 * sizeof(unsigned long long)));

    std::vector<double> raw_cycles;
    std::vector<double> clock_overhead_cycles;
    raw_cycles.reserve(reps);
    clock_overhead_cycles.reserve(reps);

    //run the kernel for X reps
    for (int rep = 0; rep < reps; ++rep) {
        constant_chase_kernel<<<1, WARP_THREADS>>>(device_output, device_clock);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long host_clock[2] = {0, 0};
        CUDA_CHECK(cudaMemcpy(host_clock, device_clock, sizeof(host_clock),
                              cudaMemcpyDeviceToHost));
        if (host_clock[1] > 0) {
            sum_clock_cycles += static_cast<double>(host_clock[0]);
            sum_clock_ns += static_cast<double>(host_clock[1]);
        }

        unsigned long long output[3]{};
        CUDA_CHECK(cudaMemcpy(output, device_output, sizeof(output),
                              cudaMemcpyDeviceToHost));

        raw_cycles.push_back(static_cast<double>(output[0]));
        clock_overhead_cycles.push_back(static_cast<double>(output[1]));
    }
    // statistics collection
    const double clock_overhead = median(clock_overhead_cycles);
    std::vector<double> per_load_cycles;
    per_load_cycles.reserve(reps);

    for (double cycles : raw_cycles) {
        per_load_cycles.push_back(
            (cycles - clock_overhead) / CHASE_LOADS);
    }

    // json file processing
    BenchmarkReport report;
    report.benchmark = "constant_memory_load.cu";
    report.parameter = PARAMETER;
    report.unit = "cycles";
    report.stats = compute_stats(per_load_cycles);
    // clock.sm_clock_ghz: total clock64 cycles / total globaltimer ns over all reps
    report.sm_clock_ghz = sum_clock_ns > 0.0 ? sum_clock_cycles / sum_clock_ns : 0.0;

    //use c++ initializer lists to add measurements that are unique to this benchmark
    report.arrays = {
        {"raw_cycles", raw_cycles},
        {"clock_overhead_samples_cycles", clock_overhead_cycles},
        {"per_load_cycles", per_load_cycles}
    };
    report.scalars = {
        {"clock_overhead_cycles", clock_overhead}
    };
    report.measurement_str = {
        {"method", "dependent uniform constant-cache-hit pointer chase"},
        {"load_instruction", "ld.const.u64"},
        {"timer", "clock64 (cycles), globaltimer (SM clock)"}
    };
    report.measurement_num = {
        {"nodes", NUM_NODES},
        {"loads_per_rep", CHASE_LOADS},
        {"threads_per_block", WARP_THREADS},
        {"reps", reps}
    };

    write_report(args, report);

    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_clock));
    return 0;
}
