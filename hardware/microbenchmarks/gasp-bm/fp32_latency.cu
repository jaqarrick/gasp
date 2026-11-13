// fp32_latency.cu
// ---------------------------------------------------------------------------
// Measures : fp32_inst_latency_cycles
// Method   : One thread runs a chain of dependent add.rn.f32 instructions
//            (each add needs the previous result). The SM cycle counter
//            (%clock64) is read inside the kernel before and after the chain,
//            so launch overhead is not included. Clock-read overhead is
//            measured in the same kernel and subtracted.
//
// Build    : (from hardware/microbenchmarks/)
//            nvcc -O3 -arch=sm_75 fp32_latency.cu -o fp32_latency   (RTX 2080 Ti)
//            nvcc -O3 -arch=sm_80 fp32_latency.cu -o fp32_latency   (A100)
//            nvcc -O3 -arch=sm_90 fp32_latency.cu -o fp32_latency   (H100)
//
// Run      : ./fp32_latency
//            Writes fp32_inst_latency_cycles.<gpu>.run.json into the folder
//            that holds the executable.
//
// Inspect  : nvcc -O3 -arch=sm_80 -cubin fp32_latency.cu -o fp32_latency.cubin
//            cuobjdump -sass fp32_latency.cubin     (checklist at bottom)
// ---------------------------------------------------------------------------
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "common.h"

#define CHAIN_OPS 1024 // number of chained FADD operations in the timed region
#define UNROLL 16 // number of FADDs per loop iteration
#define PASSES 2 // pass 0 warms the instruction cache; last pass is kept
#define WARMUP_LAUNCHES 10 // untimed launches (module load, GPU clocks up)
#define REPS 200 // timed launches (issue asks for >= 100)

// ---- Device ----------------------------------------------------------------

// One dependent add: x = x + step. "+f"(x) makes x both input and output, so every FADD must wait for the previous one.
#define FADD_ONE asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(x) : "f"(step) : "memory");

// out[0] = cycles for CHAIN_OPS dependent FADDs (clock overhead included)
// out[1] = clock overhead (two back-to-back clock reads)
// out[2] = globaltimer ns around the timed chain (read outside the clock64 pair)
// sink[0] = final x, so the host can check that every add ran
__global__ void fp32_latency_kernel(float step, long long *out, float *sink) {
    float x = 1.0f;
    unsigned long long t0 = 0;
    unsigned long long t1 = 0;
    unsigned long long g0 = 0;
    unsigned long long g1 = 0;

#pragma unroll 1
    for (int pass = 0; pass < PASSES; pass++) {
        g0 = read_globaltimer();
        t0 = read_clock64(); // start timer
        // do UNROLL=16 dependent adds (this means if we change the UNROLL value, this will no longer work)
#pragma unroll 1
        for (int i = 0; i < CHAIN_OPS; i += UNROLL) { // <-- possible overhead for loop creation -josh
            FADD_ONE FADD_ONE FADD_ONE FADD_ONE
            FADD_ONE FADD_ONE FADD_ONE FADD_ONE
            FADD_ONE FADD_ONE FADD_ONE FADD_ONE
            FADD_ONE FADD_ONE FADD_ONE FADD_ONE
        }
        t1 = read_clock64(); // end timer
        g1 = read_globaltimer();
    }

    unsigned long long b0 = read_clock64();
    unsigned long long b1 = read_clock64();

    out[0] = (long long)(t1 - t0);
    out[1] = (long long)(b1 - b0);
    out[2] = (long long)(g1 - g0);
    sink[0] = x;
}

// ---- Main ------------------------------------------------------------------

int main(int argc, char **argv) {
    (void)argc;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    long long *d_out;
    float *d_sink;
    CUDA_CHECK(cudaMalloc(&d_out, 3 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

    const float step = 1.0f;
    // Starting at 1 and adding 1 stays exact in float (below 2^24),
    // so the final value tells us whether every add executed.
    const float expected_x = 1.0f + (float)(PASSES * CHAIN_OPS) * step;

    for (int i = 0; i < WARMUP_LAUNCHES; i++) {
        fp32_latency_kernel<<<1, 1>>>(step, d_out, d_sink);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double raw_cycles[REPS];
    double sum_clock_cycles = 0.0, sum_clock_ns = 0.0;  // for the SM clock
    double overhead_samples[REPS];

    for (int r = 0; r < REPS; r++) {
        fp32_latency_kernel<<<1, 1>>>(step, d_out, d_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        long long h_out[3];
        float h_x;
        CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_x, d_sink, sizeof(float), cudaMemcpyDeviceToHost));

        if (h_x != expected_x) {
            fprintf(stderr, "ERROR: final x = %.1f, expected %.1f. The FADD chain did not run "
                            "as written; check the SASS.\n", h_x, expected_x);
            return 1;
        }
        raw_cycles[r] = (double)h_out[0];
        overhead_samples[r] = (double)h_out[1];
        if (h_out[2] > 0) {
            sum_clock_cycles += (double)h_out[0];
            sum_clock_ns += (double)h_out[2];
        }
    }

    // Subtract the median clock overhead from every run, then divide by the
    // number of adds to get cycles per FADD.
    if (sum_clock_ns <= 0.0) {
        fprintf(stderr, "ERROR: globaltimer did not advance\n");
        return 1;
    }
    // SM clock over all reps: total cycles / total ns (globaltimer updates every ~1 us,
    // so one short chain alone would be too coarse).
    const double sm_clock_ghz = sum_clock_cycles / sum_clock_ns;

    double overhead = median_of(overhead_samples, REPS);
    // Per-run breakdown: gross (clock overhead included) -> net (overhead removed) -> per op.
    double net_cycles[REPS];
    double per_op[REPS];
    double gross_per_op[REPS];
    for (int r = 0; r < REPS; r++) {
        net_cycles[r] = raw_cycles[r] - overhead;
        per_op[r] = net_cycles[r] / (double)CHAIN_OPS;
        gross_per_op[r] = raw_cycles[r] / (double)CHAIN_OPS;
    }
    double uncorrected_value = median_of(gross_per_op, REPS);  // no subtraction

    const Stats stats = compute_stats(per_op, REPS);
    const double median = stats.median;

    if (median < 2.0) {
        fprintf(stderr, "WARNING: median below 2 cycles; the adds may not be dependent. Check the SASS.\n");
    }

    // ---- Collect run metadata ----------------------------------------------
    const std::string gpu = short_gpu_name(prop.name);

    int runtime_version = 0;  // e.g. 12080 for CUDA 12.8
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

    const std::string driver = driver_version();

    const std::string timestamp = utc_timestamp();

    // ---- Write <exe dir>/fp32_inst_latency_cycles.<gpu>.run.json ------------
    std::string json;
    json += "{\n";
    json += "  \"benchmark\": \"fp32_latency.cu\",\n";
    json += "  \"parameter\": \"fp32_inst_latency_cycles\",\n";
    json += strprintf("  \"value\": %.4f,\n", median);
    json += "  \"unit\": \"cycles\",\n";
    json += strprintf("  \"gpu\": \"%s\",\n", gpu.c_str());
    json += strprintf("  \"gpu_name\": \"%s\",\n", prop.name);
    json += strprintf("  \"arch\": \"sm_%d%d\",\n", prop.major, prop.minor);
    json += strprintf("  \"num_sms\": %d,\n", prop.multiProcessorCount);
    json += json_stats(stats);
    // raw_cycles           : per run, clock64 delta around the chain, clock-read cost INCLUDED
    // overhead_samples     : per run, clock64 delta of two back-to-back reads (cost of the clock)
    // overhead_cycles      : median of overhead_samples; the value subtracted from every run
    // net_cycles           : raw_cycles - overhead_cycles (per run)
    // per_op_cycles        : net_cycles / CHAIN_OPS (per run); stats are computed over these
    // uncorrected_value    : median of raw_cycles / CHAIN_OPS (no overhead subtracted)
    json += strprintf("  \"raw_cycles\": %s,\n", json_array(raw_cycles, REPS, 0).c_str());
    json += strprintf("  \"overhead_samples_cycles\": %s,\n", json_array(overhead_samples, REPS, 0).c_str());
    json += strprintf("  \"overhead_cycles\": %.4f,\n", overhead);
    json += "  \"overhead_subtracted\": true,\n";
    json += strprintf("  \"net_cycles\": %s,\n", json_array(net_cycles, REPS).c_str());
    json += strprintf("  \"per_op_cycles\": %s,\n", json_array(per_op, REPS).c_str());
    json += strprintf("  \"uncorrected_value\": %.4f,\n", uncorrected_value);
    // clock.sm_clock_ghz: total clock64 cycles / total globaltimer ns over all reps
    json += strprintf("  \"clock\": {\"sm_clock_ghz\": %.4f},\n", sm_clock_ghz);
    json += strprintf("  \"cuda_version\": \"%s\",\n", cuda_version_string(runtime_version).c_str());
    json += strprintf("  \"nvcc_version\": \"%d.%d\",\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
    json += strprintf("  \"driver_version\": \"%s\",\n", driver.c_str());
    json += strprintf("  \"timestamp\": \"%s\"\n", timestamp.c_str());
    json += "}\n";

    const std::string out_dir = executable_dir(argv[0]);
    const std::string out_path = write_run_json(out_dir, "fp32_inst_latency_cycles", gpu, json);
    if (out_path.empty()) return 1;

    fprintf(stderr, "Saved %s (median %.3f cycles per FADD)\n", out_path.c_str(), median);

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}

// ---------------------------------------------------------------------------
// SASS inspection checklist (required by issue #11):
//   cuobjdump -sass fp32_latency.cubin
// In fp32_latency_kernel you should see:
//   * inside the inner loop, 16 consecutive FADD instructions, each reading
//     and writing the same register (FADD R2, R2, R3 ...)
//   * no FFMA / FMUL replacing the chain, and no constant-folded result
//   * CS2R clock reads before and after the inner loop, plus a back-to-back
//     pair after the pass loop
//   * a globaltimer read (SR_GLOBALTIMER) just before the first CS2R and just after
//     the second, in the pass loop
//   * a branch after the 16 FADDs (the loop was kept, not fully unrolled)
// ---------------------------------------------------------------------------
