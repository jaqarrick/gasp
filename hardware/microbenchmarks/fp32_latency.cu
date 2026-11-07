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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHAIN_OPS 1024 // number of chained FADD operations in the timed region
#define UNROLL 16 // number of FADDs per loop iteration
#define PASSES 2 // pass 0 warms the instruction cache; last pass is kept
#define WARMUP_LAUNCHES 10 // untimed launches (module load, GPU clocks up)
#define REPS 200 // timed launches (issue asks for >= 100)

// ---- Device ----------------------------------------------------------------

// Read the SM cycle counter. Written as volatile asm (instead of calling clock64()) so the compiler keeps it in order with the volatile asm FADDs.
__device__ unsigned long long read_clock64(void) {
    unsigned long long c;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c) :: "memory");
    return c;
}

// One dependent add: x = x + step. "+f"(x) makes x both input and output, so every FADD must wait for the previous one.
#define FADD_ONE asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(x) : "f"(step) : "memory");

// out[0] = cycles for CHAIN_OPS dependent FADDs (clock overhead included)
// out[1] = clock overhead (two back-to-back clock reads)
// sink[0] = final x, so the host can check that every add ran
__global__ void fp32_latency_kernel(float step, long long *out, float *sink) {
    float x = 1.0f;
    unsigned long long t0 = 0;
    unsigned long long t1 = 0;

#pragma unroll 1
    for (int pass = 0; pass < PASSES; pass++) {
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
    }

    unsigned long long b0 = read_clock64();
    unsigned long long b1 = read_clock64();

    out[0] = (long long)(t1 - t0);
    out[1] = (long long)(b1 - b0);
    sink[0] = x;
}

// ---- Host helpers ----------------------------------------------------------

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

// Median of n values. Sorts a copy, so the caller's array is unchanged.
static double median_of(const double *values, int n) {
    double *sorted = (double *)malloc(n * sizeof(double));
    memcpy(sorted, values, n * sizeof(double));
    qsort(sorted, n, sizeof(double), compare_doubles);
    double m = (n % 2) ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    free(sorted);
    return m;
}

// Run a shell command and copy the first line of its output into buf.
// buf is "unknown" if the command fails or prints nothing.
static void run_cmd_first_line(const char *cmd, char *buf, size_t size) {
    snprintf(buf, size, "unknown");
    FILE *pipe = popen(cmd, "r");
    if (!pipe) return;
    char line[256];
    if (fgets(line, sizeof(line), pipe)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] != '\0') snprintf(buf, size, "%s", line);
    }
    pclose(pipe);
}

// "NVIDIA A100-SXM4-80GB" -> "a100". Other GPUs get the lowercased name with
// anything that isn't a letter or digit replaced by '_'
// ("NVIDIA GeForce RTX 2080 Ti" -> "nvidia_geforce_rtx_2080_ti").
static void short_gpu_name(const char *full, char *buf, size_t size) {
    const char *known[] = {"a100", "h100", "h200", "b200", "b100"};
    char lower[256];
    size_t n = 0;
    for (; full[n] != '\0' && n < sizeof(lower) - 1; n++) {
        lower[n] = (char)tolower((unsigned char)full[n]);
    }
    lower[n] = '\0';

    for (size_t k = 0; k < sizeof(known) / sizeof(known[0]); k++) {
        if (strstr(lower, known[k])) {
            snprintf(buf, size, "%s", known[k]);
            return;
        }
    }
    for (n = 0; lower[n] != '\0' && n < size - 1; n++) {
        buf[n] = isalnum((unsigned char)lower[n]) ? lower[n] : '_';
    }
    buf[n] = '\0';
}

// Folder that holds the executable, taken from argv[0]:
// "./fp32_latency" -> ".", "hardware/microbenchmarks/fp32_latency" -> "hardware/microbenchmarks".
static void executable_dir(const char *argv0, char *buf, size_t size) {
    const char *slash = strrchr(argv0, '/');
    if (!slash) {
        snprintf(buf, size, ".");
    } else if (slash == argv0) {
        snprintf(buf, size, "/");
    } else {
        snprintf(buf, size, "%.*s", (int)(slash - argv0), argv0);
    }
}

// ---- Main ------------------------------------------------------------------

int main(int argc, char **argv) {
    (void)argc;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    long long *d_out;
    float *d_sink;
    CUDA_CHECK(cudaMalloc(&d_out, 2 * sizeof(long long)));
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
    double overhead_samples[REPS];

    for (int r = 0; r < REPS; r++) {
        fp32_latency_kernel<<<1, 1>>>(step, d_out, d_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        long long h_out[2];
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
    }

    // Subtract the median clock overhead from every run, then divide by the
    // number of adds to get cycles per FADD.
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

    double sum = 0.0;
    double mn = per_op[0];
    double mx = per_op[0];
    for (int r = 0; r < REPS; r++) {
        sum += per_op[r];
        if (per_op[r] < mn) mn = per_op[r];
        if (per_op[r] > mx) mx = per_op[r];
    }
    double mean = sum / REPS;
    double var = 0.0;
    for (int r = 0; r < REPS; r++) {
        var += (per_op[r] - mean) * (per_op[r] - mean);
    }
    double stddev = sqrt(var / (REPS - 1));
    double median = median_of(per_op, REPS);

    if (median < 2.0) {
        fprintf(stderr, "WARNING: median below 2 cycles; the adds may not be dependent. Check the SASS.\n");
    }

    // ---- Collect run metadata ----------------------------------------------
    char gpu[128];
    short_gpu_name(prop.name, gpu, sizeof(gpu));

    int runtime_version = 0;  // e.g. 12080 for CUDA 12.8
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

    char driver[64];
    run_cmd_first_line("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null",
                       driver, sizeof(driver));

    char timestamp[32];
    time_t now = time(NULL);
    struct tm utc;
    gmtime_r(&now, &utc);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc);

    // ---- Write <exe dir>/fp32_inst_latency_cycles.<gpu>.run.json ------------
    char out_dir[1024];
    executable_dir(argv[0], out_dir, sizeof(out_dir));
    char out_path[1400];
    snprintf(out_path, sizeof(out_path), "%s/fp32_inst_latency_cycles.%s.run.json", out_dir, gpu);

    FILE *f = fopen(out_path, "w");
    if (!f) {
        perror(out_path);
        return 1;
    }
    fprintf(f, "{\n");
    fprintf(f, "  \"benchmark\": \"fp32_latency.cu\",\n");
    fprintf(f, "  \"parameter\": \"fp32_inst_latency_cycles\",\n");
    fprintf(f, "  \"value\": %.4f,\n", median);
    fprintf(f, "  \"unit\": \"cycles\",\n");
    fprintf(f, "  \"gpu\": \"%s\",\n", gpu);
    fprintf(f, "  \"gpu_name\": \"%s\",\n", prop.name);
    fprintf(f, "  \"arch\": \"sm_%d%d\",\n", prop.major, prop.minor);
    fprintf(f, "  \"num_sms\": %d,\n", prop.multiProcessorCount);
    fprintf(f, "  \"stats\": {\n");
    fprintf(f, "    \"median\": %.4f,\n", median);
    fprintf(f, "    \"mean\": %.4f,\n", mean);
    fprintf(f, "    \"std\": %.4f,\n", stddev);
    fprintf(f, "    \"min\": %.4f,\n", mn);
    fprintf(f, "    \"max\": %.4f\n", mx);
    fprintf(f, "  },\n");
    // raw_cycles           : per run, clock64 delta around the chain, clock-read cost INCLUDED
    // overhead_samples     : per run, clock64 delta of two back-to-back reads (cost of the clock)
    // overhead_cycles      : median of overhead_samples; the value subtracted from every run
    // net_cycles           : raw_cycles - overhead_cycles (per run)
    // per_op_cycles        : net_cycles / CHAIN_OPS (per run); stats are computed over these
    // uncorrected_value    : median of raw_cycles / CHAIN_OPS (no overhead subtracted)
    fprintf(f, "  \"raw_cycles\": [");
    for (int r = 0; r < REPS; r++) fprintf(f, "%s%.0f", r ? ", " : "", raw_cycles[r]);
    fprintf(f, "],\n");
    fprintf(f, "  \"overhead_samples_cycles\": [");
    for (int r = 0; r < REPS; r++) fprintf(f, "%s%.0f", r ? ", " : "", overhead_samples[r]);
    fprintf(f, "],\n");
    fprintf(f, "  \"overhead_cycles\": %.4f,\n", overhead);
    fprintf(f, "  \"overhead_subtracted\": true,\n");
    fprintf(f, "  \"net_cycles\": [");
    for (int r = 0; r < REPS; r++) fprintf(f, "%s%.4f", r ? ", " : "", net_cycles[r]);
    fprintf(f, "],\n");
    fprintf(f, "  \"per_op_cycles\": [");
    for (int r = 0; r < REPS; r++) fprintf(f, "%s%.4f", r ? ", " : "", per_op[r]);
    fprintf(f, "],\n");
    fprintf(f, "  \"uncorrected_value\": %.4f,\n", uncorrected_value);
    fprintf(f, "  \"cuda_version\": \"%d.%d\",\n", runtime_version / 1000, (runtime_version % 1000) / 10);
    fprintf(f, "  \"nvcc_version\": \"%d.%d\",\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
    fprintf(f, "  \"driver_version\": \"%s\",\n", driver);
    fprintf(f, "  \"timestamp\": \"%s\"\n", timestamp);
    fprintf(f, "}\n");
    fclose(f);

    fprintf(stderr, "Saved %s (median %.3f cycles per FADD)\n", out_path, median);

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
//   * a branch after the 16 FADDs (the loop was kept, not fully unrolled)
// ---------------------------------------------------------------------------
