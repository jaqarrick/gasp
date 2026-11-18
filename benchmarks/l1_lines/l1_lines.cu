// Tests how T_mem turns sectors into 128-byte lines from our assumption on how ua sectors translates to cache lines

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// launch config, set by gasp --run; a grid size of -1 means ceil(N / block)
#ifndef GASP_BLOCK_SIZE
#define GASP_BLOCK_SIZE 256
#endif
#ifndef GASP_GRID_SIZE
#define GASP_GRID_SIZE -1
#endif

// one slot holds the widest pattern (gap 32: 32 lines, 4 KB); 8 slots = 32 KB
#define WINDOW_FLOATS 1024
#define SLOTS 8

__global__ void l1_lines(float *out, const float *in, int N, int group, int gap,
                         int offset, int reads) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int lane = threadIdx.x % 32;
    int idx = (lane / group) * gap + lane % group + offset;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma unroll 1
    for (int k = 0; k < reads; k += 4) {
        int base = idx + (k % 8) * WINDOW_FLOATS;
        s0 += in[base];
        s1 += in[base + 1 * WINDOW_FLOATS];
        s2 += in[base + 2 * WINDOW_FLOATS];
        s3 += in[base + 3 * WINDOW_FLOATS];
    }
    out[i] = s0 + s1 + s2 + s3;
}

static void check_cuda(cudaError_t err, const char *context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char *argv[]) {
    int N = 1048576;
    int group = 32;
    int gap = 32;
    int offset = 0;
    int reads = 256;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) group = atoi(argv[2]);
    if (argc > 3) gap = atoi(argv[3]);
    if (argc > 4) offset = atoi(argv[4]);
    if (argc > 5) reads = atoi(argv[5]);

    // the pattern must fit one slot, so every load stays in the 32 KB window
    int last = (31 / (group > 0 ? group : 1)) * gap + group - 1 + offset;
    if (group <= 0 || 32 % group != 0 || gap < group || offset < 0 ||
        last >= WINDOW_FLOATS || reads <= 0 || reads % 4 != 0) {
        fprintf(stderr, "need group dividing 32, gap >= group, offset >= 0, "
                        "pattern within %d floats, reads a multiple of 4\n", WINDOW_FLOATS);
        return 1;
    }

    size_t in_floats = (size_t)SLOTS * WINDOW_FLOATS;
    size_t in_bytes = in_floats * sizeof(float);
    size_t out_bytes = (size_t)N * sizeof(float);
    float *h_in = (float *)malloc(in_bytes);
    float *h_out = (float *)malloc(out_bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }
    // all ones, so every thread's sum is exactly reads
    for (size_t j = 0; j < in_floats; j++) h_in[j] = 1.0f;

    float *d_in, *d_out;
    check_cuda(cudaMalloc(&d_in, in_bytes), "malloc d_in");
    check_cuda(cudaMalloc(&d_out, out_bytes), "malloc d_out");
    check_cuda(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice), "memcpy d_in");

    int block_size = GASP_BLOCK_SIZE;
    int grid_size = GASP_GRID_SIZE > 0 ? GASP_GRID_SIZE
                                       : (N + block_size - 1) / block_size;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    l1_lines<<<grid_size, block_size>>>(d_out, d_in, N, group, gap, offset, reads);
    cudaEventRecord(t1);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "synchronize");
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    check_cuda(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost), "memcpy h_out");
    int errors = 0;
    for (int j = 0; j < N; j++) {
        if (h_out[j] != (float)reads) errors++;
    }
    printf("N=%d, group=%d, gap=%d, offset=%d, reads=%d, block=%d, grid=%d, "
           "kernel: %.3f ms, errors=%d\n",
           N, group, gap, offset, reads, block_size, grid_size, ms, errors);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return errors ? 1 : 0;
}
