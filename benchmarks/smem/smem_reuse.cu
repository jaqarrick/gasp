// Shared-memory carveout test: the same global access pattern with a small or
// large static shared array, so the driver picks a different shared/L1 split.
//
// Each block re-reads its own 10 KB slice of `in` PASSES times. With 4 blocks
// per SM that is 40 KB of global data per SM: it fits a 64 KB L1 (small shared
// array, 32 KB carveout) but not a 32 KB L1 (large array, 64 KB carveout).
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK 256

// floats of static shared memory per block, set per kernel with -D in smem.toml
#ifndef SMEM_FLOATS
#define SMEM_FLOATS 1024  // 4 KB
#endif
#ifndef SLICE_STEPS
#define SLICE_STEPS 10  // elements per thread per pass: a 10 KB slice per block
#endif
#ifndef PASSES
#define PASSES 4
#endif

#ifndef GASP_BLOCK_SIZE
#define GASP_BLOCK_SIZE BLOCK
#endif
#ifndef GASP_GRID_SIZE
#define GASP_GRID_SIZE 1088  // 68 SMs x 4 blocks x 4 waves on the RTX 2080 Ti
#endif

static_assert(SMEM_FLOATS % BLOCK == 0, "SMEM_FLOATS must be a multiple of BLOCK");

__global__ void smem_reuse(float *out, const float *in) {
    __shared__ float buf[SMEM_FLOATS];

    const float *slice = in + (size_t)blockIdx.x * BLOCK * SLICE_STEPS + threadIdx.x;
    float *dst = out + (size_t)blockIdx.x * BLOCK + threadIdx.x;

    float acc = 0.0f;
#pragma unroll 1
    for (int p = 0; p < PASSES; p++) {
#pragma unroll
        for (int j = 0; j < SLICE_STEPS; j++) acc += slice[j * BLOCK];
        // out may alias in, so the next pass has to load the slice again
        *dst = acc;
    }

    // fill and read back the whole shared array, so the compiler keeps all of it
#pragma unroll
    for (int k = 0; k < SMEM_FLOATS / BLOCK; k++) buf[k * BLOCK + threadIdx.x] = acc + k;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < SMEM_FLOATS / BLOCK; k++)
        s += buf[k * BLOCK + (threadIdx.x + 1) % BLOCK];
    *dst = acc + s;
}

static void check_cuda(cudaError_t err, const char *context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(err));
        exit(1);
    }
}

int main(void) {
    if (GASP_BLOCK_SIZE != BLOCK) {
        fprintf(stderr, "this kernel needs block_size = %d\n", BLOCK);
        return 1;
    }
    const int grid = GASP_GRID_SIZE;
    const size_t threads = (size_t)grid * BLOCK;
    const size_t in_elems = threads * SLICE_STEPS;
    const size_t in_bytes = in_elems * sizeof(float);
    const size_t out_bytes = threads * sizeof(float);

    float *h_in = (float *)malloc(in_bytes);
    float *h_out = (float *)malloc(out_bytes);
    float *h_acc = (float *)malloc(out_bytes);
    if (!h_in || !h_out || !h_acc) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }
    // small integers, so every sum is exact in float
    for (size_t i = 0; i < in_elems; i++) h_in[i] = (float)(i % 4);

    float *d_in, *d_out;
    check_cuda(cudaMalloc(&d_in, in_bytes), "malloc d_in");
    check_cuda(cudaMalloc(&d_out, out_bytes), "malloc d_out");
    check_cuda(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice), "memcpy d_in");

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    smem_reuse<<<grid, BLOCK>>>(d_out, d_in);
    cudaEventRecord(t1);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "synchronize");

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("shared=%d B/block, slice=%d B/block, passes=%d, grid=%d, kernel: %.3f ms\n",
           (int)(SMEM_FLOATS * sizeof(float)), (int)(BLOCK * SLICE_STEPS * sizeof(float)),
           PASSES, grid, ms);

    check_cuda(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost), "memcpy h_out");
    for (size_t b = 0; b < (size_t)grid; b++) {
        for (int t = 0; t < BLOCK; t++) {
            float acc = 0.0f;
            for (int p = 0; p < PASSES; p++)
                for (int j = 0; j < SLICE_STEPS; j++)
                    acc += h_in[b * BLOCK * SLICE_STEPS + (size_t)j * BLOCK + t];
            h_acc[b * BLOCK + t] = acc;
        }
    }
    size_t errors = 0;
    for (size_t b = 0; b < (size_t)grid; b++) {
        for (int t = 0; t < BLOCK; t++) {
            float neighbor = h_acc[b * BLOCK + (t + 1) % BLOCK];
            float s = 0.0f;
            for (int k = 0; k < SMEM_FLOATS / BLOCK; k++) s += neighbor + k;
            if (h_out[b * BLOCK + t] != h_acc[b * BLOCK + t] + s) errors++;
        }
    }
    printf("errors=%zu\n", errors);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    free(h_acc);
    return errors != 0;
}
