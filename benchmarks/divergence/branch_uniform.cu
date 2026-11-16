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

// every thread of a block takes the same side, so each warp runs one
__global__ void branch_uniform(float *out, const float *in, int N, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = in[i];
    if (blockIdx.x & 1) {
#pragma unroll 1
        for (int k = 0; k < n; ++k) x = __sinf(x) * 1.01f + 0.5f;
    } else {
#pragma unroll 1
        for (int k = 0; k < n; ++k) x = __cosf(x) * 0.99f - 0.25f;
    }
    out[i] = x;
}

static void check_cuda(cudaError_t err, const char *context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char *argv[]) {
    int N = 1048576;
    int n = 256;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);

    size_t bytes = (size_t)N * sizeof(float);
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 1000) * 0.001f;

    float *d_in, *d_out;
    check_cuda(cudaMalloc(&d_in, bytes), "malloc d_in");
    check_cuda(cudaMalloc(&d_out, bytes), "malloc d_out");
    check_cuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "memcpy d_in");

    int block_size = GASP_BLOCK_SIZE;
    int grid_size = GASP_GRID_SIZE > 0 ? GASP_GRID_SIZE
                                       : (N + block_size - 1) / block_size;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    branch_uniform<<<grid_size, block_size>>>(d_out, d_in, N, n);
    cudaEventRecord(t1);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "synchronize");

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    check_cuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "memcpy h_out");
    double checksum = 0.0;
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (!isfinite(h_out[i])) errors++;
        checksum += h_out[i];
    }
    printf("N=%d, n=%d, block=%d, grid=%d, kernel: %.3f ms, checksum=%.3f, errors=%d\n",
           N, n, block_size, grid_size, ms, checksum, errors);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return errors ? 1 : 0;
}
