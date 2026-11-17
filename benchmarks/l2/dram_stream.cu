#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef GASP_BLOCK_SIZE
#define GASP_BLOCK_SIZE 256
#endif
#ifndef GASP_GRID_SIZE
#define GASP_GRID_SIZE -1
#endif

// 256 MB: far over L2, each element read once
#define ARRAY_FLOATS (1 << 26)
// 64 reads per thread step across the whole array
#define STEP (ARRAY_FLOATS / 64)

// built with -dlcm=cg (l2.toml), so every load skips L1
__global__ void dram_stream(float *out, const float *a, int N, int reads) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma unroll 1
    for (int k = 0; k < reads; k += 4) {
        s0 += a[(i + (k + 0) * STEP) & (ARRAY_FLOATS - 1)];
        s1 += a[(i + (k + 1) * STEP) & (ARRAY_FLOATS - 1)];
        s2 += a[(i + (k + 2) * STEP) & (ARRAY_FLOATS - 1)];
        s3 += a[(i + (k + 3) * STEP) & (ARRAY_FLOATS - 1)];
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
    int reads = 64;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) reads = atoi(argv[2]);

    size_t a_bytes = (size_t)ARRAY_FLOATS * sizeof(float);
    size_t out_bytes = (size_t)N * sizeof(float);
    float *h_a = (float *)malloc(a_bytes);
    float *h_out = (float *)malloc(out_bytes);
    if (!h_a || !h_out) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }
    for (size_t j = 0; j < (size_t)ARRAY_FLOATS; j++) h_a[j] = (float)(j % 1000) * 0.001f;

    float *d_a, *d_out;
    check_cuda(cudaMalloc(&d_a, a_bytes), "malloc d_a");
    check_cuda(cudaMalloc(&d_out, out_bytes), "malloc d_out");
    check_cuda(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice), "memcpy d_a");

    int block_size = GASP_BLOCK_SIZE;
    int grid_size = GASP_GRID_SIZE > 0 ? GASP_GRID_SIZE
                                       : (N + block_size - 1) / block_size;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    dram_stream<<<grid_size, block_size>>>(d_out, d_a, N, reads);
    cudaEventRecord(t1);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "synchronize");

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    check_cuda(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost), "memcpy h_out");
    double checksum = 0.0;
    int errors = 0;
    for (int j = 0; j < N; j++) {
        if (!isfinite(h_out[j])) errors++;
        checksum += h_out[j];
    }
    printf("N=%d, reads=%d, block=%d, grid=%d, kernel: %.3f ms, checksum=%.3f, errors=%d\n",
           N, reads, block_size, grid_size, ms, checksum, errors);

    cudaFree(d_a);
    cudaFree(d_out);
    free(h_a);
    free(h_out);
    return errors ? 1 : 0;
}
