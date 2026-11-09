#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE 32

__global__ void transpose_naive(float *out, const float *in, int n) {
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * n + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    out[y * n + x] = tile[threadIdx.x][threadIdx.y];
}

static void check_cuda(cudaError_t err, const char *context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char **argv) {
    int n = 4096;
    if (argc > 1) n = atoi(argv[1]);
    if (n <= 0 || n % TILE != 0) {
        fprintf(stderr, "N must be a positive multiple of %d\n", TILE);
        return 1;
    }
#if defined(GASP_BLOCK_DIM_X) && defined(GASP_BLOCK_DIM_Y)
    if (GASP_BLOCK_DIM_X != TILE || GASP_BLOCK_DIM_Y != TILE) {
        fprintf(stderr, "this kernel needs block_size = [%d, %d]\n", TILE, TILE);
        return 1;
    }
#endif

    size_t elems = (size_t)n * n;
    size_t bytes = elems * sizeof(float);
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }
    for (size_t i = 0; i < elems; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    check_cuda(cudaMalloc(&d_in, bytes), "malloc d_in");
    check_cuda(cudaMalloc(&d_out, bytes), "malloc d_out");
    check_cuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "memcpy d_in");

    dim3 threads(TILE, TILE);
    dim3 blocks(n / TILE, n / TILE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    transpose_naive<<<blocks, threads>>>(d_out, d_in, n);
    cudaEventRecord(t1);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "synchronize");

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("N=%d, block=%dx%d, grid=%dx%d, kernel: %.3f ms (%.1f GB/s)\n", n, TILE, TILE,
           blocks.x, blocks.y, ms, 2.0 * bytes / (ms * 1e6));

    check_cuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "memcpy h_out");
    size_t errors = 0;
    for (size_t r = 0; r < (size_t)n; r++)
        for (size_t c = 0; c < (size_t)n; c++)
            if (h_out[r * n + c] != h_in[c * n + r]) errors++;
    printf("errors=%zu\n", errors);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return errors != 0;
}
