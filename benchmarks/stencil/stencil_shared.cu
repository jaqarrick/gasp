#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// launch config, set by gasp --run; a grid size of -1 means N / block
#ifndef GASP_BLOCK_SIZE
#define GASP_BLOCK_SIZE 256
#endif
#ifndef GASP_GRID_SIZE
#define GASP_GRID_SIZE -1
#endif

// stencil radius: each output sums 2 * RADIUS + 1 inputs
#ifndef RADIUS
#define RADIUS 8
#endif

// Same result as stencil_global, with the block's inputs staged in shared
// memory: one global load per thread, plus one more for the 2 * RADIUS threads
// that load the halo, then 17 shared loads. The few-loads side of the L1 pair.
// Assumes blockDim.x == GASP_BLOCK_SIZE; `in` holds N + 2 * RADIUS floats.
__global__ void stencil_shared(const float *in, float *out, int N) {
    __shared__ float tile[GASP_BLOCK_SIZE + 2 * RADIUS];
    int t = threadIdx.x;
    int base = blockIdx.x * blockDim.x;

    tile[t] = in[base + t];
    if (t < 2 * RADIUS) {
        tile[GASP_BLOCK_SIZE + t] = in[base + GASP_BLOCK_SIZE + t];
    }
    __syncthreads();

    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k <= 2 * RADIUS; k++) {
        sum += tile[t + k];
    }
    out[base + t] = sum;
}

int main(int argc, char *argv[]) {
    int N = 1 << 20;
    if (argc > 1) N = atoi(argv[1]);

    int block_size = GASP_BLOCK_SIZE;
    if (N % block_size != 0) {
        printf("N=%d must be a multiple of the block size %d\n", N, block_size);
        return 1;
    }
    int grid_size = GASP_GRID_SIZE > 0 ? GASP_GRID_SIZE : N / block_size;

    size_t in_count = N + 2 * RADIUS;
    float *h_in = (float *)malloc(in_count * sizeof(float));
    float *h_out = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < in_count; i++) h_in[i] = (float)(i % 1000);

    float *d_in, *d_out;
    cudaMalloc(&d_in, in_count * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, in_count * sizeof(float), cudaMemcpyHostToDevice);

    stencil_shared<<<grid_size, block_size>>>(d_in, d_out, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // same order of additions as the kernel, so the sums match exactly
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int k = 0; k <= 2 * RADIUS; k++) sum += h_in[i + k];
        if (h_out[i] != sum) errors++;
    }
    printf("N=%d, block=%d, grid=%d, errors=%d\n", N, block_size, grid_size, errors);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return errors ? 1 : 0;
}
