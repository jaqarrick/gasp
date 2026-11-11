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

// reads per thread
#ifndef PASSES
#define PASSES 16
#endif

// Each thread sums PASSES neighbouring elements, (t + p) mod N: the lines a
// warp reads were just read by itself or its neighbours, so most loads hit L1.
// The L1 side of the L2 pair, against sweep_far.cu: same instructions, same
// bytes, same 2 MB array, only the distance between a thread's reads differs.
// N must be a power of two (the mask is the mod).
__global__ void sweep_near(const float *x, float *out, int N) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int mask = N - 1;
    float sum = 0.0f;
#pragma unroll
    for (int p = 0; p < PASSES; p++) {
        sum += x[(t + p) & mask];
    }
    out[t] = sum;
}

int main(int argc, char *argv[]) {
    int N = 1 << 19;
    if (argc > 1) N = atoi(argv[1]);

    int block_size = GASP_BLOCK_SIZE;
    if ((N & (N - 1)) != 0 || N % block_size != 0) {
        printf("N=%d must be a power of two and a multiple of the block size %d\n",
               N, block_size);
        return 1;
    }
    int grid_size = GASP_GRID_SIZE > 0 ? GASP_GRID_SIZE : N / block_size;

    float *h_x = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_x[i] = (float)(i % 1000);

    float *d_x, *d_out;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    sweep_near<<<grid_size, block_size>>>(d_x, d_out, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // same order of additions as the kernel, so the sums match exactly
    int errors = 0;
    for (int t = 0; t < N; t++) {
        float sum = 0.0f;
        for (int p = 0; p < PASSES; p++) sum += h_x[(t + p) & (N - 1)];
        if (h_out[t] != sum) errors++;
    }
    printf("N=%d, block=%d, grid=%d, errors=%d\n", N, block_size, grid_size, errors);

    cudaFree(d_x);
    cudaFree(d_out);
    free(h_x);
    free(h_out);
    return errors ? 1 : 0;
}
