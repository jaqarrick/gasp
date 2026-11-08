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

// Degree 12 polynomial with Horner's rule: one multiply per degree.
__global__ void poly_good(float *A, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = A[i];
        float y = 1.0f / 479001600.0f;
        y = y * x + 1.0f / 39916800.0f;
        y = y * x + 1.0f / 3628800.0f;
        y = y * x + 1.0f / 362880.0f;
        y = y * x + 1.0f / 40320.0f;
        y = y * x + 1.0f / 5040.0f;
        y = y * x + 1.0f / 720.0f;
        y = y * x + 1.0f / 120.0f;
        y = y * x + 1.0f / 24.0f;
        y = y * x + 1.0f / 6.0f;
        y = y * x + 1.0f / 2.0f;
        y = y * x + 1.0f;
        y = y * x + 1.0f;
        C[i] = y;
    }
}

int main(int argc, char *argv[]) {
    int N = 1000000;
    if (argc > 1) N = atoi(argv[1]);

    size_t bytes = N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i / (float)N;
    }

    float *d_A, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    int block_size = GASP_BLOCK_SIZE;
    int grid_size = GASP_GRID_SIZE > 0 ? GASP_GRID_SIZE
                                       : (N + block_size - 1) / block_size;

    poly_good<<<grid_size, block_size>>>(d_A, d_C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // the good and poor kernels round differently, so compare with a tolerance
    int errors = 0;
    for (int i = 0; i < N; i++) {
        double x = h_A[i];
        double expected = 0.0;
        for (int k = 12; k >= 0; k--) {
            double coefficient = 1.0;
            for (int j = 2; j <= k; j++) coefficient /= j;
            expected = expected * x + coefficient;
        }
        if (fabs(h_C[i] - expected) > 1e-3) errors++;
    }
    printf("N=%d, block=%d, grid=%d, errors=%d\n", N, block_size, grid_size, errors);

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);

    return errors ? 1 : 0;
}
