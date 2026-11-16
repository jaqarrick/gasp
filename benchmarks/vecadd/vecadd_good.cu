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

__global__ void vecadd_good(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[]) {
    int N = 1000000;
    if (argc > 1) N = atoi(argv[1]);

    size_t bytes = N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int block_size = GASP_BLOCK_SIZE;
    int grid_size = GASP_GRID_SIZE > 0 ? GASP_GRID_SIZE
                                       : (N + block_size - 1) / block_size;

    vecadd_good<<<grid_size, block_size>>>(d_A, d_B, d_C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) errors++;
    }
    printf("N=%d, block=%d, grid=%d, errors=%d\n", N, block_size, grid_size, errors);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return errors ? 1 : 0;
}
