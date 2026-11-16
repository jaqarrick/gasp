#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#define TILE_WIDTH 16

__global__ void mm_tiled(float *a, float *b, float *c, int M, int N, int K) {

    __shared__ float a_tile [TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_tile [TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        int a_col = t * TILE_WIDTH + threadIdx.x;
        int b_row = t * TILE_WIDTH + threadIdx.y;

        // load one element of a and one of b into shared memory
        // check bounds
        a_tile[threadIdx.y][threadIdx.x] =  (row < M && a_col < K) ? a[row * K + a_col] : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? b[b_row * N + col] : 0.0f;

        // wait till all threads in block have loaded elements
        __syncthreads();

        // perform multiplication from tile
        for (int k = 0; k < TILE_WIDTH; k++){
            sum += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }

        __syncthreads();

    }

    if (row < M && col < N){
        c[row * N + col] = sum;
    }
}
static void check_cuda(cudaError_t err, const char *context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char **argv) {
    // A (M x K) x B (K x N) = C (M x N)
    // M is how many rows A
    int M = 1024, N = 1024, K = 1024;
    int bx = 16, by = 16;

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) bx = atoi(argv[4]);
    if (argc > 5) by = atoi(argv[5]);

    if (bx * by > 1024) {
        fprintf(stderr, "block size %dx%d = %d exceeds 1024 thread limit\n", bx, by, bx * by);
        return 1;
    }

    size_t elems_a = (size_t)M * K;
    size_t elems_b = (size_t)K * N;
    size_t elems_c = (size_t)M * N;
    size_t bytes_a = elems_a * sizeof(float);
    size_t bytes_b = elems_b * sizeof(float);
    size_t bytes_c = elems_c * sizeof(float);

    printf("A=%dx%d, B=%dx%d, C=%dx%d, block=%dx%d (%d threads)\n",
           M, K, K, N, M, N, bx, by, bx * by);

    float *h_a = (float *)malloc(bytes_a);
    float *h_b = (float *)malloc(bytes_b);
    float *h_c = (float *)malloc(bytes_c);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < elems_a; i++) h_a[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < elems_b; i++) h_b[i] = (float)rand() / RAND_MAX;

    float *d_a, *d_b, *d_c;
    check_cuda(cudaMalloc(&d_a, bytes_a), "malloc d_a");
    check_cuda(cudaMalloc(&d_b, bytes_b), "malloc d_b");
    check_cuda(cudaMalloc(&d_c, bytes_c), "malloc d_c");

    check_cuda(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice), "memcpy d_a");
    check_cuda(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice), "memcpy d_b");

    dim3 threads(bx, by);
    dim3 blocks((N + bx - 1) / bx, (M + by - 1) / by);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    mm_tiled<<<blocks, threads>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(t1);

    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "synchronize");

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    double gflops = (2.0 * M * N * (double)K) / (ms * 1e6);
    printf("kernel: %.3f ms  (%.1f GFLOP/s)\n", ms, gflops);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    check_cuda(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost), "memcpy h_c");

    int checks[][2] = {
        {0, 0}, {0, N-1}, {M-1, 0}, {M-1, N-1}, {M/2, N/3}
    };
    int errors = 0;
    for (int t = 0; t < 5; t++) {
        int r = checks[t][0], c = checks[t][1];
        float ref = 0.0f;
        for (int k = 0; k < K; k++)
            ref += h_a[(size_t)r * K + k] * h_b[(size_t)k * N + c];
        float got = h_c[(size_t)r * N + c];
        if (fabsf(got - ref) > 1e-3f * fabsf(ref)) {
            fprintf(stderr, "  MISMATCH [%d,%d]: got=%.6f ref=%.6f\n", r, c, got, ref);
            errors++;
        }
    }

    printf("grid=%dx%d, spot-checks=%d, errors=%d\n", blocks.x, blocks.y, 5, errors);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return errors != 0;
}