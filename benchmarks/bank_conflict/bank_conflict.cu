// This file is a benchmark to observe shared memory bank conflicts,
// which is observed through varying degrees of conflict from 1 to 32.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// launch config, set by gasp --run (-DGASP_BLOCK_SIZE=... -DGASP_GRID_SIZE=...)
#ifndef GASP_BLOCK_SIZE
#define GASP_BLOCK_SIZE 256
#endif
#ifndef GASP_GRID_SIZE
#define GASP_GRID_SIZE 4096
#endif

constexpr int SHARED_WORDS = 1024;
constexpr int REPEATS = 64; // feel free to change this- 64 is arbitrary

// Bank conflict kernel accesses shared memory with a configurable stride
__global__ void bank_conflict(float *out, int stride) {
    // initialize words in shared memory with volatile so compiler doesn't move it anywhere else
    __shared__ volatile float words[SHARED_WORDS];

    int tid = threadIdx.x;
    int lane = tid % 32;

    // initializing words array with data, for any block size
    for (int i = tid; i < SHARED_WORDS; i += blockDim.x) {
        words[i] = (float)i;
    }

    __syncthreads(); // sync threads before accessing shared memory

    float value = 0.0f;
// prevent loop unrolling to keep repeated memory accesses consistent
#pragma unroll 1
    for (int i = 0; i < REPEATS; ++i) {
        // here, stride determines which shared memory bank each warp lane accesses
        value = words[lane * stride];
    }
    // store result in global memory so the CPU can read it after the kernel finishes
    // we need this so the compiler doesn't eliminate shared memory reads entirely, thinking we don't need it
    out[blockIdx.x * blockDim.x + tid] = value;
}

// helper method for error handling
static void check_cuda(cudaError_t error, const char *where) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(error));
        exit(1);
    }
}

int main(int argc, char **argv) {
    // read the requested memory stride from command line (default is 1)
    int stride = argc > 1 ? atoi(argv[1]) : 1;

    // restrict strides to powers of two for controlled bank conflict testing
    if (stride != 1 && stride != 2 && stride != 4 &&
        stride != 8 && stride != 16 && stride != 32) {
        fprintf(stderr, "stride must be one of 1, 2, 4, 8, 16, 32\n");
        return 1;
    }
    int block_size = GASP_BLOCK_SIZE;
    int grid_size = GASP_GRID_SIZE;

    // calculate total number of output elements across all blocks
    const size_t count = (size_t)grid_size * block_size;
    float *device_out;
    float *host_out = (float *)malloc(count * sizeof(float));

    // allocate host memory to receive and validate kernel results
    if (!host_out) {
        fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    check_cuda(cudaMalloc(&device_out, count * sizeof(float)), "cudaMalloc");
    bank_conflict<<<grid_size, block_size>>>(device_out, stride);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel execution");
    check_cuda(cudaMemcpy(host_out, device_out, count * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy result");

    // verify every thread accessed the expected shared memory value
    int errors = 0;
    for (size_t i = 0; i < count; ++i) {
        float expected = (float)((i % block_size % 32) * stride);
        if (host_out[i] != expected) {
            ++errors;
        }
    }
    
    printf("stride=%d, repeats=%d, errors=%d\n", stride, REPEATS, errors);

    check_cuda(cudaFree(device_out), "cudaFree");
    free(host_out);
    return errors ? 1 : 0;
}