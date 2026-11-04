// Measures effective GPU-timeline cycles per minimal kernel launch
// Does not measure CPU-side launch-call times
#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>


//minimum empty kernel
__global__ void launch_empty_kernel() {
}

// grid and block as CLI arguments
int main(int argc, char** argv){
    if(argc != 3){
        fprintf(stderr, "Usage: %s GRID BLOCK\n", argv[0]);
        return 1;
    }

    int grid = atoi(argv[1]);
    int block = atoi(argv[2]);

    // reject non-positive values
    if(grid <= 0 || block <= 0){
        fprintf(stderr, "Usage: %s GRID BLOCK\n", argv[0]);
        return 1;
    }

    //validate grid + block

    launch_empty_kernel<<<grid, block>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
