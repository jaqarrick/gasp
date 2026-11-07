// FFMA, grid-stride loop (back edge, loop-carried index)

__global__ void saxpy(float* y, const float* x, float a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

// trip count unknown: ptxas unrolls by 4 plus a remainder loop
__global__ void saxpy_grid_stride(float* y, const float* x, float a, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        y[i] = a * x[i] + y[i];
}

// fixed trip count: fully unrolled, no back edge
__global__ void saxpy_fixed(float* y, const float* x, float a)
{
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
#pragma unroll
    for (int k = 0; k < 8; ++k)
        y[base + k] = a * x[base + k] + y[base + k];
}
