// shared memory: STS / LDS, BAR.SYNC, static vs dynamic allocation
#define BLOCK 256

__global__ void reverse_static(float* out, const float* in)
{
    __shared__ float s[BLOCK];
    int t = threadIdx.x;
    int base = blockIdx.x * BLOCK;
    s[t] = in[base + t];
    __syncthreads();
    out[base + t] = s[BLOCK - 1 - t];
}

// extern: size set at launch, base address from a constant / uniform register
__global__ void reverse_dynamic(float* out, const float* in)
{
    extern __shared__ float s[];
    int t = threadIdx.x;
    int n = blockDim.x;
    s[t] = in[blockIdx.x * n + t];
    __syncthreads();
    out[blockIdx.x * n + t] = s[n - 1 - t];
}

// stride-2 access: 2-way bank conflict
__global__ void strided(float* out, const float* in)
{
    __shared__ float s[2 * BLOCK];
    int t = threadIdx.x;
    s[t] = in[blockIdx.x * BLOCK + t];
    s[t + BLOCK] = 0.0f;
    __syncthreads();
    out[blockIdx.x * BLOCK + t] = s[2 * t];
}
