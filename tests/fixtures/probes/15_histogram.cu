// 256-bin histogram: per-block private copy in shared (ATOMS), merged with RED
#define BINS 256

// global atomics only: heavy contention on hot bins
__global__ void histogram_global(unsigned* bins, const unsigned char* in, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&bins[in[i]], 1u);
}

// privatized: ATOMS.ADD in the loop, one RED per bin per block at the end
__global__ void histogram_shared(unsigned* bins, const unsigned char* in, int n)
{
    __shared__ unsigned s[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        s[b] = 0;
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&s[in[i]], 1u);
    __syncthreads();

    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        if (s[b])
            atomicAdd(&bins[b], s[b]);
}

// four bytes per load: LDG.E then byte selectors / PRMT feeding ATOMS
__global__ void histogram_uchar4(unsigned* bins, const uchar4* in, int n4)
{
    __shared__ unsigned s[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        s[b] = 0;
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
        uchar4 v = in[i];
        atomicAdd(&s[v.x], 1u);
        atomicAdd(&s[v.y], 1u);
        atomicAdd(&s[v.z], 1u);
        atomicAdd(&s[v.w], 1u);
    }
    __syncthreads();

    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        atomicAdd(&bins[b], s[b]);
}
