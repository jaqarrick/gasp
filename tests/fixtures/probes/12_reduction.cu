// sum reduction: shared tree, warp shuffle, atomic combine across blocks
#define BLOCK 256

// classic shared-memory tree: loop with BAR.SYNC in the body
__global__ void reduce_shared(float* out, const float* in)
{
    __shared__ float s[BLOCK];
    int t = threadIdx.x;
    s[t] = in[blockIdx.x * BLOCK + t];
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (t < stride)
            s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0)
        out[blockIdx.x] = s[0];
}

// warp shuffle within each warp, then shared for the 8 warp sums
__global__ void reduce_shfl(float* out, const float* in, int n)
{
    __shared__ float warp_sums[BLOCK / 32];
    int i = blockIdx.x * BLOCK + threadIdx.x;
    float v = i < n ? in[i] : 0.0f;

    for (int s = 16; s > 0; s >>= 1)
        v += __shfl_down_sync(0xffffffff, v, s);

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0)
        warp_sums[warp] = v;
    __syncthreads();

    if (warp == 0) {
        v = lane < BLOCK / 32 ? warp_sums[lane] : 0.0f;
        for (int s = 4; s > 0; s >>= 1)
            v += __shfl_down_sync(0xffffffff, v, s);
        if (lane == 0)
            atomicAdd(out, v);  // RED.E.ADD.F32
    }
}

// grid-stride accumulate first, so each thread does many loads
__global__ void reduce_grid_stride(float* out, const float* in, int n)
{
    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        v += in[i];
    for (int s = 16; s > 0; s >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, s);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(out, v);
}
