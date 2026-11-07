// inclusive prefix sum: Hillis-Steele in shared, then warp-shuffle version
#define BLOCK 256

// double-buffered shared memory, log2(BLOCK) steps with BAR.SYNC each
__global__ void scan_hillis_steele(float* out, const float* in)
{
    __shared__ float buf[2][BLOCK];
    int t = threadIdx.x;
    int base = blockIdx.x * BLOCK;
    int src = 0;
    buf[src][t] = in[base + t];
    __syncthreads();

    for (int offset = 1; offset < BLOCK; offset <<= 1) {
        int dst = 1 - src;
        float v = buf[src][t];
        if (t >= offset)
            v += buf[src][t - offset];
        buf[dst][t] = v;
        src = dst;
        __syncthreads();
    }
    out[base + t] = buf[src][t];
}

// SHFL.UP within each warp, warp totals scanned by warp 0, then added back
__global__ void scan_shfl(float* out, const float* in)
{
    __shared__ float warp_totals[BLOCK / 32];
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    float v = in[blockIdx.x * BLOCK + t];

    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, v, offset);
        if (lane >= offset)
            v += n;
    }
    if (lane == 31)
        warp_totals[warp] = v;
    __syncthreads();

    if (warp == 0) {
        float w = lane < BLOCK / 32 ? warp_totals[lane] : 0.0f;
        for (int offset = 1; offset < BLOCK / 32; offset <<= 1) {
            float n = __shfl_up_sync(0xffffffff, w, offset);
            if (lane >= offset)
                w += n;
        }
        if (lane < BLOCK / 32)
            warp_totals[lane] = w;
    }
    __syncthreads();

    if (warp > 0)
        v += warp_totals[warp - 1];
    out[blockIdx.x * BLOCK + t] = v;
}
