// global -> shared async copy: LDGSTS, LDGDEPBAR, DEPBAR (sm_80+); launch: block (TILE)
#include <cuda_pipeline.h>
#define TILE 256

// double-buffered: copy tile k+1 while computing on tile k
__global__ void async_sum(float* out, const float* in, int tiles)
{
    __shared__ float buf[2][TILE];
    int t = threadIdx.x;
    const float* src = in + (size_t)blockIdx.x * tiles * TILE;
    float acc = 0.0f;

    __pipeline_memcpy_async(&buf[0][t], &src[t], sizeof(float));
    __pipeline_commit();

    for (int k = 0; k < tiles; ++k) {
        if (k + 1 < tiles)
            __pipeline_memcpy_async(&buf[(k + 1) & 1][t], &src[(k + 1) * TILE + t], sizeof(float));
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        acc += buf[k & 1][TILE - 1 - t];
        __syncthreads();
    }
    out[blockIdx.x * TILE + t] = acc;
}

// 16-byte copies: LDGSTS.E.128
__global__ void async_copy_float4(float4* out, const float4* in)
{
    __shared__ float4 buf[TILE];
    int t = threadIdx.x;
    __pipeline_memcpy_async(&buf[t], &in[blockIdx.x * TILE + t], sizeof(float4));
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    out[blockIdx.x * TILE + t] = buf[TILE - 1 - t];
}
