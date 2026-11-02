// FP16 / BF16: HADD2 / HMUL2 / HFMA2, half selectors (R2.H0_H0, R2.H1_H1), F2FP packing
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// HFMA2
__global__ void half2_fma(__half2* out, const __half2* a, const __half2* b, const __half2* c)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = __hfma2(a[t], b[t], c[t]);
}

// lane selectors on the operands
__global__ void half2_lanes(__half2* swapped, __half2* broadcast, const __half2* a)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    __half2 x = a[t];
    swapped[t] = __hadd2(x, __lowhigh2highlow(x));  // R.H1_H0
    broadcast[t] = __hmul2(x, __high2half2(x));     // R.H1_H1
}

// HADD2.F32 (half -> float), F2FP.F16.F32.PACK_AB (float -> half)
__global__ void half_convert(float* f, __half2* h, const __half* hin, const float2* fin)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    f[t] = __half2float(hin[t]);
    h[t] = __float22half2_rn(fin[t]);
}

// HFMA2.BF16
__global__ void bf16_fma(__nv_bfloat162* out, const __nv_bfloat162* a, const __nv_bfloat162* b,
                         const __nv_bfloat162* c)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = __hfma2(a[t], b[t], c[t]);
}

// fp16 dot product accumulated in fp32
__global__ void half_dot(float* out, const __half2* a, const __half2* b, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
        float2 p = __half22float2(__hmul2(a[k * 1024 + t], b[k * 1024 + t]));
        acc += p.x + p.y;
    }
    out[t] = acc;
}
