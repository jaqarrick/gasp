// constant memory: c[0x3][...] operands, LDC with a register index
__constant__ float coeffs[64];

// fixed indices: folded into FFMA operands, c[0x3][0x...]
__global__ void const_poly(float* out, const float* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[t];
    float p = coeffs[0];
#pragma unroll
    for (int k = 1; k < 8; ++k)
        p = p * x + coeffs[k];
    out[t] = p;
}

// uniform runtime index (param): LDC or ULDC with a register address
__global__ void const_uniform(float* out, const float* in, int k)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = in[t] * coeffs[k & 63];
}

// per-thread index: LDC R, c[0x3][R], serialized when lanes differ
__global__ void const_divergent(float* out, const float* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = in[t] * coeffs[threadIdx.x & 63];
}
