// local memory: runtime-indexed per-thread arrays -> STL / LDL off the stack pointer R1

// runtime index: the array can't live in registers
__global__ void local_array(float* out, const float* in, const int* idx)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float buf[32];
    for (int k = 0; k < 32; ++k)
        buf[k] = in[k * 1024 + t];
    out[t] = buf[idx[t] & 31];
}

// compile-time indices only: stays in registers, no LDL / STL
__global__ void register_array(float* out, const float* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float buf[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
        buf[k] = in[k * 1024 + t];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k)
        acc += buf[k] * buf[7 - k];
    out[t] = acc;
}
