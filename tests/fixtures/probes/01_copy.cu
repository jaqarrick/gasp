// LDG / STG, IMAD.WIDE address math, ISETP + @P EXIT bounds guard

__global__ void copy(float* out, const float* in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = in[i];
}

// scalar param read straight from c[0x0][...]
__global__ void scale(float* out, const float* in, float s, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = in[i] * s;
}

// __restrict__ const input -> LDG.E.CONSTANT (read-only path)
__global__ void copy_restrict(float* __restrict__ out, const float* __restrict__ in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = in[i];
}
