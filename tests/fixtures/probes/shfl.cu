// P7: SHFL.<mode> Pout, Rout, Rsrc, lane, clamp
// slot 0 = in-bounds predicate (written, usually PT), slot 1 = value (written)

// SHFL.BFLY, each FADD reads the previous SHFL's slot 1
__global__ void shfl_reduce(float* out, const float* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float v = in[t];

    for (int s = 16; s > 0; s >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, s);

    out[t] = v;
}

__global__ void shfl_modes(float* out, const float* in, int lane)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float v = in[t];

    float a = __shfl_sync(0xffffffff, v, lane);    // SHFL.IDX
    float b = __shfl_up_sync(0xffffffff, v, 1);    // SHFL.UP
    float c = __shfl_down_sync(0xffffffff, v, 2);  // SHFL.DOWN

    out[t] = a + b * c;
}

// using the predicate output should put a real P in slot 0
__global__ void shfl_pred(int* values, int* in_bounds, const int* in, int delta)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int v, ok;

    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "shfl.sync.down.b32 %0|p, %2, %3, 0x1f, 0xffffffff;\n\t"
        "selp.s32 %1, 1, 0, p;\n\t"
        "}"
        : "=r"(v), "=r"(ok)
        : "r"(in[t]), "r"(delta));

    values[t] = v;
    in_bounds[t] = ok;
}
