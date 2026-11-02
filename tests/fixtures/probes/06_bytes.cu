// narrow types: LDG.E.U8 / .S16, byte selectors (R2.B1), I2F.U8, PRMT packing

// LDG.E.U8, I2F.U32 or I2F.U8
__global__ void bytes_to_float(float* out, const unsigned char* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = in[t] * (1.0f / 255.0f);
}

// one 32-bit load, four bytes taken by selector (R.B1, R.B2, R.B3)
__global__ void unpack_uchar4(float4* out, const uchar4* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    uchar4 v = in[t];
    out[t] = make_float4(v.x, v.y, v.z, v.w);
}

// F2I.U32 + PRMT / LOP3 to pack four bytes
__global__ void pack_uchar4(uchar4* out, const float4* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float4 v = in[t];
    out[t] = make_uchar4(__float2uint_rn(v.x), __float2uint_rn(v.y), __float2uint_rn(v.z),
                         __float2uint_rn(v.w));
}

// LDG.E.S16 sign extension, STG.E.U16
__global__ void shorts(short* out, const short* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = (short)(in[t] * 3 - 1);
}
