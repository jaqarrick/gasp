// wide loads / stores: .64 and .128 data registers

// LDG.E.64 / STG.E.64
__global__ void copy_float2(float2* out, const float2* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float2 v = in[t];
    out[t] = make_float2(v.x * 2.0f, v.y * 2.0f);
}

// LDG.E.128 / STG.E.128, four FMULs reading R, R+1, R+2, R+3
__global__ void copy_float4(float4* out, const float4* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float4 v = in[t];
    out[t] = make_float4(v.x * 2.0f, v.y * 2.0f, v.z * 2.0f, v.w * 2.0f);
}

// LDG.E.128 holding two doubles: two DADDs on pairs inside one quad
__global__ void add_double2(double2* out, const double2* a, const double2* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    double2 x = a[t], y = b[t];
    out[t] = make_double2(x.x + y.x, x.y + y.y);
}

// STS.128 / LDS.128
__global__ void shared_float4(float4* out, const float4* in)
{
    __shared__ float4 s[256];
    int t = threadIdx.x;
    s[t] = in[blockIdx.x * 256 + t];
    __syncthreads();
    out[blockIdx.x * 256 + t] = s[255 - t];
}
