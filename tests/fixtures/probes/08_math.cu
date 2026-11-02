// MUFU (SIN, COS, EX2, LG2, RSQ, RCP, SQRT); accurate versions add range reduction and CALL slow paths

// fast intrinsics: mostly one MUFU each, plus FMUL scaling
__global__ void intrinsics(float* out, const float* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[t];
    out[6 * t + 0] = __sinf(x);
    out[6 * t + 1] = __cosf(x);
    out[6 * t + 2] = __expf(x);
    out[6 * t + 3] = __logf(x);
    out[6 * t + 4] = rsqrtf(x);
    out[6 * t + 5] = __fdividef(1.0f, x);
}

// IEEE versions: FCHK + CALL for division, FSETP denormal checks for sqrt
__global__ void accurate(float* out, const float* a, const float* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float x = a[t], y = b[t];
    out[5 * t + 0] = sinf(x);
    out[5 * t + 1] = expf(x);
    out[5 * t + 2] = logf(x);
    out[5 * t + 3] = sqrtf(x);
    out[5 * t + 4] = x / y;
}

// FP64: MUFU.RCP64H / RSQ64H + DFMA Newton steps, CALL slow paths
__global__ void double_math(double* out, const double* a, const double* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    double x = a[t], y = b[t];
    out[3 * t + 0] = sqrt(x);
    out[3 * t + 1] = x / y;
    out[3 * t + 2] = exp(x);
}

// polynomial: FFMA chain, each FFMA depends on the previous one (ILP = 1)
__global__ void horner(float* out, const float* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[t];
    float p = 0.1f;
    p = p * x + 0.2f;
    p = p * x + 0.3f;
    p = p * x + 0.4f;
    p = p * x + 0.5f;
    p = p * x + 0.6f;
    p = p * x + 0.7f;
    out[t] = p;
}
