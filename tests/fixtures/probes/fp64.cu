// P4 / P5: FP64 ops and 64-bit conversions use register pairs with no .64,
// MUFU.RCP64H / RSQ64H are 32-bit, CS2R is a pair unless .32

// DFMA, DADD
__global__ void fp64_arith(double* fma, double* sum, const double* a, const double* b,
                           const double* c)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    fma[t] = a[t] * b[t] + c[t];
    sum[t] = a[t] + b[t];
}

// MUFU.RCP64H / MUFU.RSQ64H + DFMA refinement
__global__ void fp64_div_rsqrt(double* q, double* rs, const double* a, const double* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    q[t] = a[t] / b[t];
    rs[t] = rsqrt(a[t]);
}

// expected forms on the right
__global__ void conversions(double* d_from_f, float* f_from_d, int* i_from_d,
                            long long* l_from_d, unsigned long long* u_from_f,
                            double* d_from_i, double* d_from_l, float* f_from_l,
                            double* d_trunc, const double* d, const float* f,
                            const int* i, const long long* l)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    double x = d[t];
    float y = f[t];

    d_from_f[t] = (double)y;              // F2F.F64.F32
    f_from_d[t] = (float)x;               // F2F.F32.F64
    i_from_d[t] = (int)x;                 // F2I.F64.TRUNC
    l_from_d[t] = (long long)x;           // F2I.S64.F64.TRUNC
    u_from_f[t] = (unsigned long long)y;  // F2I.U64.TRUNC
    d_from_i[t] = (double)i[t];           // I2F.F64
    d_from_l[t] = (double)l[t];           // I2F.F64.S64
    f_from_l[t] = (float)l[t];            // I2F.S64
    d_trunc[t] = trunc(x);                // FRND.F64.TRUNC
}

// clock64 -> CS2R (pair), clock -> CS2R.32 or S2R
__global__ void clocks(long long* out64, unsigned* out32, const float* in, float* sink)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    long long start64 = clock64();
    unsigned start32 = (unsigned)clock();

    sink[t] = in[t] * 3.0f;

    out64[t] = clock64() - start64;
    out32[t] = (unsigned)clock() - start32;
}
