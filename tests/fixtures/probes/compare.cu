// P3: FSET / HSET2 write one register, *SETP / PLOP3 / PSETP write two predicates
// Confirmed when every leading output is read later (store, SEL, @P guard)

// FSET.BF and FSET
__global__ void fset(float* as_float, unsigned* as_mask, const float* a, const float* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float r;
    unsigned m;

    asm volatile("set.gt.f32.f32 %0, %1, %2;" : "=f"(r) : "f"(a[t]), "f"(b[t]));
    asm volatile("set.gt.u32.f32 %0, %1, %2;" : "=r"(m) : "f"(a[t]), "f"(b[t]));

    as_float[t] = r;
    as_mask[t] = m;
}

// HSET2
__global__ void hset2(unsigned* out, const unsigned* a, const unsigned* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned r;

    asm volatile("set.gt.f16x2.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[t]), "r"(b[t]));

    out[t] = r;
}

// FSETP / DSETP / ISETP P, P
// p = t AND c, q = !t AND c, so ptxas can't drop q
__global__ void setp_pairs(int* out, const float* f, const double* d, const int* i,
                           const int* c)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int r0, r1, r2, r3, r4, r5;

    asm volatile(
        "{\n\t"
        ".reg .pred c, p, q;\n\t"
        "setp.ne.s32 c, %6, 0;\n\t"
        "setp.gt.and.f32 p|q, %7, 0f3F000000, c;\n\t"
        "selp.s32 %0, 1, 2, p;\n\t"
        "selp.s32 %1, 3, 4, q;\n\t"
        "setp.gt.and.f64 p|q, %8, 0d3FE0000000000000, c;\n\t"
        "selp.s32 %2, 5, 6, p;\n\t"
        "selp.s32 %3, 7, 8, q;\n\t"
        "setp.lt.and.s32 p|q, %9, 16, c;\n\t"
        "selp.s32 %4, 9, 10, p;\n\t"
        "selp.s32 %5, 11, 12, q;\n\t"
        "}"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5)
        : "r"(c[t]), "f"(f[t]), "d"(d[t]), "r"(i[t]));

    out[6 * t + 0] = r0;
    out[6 * t + 1] = r1;
    out[6 * t + 2] = r2;
    out[6 * t + 3] = r3;
    out[6 * t + 4] = r4;
    out[6 * t + 5] = r5;
}

// HSETP2 P, P (low half, high half)
__global__ void hsetp2(int* out, const unsigned* a, const unsigned* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int lo, hi;

    asm volatile(
        "{\n\t"
        ".reg .pred p, q;\n\t"
        "setp.gt.f16x2 p|q, %2, %3;\n\t"
        "selp.s32 %0, 1, 2, p;\n\t"
        "selp.s32 %1, 3, 4, q;\n\t"
        "}"
        : "=r"(lo), "=r"(hi)
        : "r"(a[t]), "r"(b[t]));

    out[2 * t + 0] = lo;
    out[2 * t + 1] = hi;
}

// PLOP3 / PSETP (ptxas may fold into ISETP instead)
__global__ void pred_logic(int* out, const int* a, const int* b, const int* c)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int x_out, y_out;

    asm volatile(
        "{\n\t"
        ".reg .pred p, q, r, x, y;\n\t"
        "setp.gt.s32 p, %2, 0;\n\t"
        "setp.gt.s32 q, %3, 0;\n\t"
        "setp.gt.s32 r, %4, 0;\n\t"
        "and.pred x, p, q;\n\t"
        "or.pred x, x, r;\n\t"
        "xor.pred y, p, r;\n\t"
        "and.pred y, y, q;\n\t"
        "selp.s32 %0, 1, 0, x;\n\t"
        "selp.s32 %1, 1, 0, y;\n\t"
        "}"
        : "=r"(x_out), "=r"(y_out)
        : "r"(a[t]), "r"(b[t]), "r"(c[t]));

    out[2 * t + 0] = x_out;
    out[2 * t + 1] = y_out;
}

// UISETP / UPLOP3 (conditions on params only)
__global__ void uniform_compare(int* out, int n, int m, int k)
{
    int acc = 0;
    for (int j = 0; j < n; ++j) {
        if (j < m && (j ^ k) > 2)
            acc += j;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
