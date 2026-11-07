// P2: carry-out in slot 1 (written), carry-in last predicate of the .X op (read)
//   IADD3 R2, P0, PT, R4, R6, RZ   ->   IADD3.X R3, PT, PT, R5, R7, RZ, P0, !PT
//   LEA R4, P0, R17, R2, 0x2       ->   LEA.HI.X R5, R17, R3, R19, 0x2, P0

// IADD3 / IADD3.X
__global__ void add64(long long* out, const long long* a, const long long* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = a[t] + b[t];
}

// LEA / LEA.HI.X
__global__ void index64(float* out, const float* in, long long base, long long stride)
{
    long long i = base + (long long)threadIdx.x * stride;
    out[i] = in[i] * 2.0f;
}

// carry chain inside a loop
__global__ void pointer_walk(float* out, const float* in, size_t stride, int n)
{
    const float* p = in + blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
        acc += *p;
        p += stride;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// does IADD3.X also write a carry-out?
__global__ void add128(unsigned* out, const unsigned* a, const unsigned* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned* x = a + 4 * t;
    const unsigned* y = b + 4 * t;
    unsigned r0, r1, r2, r3;

    asm volatile(
        "add.cc.u32  %0, %4, %8;\n\t"
        "addc.cc.u32 %1, %5, %9;\n\t"
        "addc.cc.u32 %2, %6, %10;\n\t"
        "addc.u32    %3, %7, %11;"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]),
          "r"(y[0]), "r"(y[1]), "r"(y[2]), "r"(y[3]));

    out[4 * t + 0] = r0;
    out[4 * t + 1] = r1;
    out[4 * t + 2] = r2;
    out[4 * t + 3] = r3;
}

// IMAD.WIDE.U32 / IMAD.X carry forms
__global__ void mul64(unsigned long long* lo, unsigned long long* hi,
                      const unsigned long long* a, const unsigned long long* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    lo[t] = a[t] * b[t];
    hi[t] = __umul64hi(a[t], b[t]);
}

// UIADD3 UP0 carry-out (params are uniform)
__global__ void uniform_add64(long long* out, long long a, long long b)
{
    out[blockIdx.x * blockDim.x + threadIdx.x] = a + b;
}
