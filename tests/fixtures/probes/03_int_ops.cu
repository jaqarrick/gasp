// integer ALU: LOP3, SHF, POPC, FLO, BREV, IMNMX / VIMNMX, IMAD.HI

// runtime divisor: I2F / MUFU.RCP / F2I / IMAD.HI correction sequence
__global__ void int_div(int* q, int* r, const int* a, const int* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    q[t] = a[t] / b[t];
    r[t] = a[t] % b[t];
}

// constant divisor: IMAD.HI with a magic number + SHF
__global__ void int_div_const(int* q, unsigned* r, const int* a, const unsigned* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    q[t] = a[t] / 7;
    r[t] = b[t] % 10;
}

__global__ void bit_ops(unsigned* out, const unsigned* a, const unsigned* b, const unsigned* c)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned x = a[t], y = b[t], z = c[t];

    out[8 * t + 0] = (x & y) | (~x & z);         // LOP3.LUT
    out[8 * t + 1] = (x << 3) | (y >> 29);       // SHF / LEA
    out[8 * t + 2] = __popc(x);                  // POPC
    out[8 * t + 3] = __clz(y);                   // FLO
    out[8 * t + 4] = __brev(z);                  // BREV
    out[8 * t + 5] = __ffs(x);                   // BREV + FLO
    out[8 * t + 6] = __funnelshift_l(x, y, z);   // SHF.L.W
    out[8 * t + 7] = __byte_perm(x, y, 0x5140);  // PRMT
}

__global__ void min_max(int* out, const int* a, const int* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[3 * t + 0] = min(a[t], b[t]);
    out[3 * t + 1] = max(a[t], b[t]);
    out[3 * t + 2] = abs(a[t]);  // IABS
}

// 64-bit integer ops on register pairs
__global__ void int64_ops(long long* out, const long long* a, const long long* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    long long x = a[t], y = b[t];
    out[3 * t + 0] = x * y;
    out[3 * t + 1] = x >> 7;
    out[3 * t + 2] = x ^ y;
}
