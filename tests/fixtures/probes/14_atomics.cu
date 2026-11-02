// atomics: result unused -> RED, result used -> ATOMG, shared -> ATOMS

// RED.E.ADD (int, F32, F64); the F64 data is a register pair
__global__ void add_unused(int* ci, float* cf, double* cd, const float* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[t];
    atomicAdd(ci, (int)x);
    atomicAdd(cf, x);
    atomicAdd(cd, (double)x);
}

// ATOMG.E.ADD with a destination register
__global__ void add_used(int* out, int* counter, const int* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = atomicAdd(counter, in[t]);
}

// ATOMG.E.CAS.64: compare and swap values are both pairs
__global__ void cas64(unsigned long long* out, unsigned long long* p, const unsigned long long* cmp,
                      const unsigned long long* val)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = atomicCAS(p + (t & 15), cmp[t], val[t]);
}

// CAS loop: atomic max on doubles (no native instruction)
__global__ void max_double(double* m, const double* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    double x = in[t];
    unsigned long long* addr = (unsigned long long*)m;
    unsigned long long old = *addr, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= x)
            break;
        old = atomicCAS(addr, assumed, __double_as_longlong(x));
    } while (old != assumed);
}

// ATOMS.ADD / ATOMS.MAX / ATOMS.EXCH
__global__ void shared_atomics(int* out, const int* in)
{
    __shared__ int s[32];
    int t = threadIdx.x;
    if (t < 32)
        s[t] = 0;
    __syncthreads();
    int v = in[blockIdx.x * blockDim.x + t];
    atomicAdd(&s[v & 31], 1);
    atomicMax(&s[(v >> 5) & 31], v);
    __syncthreads();
    if (t < 32)
        out[blockIdx.x * 32 + t] = atomicExch(&s[t], 0);
}
