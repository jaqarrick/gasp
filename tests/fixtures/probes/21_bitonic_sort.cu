// bitonic sort of SORT_N floats per block in shared memory; launch: block (SORT_N / 2)
#define SORT_N 1024

// nested k / j loops with BAR.SYNC inside, FSETP + SEL compare-exchange
__global__ void bitonic_sort(float* data)
{
    __shared__ float s[SORT_N];
    int t = threadIdx.x;
    float* base = data + blockIdx.x * SORT_N;
    s[t] = base[t];
    s[t + SORT_N / 2] = base[t + SORT_N / 2];
    __syncthreads();

    for (int k = 2; k <= SORT_N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int i = 2 * t - (t & (j - 1));  // lower index of this thread's pair
            bool ascending = (i & k) == 0;
            float a = s[i], b = s[i + j];
            if ((a > b) == ascending) {
                s[i] = b;
                s[i + j] = a;
            }
            __syncthreads();
        }
    }
    base[t] = s[t];
    base[t + SORT_N / 2] = s[t + SORT_N / 2];
}

// warp-level bitonic sort of 32 values with shuffles, no shared memory
__global__ void bitonic_warp(float* data)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    float v = data[t];

    for (int k = 2; k <= 32; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, v, j);
            bool ascending = (lane & k) == 0;
            bool lower = (lane & j) == 0;
            v = (lower == ascending) ? fminf(v, other) : fmaxf(v, other);
        }
    }
    data[t] = v;
}
