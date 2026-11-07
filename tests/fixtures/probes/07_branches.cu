// control flow: @P BRA, BSSY / BSYNC reconvergence, SEL / FSEL, @P EXIT

// short if/else: if-converted to FSEL, no branch
__global__ void select_value(float* out, const float* in, float threshold)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[t];
    out[t] = x > threshold ? x : 0.0f;
}

// divergent paths with loops: BSSY B0 / BSYNC B0 around the if/else
__global__ void divergent(float* out, const float* in, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[t];
    if (threadIdx.x & 1) {
        for (int k = 0; k < n; ++k)
            x = x * 1.01f + 0.5f;
    } else {
        for (int k = 0; k < 2 * n; ++k)
            x = x * 0.99f - 0.25f;
    }
    out[t] = x;
}

// data-dependent early return: @P EXIT
__global__ void early_exit(float* out, const float* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[t];
    if (x < 0.0f)
        return;
    out[t] = sqrtf(x);
}

// dense switch: jump table (BRX) or a compare chain
__global__ void switch_case(float* out, const int* sel, const float* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[t];
    switch (sel[t]) {
    case 0: x = x + 1.0f; break;
    case 1: x = x * 3.0f; break;
    case 2: x = x - 7.0f; break;
    case 3: x = x * x; break;
    case 4: x = 1.0f / x; break;
    case 5: x = -x; break;
    default: x = 0.0f; break;
    }
    out[t] = x;
}

// nested loops with a break: multiple back edges and exits
__global__ void nested_loops(int* out, const int* in, int rows, int cols)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int v = in[(r * cols + c) * 1024 + t];
            if (v < 0)
                break;
            acc += v;
        }
    }
    out[t] = acc;
}
