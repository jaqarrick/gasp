// stencils: neighbours from global, then from a shared tile with a halo
#define BX 32
#define BY 8

// 1D 3-point, boundary handled by a guard
__global__ void stencil_1d(float* out, const float* in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < n - 1)
        out[i] = 0.25f * in[i - 1] + 0.5f * in[i] + 0.25f * in[i + 1];
}

// 2D 5-point Jacobi step, all loads from global
__global__ void jacobi_global(float* out, const float* in, int nx, int ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1)
        return;
    int i = y * nx + x;
    out[i] = 0.25f * (in[i - 1] + in[i + 1] + in[i - nx] + in[i + nx]);
}

// shared tile + 1-cell halo: edge threads do extra loads (divergent)
// launch: block (BX, BY)
__global__ void jacobi_shared(float* out, const float* in, int nx, int ny)
{
    __shared__ float s[BY + 2][BX + 2];
    int tx = threadIdx.x + 1, ty = threadIdx.y + 1;
    int x = blockIdx.x * BX + threadIdx.x;
    int y = blockIdx.y * BY + threadIdx.y;
    int i = y * nx + x;
    bool inside = x < nx && y < ny;

    s[ty][tx] = inside ? in[i] : 0.0f;
    if (threadIdx.x == 0)
        s[ty][0] = x > 0 && inside ? in[i - 1] : 0.0f;
    if (threadIdx.x == BX - 1)
        s[ty][BX + 1] = x + 1 < nx && inside ? in[i + 1] : 0.0f;
    if (threadIdx.y == 0)
        s[0][tx] = y > 0 && inside ? in[i - nx] : 0.0f;
    if (threadIdx.y == BY - 1)
        s[BY + 1][tx] = y + 1 < ny && inside ? in[i + nx] : 0.0f;
    __syncthreads();

    if (x >= 1 && y >= 1 && x < nx - 1 && y < ny - 1)
        out[i] = 0.25f * (s[ty][tx - 1] + s[ty][tx + 1] + s[ty - 1][tx] + s[ty + 1][tx]);
}
