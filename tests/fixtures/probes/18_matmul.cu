// C = A * B, row-major, n a multiple of the tile size
#define TILE 16

// one output per thread, every operand from global
__global__ void matmul_naive(float* c, const float* a, const float* b, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int k = 0; k < n; ++k)
        acc += a[row * n + k] * b[k * n + col];
    c[row * n + col] = acc;
}

// shared tiles: LDS feeding an unrolled FFMA chain, two BAR.SYNC per tile
// launch: block (TILE, TILE)
__global__ void matmul_tiled(float* c, const float* a, const float* b, int n)
{
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    for (int t = 0; t < n / TILE; ++t) {
        as[ty][tx] = a[row * n + t * TILE + tx];
        bs[ty][tx] = b[(t * TILE + ty) * n + col];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += as[ty][k] * bs[k][tx];
        __syncthreads();
    }
    c[row * n + col] = acc;
}

// register blocking: each thread computes 4 outputs in a column, 4 independent FFMA chains
// launch: block (TILE, TILE / 4)
__global__ void matmul_reg4(float* c, const float* a, const float* b, int n)
{
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;
    int row0 = blockIdx.y * TILE + ty * 4;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int t = 0; t < n / TILE; ++t) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            as[ty * 4 + r][tx] = a[(row0 + r) * n + t * TILE + tx];
            bs[ty * 4 + r][tx] = b[(t * TILE + ty * 4 + r) * n + col];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float bv = bs[k][tx];
#pragma unroll
            for (int r = 0; r < 4; ++r)
                acc[r] += as[ty * 4 + r][k] * bv;
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; ++r)
        c[(row0 + r) * n + col] = acc[r];
}
