// matrix transpose: uncoalesced naive, shared tile with bank conflicts, padded tile
// launch: block (TILE, ROWS), grid (cols / TILE, rows / TILE)
#define TILE 32
#define ROWS 8

// coalesced reads, strided (uncoalesced) writes
__global__ void transpose_naive(float* out, const float* in, int width, int height)
{
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS)
        out[x * height + (y + j)] = in[(y + j) * width + x];
}

// coalesced both ways, but column reads of the tile hit one bank (32-way conflict)
__global__ void transpose_shared(float* out, const float* in, int width, int height)
{
    __shared__ float tile[TILE][TILE];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS)
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS)
        out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
}

// one padding column shifts each row by a bank: conflict-free
__global__ void transpose_padded(float* out, const float* in, int width, int height)
{
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS)
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS)
        out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
}
