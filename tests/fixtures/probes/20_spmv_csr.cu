// sparse matrix-vector product, CSR: data-dependent trip counts, indirect loads (x[col[j]])

// one thread per row: uncoalesced, divergent loop lengths
__global__ void spmv_scalar(float* y, const int* row_ptr, const int* col, const float* val,
                            const float* x, int rows)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows)
        return;
    float acc = 0.0f;
    for (int j = row_ptr[r]; j < row_ptr[r + 1]; ++j)
        acc += val[j] * x[col[j]];
    y[r] = acc;
}

// one warp per row: coalesced over the row, SHFL reduction at the end
__global__ void spmv_vector(float* y, const int* row_ptr, const int* col, const float* val,
                            const float* x, int rows)
{
    int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (r >= rows)
        return;
    float acc = 0.0f;
    for (int j = row_ptr[r] + lane; j < row_ptr[r + 1]; j += 32)
        acc += val[j] * __ldg(&x[col[j]]);
    for (int s = 16; s > 0; s >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, s);
    if (lane == 0)
        y[r] = acc;
}

// ELL format: fixed width per row, column-major, so loads coalesce and loops are uniform
__global__ void spmv_ell(float* y, const int* col, const float* val, const float* x, int rows,
                         int width)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows)
        return;
    float acc = 0.0f;
    for (int k = 0; k < width; ++k) {
        int c = col[k * rows + r];
        if (c >= 0)
            acc += val[k * rows + r] * x[c];
    }
    y[r] = acc;
}
