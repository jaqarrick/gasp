// warp-wide ops: VOTE.ANY / VOTE.ALL, ballot, MATCH.ANY (32 and .U64), REDUX
#define FULL 0xffffffffu

__global__ void votes(unsigned* out, const float* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    bool p = in[t] > 0.0f;
    unsigned ballot = __ballot_sync(FULL, p);  // VOTE.ANY R, PT, P
    out[4 * t + 0] = ballot;
    out[4 * t + 1] = __any_sync(FULL, p);      // VOTE.ANY with predicate out
    out[4 * t + 2] = __all_sync(FULL, p);      // VOTE.ALL
    out[4 * t + 3] = __popc(ballot);
}

// MATCH.ANY and MATCH.ANY.U64 (64-bit source is a register pair)
__global__ void match(unsigned* out32, unsigned* out64, const int* a, const long long* b)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    out32[t] = __match_any_sync(FULL, a[t]);
    out64[t] = __match_any_sync(FULL, b[t]);
}

// REDUX.SUM / REDUX.MAX (warp reduce in one instruction)
__global__ void redux(unsigned* sum, unsigned* mx, const unsigned* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    sum[t] = __reduce_add_sync(FULL, in[t]);
    mx[t] = __reduce_max_sync(FULL, in[t]);
}

// warp-aggregated atomic: one atomic per warp, each lane gets its own slot
__global__ void compact(float* out, int* count, const float* in)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[t];
    bool keep = x > 0.0f;
    unsigned ballot = __ballot_sync(FULL, keep);
    int lane = threadIdx.x & 31;
    int leader = __ffs(ballot) - 1;
    int base = 0;
    if (lane == leader)
        base = atomicAdd(count, __popc(ballot));
    base = __shfl_sync(FULL, base, leader);
    if (keep)
        out[base + __popc(ballot & ((1u << lane) - 1))] = x;
}
