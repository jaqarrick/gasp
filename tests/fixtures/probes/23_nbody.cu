// all-pairs n-body forces: shared tiles of float4 positions, rsqrtf, long FFMA chains
// n a multiple of TILE; launch: block (TILE)
#define TILE 128

__global__ void nbody_forces(float3* accel, const float4* pos, int n, float softening)
{
    __shared__ float4 tile[TILE];
    int i = blockIdx.x * TILE + threadIdx.x;
    float4 p = pos[i];
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    for (int base = 0; base < n; base += TILE) {
        tile[threadIdx.x] = pos[base + threadIdx.x];
        __syncthreads();
#pragma unroll 4
        for (int j = 0; j < TILE; ++j) {
            float4 q = tile[j];  // LDS.128 broadcast: every lane reads the same address
            float dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
            float d2 = dx * dx + dy * dy + dz * dz + softening;
            float inv = rsqrtf(d2);  // MUFU.RSQ
            float s = q.w * inv * inv * inv;
            ax += dx * s;
            ay += dy * s;
            az += dz * s;
        }
        __syncthreads();
    }
    accel[i] = make_float3(ax, ay, az);
}
