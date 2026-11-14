#!/usr/bin/env bash
# Check a hardware profile's spec fields against the GPU this runs on.
#
#   scripts/check_gpu_spec.sh hardware/profiles/titanv.placeholder.json
#   scripts/check_gpu_spec.sh            # just print what the GPU reports
#
# Needs nvcc (module load cuda-12.4 on the cuda machines) and python3.
# simd_width, sfu_width, l1_shared_array_bytes and shared_mem_carveouts_bytes are
# not queryable from CUDA; check those against the tuning guide / whitepaper.
# The max SM clock is printed for reference: the profile's clock_rate_ghz is the
# base clock, which a microbenchmark run reports as "SM clock: median".
set -euo pipefail

profile="${1:-}"
dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT

cat > "$dir/query.cu" <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { fprintf(stderr, "no CUDA device\n"); return 1; }
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    printf("name=%s\n", p.name);
    printf("arch=sm_%d%d\n", p.major, p.minor);
    printf("warp_size=%d\n", p.warpSize);
    printf("sms_per_gpu=%d\n", p.multiProcessorCount);
    printf("max_warps_per_sm=%d\n", p.maxThreadsPerMultiProcessor / p.warpSize);
    printf("max_blocks_per_sm=%d\n", p.maxBlocksPerMultiProcessor);
    printf("max_registers_per_sm=%d\n", p.regsPerMultiprocessor);
    printf("max_shared_mem_per_sm_bytes=%zu\n", p.sharedMemPerMultiprocessor);
    printf("l2_cache_size_bytes=%d\n", p.l2CacheSize);
    printf("max_sm_clock_ghz=%.3f\n", clock_khz / 1e6);
    return 0;
}
EOF

nvcc -O2 -arch=native "$dir/query.cu" -o "$dir/query"
"$dir/query" > "$dir/gpu.txt"

if [[ -z "$profile" ]]; then
    cat "$dir/gpu.txt"
    exit 0
fi

python3 - "$profile" "$dir/gpu.txt" <<'EOF'
import json, sys
profile = json.load(open(sys.argv[1]))
gpu = dict(line.rstrip("\n").split("=", 1) for line in open(sys.argv[2]))
print(f"GPU: {gpu.pop('name')}   profile: {sys.argv[1]}")
print(f"(max SM clock {gpu.pop('max_sm_clock_ghz')} GHz; profile clock_rate_ghz "
      f"{profile['profiled'].get('clock_rate_ghz')} should be the base clock)\n")
bad = 0
for key, actual in gpu.items():
    expected = profile["arch"] if key == "arch" else profile["spec"].get(key)
    ok = str(expected) == actual
    bad += not ok
    print(f"  {'ok      ' if ok else 'MISMATCH'}  {key:30} profile {expected!s:>10}   gpu {actual:>10}")
print(f"\n{bad} mismatch(es)")
sys.exit(1 if bad else 0)
EOF
