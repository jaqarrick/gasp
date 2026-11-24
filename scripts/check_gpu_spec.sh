#!/usr/bin/env bash
# Check a hardware profile's spec fields against the GPU this runs on.
# Needs nvcc and python3. simd_width, sfu_width and the shared-memory sizes can't be
# queried from CUDA; check those against the whitepaper.
#
#   scripts/check_gpu_spec.sh hardware/profiles/titanv.json
#   scripts/check_gpu_spec.sh            # just print what the GPU reports
#   scripts/check_gpu_spec.sh --output spec.txt [profile]
set -euo pipefail

profile=""
output=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            [[ $# -ge 2 ]] || { echo "--output needs a file path" >&2; exit 2; }
            output="$2"
            shift 2
            ;;
        --output=*)
            output="${1#--output=}"
            shift
            ;;
        *)
            profile="$1"
            shift
            ;;
    esac
done

if [[ -n "$output" ]]; then
    mkdir -p "$(dirname "$output")"
    exec > "$output"
fi

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

field() {
    local value
    value="$("${@:2}" 2>/dev/null | head -1 | sed 's/^[[:space:]]*//')" || true
    printf '%-22s %s\n' "$1" "${value:-unknown}"
}
smi() { nvidia-smi --query-gpu="$1" --format=csv,noheader -i 0; }

echo "machine"
field "date:" date '+%Y-%m-%d %H:%M:%S %Z'
field "host:" hostname -f
field "os:" uname -srm
field "cpu:" bash -c "lscpu | sed -n 's/^Model name:[[:space:]]*//p'"
field "cpu cores:" nproc
field "memory:" bash -c "free -h | awk '/^Mem:/ {print \$2}'"
field "gpu:" smi name
field "gpu memory:" smi memory.total
field "vbios:" smi vbios_version
field "driver:" smi driver_version
field "cuda (driver):" bash -c "nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | cut -d' ' -f3"
field "nvcc:" bash -c "nvcc --version | grep release"
field "ncu:" bash -c "ncu --version | tail -1"
field "sm clock (max):" smi clocks.max.sm
field "mem clock (max):" smi clocks.max.mem
field "sm clock (now):" smi clocks.sm
field "mem clock (now):" smi clocks.mem
field "power limit:" smi power.limit
field "persistence mode:" smi persistence_mode
field "ecc mode:" smi ecc.mode.current
field "compute mode:" smi compute_mode
field "other gpu processes:" bash -c "command -v nvidia-smi > /dev/null && nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l"
echo

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
