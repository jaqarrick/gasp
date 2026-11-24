#!/bin/bash
#SBATCH --job-name=microbenchmarks_a100
#SBATCH --account=csci_ga_3033_025-2026fa
#SBATCH --partition=c12m85-a100-1        # 1x A100; change to match the GPU you want
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00                   # ncu replays each kernel many times; the full suite is slow
#SBATCH --output=%x_%j.out

set -euo pipefail

# ---- what to run -----------------------------------------------------------
# GPU and VERSION name the output dir hardware/runs/$GPU/$VERSION/
#   A100 -> sm_80   H100 -> sm_90   B200 -> sm_100
#   V100 -> sm_70   RTX2080Ti -> sm_75   TitanV -> sm_70   RTX4070 -> sm_89
GPU=a100
VERSION=1.1
ARCH=sm_80
EXTRA_ARGS=""                             # e.g. "--only hpmll,dram_latency --force"

# ---- container -------------------------------------------------------------
IMG=/share/apps/images/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif
# IMG=/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif   # fallback if driver < 12.6

# Run from the repo root (submit this script from there):  sbatch hardware/microbenchmarks/microbenchmarks.sbatch
REPO="$SLURM_SUBMIT_DIR"
cd "$REPO"

# ---- preflight -------------------------------------------------------------
# both submodules must be checked out; fetch them on a login node:
#   git submodule update --init hardware/microbenchmarks/hpmll hardware/microbenchmarks/gpu-benches
missing=0
if [ ! -d hardware/microbenchmarks/hpmll/RegularUnits ]; then
  echo "ERROR: hpmll submodule not checked out (hpmll/RegularUnits missing)." >&2
  missing=1
fi
if [ ! -f hardware/microbenchmarks/gpu-benches/gpu-stream/main.cu ]; then
  echo "ERROR: gpu-benches submodule not checked out (gpu-stream/main.cu missing)." >&2
  missing=1
fi
if [ "$missing" -ne 0 ]; then
  echo "On a login node (has internet), run from the repo root:" >&2
  echo "  git submodule update --init hardware/microbenchmarks/hpmll hardware/microbenchmarks/gpu-benches" >&2
  echo "then resubmit." >&2
  exit 1
fi

nvidia-smi

# ---- run -------------------------------------------------------------------
# runs under ncu --clock-control base; output: hardware/runs/$GPU/$VERSION/
singularity exec --nv "$IMG" \
  python3 hardware/microbenchmarks/gasp-bm/run_all.py \
    --gpu "$GPU" --version "$VERSION" --arch "$ARCH" $EXTRA_ARGS

echo "Done. Profile: hardware/runs/$GPU/$VERSION/$GPU.$VERSION.json"
