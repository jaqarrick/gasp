#!/bin/bash
#SBATCH --job-name=ablation_a100_mm
#SBATCH --account=csci_ga_3033_025-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

GPU=a100
DESC='ablation (with e2e results included) on mm (A100)'
PROFILE=""
BMKS="mm"
IMG=/share/apps/images/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif

REPO="$SLURM_SUBMIT_DIR"
cd "$REPO"

FAIAL_COST="$REPO/.faial-bin/faial-cost"

# run the following from an interactive jupyter notebook before submitting batch script:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   singularity exec "$IMG" bash -c 'export PATH="$HOME/.local/bin:$PATH"; unset VIRTUAL_ENV; uv venv .venv --python /usr/bin/python3 && uv pip install --python .venv/bin/python -r requirements.txt'
#   singularity exec "$IMG" env FAIAL_MODE=native FAIAL_PREFIX="$PWD/.faial" BIN_DIR="$PWD/.faial-bin" bash scripts/install_faial.sh   # only if .faial-bin/faial-cost isn't there yet
#   singularity exec "$IMG" env GASP_FAIAL_COST="$PWD/.faial-bin/faial-cost" ./scripts/check_requirements.sh   # expect "All requirements satisfied."

nvidia-smi

MAKE_ARGS=(batch-ablation-experiment GPU="$GPU" DESC="$DESC")
[ -n "$PROFILE" ] && MAKE_ARGS+=(PROFILE="$PROFILE")
[ -n "$BMKS" ] && MAKE_ARGS+=(BMKS="$BMKS")

singularity exec --nv --env GASP_FAIAL_COST="$FAIAL_COST" "$IMG" bash -c '
  set -euo pipefail
  cd "$1"; shift
  command -v make >/dev/null || { echo "ERROR: make is not in the container image; this workflow needs it." >&2; exit 1; }
  ./scripts/check_requirements.sh || { echo "ERROR: requirements not met in the container; do the one-time login-node setup above." >&2; exit 1; }
  make "$@"
' _ "$REPO" "${MAKE_ARGS[@]}"

echo "Done. See experiments/results/ablation/$GPU/"
