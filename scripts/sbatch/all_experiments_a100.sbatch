#!/bin/bash
#SBATCH --job-name=all_experiments_a100
#SBATCH --account=csci_ga_3033_025-2026fa
#SBATCH --partition=c12m85-a100-1        # 1x A100; change to match the GPU you want
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00                   # ablation (4 bmks) + e2e (every bmk), each measured with ncu -> long
#SBATCH --output=%x_%j.out

set -euo pipefail

# ---- what to run -----------------------------------------------------------
GPU=a100

# Descriptions are REQUIRED. `make all-experiments` asks for them on a terminal,
# and a batch job has none -- it would exit with "no terminal to ask for the
# descriptions". Passing them up front lets the whole run proceed unattended.
ABLATION_DESC="ablation batch on mm, bank_conflict, transpose, l2 (A100)"
E2E_DESC="e2e accuracy batch on every benchmark (A100)"

PROFILE=""             # optional profile path (e.g. hardware/runs/a100/1_1/a100.1_1.json); empty = GPU's built-in
ABLATION_BMKS=""       # optional override of the ablation list (default: mm bank_conflict transpose l2)

# ---- container -------------------------------------------------------------
IMG=/share/apps/images/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif
# IMG=/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif   # fallback if driver < 12.6

REPO="$SLURM_SUBMIT_DIR"
cd "$REPO"

# faial-cost (pico) wrapper installed with the container's glibc (from the
# speedup setup). gasp reads GASP_FAIAL_COST; we forward it into the container.
FAIAL_COST="$REPO/.faial-bin/faial-cost"

# ---- one-time setup (do this on a LOGIN node, which has internet) -----------
# Everything runs INSIDE the container: nvcc, ncu, make, and the venv. The
# compute node has no internet, so build the venv and faial ahead of time.
# The image's base python3 has no pip/venv (and you can't apt-install without
# admin), so use uv -- a no-root installer the repo already recommends -- and run
# it INSIDE the container so the venv points at the container's python3. Run with
# no venv active (prompt must NOT show "(.venv)"). From the repo root:
#
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   singularity exec "$IMG" bash -c 'export PATH="$HOME/.local/bin:$PATH"; unset VIRTUAL_ENV; uv venv .venv --python /usr/bin/python3 && uv pip install --python .venv/bin/python -r requirements.txt'
#   singularity exec "$IMG" env FAIAL_MODE=native FAIAL_PREFIX="$PWD/.faial" BIN_DIR="$PWD/.faial-bin" bash scripts/install_faial.sh   # only if .faial-bin/faial-cost isn't there yet
#   singularity exec "$IMG" env GASP_FAIAL_COST="$PWD/.faial-bin/faial-cost" ./scripts/check_requirements.sh   # expect "All requirements satisfied."
#
# (If check_requirements reports `make` missing, the image has no make and this
# make-based workflow can't run in it -- tell me and I'll adapt.)

nvidia-smi                                # host driver info, for your records

# ---- run -------------------------------------------------------------------
# Build the make args in the host shell so the descriptions (with spaces) pass
# cleanly, then run make INSIDE the container. Descriptions passed up front so
# nothing blocks on a prompt. all-experiments runs the ablation batch first; if
# it fails, the e2e batch does not start.
# Results: experiments/results/ablation/$GPU/batch_* and experiments/results/e2e/$GPU/batch_*
MAKE_ARGS=(all-experiments GPU="$GPU" ABLATION_DESC="$ABLATION_DESC" E2E_DESC="$E2E_DESC")
[ -n "$PROFILE" ] && MAKE_ARGS+=(PROFILE="$PROFILE")
[ -n "$ABLATION_BMKS" ] && MAKE_ARGS+=(ABLATION_BMKS="$ABLATION_BMKS")

singularity exec --nv --env GASP_FAIAL_COST="$FAIAL_COST" "$IMG" bash -c '
  set -euo pipefail
  cd "$1"; shift
  command -v make >/dev/null || { echo "ERROR: make is not in the container image; this workflow needs it." >&2; exit 1; }
  ./scripts/check_requirements.sh || { echo "ERROR: requirements not met in the container; do the one-time login-node setup above." >&2; exit 1; }
  make "$@"
' _ "$REPO" "${MAKE_ARGS[@]}"

echo "Done. See experiments/results/ablation/$GPU/ and experiments/results/e2e/$GPU/"
