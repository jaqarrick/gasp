#!/bin/bash
# Run the e2e accuracy experiment for every config in benchmarks/*/configs/*.toml
# on one GPU. Benchmark names after the GPU limit the run to those benchmarks.
# A failed config doesn't stop the others.
#
# Results go in one folder per batch, a subfolder per config, and a summary:
#   experiments/results/e2e/<gpu>/batch_<timestamp>/<benchmark>_<config>/
#   experiments/results/e2e/<gpu>/batch_<timestamp>/summary.txt
#
# Example usage:
#   scripts/batch_experiment.sh [--dry-run] <gpu> [benchmark ...]
#   scripts/batch_experiment.sh rtx2080ti
#   scripts/batch_experiment.sh --dry-run rtx2080ti mm vecadd
set -euo pipefail

usage() {
    echo "usage: $0 [--dry-run] <gpu> [benchmark ...]   (e.g. $0 rtx2080ti mm)" >&2
    exit 1
}

DRY_RUN=0
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -*) echo "error: unknown option $arg" >&2; usage ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

if [ "${#POSITIONAL[@]}" -lt 1 ]; then
    usage
fi

GPU="${POSITIONAL[0]}"
BMKS=("${POSITIONAL[@]:1}")
DRY_RUN_FLAG=()
if [ "$DRY_RUN" -eq 1 ]; then
    DRY_RUN_FLAG=(--dry-run)
fi

cd "$(dirname "$0")/.."

GPU_PROFILE="./hardware/profiles/$GPU.json"
if [ ! -f "$GPU_PROFILE" ]; then
    echo "error: GPU profile $GPU_PROFILE not found" >&2
    exit 1
fi

# no names given: every benchmark with a configs/ directory
if [ "${#BMKS[@]}" -eq 0 ]; then
    for dir in ./benchmarks/*/configs/; do
        [ -d "$dir" ] && BMKS+=("$(basename "$(dirname "$dir")")")
    done
fi
if [ "${#BMKS[@]}" -eq 0 ]; then
    echo "error: no benchmarks/*/configs/ directories found" >&2
    exit 1
fi

# init the repo and python once, not per config
if ! ./scripts/check_requirements.sh > /dev/null; then
    echo "Requirements not met, running make install"
    make install
fi

BATCH_DIR="./experiments/results/e2e/${GPU}/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BATCH_DIR"

echo "Running config sweeps for ${BMKS[*]} on $GPU into $BATCH_DIR"
PASSED=()
FAILED=()
for BMK in "${BMKS[@]}"; do
    CONFIG_DIR="./benchmarks/$BMK/configs"
    if [ ! -d "$CONFIG_DIR" ]; then
        echo "error: $CONFIG_DIR not found" >&2
        FAILED+=("$BMK/configs")
        continue
    fi
    shopt -s nullglob
    CONFIGS=("$CONFIG_DIR"/*.toml)
    shopt -u nullglob
    if [ "${#CONFIGS[@]}" -eq 0 ]; then
        echo "error: no configs in $CONFIG_DIR" >&2
        FAILED+=("$BMK/configs")
        continue
    fi
    for CONFIG in "${CONFIGS[@]}"; do
        NAME="$BMK/$(basename "$CONFIG" .toml)"
        if ./scripts/run_e2e_accuracy_experiment.sh "${DRY_RUN_FLAG[@]}" \
            --config "$CONFIG" --out-dir "$BATCH_DIR/${BMK}_$(basename "$CONFIG" .toml)" \
            "$BMK" "$GPU"; then
            PASSED+=("$NAME")
        else
            FAILED+=("$NAME")
        fi
    done
done

{
    echo "config sweeps on $GPU$( [ "$DRY_RUN" -eq 1 ] && echo " (dry run)"): ${#PASSED[@]} passed, ${#FAILED[@]} failed"
    [ "${#PASSED[@]}" -eq 0 ] || printf '  passed: %s\n' "${PASSED[@]}"
    [ "${#FAILED[@]}" -eq 0 ] || printf '  failed: %s\n' "${FAILED[@]}"
} > "$BATCH_DIR/summary.txt"

echo
cat "$BATCH_DIR/summary.txt"
echo "results: $BATCH_DIR"
if [ "${#FAILED[@]}" -gt 0 ]; then
    exit 1
fi
