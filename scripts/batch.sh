#!/bin/bash
# Single script to run all experiments on a single GPU in one go
#
# Example usage:
#   scripts/batch.sh [--dry-run] <gpu>
#   scripts/batch.sh rtx2080ti
#   scripts/batch.sh --dry-run rtx2080ti
set -euo pipefail

usage() {
    echo "usage: $0 [--dry-run] <gpu>   (e.g. $0 rtx2080ti)" >&2
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

if [ "${#POSITIONAL[@]}" -ne 1 ]; then
    usage
fi

GPU="${POSITIONAL[0]}"
DRY_RUN_FLAG=()
if [ "$DRY_RUN" -eq 1 ]; then
    DRY_RUN_FLAG=(--dry-run)
fi

cd "$(dirname "$0")/.."

echo "Running all experiments on $GPU"
GPU_PROFILE="./hardware/profiles/$GPU.json"
if [ ! -f "$GPU_PROFILE" ]; then
    echo "error: GPU profile $GPU_PROFILE not found" >&2
    exit 1
fi

# init the repo and python
if ! ./scripts/check_requirements.sh > /dev/null; then
    echo "Requirements not met, running make install"
    make install
fi

# EXPERIMENT 1 (we can add more below)
# for each benchmark in ./benchmarks/* run the e2e accuracy experiment. Only
# directories with a <name>/<name>.toml config are experiments; the rest are
# skipped. A failed benchmark doesn't stop the others.
PASSED=()
FAILED=()
SKIPPED=()
for dir in ./benchmarks/*/; do
    BMK="$(basename "$dir")"
    if [ ! -f "$dir$BMK.toml" ]; then
        SKIPPED+=("$BMK")
        continue
    fi
    if ./scripts/run_e2e_accuracy_experiment.sh "${DRY_RUN_FLAG[@]}" "$BMK" "$GPU"; then
        PASSED+=("$BMK")
    else
        FAILED+=("$BMK")
    fi
done

echo
echo "e2e accuracy on $GPU:"
echo "  passed:  ${PASSED[*]:-none}"
echo "  failed:  ${FAILED[*]:-none}"
echo "  skipped: ${SKIPPED[*]:-none} (no <name>.toml)"

if [ "${#FAILED[@]}" -gt 0 ]; then
    exit 1
fi
