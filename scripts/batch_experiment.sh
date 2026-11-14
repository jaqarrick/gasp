#!/bin/bash
# Run the e2e accuracy experiment for every config in benchmarks/*/configs/*.toml
# on one GPU. Benchmark names after the GPU limit the run to those benchmarks.
# --profile <path> uses that hardware profile (such as one from make microbenchmarks)
# instead of the built-in one for <gpu>; <gpu> then only names the results folder.
# A failed config doesn't stop the others.
#
# Results go in one folder per batch, a subfolder per config, and a summary:
#   experiments/results/e2e/<gpu>/batch_<timestamp>/<benchmark>_<config>/
#   experiments/results/e2e/<gpu>/batch_<timestamp>/summary.txt
#
# Example usage:
#   scripts/batch_experiment.sh [--dry-run] [--profile <path>] <gpu> [benchmark ...]
#   scripts/batch_experiment.sh rtx2080ti
#   scripts/batch_experiment.sh --dry-run rtx2080ti mm vecadd
#   scripts/batch_experiment.sh --profile hardware/runs/rtx2080ti/1_2/rtx2080ti.1_2.json rtx2080ti
set -euo pipefail

usage() {
    echo "usage: $0 [--dry-run] [--profile <path>] <gpu> [benchmark ...]   (e.g. $0 rtx2080ti mm)" >&2
    exit 1
}

DRY_RUN=0
PROFILE=""
POSITIONAL=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --profile)
            [ "$#" -ge 2 ] || { echo "error: --profile needs a path" >&2; usage; }
            PROFILE="$2"
            shift
            ;;
        --profile=*) PROFILE="${1#--profile=}" ;;
        -*) echo "error: unknown option $1" >&2; usage ;;
        *) POSITIONAL+=("$1") ;;
    esac
    shift
done

if [ "${#POSITIONAL[@]}" -lt 1 ]; then
    usage
fi

GPU="${POSITIONAL[0]}"
BMKS=("${POSITIONAL[@]:1}")
# flags passed on to every e2e run
E2E_FLAGS=()
if [ "$DRY_RUN" -eq 1 ]; then
    E2E_FLAGS+=(--dry-run)
fi
# a relative --profile is relative to where the script was called from
if [ -n "$PROFILE" ] && [[ "$PROFILE" != /* ]]; then
    PROFILE="$PWD/$PROFILE"
fi

cd "$(dirname "$0")/.."

# check the profile once here, so a bad one doesn't fail every config
if [ -n "$PROFILE" ]; then
    if [ ! -f "$PROFILE" ]; then
        echo "error: profile $PROFILE not found" >&2
        exit 1
    fi
    E2E_FLAGS+=(--profile "$PROFILE")
elif [ ! -f "./hardware/profiles/$GPU.json" ] && [ ! -f "./hardware/profiles/$GPU.placeholder.json" ]; then
    echo "error: no hardware/profiles/$GPU.json or $GPU.placeholder.json; pass --profile" >&2
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

echo "Running config sweeps for ${BMKS[*]} on $GPU${PROFILE:+ with profile $PROFILE} into $BATCH_DIR"
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
        if ./scripts/run_e2e_accuracy_experiment.sh "${E2E_FLAGS[@]}" \
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
    echo "profile: ${PROFILE:-built-in for $GPU}"
    [ "${#PASSED[@]}" -eq 0 ] || printf '  passed: %s\n' "${PASSED[@]}"
    [ "${#FAILED[@]}" -eq 0 ] || printf '  failed: %s\n' "${FAILED[@]}"
} > "$BATCH_DIR/summary.txt"

echo
cat "$BATCH_DIR/summary.txt"
echo "results: $BATCH_DIR"
if [ "${#FAILED[@]}" -gt 0 ]; then
    exit 1
fi
