#!/bin/bash
# Run the e2e accuracy experiment for every config in benchmarks/*/configs/*.toml
# on one GPU. Benchmark names after the GPU limit the run to those benchmarks.
# --profile <path> uses that hardware profile (such as one from make microbenchmarks)
# instead of the built-in one for <gpu>; <gpu> then only names the results folder.
# A failed config doesn't stop the others.
#
# Before it starts, it asks for a short description of the experiment and saves it,
# with the date, GPU, profile, code version and configs, in experiment.txt.
# --description "<text>" gives it without the prompt (needed without a terminal).
#
# Results go in one folder per batch, a subfolder per config, and two text files:
#   experiments/results/e2e/<gpu>/batch_<timestamp>/<benchmark>_<config>/
#   experiments/results/e2e/<gpu>/batch_<timestamp>/experiment.txt
#   experiments/results/e2e/<gpu>/batch_<timestamp>/summary.txt
#
# Example usage:
#   scripts/batch_experiment.sh [--dry-run] [--profile <path>] [--description <text>] <gpu> [benchmark ...]
#   scripts/batch_experiment.sh rtx2080ti
#   scripts/batch_experiment.sh --dry-run rtx2080ti mm vecadd
#   scripts/batch_experiment.sh --profile hardware/runs/rtx2080ti/1_2/rtx2080ti.1_2.json rtx2080ti
#   scripts/batch_experiment.sh --description "mm block sweep, new L1 model" rtx2080ti mm
set -euo pipefail

# the command as given, for experiment.txt
INVOCATION="$0 $*"

usage() {
    echo "usage: $0 [--dry-run] [--profile <path>] [--description <text>] <gpu> [benchmark ...]   (e.g. $0 rtx2080ti mm)" >&2
    exit 1
}

DRY_RUN=0
PROFILE=""
DESCRIPTION=""
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
        --description)
            [ "$#" -ge 2 ] || { echo "error: --description needs a text" >&2; usage; }
            DESCRIPTION="$2"
            shift
            ;;
        --description=*) DESCRIPTION="${1#--description=}" ;;
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

# ask before creating anything, so Ctrl-C here leaves no batch folder behind
if [ -z "$DESCRIPTION" ]; then
    if [ ! -t 0 ]; then
        echo "error: no terminal to ask for a description; pass --description \"<text>\"" >&2
        exit 1
    fi
    while [ -z "$DESCRIPTION" ]; do
        read -r -p "Explain briefly what this experiment does: " DESCRIPTION
    done
fi

# the profile file GASP will load: the given one, or the built-in one for GPU
if [ -n "$PROFILE" ]; then
    PROFILE_FILE="$PROFILE"
    PROFILE_KIND="given"
else
    PROFILE_FILE="$(PYTHONPATH=src .venv/bin/python -c \
        "import sys; from gasp.cli.utils import GPU_PROFILES; print(GPU_PROFILES.get(sys.argv[1], ''))" \
        "$GPU" 2>/dev/null || true)"
    PROFILE_KIND="built-in for $GPU"
fi

STARTED="$(date '+%Y-%m-%d %H:%M:%S %Z')"
BATCH_DIR="./experiments/results/e2e/${GPU}/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BATCH_DIR"

# what was run, where and with what, so the batch can be told apart and repeated later
{
    echo "description: $DESCRIPTION"
    echo
    echo "started:     $STARTED"
    echo "gpu:         $GPU"
    if command -v nvidia-smi > /dev/null; then
        echo "device:      $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
    fi
    if command -v nvcc > /dev/null; then
        echo "nvcc:        $(nvcc --version 2>/dev/null | grep -m1 release || echo unknown)"
    fi
    echo "host:        $(hostname)"
    echo "profile:     ${PROFILE_FILE:-unknown} ($PROFILE_KIND)"
    if [ -n "$PROFILE_FILE" ] && [ -f "$PROFILE_FILE" ]; then
        echo "profile sha: $(sha256sum "$PROFILE_FILE" | cut -d' ' -f1)"
    fi
    echo "mode:        $( [ "$DRY_RUN" -eq 1 ] && echo "dry run (predict only)" || echo "full (predict and measure with ncu)")"
    echo "commit:      $(git rev-parse --short HEAD 2>/dev/null || echo unknown) on $(git branch --show-current 2>/dev/null || echo unknown)$( [ -n "$(git status --porcelain 2>/dev/null)" ] && echo " (uncommitted changes)")"
    echo "command:     $INVOCATION"
    echo "benchmarks:  ${BMKS[*]}"
    echo "configs:"
    for BMK in "${BMKS[@]}"; do
        for CONFIG in ./benchmarks/"$BMK"/configs/*.toml; do
            [ -f "$CONFIG" ] && echo "  ${CONFIG#./}"
        done
    done
} > "$BATCH_DIR/experiment.txt"

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

echo "finished:    $(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$BATCH_DIR/experiment.txt"

echo
cat "$BATCH_DIR/summary.txt"
echo "results: $BATCH_DIR"
if [ "${#FAILED[@]}" -gt 0 ]; then
    exit 1
fi
