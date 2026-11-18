#!/bin/bash
# Run the ablation for every config in benchmarks/*/configs/*.toml on one GPU.
# Each config is first measured end to end (scripts/run_e2e_accuracy_experiment.sh:
# predict, then --run --save-metrics under ncu), then experiments/ablation/run_ablation.py
# replays its metrics.json through every ablation row and saves ablation.json.
# Benchmark names after the GPU limit the run to those benchmarks.
# --profile <path> uses that hardware profile (such as one from make microbenchmarks)
# instead of the built-in one for <gpu>; <gpu> then only names the results folder.
# A failed config doesn't stop the others. Needs the GPU, nvcc and ncu (no dry run:
# the ablation compares every row with the measurements).
#
# --replay <e2e batch dir> skips the measuring: it runs the ablation on the metrics.json
# each config of that batch (from scripts/batch_experiment.sh, not a dry run) already
# saved, and copies its metrics.json and result.json into the new batch. No GPU or ncu,
# only nvcc. <gpu> has to be the GPU the batch was measured on.
#
# Before it starts, it asks for a short description of the experiment and saves it,
# with the date, GPU, profile, code version and configs, in experiment.txt.
# --description "<text>" gives it without the prompt (needed without a terminal).
#
# Results go in one folder per batch, a subfolder per config, and two text files:
#   experiments/results/ablation/<gpu>/batch_<timestamp>/<benchmark>_<config>/
#       metrics.json, result.json, ablation.json, run.log, ablation.log, traces.log
#   experiments/results/ablation/<gpu>/batch_<timestamp>/experiment.txt
#   experiments/results/ablation/<gpu>/batch_<timestamp>/summary.txt
# Under Slurm the folder is batch_<timestamp>_<job id>, so parallel jobs that start in the
# same second get their own folders. If the folder already exists, the run stops.
# summary.txt holds the pass/fail list and the error decomposition from
# experiments/ablation/read_ablation.py (GASPerf only); it is printed at the end of the run.
# experiments/ablation/read_ablation.py <batch dir> (make ablation-results) reads them.
#
# Example usage:
#   scripts/batch_ablation.sh [--profile <path>] [--replay <e2e batch dir>] [--description <text>] <gpu> [benchmark ...]
#   scripts/batch_ablation.sh rtx2080ti
#   scripts/batch_ablation.sh --replay experiments/results/e2e/rtx2080ti/batch_20261117_220527 rtx2080ti bank_conflict transpose l2
#   scripts/batch_ablation.sh --description "ablation, vecadd smoke test" rtx2080ti vecadd
#   scripts/batch_ablation.sh --profile hardware/runs/rtx2080ti/1_2/rtx2080ti.1_2.json rtx2080ti
set -euo pipefail

# the command as given, for experiment.txt
INVOCATION="$0 $*"

usage() {
    echo "usage: $0 [--profile <path>] [--replay <e2e batch dir>] [--description <text>] <gpu> [benchmark ...]   (e.g. $0 rtx2080ti mm)" >&2
    exit 1
}

PROFILE=""
REPLAY=""
DESCRIPTION=""
POSITIONAL=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile)
            [ "$#" -ge 2 ] || { echo "error: --profile needs a path" >&2; usage; }
            PROFILE="$2"
            shift
            ;;
        --profile=*) PROFILE="${1#--profile=}" ;;
        --replay)
            [ "$#" -ge 2 ] || { echo "error: --replay needs an e2e batch folder" >&2; usage; }
            REPLAY="$2"
            shift
            ;;
        --replay=*) REPLAY="${1#--replay=}" ;;
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
# flags passed on to every e2e run, and the hardware flag for the ablation
E2E_FLAGS=()
HW_ARGS="--gpu $GPU"
# a relative --profile is relative to where the script was called from
if [ -n "$PROFILE" ] && [[ "$PROFILE" != /* ]]; then
    PROFILE="$PWD/$PROFILE"
fi
# the same for --replay
if [ -n "$REPLAY" ] && [[ "$REPLAY" != /* ]]; then
    REPLAY="$PWD/$REPLAY"
fi

cd "$(dirname "$0")/.."

if [ -n "$REPLAY" ] && [ ! -d "$REPLAY" ]; then
    echo "error: --replay folder $REPLAY not found" >&2
    exit 1
fi

# check the profile once here, so a bad one doesn't fail every config
if [ -n "$PROFILE" ]; then
    if [ ! -f "$PROFILE" ]; then
        echo "error: profile $PROFILE not found" >&2
        exit 1
    fi
    E2E_FLAGS+=(--profile "$PROFILE")
    HW_ARGS="--profile $PROFILE"
elif [ ! -f "./hardware/profiles/$GPU.json" ]; then
    echo "error: no hardware/profiles/$GPU.json; pass --profile" >&2
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
BATCH_PARENT="./experiments/results/ablation/${GPU}"
BATCH_DIR="$BATCH_PARENT/batch_$(date +%Y%m%d_%H%M%S)${SLURM_JOB_ID:+_$SLURM_JOB_ID}"
mkdir -p "$BATCH_PARENT"
mkdir "$BATCH_DIR" || { echo "error: $BATCH_DIR already exists" >&2; exit 1; }

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
    if [ -n "$REPLAY" ]; then
        echo "mode:        ablation, replayed from the measurements in $REPLAY (no ncu)"
    else
        echo "mode:        ablation (measure with ncu, then every ablation row)"
    fi
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

echo "Running the ablation for ${BMKS[*]} on $GPU${PROFILE:+ with profile $PROFILE}${REPLAY:+ from $REPLAY} into $BATCH_DIR"
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
        OUT="$BATCH_DIR/${BMK}_$(basename "$CONFIG" .toml)"
        if [ -n "$REPLAY" ]; then
            # the measurements from the e2e batch, copied so this batch stands on its own
            SRC="$REPLAY/${BMK}_$(basename "$CONFIG" .toml)"
            if [ ! -f "$SRC/metrics.json" ]; then
                echo "error: no $SRC/metrics.json (config failed in that batch, or it was a dry run)" >&2
                FAILED+=("$NAME")
                continue
            fi
            mkdir -p "$OUT"
            cp "$SRC/metrics.json" "$OUT/"
            [ -f "$SRC/result.json" ] && cp "$SRC/result.json" "$OUT/"
            MEASURE=(true)
        else
            # measure first; the ablation needs the metrics.json it saves
            MEASURE=(./scripts/run_e2e_accuracy_experiment.sh ${E2E_FLAGS[@]+"${E2E_FLAGS[@]}"} \
                --config "$CONFIG" --out-dir "$OUT" "$BMK" "$GPU")
        fi
        if "${MEASURE[@]}" \
            && make --no-print-directory ablation \
                ARGS="$HW_ARGS --config $CONFIG --metrics $OUT/metrics.json --out $OUT/ablation.json" \
                2>&1 | tee "$OUT/ablation.log"; then
            PASSED+=("$NAME")
        else
            FAILED+=("$NAME")
        fi
    done
done

{
    echo "ablation on $GPU: ${#PASSED[@]} passed, ${#FAILED[@]} failed"
    echo "profile: ${PROFILE:-built-in for $GPU}"
    [ -z "$REPLAY" ] || echo "replayed from: $REPLAY"
    [ "${#PASSED[@]}" -eq 0 ] || printf '  passed: %s\n' "${PASSED[@]}"
    [ "${#FAILED[@]}" -eq 0 ] || printf '  failed: %s\n' "${FAILED[@]}"
    echo
    echo "error decomposition (GASPerf; make ablation-results BATCH=$BATCH_DIR ARGS='--model all' for every model):"
    # read_ablation exits 1 when a config left no ablation.json; the FAILED list above already covers that
    .venv/bin/python experiments/ablation/read_ablation.py "$BATCH_DIR" 2>&1 || true
} > "$BATCH_DIR/summary.txt"

echo "finished:    $(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$BATCH_DIR/experiment.txt"

echo
cat "$BATCH_DIR/summary.txt"
echo "results: $BATCH_DIR"
if [ "${#FAILED[@]}" -gt 0 ]; then
    exit 1
fi
