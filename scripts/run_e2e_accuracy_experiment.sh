#!/bin/bash
# Run one benchmark config end to end (predict, then measure under ncu) and save the JSON results.
# --dry-run only predicts: no --run, so no GPU or ncu is needed.
#
# Example usage:
#   scripts/run_e2e_accuracy_experiment.sh [--dry-run] <benchmark> <gpu>
#   scripts/run_e2e_accuracy_experiment.sh mm rtx2080ti
#   scripts/run_e2e_accuracy_experiment.sh --dry-run mm rtx2080ti
set -euo pipefail

usage() {
    echo "usage: $0 [--dry-run] <benchmark> <gpu>   (e.g. $0 mm rtx2080ti)" >&2
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

if [ "${#POSITIONAL[@]}" -ne 2 ]; then
    usage
fi

BMK="${POSITIONAL[0]}"
GPU="${POSITIONAL[1]}"

cd "$(dirname "$0")/.."

echo "Running $BMK on $GPU"
BMK_DIR="./benchmarks/$BMK"
BMK_CONFIG="$BMK_DIR/$BMK.toml"
GPU_PROFILE="./hardware/profiles/$GPU.json"

# First check if the benchmark config and path exists in ./benchmarks
# e.g. argument is "mm" we look for mm.toml in ./benchmarks/mm/
if [ ! -d "$BMK_DIR" ]; then
    echo "error: benchmark directory $BMK_DIR not found" >&2
    exit 1
fi
if [ ! -f "$BMK_CONFIG" ]; then
    echo "error: benchmark config $BMK_CONFIG not found" >&2
    exit 1
fi
if [ ! -f "$GPU_PROFILE" ]; then
    echo "error: GPU profile $GPU_PROFILE not found" >&2
    exit 1
fi

# init the repo and python
if ! ./scripts/check_requirements.sh > /dev/null; then
    echo "Requirements not met, running make install"
    make install
fi

# create the output / target directory if not there
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_OUT_DIR="./experiments/results/e2e/${GPU}/${BMK}_${STAMP}"
mkdir -p "$EXP_OUT_DIR"
EXP_OUT_FNAME="result.json"

GASP_ARGS="--gpu $GPU --json --verbose --config $BMK_CONFIG --trace-file $EXP_OUT_DIR/traces.log"
if [ "$DRY_RUN" -eq 0 ]; then
    GASP_ARGS="$GASP_ARGS --run --save-metrics $EXP_OUT_DIR/metrics.json"
fi

# save JSON stdout to file and other logs to same dir
make --no-print-directory run ARGS="$GASP_ARGS" \
    > "$EXP_OUT_DIR/$EXP_OUT_FNAME"              \
    2> >(tee "$EXP_OUT_DIR/run.log" >&2)

echo "Experiment for $BMK completed: $EXP_OUT_DIR/$EXP_OUT_FNAME"
