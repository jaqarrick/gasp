#!/bin/bash
# Run one benchmark config end to end (predict, then measure under ncu) and save the JSON results.
# --dry-run only predicts: no --run, so no GPU or ncu is needed.
# --config <path> runs that config instead of benchmarks/<benchmark>/<benchmark>.toml,
# such as one from benchmarks/<benchmark>/configs/.
# --out-dir <dir> saves the results in <dir> instead of a new
# experiments/results/e2e/<gpu>/<benchmark>_<timestamp>/ (scripts/batch_experiment.sh uses it).
#
# Example usage:
#   scripts/run_e2e_accuracy_experiment.sh [--dry-run] [--config <path>] [--out-dir <dir>] <benchmark> <gpu>
#   scripts/run_e2e_accuracy_experiment.sh mm rtx2080ti
#   scripts/run_e2e_accuracy_experiment.sh --dry-run mm rtx2080ti
#   scripts/run_e2e_accuracy_experiment.sh --config benchmarks/mm/configs/n1024_b8x8.toml mm rtx2080ti
set -euo pipefail

usage() {
    echo "usage: $0 [--dry-run] [--config <path>] [--out-dir <dir>] <benchmark> <gpu>   (e.g. $0 mm rtx2080ti)" >&2
    exit 1
}

DRY_RUN=0
CONFIG=""
OUT_DIR=""
POSITIONAL=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --config)
            [ "$#" -ge 2 ] || { echo "error: --config needs a path" >&2; usage; }
            CONFIG="$2"
            shift
            ;;
        --config=*) CONFIG="${1#--config=}" ;;
        --out-dir)
            [ "$#" -ge 2 ] || { echo "error: --out-dir needs a path" >&2; usage; }
            OUT_DIR="$2"
            shift
            ;;
        --out-dir=*) OUT_DIR="${1#--out-dir=}" ;;
        -*) echo "error: unknown option $1" >&2; usage ;;
        *) POSITIONAL+=("$1") ;;
    esac
    shift
done

if [ "${#POSITIONAL[@]}" -ne 2 ]; then
    usage
fi

BMK="${POSITIONAL[0]}"
GPU="${POSITIONAL[1]}"

# a relative --config or --out-dir is relative to where the script was called from
if [ -n "$CONFIG" ] && [[ "$CONFIG" != /* ]]; then
    CONFIG="$PWD/$CONFIG"
fi
if [ -n "$OUT_DIR" ] && [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$PWD/$OUT_DIR"
fi

cd "$(dirname "$0")/.."

BMK_DIR="./benchmarks/$BMK"
BMK_CONFIG="${CONFIG:-$BMK_DIR/$BMK.toml}"
echo "Running $BMK on $GPU with $BMK_CONFIG"
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
# a --config run is named after its config too, so runs of one benchmark don't collide
if [ -n "$OUT_DIR" ]; then
    EXP_OUT_DIR="$OUT_DIR"
elif [ -n "$CONFIG" ]; then
    EXP_OUT_DIR="./experiments/results/e2e/${GPU}/${BMK}_$(basename "$BMK_CONFIG" .toml)_${STAMP}"
else
    EXP_OUT_DIR="./experiments/results/e2e/${GPU}/${BMK}_${STAMP}"
fi
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

echo "Experiment for $BMK ($BMK_CONFIG) completed: $EXP_OUT_DIR/$EXP_OUT_FNAME"
