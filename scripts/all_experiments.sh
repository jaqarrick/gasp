#!/bin/bash
# Run every experiment on one GPU: the ablation batch on mm (scripts/batch_ablation.sh),
# then the e2e batch on every benchmark (scripts/batch_experiment.sh).
#
# Both descriptions are asked for here, before anything runs, and passed to the batches,
# so the run never stops halfway to wait for input. --ablation-description and
# --e2e-description give them up front instead (needed without a terminal, e.g. nohup).
# --profile <path> goes to both batches.
#
# If the ablation batch fails (any config), the e2e batch doesn't start.
#
# Example usage:
#   scripts/all_experiments.sh [--profile <path>] [--ablation-description <text>] [--e2e-description <text>] <gpu>
#   scripts/all_experiments.sh rtx2080ti
#   scripts/all_experiments.sh --ablation-description "mm ablation" --e2e-description "full e2e" rtx2080ti
set -euo pipefail

usage() {
    echo "usage: $0 [--profile <path>] [--ablation-description <text>] [--e2e-description <text>] <gpu>   (e.g. $0 rtx2080ti)" >&2
    exit 1
}

PROFILE=""
ABLATION_DESC=""
E2E_DESC=""
POSITIONAL=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile)
            [ "$#" -ge 2 ] || { echo "error: --profile needs a path" >&2; usage; }
            PROFILE="$2"
            shift
            ;;
        --profile=*) PROFILE="${1#--profile=}" ;;
        --ablation-description)
            [ "$#" -ge 2 ] || { echo "error: --ablation-description needs a text" >&2; usage; }
            ABLATION_DESC="$2"
            shift
            ;;
        --ablation-description=*) ABLATION_DESC="${1#--ablation-description=}" ;;
        --e2e-description)
            [ "$#" -ge 2 ] || { echo "error: --e2e-description needs a text" >&2; usage; }
            E2E_DESC="$2"
            shift
            ;;
        --e2e-description=*) E2E_DESC="${1#--e2e-description=}" ;;
        -*) echo "error: unknown option $1" >&2; usage ;;
        *) POSITIONAL+=("$1") ;;
    esac
    shift
done

if [ "${#POSITIONAL[@]}" -ne 1 ]; then
    usage
fi
GPU="${POSITIONAL[0]}"

# ask for every missing description now, before either batch starts
if [ -z "$ABLATION_DESC" ] || [ -z "$E2E_DESC" ]; then
    if [ ! -t 0 ]; then
        echo "error: no terminal to ask for the descriptions; pass --ablation-description \"<text>\" and --e2e-description \"<text>\"" >&2
        exit 1
    fi
fi
while [ -z "$ABLATION_DESC" ]; do
    read -r -p "Explain briefly what the ablation experiment (mm) does: " ABLATION_DESC
done
while [ -z "$E2E_DESC" ]; do
    read -r -p "Explain briefly what the e2e experiment (every benchmark) does: " E2E_DESC
done

# a relative --profile is relative to where the script was called from
if [ -n "$PROFILE" ] && [[ "$PROFILE" != /* ]]; then
    PROFILE="$PWD/$PROFILE"
fi
PROFILE_ARGS=()
if [ -n "$PROFILE" ]; then
    PROFILE_ARGS=(--profile "$PROFILE")
fi

cd "$(dirname "$0")/.."

echo "=== 1/2: ablation batch on mm ==="
./scripts/batch_ablation.sh ${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"} --description "$ABLATION_DESC" "$GPU" mm

echo
echo "=== 2/2: e2e batch on every benchmark ==="
./scripts/batch_experiment.sh ${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"} --description "$E2E_DESC" "$GPU"
