#!/bin/bash
# Predict every kernel of one or more configs once per set of --model-opts,
# each run with its own trace file, to compare the model with and without
# each formula change.
#
#   scripts/run_model_opts.sh [--gpu GPU | --hw-profile FILE] [--out DIR] [--run] CONFIG...
#
#   --gpu GPU          built-in hardware profile (default: rtx2080ti)
#   --hw-profile FILE  hardware profile JSON instead of --gpu, passed to
#                      GASP as --profile FILE (e.g. a *.placeholder.json)
#   --out DIR          where metrics, traces and logs go (default: traces)
#   --run              run each config under ncu first and save its metrics
#                      (needs the GPU); without it, a metrics file already in
#                      DIR is used, and with none the runs are predictions only
#
# For benchmarks/vecadd/vecadd.toml this writes, in DIR:
#   vecadd.metrics.json              with --run
#   traces.vecadd.run.log            with --run
#   traces.vecadd.<options>.log      one per option set
#   model-opts.vecadd.log            the tables from every run
# With --hw-profile FILE the per-option traces and the tables log get the
# profile's file name added (traces.vecadd.rtx2080ti.placeholder.none.log);
# the metrics file is shared, so a --gpu run's measurements are reused.
#
# OPT_SETS overrides the option sets, space-separated, "none" for no options:
#   OPT_SETS="none dram-floor all" scripts/run_model_opts.sh CONFIG

set -euo pipefail

gpu=""
hw_profile=""
out="traces"
run=false
configs=()

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) gpu="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --hw-profile) hw_profile="$2"; shift 2 ;;
        --run) run=true; shift ;;
        --profile)
            echo "--profile is now --run; use --hw-profile FILE for a hardware profile" >&2
            exit 1 ;;
        -h|--help) sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 1 ;;
        *) configs+=("$1"); shift ;;
    esac
done

if [ ${#configs[@]} -eq 0 ]; then
    echo "usage: $0 [--gpu GPU | --hw-profile FILE] [--out DIR] [--run] CONFIG..." >&2
    exit 1
fi

if [ -n "$gpu" ] && [ -n "$hw_profile" ]; then
    echo "--gpu and --hw-profile are mutually exclusive" >&2
    exit 1
fi
if [ -n "$hw_profile" ]; then
    if [ ! -f "$hw_profile" ]; then
        echo "hardware profile not found: $hw_profile" >&2
        exit 1
    fi
    hw_profile="$(realpath "$hw_profile")"
    hw_args="--profile $hw_profile"
    hw_name="$(basename "$hw_profile" .json)"
    tag=".$hw_name"
else
    gpu="${gpu:-rtx2080ti}"
    hw_args="--gpu $gpu"
    hw_name="$gpu"
    tag=""
fi

opt_sets="${OPT_SETS:-none dram-floor mem-classes,amat-mem-l mem-classes,amat-mem-l,dram-floor}"

cd "$(dirname "$0")/.."
mkdir -p "$out"

for config in "${configs[@]}"; do
    if [ ! -f "$config" ]; then
        echo "config not found: $config" >&2
        exit 1
    fi
    name="$(basename "$config" .toml)"
    metrics="$out/$name.metrics.json"
    log="$out/model-opts.$name$tag.log"
    : > "$log"

    if $run; then
        echo "=== $name: profiling with $hw_name ===" | tee -a "$log"
        make -s run ARGS="$hw_args --run --config $config --save-metrics $metrics --trace-file $out/traces.$name.run.log" 2>&1 | tee -a "$log"
    fi

    metrics_args=""
    if [ -f "$metrics" ]; then
        metrics_args="--metrics $metrics"
    else
        echo "$name: no $metrics, so predictions only (pass --run on the GPU to measure)" | tee -a "$log"
    fi

    for opts in $opt_sets; do
        opts_args=""
        if [ "$opts" != "none" ]; then
            opts_args="--model-opts $opts"
        fi
        echo "=== $name: model-opts $opts ===" | tee -a "$log"
        make -s run ARGS="$hw_args --config $config $metrics_args --trace-file $out/traces.$name$tag.${opts//,/+}.log $opts_args" 2>&1 | tee -a "$log"
    done
    echo "Wrote $log"
done
