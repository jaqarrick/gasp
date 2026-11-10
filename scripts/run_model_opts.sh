#!/bin/bash
# Predict every kernel of one or more configs once per set of --model-opts,
# each run with its own trace file, to compare the model with and without
# each formula change.
#
#   scripts/run_model_opts.sh [--gpu GPU] [--out DIR] [--profile] CONFIG...
#
#   --gpu GPU    hardware profile (default: rtx2080ti)
#   --out DIR    where metrics, traces and logs go (default: traces)
#   --profile    run each config under ncu first and save its metrics
#                (needs the GPU); without it, a metrics file already in DIR
#                is used, and with none the runs are predictions only
#
# For benchmarks/vecadd/vecadd.toml this writes, in DIR:
#   vecadd.metrics.json              with --profile
#   traces.vecadd.run.log            with --profile
#   traces.vecadd.<options>.log      one per option set
#   model-opts.vecadd.log            the tables from every run
#
# OPT_SETS overrides the option sets, space-separated, "none" for no options:
#   OPT_SETS="none dram-floor all" scripts/run_model_opts.sh CONFIG

set -euo pipefail

gpu="rtx2080ti"
out="traces"
profile=false
configs=()

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) gpu="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --profile) profile=true; shift ;;
        -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 1 ;;
        *) configs+=("$1"); shift ;;
    esac
done

if [ ${#configs[@]} -eq 0 ]; then
    echo "usage: $0 [--gpu GPU] [--out DIR] [--profile] CONFIG..." >&2
    exit 1
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
    log="$out/model-opts.$name.log"
    : > "$log"

    if $profile; then
        echo "=== $name: profiling on $gpu ===" | tee -a "$log"
        make -s run ARGS="--gpu $gpu --run --config $config --save-metrics $metrics --trace-file $out/traces.$name.run.log" 2>&1 | tee -a "$log"
    fi

    metrics_args=""
    if [ -f "$metrics" ]; then
        metrics_args="--metrics $metrics"
    else
        echo "$name: no $metrics, so predictions only (pass --profile on the GPU to measure)" | tee -a "$log"
    fi

    for opts in $opt_sets; do
        opts_args=""
        if [ "$opts" != "none" ]; then
            opts_args="--model-opts $opts"
        fi
        echo "=== $name: model-opts $opts ===" | tee -a "$log"
        make -s run ARGS="--gpu $gpu --config $config $metrics_args --trace-file $out/traces.$name.${opts//,/+}.log $opts_args" 2>&1 | tee -a "$log"
    done
    echo "Wrote $log"
done
