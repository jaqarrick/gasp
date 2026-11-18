# Ablation

Measures which static inputs matter: each model (GPUPerf and GASPerf) is run once per row below, each row replacing some static inputs with profiled (ncu) or crude values, and every row is compared with the same measurements.

| Row | Overrides | What it shows |
|---|---|---|
| `all-static` | none | GASP's accuracy as is |
| `crude` | `coalescing=crude`, `bank=crude` | Accuracy without Pico |
| `profiled-cache` | `cache=profiled` | Error from the static cache model |
| `profiled-coalescing` | `coalescing=profiled` | Error left after Pico's coalescing |
| `profiled-bank` | `bank=profiled` | Error left after Pico's bank conflicts |
| `profiled-counts` | `counts=profiled` | Error from static instruction counts |
| `all-profiled` | `all=profiled` | The ceiling: how accurate the formulas are (all profiled except ILP, MLP and the sync count) |

`crude`, `all-static` and `profiled-coalescing` (or `profiled-bank`) are the three levels for coalescing and bank conflicts. The rows are the `ROWS` table at the top of `run_ablation.py`. ILP, MLP and the sync instruction count have no profiled source (ncu has no counter for them), so there is no ILP/MLP row and `all-profiled` keeps the static ones. Instead, every record says whether ITILP or ITMLP sits at its cap (`itilp_capped`, `itmlp_capped`). Where it does, a different ILP or MLP could not change the prediction, so the missing row costs nothing for that kernel. GASPerf has no cap on ITMLP, so its `itmlp_capped` is always false.

## Workflow: every config on a GPU

```bash
make batch-ablation-experiment GPU=rtx2080ti                  # needs the GPU: measures each config, then runs every row
make ablation-results BATCH=experiments/results/ablation/rtx2080ti/batch_<timestamp>
```

`batch-ablation-experiment` (`scripts/batch_ablation.sh`) takes the same `BMKS`, `PROFILE` and `DESC` as `make batch-e2e-experiment`. Each config gets a folder with the e2e outputs (`metrics.json`, `result.json`, `run.log`) and the ablation's `ablation.json` and `ablation.log`. `ablation-results` (`read_ablation.py`) prints, per benchmark and over all of them, per model and row: the mean and median absolute T_GPU error, the same for the speedup (leaving out each config's reference kernel), and how many kernels had ITILP or ITMLP capped. `ARGS='--csv'`, `--model all` and `--records` work like `make batch-results`. Every row names its GPU, so the CSVs of several GPUs can be concatenated.

## Workflow: replay an e2e batch (no GPU)

An e2e batch (`make batch-e2e-experiment`, not a dry run) already saved a `metrics.json` for every config. `REPLAY` runs the ablation on those instead of measuring again:

```bash
make batch-ablation-experiment GPU=rtx2080ti BMKS='bank_conflict transpose l2' \
    REPLAY=experiments/results/e2e/rtx2080ti/batch_20261117_220527
```

It needs `nvcc` (for the static analysis) but no GPU or ncu, and takes minutes. Each config's `metrics.json` and `result.json` are copied into the new ablation batch, so the batch stands on its own, and its `all-static` row matches the e2e batch exactly, since they share the measurements. `GPU` has to be the GPU the e2e batch was measured on. A config that failed in the e2e batch fails here too. Without `BMKS`, every benchmark is replayed. Replaying again after a model change needs no new measurements, as long as the kernels and launches are the same.

## Workflow: one config

1. Profile once, on the GPU. This saves the ncu measurements:

   ```bash
   make run ARGS="--gpu rtx2080ti --run --config benchmarks/vecadd/configs/vecadd.toml --save-metrics vecadd.metrics.json"
   ```

2. Run the ablation, anywhere (no GPU needed):

   ```bash
   make ablation ARGS="--gpu rtx2080ti --config benchmarks/vecadd/configs/vecadd.toml --metrics vecadd.metrics.json --out vecadd.ablation.json"
   ```

It prints one line per (row, kernel, model): predicted and measured cycles, their error, predicted and measured speedup against the reference kernel under the same model, whether Pico fell back, and whether ITILP and ITMLP are capped. When Pico fell back on a kernel, its static coalescing and bank values already are the crude ones.

`--out` saves the same as JSON:

```json
{
  "gasp_version": "0.1.0", "commit": "e9f2176", "gpu": "rtx2080ti", "arch": "sm_75",
  "profile": "...", "config": "...", "metrics": "...", "reference": "good",
  "rows": {"all-static": [], "crude": ["coalescing=crude", "bank=crude"]},
  "not_profiled": ["ilp", "mlp", "sync_inst_count"],
  "records": [{"row": "all-static", "kernel": "good", "model": "GASPerf", "error_pct": 10.2,
               "itilp_capped": true, "itmlp_capped": false, "ilp": 1.67, "itilp": 3.67, "itilp_max": 3.67}]
}
```

Records also carry `overrides`, `pico`, the predicted and measured cycles and speedups, `mlp`, `itmlp` and `mwp_peak_bw`.

The GPU, the config's kernel names and each kernel's block and grid size have to match the metrics file, or the script stops. A single row can be tried through the CLI with `--metrics FILE --override GROUP=SOURCE` (see the main README).
