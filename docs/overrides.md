# Overrides

An override replaces a group of the model's static inputs with what ncu measured, to see how much of the error that input causes. Profile once and save the measurements, then replay them with any override. Replaying needs no GPU:

```bash
make run ARGS="--gpu rtx2080ti --run --config benchmarks/vecadd/configs/vecadd.toml --save-metrics vecadd.metrics.json"
make run ARGS="--gpu rtx2080ti --config benchmarks/vecadd/configs/vecadd.toml --metrics vecadd.metrics.json --override cache=profiled"
```

`--override GROUP=SOURCE` is repeatable. A group not named stays static.

| Group        | Inputs replaced                          | `profiled` (from the metrics file)                     | `crude`       |
|--------------|------------------------------------------|--------------------------------------------------------|---------------|
| `cache`      | L1 and L2 hit ratios                     | ncu's L1 hit rate for global loads; L2 read lookup hits / lookups | -  |
| `coalescing` | Transactions per warp access             | Global-load sectors / requests / 4, at least 1         | 1 (coalesced) |
| `bank`       | Bank conflicts per warp                  | Shared load + store bank conflicts / warps             | 0             |
| `counts`     | Total, global memory (loads + stores), SFU and FP64 instructions per warp | ncu's executed instruction counts / warps | -             |

- `all=profiled` sets every group. `crude` is what the model assumes without Pico.
- ILP, MLP and the sync instruction count have no profiled source, so they're always static.
- The metrics file holds ncu's raw values for every run, keyed by kernel name. GASP stops if the GPU, a kernel name, or a kernel's block or grid size differs from the file, or if a metric a `profiled` override needs is missing.
- `--metrics` also prints the predicted vs measured table from the file. It can't be combined with `--run`.

The full ablation, one row per override set, is in [`experiments/ablation`](../experiments/ablation/README.md): `make ablation ARGS="..."` for one config, `make batch-ablation-experiment GPU=<gpu>` to measure and ablate every config on a GPU, and `make ablation-results BATCH=<dir>` for the error decomposition.
