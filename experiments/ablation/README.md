# Ablation

Measures which static inputs matter: the model is run once per row below, each row replacing some static inputs with profiled (ncu) or crude values, and every row is compared with the same measurements.

| Row | Overrides | What it shows |
|---|---|---|
| `all-static` | none | GASP's accuracy as is |
| `crude` | `coalescing=crude`, `bank=crude` | Accuracy without Pico |
| `profiled-cache` | `cache=profiled` | Error from the static cache model |
| `profiled-coalescing` | `coalescing=profiled` | Error left after Pico's coalescing |
| `profiled-bank` | `bank=profiled` | Error left after Pico's bank conflicts |
| `profiled-counts` | `counts=profiled` | Error from static instruction counts |
| `all-profiled` | `all=profiled` | The ceiling: how accurate the formulas are |

`crude`, `all-static` and `profiled-coalescing` (or `profiled-bank`) are the three levels for coalescing and bank conflicts. The rows are the `ROWS` table at the top of `run_ablation.py`. ILP/MLP has no profiled source yet, so `all-profiled` keeps the static ILP and MLP.

## Workflow

1. Profile once, on the GPU. This saves the ncu measurements:

   ```bash
   make run ARGS="--gpu rtx2080ti --run --config benchmarks/vecadd/vecadd.toml --save-metrics vecadd.metrics.json"
   ```

2. Run the ablation, anywhere (no GPU needed):

   ```bash
   make ablation ARGS="--gpu rtx2080ti --config benchmarks/vecadd/vecadd.toml --metrics vecadd.metrics.json --out vecadd.ablation.csv"
   ```

It prints one line per (row, kernel) and writes the same to the CSV: predicted and measured cycles, their error, predicted and measured speedup against the reference kernel, and whether Pico fell back. When Pico fell back on a kernel, its static coalescing and bank values already are the crude ones.

The GPU, the config's kernel names and each kernel's block and grid size have to match the metrics file, or the script stops. A single row can be tried through the CLI with `--metrics FILE --override GROUP=SOURCE` (see the main README).
