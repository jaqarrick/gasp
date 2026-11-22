# Experiments

How to measure a GPU, check a speedup against it, and run many configs at once. Every command runs from the repo root, on a machine with the GPU, `nvcc` and `ncu`.


The first time you clone the repo, run `make install`. After that, run `source .venv/bin/activate` in each new shell.

## 1. Run the microbenchmark suite

The suite measures the GPU and writes a hardware profile.

The suite needs the benchmark submodules. Fetch them once:

```sh
git submodule update --init --recursive
```

Run the suite. `GPU` names the card; `VERSION` labels this run:

```sh
make microbenchmarks GPU=rtx2080ti VERSION=1.2
```

The profile is written to:

```
hardware/runs/rtx2080ti/1_2/rtx2080ti.1_2.json
```

The folder also has each benchmark's logs, and `rtx2080ti.1_2.sources.json`, which says where every value came from.

## 2. Test a speedup config with that profile

A speedup config lists two kernels that do the same work, such as `benchmarks/vecadd/configs/vecadd.toml` (`good` and `poor`). GASP predicts each kernel and reports every other kernel's speedup over the first.

Pass the new profile with `--profile`, and add `--run` to also measure both kernels with `ncu`:

```sh
make run ARGS='--profile hardware/runs/rtx2080ti/1_2/rtx2080ti.1_2.json --run --config benchmarks/vecadd/configs/vecadd.toml'
```

The output shows the predicted and measured speedup, and how far apart they are. Leave out `--run` to only predict (no GPU needed).

## 3. Update the built-in profile

If you see positive results across kernels, you can replace the built-in profile in `./hardware/profiles`.

## 4. Run batch experiments

A batch runs every config in `benchmarks/*/configs/`. Most configs are one pair of kernels at one problem size and block size, so a batch shows how the speedup changes with them. A few hold more kernels (`mm/configs/mm.toml`, five mm variants against `basic`) or one (`mm/configs/mm_basic.toml`, no speedup).

```sh
make batch-e2e-experiment GPU=rtx2080ti
```

- `BMKS='mm vecadd'` runs only those benchmarks.
- `DRY_RUN=1` only predicts: quick, and no GPU needed. Good for checking the configs first.
- `PROFILE=<path>` uses that profile, such as the one from step 1. Without it, the batch uses GASP's built-in profile for `GPU`.

```sh
make batch-e2e-experiment GPU=rtx2080ti PROFILE=hardware/runs/rtx2080ti/1_2/rtx2080ti.1_2.json
```

`GPU` still names the results folder.

Before it starts, the batch asks you to explain briefly what the experiment does. It saves your answer in `experiment.txt`, with the date, GPU, profile, commit and configs. `DESC='<text>'` gives the answer up front, which you need when there's no terminal (for example under `nohup`).

Results go in one folder per batch:

```
experiments/results/e2e/rtx2080ti/batch_<timestamp>/
    experiment.txt          your description, date, GPU, profile, commit, configs
    summary.txt             which configs passed and failed
    mm_n1024_b16x16/        one folder per config: result.json, run.log, metrics.json
    ...
```

Under Slurm the folder is `batch_<timestamp>_<job id>`, so jobs that run in parallel and start in the same second don't share a folder. A batch never reuses a folder that already exists: it stops with an error instead.

To see the results as a table, one row per config and kernel:

```sh
make batch-results BATCH=experiments/results/e2e/rtx2080ti/batch_<timestamp>
```

- `ARGS='--csv'` prints CSV instead.
- `ARGS='--model all'` shows every model, not only GASPerf. The batch runs GASP with `--model all`, so `result.json` always holds both.

## 5. Add your own configs

A speedup config lists two or more kernels. The first one is the base: every other kernel's speedup is measured against it.

```toml
# settings every kernel shares
[defaults]
problem_size = 1000000
block_size = 256
program_args = ["1000000"]   # what the program gets with --run; keep it equal to problem_size

# the base
[[kernel]]
name = "good"
file = "benchmarks/vecadd/vecadd_good.cu"

# compared with the base
[[kernel]]
name = "poor"
file = "benchmarks/vecadd/vecadd_poor.cu"
```

- A setting under a `[[kernel]]` overrides the one in `[defaults]` for that kernel.
- Put the file in `benchmarks/<name>/configs/` and `make batch-e2e-experiment` picks it up.
- Run one config on its own with `make run ARGS='--gpu rtx2080ti --config <path>'`.
- The main `README.md` lists every setting. Look there first if you're confused about any of the arguments.

## 6. Run the ablation

The ablation shows which of GASP's static inputs cause the most error. For every config it measures the kernels once with `ncu`, then predicts them again once per ablation row, each row replacing some static inputs with the measured (profiled) values or crude ones. [`experiments/ablation/README.md`](../experiments/ablation/README.md) lists the rows.

```sh
make batch-ablation-experiment GPU=rtx2080ti
```

- `BMKS='mm vecadd'`, `PROFILE=<path>` and `DESC='<text>'` work as for `batch-e2e-experiment`.
- There's no `DRY_RUN`: every row is compared with the measurements, so it needs the GPU.
- `REPLAY=<e2e batch dir>` skips the measuring and uses the `metrics.json` files that e2e batch already saved, so no GPU or ncu is needed (only `nvcc`). `GPU` has to match that batch: `make batch-ablation-experiment GPU=rtx2080ti BMKS='transpose l2' REPLAY=experiments/results/e2e/rtx2080ti/batch_<timestamp>`.

Results go in one folder per batch, like the e2e batch:

```
experiments/results/ablation/rtx2080ti/batch_<timestamp>/
    experiment.txt          your description, date, GPU, profile, commit, configs
    summary.txt             which configs passed and failed
    vecadd_n1000000_b256/   one folder per config: metrics.json, result.json, ablation.json, run.log, ablation.log
    ...
```

To see the error decomposition (mean and median absolute error per benchmark, model and row):

```sh
make ablation-results BATCH=experiments/results/ablation/rtx2080ti/batch_<timestamp>
```

- `ARGS='--csv'` prints CSV instead.
- `ARGS='--model all'` shows both models, not only GASPerf.
- `ARGS='--records'` prints every record instead of the summary.

Every row names its GPU, so to compare GPUs, run the ablation on each and concatenate their `--csv` outputs.


## 7. Run everything

```sh
make all-experiments GPU=rtx2080ti
```

This runs the ablation batch on `mm`, `bank_conflict`, `transpose` and `l2` (`make batch-ablation-experiment BMKS='mm bank_conflict transpose l2'`), then the e2e batch on every benchmark (`make batch-e2e-experiment`). mm is the main ablation target; the other three exercise the bank-conflict, coalescing and cache inputs that mm barely varies.

**It asks for both descriptions first, before anything runs:** one for the ablation batch, then one for the e2e batch. After that it runs to the end without asking for anything, so you can leave it.

- `ABLATION_DESC='<text>'` and `E2E_DESC='<text>'` give the descriptions up front and skip the questions. Give both when there's no terminal (for example under `nohup`), or it stops at once with an error.
- `ABLATION_BMKS='mm transpose'` replaces the ablation's list of benchmarks.
- `PROFILE=<path>` goes to both batches.
- If the ablation batch fails (any config), the e2e batch doesn't start.

```sh
nohup make all-experiments GPU=rtx2080ti ABLATION_DESC='ablation, new L1 model' E2E_DESC='full e2e, new L1 model' > all.log 2>&1 &
```

## 8. Export the report's figure data

```sh
make report-data DATA=~/path/to/data/directory
```

The report draws its figures in LaTeX (pgfplots) from CSV files next to `draft.tex`, and takes its e2e tables from them. This writes those CSVs from one ablation batch per GPU, with the scripts in `scripts/results/`. It only reads the batches, so it needs no GPU. The all-static row of each batch is its e2e result, and the numbers come from the same code as `make ablation-results`.

- `DATA=<dir>` is the folder the CSVs go in, usually the report's `data/` folder. It has to exist.
- `BATCHES='<dir> <dir> ...'` replaces the batches. The default is the final batch of each of the four GPUs. Each batch's GPU is the name of its parent folder.

| Script | Writes | For |
|---|---|---|
| `export_speedup_scatter.py` | `speedup_<gpu>.csv`: every compared kernel's measured and predicted speedup (all-static) | `fig:scatter`, and the input of the next two |
| `export_mm.py` | `mm_<gpu>.csv`: median speedup of each mm variant over `basic` | `fig:mm` |
| `export_bars.py` | `bars_<gpu>.csv`, `bank_<gpu>.csv`, `l1_<gpu>.csv`: the other benchmarks, bank-conflict degrees, L1 line patterns | `fig:bars`, `fig:bank-l1` |
| `export_ablation.py` | `ablation_<gpu>.csv`: change in median speedup error from all-static, per row and benchmark | `fig:ablation` |
| `export_e2e.py` | `e2e_<gpu>.csv`: every kernel and model (all-static) with its T_GPU and speedup errors, model terms and static and measured inputs; `e2e_summary.csv`: per GPU and model, the runtime, signed and speedup errors and the ranking; `e2e_benchmarks.csv`: the same per benchmark; `e2e_kernels.csv`: per kernel name over its configs; `e2e_bank_cost.csv`: predicted against measured time added by bank conflicts; `e2e_inputs.csv`: static against measured inputs per GPU, benchmark and input | `sec:results-e2e`, `tab:e2e-summary` |

`export_mm.py` and `export_bars.py` read the `speedup_<gpu>.csv` files in `DATA`, so `export_speedup_scatter.py` runs first.

Some configs repeat a kernel run of another config: `mm.toml` and `n1024_b16x16.toml` both run `basic` and `tiled` at 1024 with 16x16 blocks, and `mm_basic.toml` runs `basic` a third time. Every script counts each kernel run, speedup comparison and ranking pair once. Going through the configs in name order, the first config that has it keeps it (`mark_duplicates` in `experiments/ablation/read_ablation.py`, which `make ablation-results` uses too; two runs are the same if file, kernel, grid, block, problem size, footprint, reuse, loop counts and parameters match). `e2e_<gpu>.csv` keeps every row and marks the duplicates in `duplicate_run` and `duplicate_speedup`. Each script also runs on its own: `python3 scripts/results/<script> --out <dir>`.
