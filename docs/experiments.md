# Experiments

How to measure a GPU, check a speedup against it, and run many configs at once. Every command runs from the repo root, on a machine with the GPU, `nvcc` and `ncu`.


The first time you clone the repo you'll need to run `make install` the, afterwards, possible `source .venv/bin/activate`

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

A speedup config lists two kernels that do the same work, such as `benchmarks/vecadd/vecadd.toml` (`good` and `poor`). GASP predicts both and reports the second kernel's speedup over the first.

Pass the new profile with `--profile`, and add `--run` to also measure both kernels with `ncu`:

```sh
make run ARGS='--profile hardware/runs/rtx2080ti/1_2/rtx2080ti.1_2.json --run --config benchmarks/vecadd/vecadd.toml'
```

The output shows the predicted and measured speedup, and how far apart they are. Leave out `--run` to only predict (no GPU needed).

## 3. Update the built-in profile

If you see positive results across kernels, you can replace the built in profile in `./hardware/profiles`

## 4. Run batch experiments

A batch runs every config in `benchmarks/*/configs/`. Each config is the same pair of kernels at one problem size and block size, so a batch shows how the speedup changes with them.

```sh
make batch-experiment GPU=rtx2080ti
```

- `BMKS='mm vecadd'` runs only those benchmarks.
- `DRY_RUN=1` only predicts: quick, and no GPU needed. Good for checking the configs first.
- `PROFILE=<path>` uses that profile, such as the one from step 1. Without it, the batch uses GASP's built-in profile for `GPU`.

```sh
make batch-experiment GPU=rtx2080ti PROFILE=hardware/runs/rtx2080ti/1_2/rtx2080ti.1_2.json
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

To see the results as a table, one row per config and kernel:

```sh
make batch-results BATCH=experiments/results/e2e/rtx2080ti/batch_<timestamp>
```

- `ARGS='--csv'` prints CSV instead.
- `ARGS='--model all'` shows every model, not only GASPerf.

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
- Put the file in `benchmarks/<name>/configs/` and `make batch-experiment` picks it up.
- Run one config on its own with `make run ARGS='--gpu rtx2080ti --config <path>'`.
- The main `README.md` lists every setting. Look there first if you're confused about any of the arguments. 
