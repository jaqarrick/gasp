# GASP Custom Benchmarks
This directory includes custom CUDA benchmarks we have prepared for development/experimentation. They are used to evaluate how well GASP predicts kernel performance and speedups. 

Structure:
In each benchmark, the TOML config file describes a comparison.
- `[defaults]` are settings shared by the kernels (e.g. workload size, block/grid dimensions, arguments)
- `[[kernel]]` names the implementation and points to its CUDA source. The first kernel is the reference by default.
- `params` give Pico kernel argument values for analysis
- `program_args` gives arguments to the executable when using `--run`.

Running the benchmarks:
- `make batch-experiment` runs all benchmarks with TOML files in `benchmarks/*/configs`. 
- For a specific benchmark, run: `make batch-experiment GPU=rtx4070 BMKS=mm DESC="MM sweep"` (change GPU, BMKS, DESC as needed).
- To run one config, use `make run ARGS="--gpu rtx4070 --config benchmarks/mm/configs/n512_b8x8.toml"`
- To run batch experiments without running kernels or using NCU, add the `DRY_RUN=1` flag. However, running one config by default only predicts. To execute it on the GPU and measure with NCU add the `--run` flag to `make run ...`.
