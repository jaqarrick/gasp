# Microbenchmarks
These are targeted CUDA programs for specific operation measurements.

## Building a profile
`make microbenchmarks GPU=<gpu> VERSION=<label>` (`gasp-bm/run_all.py`) runs three suites and merges them into one profile, `hardware/runs/<gpu>/<version>/<gpu>.<version>.json`. Each field takes the first valid result of:

1. gpu-benches' `gpu-stream`, for `dram_bandwidth_bytes_per_cycle` only (see "gpu-benches: DRAM bandwidth").
2. HPMLL (`scripts/run/run_hpmll_bmks.py`, results in `<version>/hpmll/`), for every field it measures: fp32/fp64/int32 instruction latency, L1/L2/DRAM latency, shared load latency and throughput, and L1 and L2 bandwidth. Its `mem_bw` is not run, since `gpu-stream` supplies that field.
3. gasp-bm (`gasp-bm/*.cu`), for every other field, and for an HPMLL field whose HPMLL result was invalid or missing.

The gasp-bm benchmarks that HPMLL overrides still run, as the fallback. `shared_memory_store_cycles` also needs gasp-bm's shared load latency. `sync_cost` gets the same DRAM latency and bandwidth as the profile. `<gpu>.<version>.sources.json` records each field's source file, the gasp-bm value an HPMLL value replaced (`replaces`), and why an HPMLL result was not used (`hpmll_not_used`). Fields nothing measures are copied from the base profile. `ARGS='--only hpmll,dram_latency'` runs a subset; results already in the version directory are reused.

`make microbenchmarks-hpmll` and `make microbenchmarks-gpu-benches` still run one suite on its own.

## Citation: HPMLL NVIDIA Hopper Benchmark
The `hpmll/` submodule is [HPMLL/NVIDIA-Hopper-Benchmark](https://github.com/HPMLL/NVIDIA-Hopper-Benchmark) (GPL-3.0). Its authors ask that the project and these papers be cited:

- Weile Luo, Ruibo Fan, Zeyu Li, Dayou Du, Qiang Wang, and Xiaowen Chu. "[Benchmarking and Dissecting the Nvidia Hopper GPU Architecture.](https://ieeexplore.ieee.org/abstract/document/10579250)" In 2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS), pp. 656-667. IEEE, 2024.
- Weile Luo, Ruibo Fan, Zeyu Li, Dayou Du, Hongyuan Liu, Qiang Wang, and Xiaowen Chu. "[Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis.](https://arxiv.org/abs/2501.12084)" arXiv preprint arXiv:2501.12084 (2025).

```bibtex
@inproceedings{luo2024benchmarking,
  title={Benchmarking and dissecting the nvidia hopper gpu architecture},
  author={Luo, Weile and Fan, Ruibo and Li, Zeyu and Du, Dayou and Wang, Qiang and Chu, Xiaowen},
  booktitle={2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  pages={656--667},
  year={2024},
  organization={IEEE}
}

@article{luo2025dissecting,
  title={Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis},
  author={Luo, Weile and Fan, Ruibo and Li, Zeyu and Du, Dayou and Liu, Hongyuan and Wang, Qiang and Chu, Xiaowen},
  journal={arXiv preprint arXiv:2501.12084},
  year={2025}
}
```

## Citation: gpu-benches
The `gpu-benches/` submodule is [RRZE-HPC/gpu-benches](https://github.com/RRZE-HPC/gpu-benches) (GPL-3.0). It is built and run from the submodule as published, not copied into GASP. Its README lists these papers as using its benchmarks and welcomes citations:

- Dominik Ernst, Markus Holzer, Georg Hager, Matthias Knorr, and Gerhard Wellein. "[Analytical performance estimation during code generation on modern GPUs.](https://doi.org/10.1016/j.jpdc.2022.11.003)" Journal of Parallel and Distributed Computing 173 (2023): 152-167.
- Dominik Ernst, Georg Hager, Jonas Thies, and Gerhard Wellein. "[Performance engineering for real and complex tall & skinny matrix multiplication kernels on GPUs.](https://doi.org/10.1177/1094342020965661)" The International Journal of High Performance Computing Applications 35, no. 1 (2020): 5-19.

```bibtex
@article{ernst2023analytical,
  title={Analytical performance estimation during code generation on modern GPUs},
  author={Ernst, Dominik and Holzer, Markus and Hager, Georg and Knorr, Matthias and Wellein, Gerhard},
  journal={Journal of Parallel and Distributed Computing},
  volume={173},
  pages={152--167},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.jpdc.2022.11.003}
}

@article{ernst2020performance,
  title={Performance engineering for real and complex tall \& skinny matrix multiplication kernels on GPUs},
  author={Ernst, Dominik and Hager, Georg and Thies, Jonas and Wellein, Gerhard},
  journal={The International Journal of High Performance Computing Applications},
  volume={35},
  number={1},
  pages={5--19},
  year={2020},
  publisher={SAGE},
  doi={10.1177/1094342020965661}
}
```
