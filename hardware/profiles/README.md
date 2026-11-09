# Profiles

The files within this directory contain the hardware profiles for targeted GPUs against our microbenchmark kernels as well as the official spec from NVIDIA. All `*.placeholder.json` will be replaced once we have finished the runs. The rtx2080 placeholder profile is for running / testing on CIMS.

## Sources

- Source for rtx2080ti L1/shared size and carveout options: https://docs.nvidia.com/cuda/turing-tuning-guide/index.html
- Source for a100 L1/shared size and carveout options: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- Source for h100 L1/shared size and carveout options: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- Source for b200 L1/shared size and carveout options: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- Same values for all four in one table: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
