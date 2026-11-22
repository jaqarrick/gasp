# Citations

GASP builds on the work below. If you use GASP, please cite these too.

## GPUPerf

GASP's baseline model, and the starting point for GASPerf, is GPUPerf (`src/gasp/models/gpuperf/`):

Jaewoong Sim, Aniruddha Dasgupta, Hyesoon Kim, and Richard Vuduc. A Performance
Analysis Framework for Identifying Potential Benefits in GPGPU Applications. In
*Proc. 17th ACM SIGPLAN Symposium on Principles and Practice of Parallel
Programming (PPoPP)*, pages 11-22, 2012. https://doi.org/10.1145/2145816.2145819

```bibtex
@inproceedings{sim2012performance,
  author    = {Sim, Jaewoong and Dasgupta, Aniruddha and Kim, Hyesoon and Vuduc, Richard},
  title     = {A Performance Analysis Framework for Identifying Potential Benefits in {GPGPU} Applications},
  booktitle = {Proc. 17th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP)},
  pages     = {11--22},
  year      = {2012},
  doi       = {10.1145/2145816.2145819}
}
```

GPUPerf extends the MWP-CWP model of Hong and Kim:

Sunpyo Hong and Hyesoon Kim. An Analytical Model for a GPU Architecture with
Memory-Level and Thread-Level Parallelism Awareness. In *Proc. 36th Annual
International Symposium on Computer Architecture (ISCA)*, pages 152-163, 2009.
https://doi.org/10.1145/1555754.1555775

```bibtex
@inproceedings{hong2009analytical,
  author    = {Hong, Sunpyo and Kim, Hyesoon},
  title     = {An Analytical Model for a {GPU} Architecture with Memory-Level and Thread-Level Parallelism Awareness},
  booktitle = {Proc. 36th Annual Intl. Symposium on Computer Architecture (ISCA)},
  pages     = {152--163},
  year      = {2009},
  doi       = {10.1145/1555754.1555775}
}
```

## Faial

GASP's Pico analysis runs on Faial's `faial-cost`:

Tiago Cogumbreiro, Julien Lange, Dennis Liew, and Hannah Zicarelli. Memory access
protocols: certified data-race freedom for GPU kernels. *Formal Methods in System
Design*, 63(1):134-171, 2024. https://doi.org/10.1007/s10703-023-00415-0

```bibtex
@article{faial:fmsd23,
  title     = {Memory access protocols: certified data-race freedom for GPU kernels},
  author    = {Cogumbreiro, Tiago and Lange, Julien and Liew, Dennis and Zicarelli, Hannah},
  journal   = {Formal Methods in System Design},
  volume    = {63},
  number    = {1},
  pages     = {134--171},
  year      = {2024},
  publisher = {Springer},
  doi       = {10.1007/s10703-023-00415-0}
}
```
