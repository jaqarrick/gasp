# Matrix Multiplication Benchmarks

`mm_basic` is a standard mm kernel with no optimizations: one thread per output
element, each reading a row of `a` and a column of `b` from global memory.

`mm_transposed` is its poor counterpart. It computes the same `c` from the same
inputs, but `b` is stored transposed, so the kernel reads `bt[col * n + k]`.
Neighbouring lanes are then `n` floats apart and every `b` load is uncoalesced
(one sector per column), where `mm_basic` reads `b` side by side.
`mm.toml` runs the pair.

The program takes the matrix side `n` as its argument (default 1024) and launches
a square 2D block of `GASP_BLOCK_SIZE` threads (default 256, i.e. 16x16; it must be
a perfect square). The grid is `ceil(n / side)` in each dimension, so the program
ignores `GASP_GRID_SIZE`. Set `grid_size` in the config to match what it launches,
for example 4096 for n = 1024 with 256-thread blocks.

```sh
gasp --gpu rtx2080ti --config benchmarks/custom/mm/mm_basic.toml
gasp --gpu rtx2080ti --config benchmarks/custom/mm/mm.toml        # good vs poor
```
