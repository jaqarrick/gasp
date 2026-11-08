# Matrix Multiplication Benchmarks

Both kernels compute `C (M x N) = A (M x K) x B (K x N)` with one thread per
element of `C`.

- `mm_basic` has no optimizations: each thread reads a row of `a` and a column
  of `b` from global memory.
- `mm_tiled` loads 16x16 tiles of `a` and `b` into shared memory and computes
  from there, so each global value is read once per tile instead of once per
  thread. `TILE_WIDTH` is fixed at 16, so it needs 16x16 blocks.

`mm.toml` compares the pair; `mm_basic.toml` runs `mm_basic` on its own.

```sh
gasp --gpu rtx2080ti --config benchmarks/mm/mm.toml
gasp --gpu rtx2080ti --config benchmarks/mm/mm_basic.toml
```

## Settings

The programs take `M N K bx by` as arguments (default `1024 1024 1024 16 16`)
and launch a `bx x by` block over a `ceil(N / bx) x ceil(M / by)` grid. They
ignore the `GASP_*` macros, so the config has to match what they launch:

- `block_size = [16, 16]` and `grid_size = [64, 64]` for `M = N = 1024`.
- `params = { M = 1024, N = 1024, K = 1024 }` gives Pico the kernel arguments.
  Without them Pico's cost depends on `M`, `N` and `K`, and GASP falls back.
- `program_args = ["1024", "1024", "1024", "16", "16"]` gives the program the
  same sizes for `--run`.
- `loop_counts` gives each compiled loop its trips for K = 1024: `[64]` for
  `mm_tiled`'s tile loop, and `[64, 0, 0]` for `mm_basic`, whose `k` loop
  compiles to a 16-step main loop (1024 / 16 = 64) and 4-step and 1-step
  remainder loops that have nothing left to do.

Change all four together.
