# GASP Custom Benchmarks
This repo includes custom CUDA benchmarks we have prepared for development/experimentation


## Profiling

We can run a kernel to get the profile:

For example with the `vecadd.cu` program:

```sh
ncu --metrics gpu__time_duration.avg,smsp__inst_executed.sum,smsp__inst_executed_op_global_ld.sum,smsp__inst_executed_op_global_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,sm__warps_active.avg.pct_of_peak_sustained_active --csv ./vecadd 1000000
```

This outputs csv, which we can convert to JSON with `./scripts/csv_to_json.py`

## Comparison pairs

Each pair does the same work in two separate programs, so `gasp --run` measures one kernel at a time.

| Files | Difference in the poor kernel | Result |
|---|---|---|
| `vecadd_good.cu`, `vecadd_poor.cu` | Redundant loads: the store to `C` may alias `A` and `B`, so each is loaded twice | Same output |
| `poly_good.cu`, `poly_poor.cu` | Extra arithmetic: every power is multiplied out again instead of using Horner's rule | Same output within rounding |
| `divergence/branch_uniform.cu`, `divergence/branch_divergent.cu` | Divergence: the branch between a `__sinf` loop and a `__cosf` loop depends on `threadIdx.x`, so every warp runs both loops instead of one | Different outputs, same work per thread |

The polynomial's degree of 12 puts T_comp and T_mem of `poly_poor` about level on `--gpu rtx2080ti`, which is where the overlap term matters most. Other hardware profiles will move that balance.

```sh
gasp --gpu rtx2080ti --run benchmarks/vecadd/vecadd_good.cu benchmarks/vecadd/vecadd_poor.cu
gasp --gpu rtx2080ti --config benchmarks/vecadd/vecadd.toml
gasp --gpu rtx2080ti --run --config benchmarks/divergence/divergence.toml
```

## Writing a program for `gasp --run`

`--run` builds each program with `-DGASP_BLOCK_SIZE=<block> -DGASP_GRID_SIZE=<grid>` and runs it under `ncu`. It checks that every run launched that grid and block size, and stops if not.

The two macros are totals. For a launch with several dimensions (`block_size = [16, 16]`), `--run` also defines `GASP_BLOCK_DIM_X/Y/Z` and `GASP_GRID_DIM_X/Y/Z`, with missing dimensions set to 1. ncu only reports totals, so a program that launches another shape with the same totals passes the check.

- A program can take its launch from those macros, as these do:

  ```c
  #ifndef GASP_BLOCK_SIZE
  #define GASP_BLOCK_SIZE 256
  #endif
  #ifndef GASP_GRID_SIZE
  #define GASP_GRID_SIZE -1  // means ceil(N / block)
  #endif
  ```

- A program that computes its own launch also works, as long as the requested grid and block size match what it launches.
- `--problem-size` only sets the model's input. The program gets its size from `--program-args` (or `program_args` in a config), so keep the two equal. The pairs above default to `N = 1000000`, the same as GASP's default problem size.
- Only the kernel's first launch is profiled.
- Prefer one kernel per program. With several in one file, the macros and program arguments apply to the whole program, and the kernels you didn't select still run and can affect the one you measured.
