# Loop Counts

GASP multiplies each loop's instructions by its trip count, which it can't work out on its own yet. There are two ways to give it:

- `loop_counts = [64, 0, 0]` on a `[[kernel]]` entry gives each loop its own count, in the order the loops appear in the compiled code.
- `loop_iterations` (`--loop-iterations`) gives every loop the same count. `loop_counts` wins if both are set, and GASP warns.

The count is trips per entry into the loop and nested loops multiply. A count of 0 means the loop doesn't run.

The current limitation is that, because GASP analyzes compiled machine code, loops are often unrolled. So, you must be careful with this option:

One source loop can become several: `mm_basic`'s `k` loop is a main loop that does 16 values of `k` per trip, plus 4-step and 1-step loops for what's left. For K = 1024 that's `[64, 0, 0]`. `--verbose` prints the loops so you can work out each count, and a missing or wrong-length `loop_counts` prints them before GASP stops:

```
#  Source line                                              Inside  Insts  Loads/trip  Count
1  mm_basic.cu:13  sum += a[row * K + k] * b[k * N + col];  -       70     32 global   64
2  mm_basic.cu:13  sum += a[row * K + k] * b[k * N + col];  -       24     8 global    0
3  mm_basic.cu:13  sum += a[row * K + k] * b[k * N + col];  -       10     2 global    0
```

- **Source line:** where the loop starts in the source. GASP compiles with `-lineinfo`, which doesn't change the SASS. Several loops on the same line come from one source loop.
- **Loads/trip:** global and shared loads in one trip. Divide by the loads one source iteration does to get how many source iterations a trip covers: here 32 / 2 = 16, 8 / 2 = 4 and 2 / 2 = 1.
- **Inside:** the enclosing loop, for nested loops.

The counts depend on the problem size, and the loops can change with the architecture or nvcc version, so check the table when either changes.
