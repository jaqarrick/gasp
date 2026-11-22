# Probes
These are CUDA kernels and their resulting SASS code, that is used during development for reverse engineering. 

## Reference kernels

`01`-`23` are general kernels, ordered from simple to complex, for reading
SASS rather than for testing one issue. The first comment in each file says
what to look for.

| Files | Topic |
|---|---|
| `01`-`06` | loads/stores, loops, integer ops, vector loads, FP16/BF16, bytes |
| `07`-`10` | branches, math functions, local memory, constant memory |
| `11`-`14` | shared memory, reductions, warp ops (vote, match, redux), atomics |
| `15`-`23` | histogram, scan, transpose, matmul, stencil, SpMV, bitonic sort, async copy, n-body |

```
for f in tests/fixtures/probes/[0-9]*.cu; do
    ./scripts/extract_sass.sh "$f" --arch "$ARCH" --sass -od tests/fixtures/probes/sass
done
```
