# Matrix Multiplication Benchmarks

We attempt to demonstrate the accuracy of our speedup prediction through variations on a matrix multiplication kernel. The implementation gets progressively more efficient, so we expect our speedup prediction to match this pattern. The following kernels are included:

1. `mm_uncoalesced.cu`

In this kernel, the developer mistakenly switches indexing for rows and columns. This breaks memory coalescing.


2. `mm_double.cu`

In this kernel, the developer uses `double` (FP64) instead of `float` (FP32).

3. `mm_bounds.cu`

A bounds check is placed within an inner loop.

4. `mm_sfu.cu`

Each step of the inner loop scales the term by `rsqrtf(w)^2 * w`, which is 1, so it adds one SFU instruction (`MUFU.RSQ`) per step.

5. `mm_basic.cu`

A simple matrix multiplication kernel. No optimizations are applied, but it executes as expected and leverages L1/L2 caching.

6. `mm_tiled.cu`

A tiled matrix multiplication kernel that leverages shared memory.
