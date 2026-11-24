# Custom hardware profiles

You can run GASP with your own profile (`--profile <path>`), as long as it matches the [format](../hardware/profiles/README.md#format). [`a100.json`](../hardware/profiles/a100.json) is an example.

Every field is required except the launch overhead terms (`launch_overhead_base_cycles`, `launch_overhead_per_thread_cycles`), since our first release does not model launch overhead. We only used them in experimentation.
