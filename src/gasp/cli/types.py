from dataclasses import dataclass


@dataclass
class Args:
    inputs: list[tuple[str, str | None]]
    config: str | None
    reference: str | None
    repeats: int | None
    gpu: str | None
    verbose: bool
    trace: set[str] | None
    trace_file: str | None
    profile: str | None
    block_size: tuple[int, ...] | None
    grid_size: tuple[int, ...] | None
    problem_size: int | None
    loop_iterations: int | None
    include_dir: list[str]
    nvcc_flags: str | None
    run: bool
    program_args: str | None
    param: list[tuple[str, int]]
    footprint: int | None
    reuse: float | None
    save_metrics: str | None
    metrics: str | None
    override: list[str]
    json: bool
