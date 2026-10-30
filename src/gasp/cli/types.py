from dataclasses import dataclass


@dataclass
class Args:
    input: str
    gpu: str | None
    verbose: bool
    profile: str | None
    block_size: int | None
    grid_size: int | None
    problem_size: int | None
    loop_iterations: int | None
    include_dir: list[str]
    nvcc_flags: str | None
    run: bool
    program_args: str | None
