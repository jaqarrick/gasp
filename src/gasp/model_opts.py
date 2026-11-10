"""Model options: changes to the model's formulas, each off by default so the
model without them is GPUPerf as published.

    --model-opts mem-classes,amat-mem-l --model-opts dram-floor

Overrides (gasp.overrides) replace the model's inputs; these replace its
equations."""

from dataclasses import dataclass, fields

ALL = "all"


class ModelOptsError(ValueError):
    pass


@dataclass(frozen=True)
class ModelOptions:
    # T_mem sums one AMAT per access class: global loads, and global stores
    # with no L1 hits (GitHub #62)
    mem_classes: bool = False
    # Mem_L in MWP and CWP is AMAT, not the DRAM latency (GitHub #64)
    amat_mem_l: bool = False
    # DRAM bandwidth limits only the bytes that miss L2: the cap leaves ITMLP
    # and becomes a floor on T_mem
    dram_floor: bool = False
    # T_mem >= bytes that miss L1 / L2 bandwidth
    l2_floor: bool = False
    # T_mem >= global load sectors per SM / L1 sector rate
    l1_floor: bool = False
    # W_parallel >= shared loads and stores per SM x their cycles each
    shared_floor: bool = False


# the names --model-opts takes, in ModelOptions order
NAMES: tuple[str, ...] = tuple(f.name.replace("_", "-") for f in fields(ModelOptions))


def parse_model_opts(values: list[str]) -> ModelOptions:
    """`NAME[,NAME...]` strings as ModelOptions. `all` turns every option on."""
    chosen: set[str] = set()
    for value in values:
        for name in (part.strip() for part in value.split(",")):
            if not name:
                raise ModelOptsError(
                    f"expected NAME[,NAME...], such as mem-classes, got '{value}'"
                )
            if name == ALL:
                chosen.update(NAMES)
            elif name in NAMES:
                chosen.add(name)
            else:
                raise ModelOptsError(
                    f"unknown model option '{name}', expected one of: "
                    f"{', '.join([*NAMES, ALL])}"
                )
    return ModelOptions(**{name.replace("-", "_"): True for name in chosen})


def format_model_opts(options: ModelOptions) -> str:
    """Short label for tables: none, all, or the names that are on."""
    active = [name for name in NAMES if getattr(options, name.replace("-", "_"))]
    if not active:
        return "none"
    if len(active) == len(NAMES):
        return ALL
    return ", ".join(active)
