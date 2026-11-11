"""Overrides: replace a group of the model's static inputs with profiled (ncu)
or crude values, for the ablation study.

    --override cache=profiled --override coalescing=crude

Each group names the model inputs it replaces. ModelOverrides has one optional
field per replaceable input; predict() applies it in a single place."""

from dataclasses import dataclass

from gasp.common.constants import SECTORS_PER_LINE

STATIC = "static"
PROFILED = "profiled"
CRUDE = "crude"
ALL = "all"

# group -> the sources it accepts
GROUPS: dict[str, tuple[str, ...]] = {
    "cache": (STATIC, PROFILED),
    "coalescing": (STATIC, PROFILED, CRUDE),
    "bank": (STATIC, PROFILED, CRUDE),
    "counts": (STATIC, PROFILED),
}

# ncu metrics the profiled values come from (all in profiler.METRICS)
L1_HIT_RATE_PCT = "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct"
L2_READ_HITS = "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"
L2_READ_MISSES = "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"
GLOBAL_LOAD_SECTORS = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
GLOBAL_LOAD_REQUESTS = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
SHARED_LOAD_BANK_CONFLICTS = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
SHARED_STORE_BANK_CONFLICTS = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
INSTS_EXECUTED = "smsp__inst_executed.sum"
GLOBAL_LOADS_EXECUTED = "smsp__inst_executed_op_global_ld.sum"
GLOBAL_STORES_EXECUTED = "smsp__inst_executed_op_global_st.sum"
SFU_INSTS_EXECUTED = "smsp__inst_executed_pipe_xu.sum"
PROFILED_METRICS: tuple[str, ...] = (
    L1_HIT_RATE_PCT,
    L2_READ_HITS,
    L2_READ_MISSES,
    GLOBAL_LOAD_SECTORS,
    GLOBAL_LOAD_REQUESTS,
    SHARED_LOAD_BANK_CONFLICTS,
    SHARED_STORE_BANK_CONFLICTS,
    INSTS_EXECUTED,
    GLOBAL_LOADS_EXECUTED,
    GLOBAL_STORES_EXECUTED,
    SFU_INSTS_EXECUTED,
)


class OverrideError(ValueError):
    pass


@dataclass
class ModelOverrides:
    """Replacement values for models.inputs.ModelInputs, same field names.
    None keeps the static value. Counts are per warp."""

    # cache
    l1_hit_ratio: float | None = None
    l2_hit_ratio: float | None = None
    # coalescing
    average_trans_warp: float | None = None
    # bank
    bank_conflict_count: float | None = None
    # counts
    total_inst_count: float | None = None
    mem_inst_count: float | None = None
    sfu_inst_count: float | None = None


def parse_overrides(values: list[str]) -> dict[str, str]:
    """`GROUP=SOURCE` strings as {group: source}. `all=profiled` names every group."""
    choices: dict[str, str] = {}
    for value in values:
        group, separator, source = (part.strip() for part in value.partition("="))
        if not separator or not group or not source:
            raise OverrideError(
                f"expected GROUP=SOURCE, such as cache=profiled, got '{value}'"
            )
        if group == ALL:
            if source != PROFILED:
                raise OverrideError(f"all only takes {PROFILED}, got '{value}'")
            groups = list(GROUPS)
        elif group in GROUPS:
            groups = [group]
        else:
            raise OverrideError(
                f"unknown override group '{group}', expected one of: "
                f"{', '.join([*GROUPS, ALL])}"
            )
        for name in groups:
            if source not in GROUPS[name]:
                raise OverrideError(
                    f"{name} has no '{source}' source, expected one of: "
                    f"{', '.join(GROUPS[name])}"
                )
            if name in choices:
                raise OverrideError(f"override group {name} is given twice")
            choices[name] = source
    return choices


def uses_profiled(choices: dict[str, str]) -> bool:
    return PROFILED in choices.values()


def format_overrides(choices: dict[str, str]) -> str:
    """Short label for tables: none, cache=profiled, all=profiled."""
    active = {g: s for g, s in choices.items() if s != STATIC}
    if not active:
        return "none"
    if all(active.get(group) == PROFILED for group in GROUPS):
        return f"{ALL}={PROFILED}"
    return ", ".join(f"{g}={active[g]}" for g in GROUPS if g in active)


def _metric(metrics: dict[str, float], name: str, group: str) -> float:
    if name not in metrics:
        raise OverrideError(
            f"{group}={PROFILED} needs ncu metric {name}, which the metrics "
            "file doesn't have for this kernel"
        )
    return metrics[name]


def _cache(metrics: dict[str, float]) -> dict[str, float]:
    hits = _metric(metrics, L2_READ_HITS, "cache")
    lookups = hits + _metric(metrics, L2_READ_MISSES, "cache")
    return {
        "l1_hit_ratio": _metric(metrics, L1_HIT_RATE_PCT, "cache") / 100,
        # of the reads that missed L1
        "l2_hit_ratio": hits / lookups if lookups > 0 else 0.0,
    }


def _coalescing(metrics: dict[str, float]) -> dict[str, float]:
    sectors = _metric(metrics, GLOBAL_LOAD_SECTORS, "coalescing")
    requests = _metric(metrics, GLOBAL_LOAD_REQUESTS, "coalescing")
    if requests <= 0:
        return {"average_trans_warp": 1.0}
    # the same sectors -> lines rule as Pico's counts (predict.get_average_trans_warp)
    return {"average_trans_warp": max(1.0, sectors / requests / SECTORS_PER_LINE)}


def _bank(metrics: dict[str, float], total_warps: int) -> dict[str, float]:
    conflicts = _metric(metrics, SHARED_LOAD_BANK_CONFLICTS, "bank") + _metric(
        metrics, SHARED_STORE_BANK_CONFLICTS, "bank"
    )
    return {"bank_conflict_count": conflicts / total_warps}


def _counts(metrics: dict[str, float], total_warps: int) -> dict[str, float]:
    global_accesses = _metric(metrics, GLOBAL_LOADS_EXECUTED, "counts") + _metric(
        metrics, GLOBAL_STORES_EXECUTED, "counts"
    )
    return {
        "total_inst_count": _metric(metrics, INSTS_EXECUTED, "counts") / total_warps,
        "mem_inst_count": global_accesses / total_warps,
        "sfu_inst_count": _metric(metrics, SFU_INSTS_EXECUTED, "counts") / total_warps,
    }


# what the model assumes without Pico (predict's fallback)
CRUDE_VALUES: dict[str, dict[str, float]] = {
    "coalescing": {"average_trans_warp": 1.0},
    "bank": {"bank_conflict_count": 0.0},
}


def build_overrides(
    choices: dict[str, str], metrics: dict[str, float] | None, total_warps: int
) -> ModelOverrides:
    """The values each chosen group replaces, for one kernel.

    metrics: the kernel's ncu metrics (one value each), needed for `profiled`.
    total_warps turns whole-kernel counts into per-warp ones."""
    profiled = {
        "cache": _cache,
        "coalescing": _coalescing,
        "bank": lambda m: _bank(m, total_warps),
        "counts": lambda m: _counts(m, total_warps),
    }
    values: dict[str, float] = {}
    for group, source in choices.items():
        if source == CRUDE:
            values.update(CRUDE_VALUES[group])
        elif source == PROFILED:
            if metrics is None:
                raise OverrideError(f"{group}={PROFILED} needs a metrics file")
            values.update(profiled[group](metrics))
    return ModelOverrides(**values)
