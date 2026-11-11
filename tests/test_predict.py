"""Tests for gasp.predict — Pico counts → average_trans_warp, T_GPU."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gasp.dependency import DependencyOutput
from gasp.models import MODELS
from gasp.models.gpuperf import GPUPerf
from gasp.models.gpuperf.types import HardwareParameters, LaunchParameters
from gasp.occupancy import OccupancyOutput
from gasp.overrides import ModelOverrides, build_overrides
from gasp.pico import PicoOutput
from gasp.predict import (
    PredictionInput,
    get_average_inst_latency,
    get_average_trans_warp,
    predict,
)


def pico(ua: int, count: int, error: bool = False) -> PicoOutput:
    return PicoOutput(
        uncoalesced_count=ua,
        bank_conflict_count=0,
        active_threads=32 * count,
        memory_accesses_count=count,
        error=error,
    )


@pytest.mark.parametrize(
    "ua,count,expected",
    [
        (12, 3, 1.0),  # vecadd: coalesced 32-bit, 4 sectors per access
        (24, 3, 2.0),  # coalesced 64-bit
        (96, 3, 8.0),  # stride 32: one sector per thread
        (3, 3, 1.0),  # under one line, floored at 1
    ],
)
def test_average_trans_warp(ua, count, expected):
    assert get_average_trans_warp(pico(ua, count), mem_inst_count=3) == expected


def test_average_trans_warp_pico_error_is_coalesced():
    assert get_average_trans_warp(pico(96, 3, error=True), mem_inst_count=3) == 1.0


def test_average_trans_warp_no_memory_accesses():
    assert get_average_trans_warp(pico(0, 0), mem_inst_count=3) == 1.0
    assert get_average_trans_warp(pico(12, 3), mem_inst_count=0) == 1.0


LAUNCH_BASE = 1000
LAUNCH_PER_THREAD = 0.01
SHARED_THROUGHPUT = 2.0
L1_CYCLES_PER_REQUEST = 0.1
L1_SECTORS_PER_CYCLE = 64.0


def prediction_input(mem_inst_count: int = 2, grid_size: int = 40) -> PredictionInput:
    return PredictionInput(
        dependency=DependencyOutput(
            ilp=2.0,
            mlp=1.0,
            total_inst_count=10,
            mem_inst_count=mem_inst_count,
            sfu_inst_count=0,
            sync_inst_count=0,
            total_latency=40.0,
        ),
        occupancy=OccupancyOutput(
            active_warps_per_sm=32,
            total_warps=320,
            active_sms=10,
            num_reps=1,
            mwp=32.0,
            warps_per_block=8,
            blocks_per_sm=4,
        ),
        pico=pico(12, 3),
        hardware=HardwareParameters(
            warp_size=32,
            simd_width=32,
            sfu_width=16,
            departure_delay=6,
            sync_factor=24,
            dram_latency=300,
            l1_hit_latency=30,
            l2_hit_latency=200,
            l1_shared_array_bytes=96 * 1024,
            shared_mem_carveouts_bytes=[32 * 1024, 64 * 1024],
            l2_cache_size_bytes=4 * 1024 * 1024,
            bank_penalty=2,
            launch_overhead_base=LAUNCH_BASE,
            launch_overhead_per_thread=LAUNCH_PER_THREAD,
            dram_bandwidth=450,
            shared_mem_throughput=SHARED_THROUGHPUT,
            # small enough that GASPerf's L1 limit never binds unless a test
            # raises it
            l1_cycles_per_request=L1_CYCLES_PER_REQUEST,
            l1_sectors_per_cycle=L1_SECTORS_PER_CYCLE,
        ),
        launch=LaunchParameters(
            block_size=256,
            grid_size=grid_size,
            problem_size=10240,
            registers_per_thread=12,
            shared_mem_per_block_bytes=0,
        ),
    )


def test_predict_t_gpu_adds_launch_overhead_to_eq_1():
    p = predict(prediction_input(), logger=MagicMock())

    assert p.t_exec == pytest.approx(p.t_comp + p.t_mem - p.t_overlap)
    overhead = LAUNCH_BASE + LAUNCH_PER_THREAD * 256 * 40
    assert p.launch_overhead == pytest.approx(overhead)
    assert p.t_gpu == pytest.approx(p.t_exec + overhead)


def test_predict_launch_overhead_grows_with_threads():
    small = predict(prediction_input(grid_size=40), logger=MagicMock())
    large = predict(prediction_input(grid_size=80), logger=MagicMock())

    assert large.launch_overhead - small.launch_overhead == pytest.approx(
        LAUNCH_PER_THREAD * 256 * 40
    )


def test_predict_has_no_one_thread_baseline():
    p = predict(prediction_input(), logger=MagicMock())

    assert not hasattr(p, "t_one_thread")
    assert not hasattr(p, "speedup")


def test_predict_bank_conflicts_scale_by_warps_per_sm():
    base = prediction_input()
    conflicted = prediction_input()
    conflicted.pico = replace(conflicted.pico, bank_conflict_count=31)

    clean = predict(base, logger=MagicMock())
    banked = predict(conflicted, logger=MagicMock())

    # 31 extra accesses per warp x 320 warps / 10 SMs x penalty 2
    assert banked.t_comp - clean.t_comp == pytest.approx(31 * 320 / 10 * 2)


def test_predict_no_divergence_no_o_cfdiv():
    p = predict(prediction_input(), logger=MagicMock())

    assert p.o_cfdiv == 0


def test_predict_o_cfdiv_moves_divergent_instructions_out_of_w_parallel():
    base = prediction_input()
    divergent = prediction_input()
    divergent.dependency = replace(divergent.dependency, divergent_inst_count=4)

    clean = predict(base, logger=MagicMock())
    split = predict(divergent, logger=MagicMock())

    # 4 of the 10 instructions per warp x 320 warps / 10 SMs x latency 4 / ITILP
    assert split.o_cfdiv == pytest.approx(4 * 320 / 10 * 4 / clean.itilp)
    # same cost per instruction as W_parallel, so T_comp is unchanged
    assert split.t_comp == pytest.approx(clean.t_comp)


def test_predict_more_memory_instructions_raise_t_mem():
    fewer = predict(prediction_input(mem_inst_count=2), logger=MagicMock())
    more = predict(prediction_input(mem_inst_count=4), logger=MagicMock())

    assert more.t_mem > fewer.t_mem


# ---------------------------------------------------------------------------
# two-level AMAT (each latency end-to-end, so no lookup is added on a miss)
# ---------------------------------------------------------------------------

L1_LAT, L2_LAT, DRAM_LAT, DELTA = 30, 200, 300, 6


def amat(h1: float, h2: float, average_trans_warp: float = 1.0) -> float:
    h = SimpleNamespace(
        l1_hit_latency=L1_LAT,
        l2_hit_latency=L2_LAT,
        dram_latency=DRAM_LAT,
        departure_delay=DELTA,
    )
    o = SimpleNamespace(
        l1_hit_ratio=h1, l2_hit_ratio=h2, average_trans_warp=average_trans_warp
    )
    return GPUPerf(MagicMock(), o, MagicMock(), h, MagicMock())._get_amat()


@pytest.mark.parametrize(
    "h1,h2,expected",
    [
        (0.0, 0.0, DRAM_LAT),  # all misses: DRAM only, no L1 lookup on top
        (1.0, 0.0, L1_LAT),
        (1.0, 1.0, L1_LAT),
        (0.0, 1.0, L2_LAT),
        (0.5, 0.0, (L1_LAT + DRAM_LAT) / 2),
    ],
)
def test_amat_two_level(h1, h2, expected):
    assert amat(h1, h2) == pytest.approx(expected)


def test_amat_uncoalesced_term_only_on_dram():
    assert amat(0.0, 0.0, average_trans_warp=3.0) == pytest.approx(DRAM_LAT + 2 * DELTA)
    assert amat(1.0, 0.0, average_trans_warp=3.0) == pytest.approx(L1_LAT)


# ---------------------------------------------------------------------------
# working set -> hit ratios
# ---------------------------------------------------------------------------


def vecadd_input(accesses: int, reusable: int) -> PredictionInput:
    i = prediction_input(mem_inst_count=accesses)
    i.dependency.global_bytes_per_thread = 12  # 2 loads + 1 store of 4 B
    i.dependency.global_access_count = accesses
    i.dependency.reusable_access_count = reusable
    return i


def test_predict_working_set_scales_per_thread_bytes():
    p = predict(vecadd_input(accesses=3, reusable=0), logger=MagicMock())

    assert p.working_set_per_sm_bytes == 12 * 256 * 4  # x block_size x blocks/SM
    assert p.working_set_total_bytes == 12 * 256 * 40  # x block_size x grid_size


def test_predict_streaming_kernel_misses_though_it_fits():
    # vecadd good: 12 KB per SM fits in L1, but every access is a first touch
    p = predict(vecadd_input(accesses=3, reusable=0), logger=MagicMock())

    assert p.reuse_fraction == 0.0
    assert p.l1_hit_ratio == 0.0
    assert p.l2_hit_ratio == 0.0
    assert p.amat == pytest.approx(DRAM_LAT)


def test_predict_repeat_accesses_hit_l1():
    # vecadd poor: 6 accesses to 3 addresses, the repeats hit (ncu: 47% L1)
    p = predict(vecadd_input(accesses=6, reusable=3), logger=MagicMock())

    assert p.reuse_fraction == 0.5
    assert p.l1_hit_ratio == 0.5
    assert p.l2_hit_ratio == 0.0  # what L1 misses is all first touches
    assert p.amat == pytest.approx((L1_LAT + DRAM_LAT) / 2)


def test_predict_no_global_accesses_has_no_hits():
    p = predict(prediction_input(), logger=MagicMock())
    assert p.reuse_fraction == 0.0
    assert p.amat == pytest.approx(DRAM_LAT)


def test_predict_traces_cache_line():
    logger = MagicMock()
    predict(vecadd_input(accesses=6, reusable=3), logger=logger)

    cache_trace = next(
        c.args[1] for c in logger.trace.call_args_list if c.args[1].startswith("cache:")
    )
    assert cache_trace == (
        "cache: L1 65,536 B (carveout 32,768 B); r = 0.5000 (per thread 3/6 reusable); L1 cap 1.000 (resident); "
        "L2 cap 1.000 (resident), concurrent 122,880 B cap 1.000 (resident); "
        "h1 0.500, h2 0.000, h2 concurrent 0.000"
    )


# ---------------------------------------------------------------------------
# footprint: whole-kernel reuse
# ---------------------------------------------------------------------------

THREADS = 256 * 40  # prediction_input's block_size x grid_size


def footprint_input(bytes_per_thread: int, footprint: int) -> PredictionInput:
    i = vecadd_input(accesses=3, reusable=0)  # per-thread count says no reuse
    i.dependency.global_bytes_accessed_per_thread = bytes_per_thread
    i.dependency.global_bytes_per_thread = bytes_per_thread  # no repeats
    i.footprint_bytes = footprint
    return i


def test_footprint_streaming_has_no_reuse():
    p = predict(footprint_input(12, 12 * THREADS), logger=MagicMock())
    assert p.reuse_fraction == 0.0
    assert p.amat == pytest.approx(DRAM_LAT)


def test_footprint_sees_reuse_between_threads():
    # threads read 1 KB each, but share 8 KB in all: mm-style sharing that the
    # per-thread count (reusable=0) can't see
    p = predict(footprint_input(1024, 8 * 1024), logger=MagicMock())

    assert p.reuse_fraction == pytest.approx(1 - 8 * 1024 / (1024 * THREADS))
    # L1 (64 KB) can't hold the 4 blocks' 1 MB; the 8 KB footprint fits in L2
    # (4 MB), so what L1 drops hits L2 and only first touches go to DRAM
    assert p.working_set_total_bytes == 8 * 1024
    dram_fraction = (1 - p.l1_hit_ratio) * (1 - p.l2_hit_ratio)
    assert dram_fraction == pytest.approx(1 - p.reuse_fraction)


def test_footprint_larger_than_l2_uses_capacity():
    # 16 MB footprint, 4 MB L2: Jarmusch's streaming branch, cap2 = C / W = 0.25
    l2 = 4 * 1024 * 1024
    footprint = 4 * l2
    p = predict(footprint_input(1024 * 1024, footprint), logger=MagicMock())

    assert p.working_set_total_bytes == footprint
    r = p.reuse_fraction
    # L1 = 96 KB array - 32 KB carveout, vs the 4 blocks' bytes
    cap1 = 64 * 1024 / (1024 * 1024 * 256 * 4)
    reusable_l1_misses = r * (1 - cap1)
    assert p.l2_hit_ratio == pytest.approx(
        reusable_l1_misses * 0.25 / (1 - p.l1_hit_ratio)
    )


def test_no_footprint_falls_back_to_per_thread_reuse():
    i = footprint_input(1024, 8 * 1024)
    i.footprint_bytes = None
    p = predict(i, logger=MagicMock())
    assert p.reuse_fraction == 0.0  # reusable=0 from the per-thread count


# ---------------------------------------------------------------------------
# --reuse: L1 hit ratio given
# ---------------------------------------------------------------------------


def test_reuse_sets_l1_hit_ratio():
    # mm_basic-style: L1 capacity term sees far too much, the block reuses 15/16
    i = footprint_input(1024 * 1024, 8 * 1024)
    i.block_reuse = 15 / 16
    p = predict(i, logger=MagicMock())

    assert p.l1_hit_ratio == pytest.approx(15 / 16)
    # 8 KB footprint fits the 4 MB L2: the rest of the reuse all hits there
    assert p.l2_hit_ratio == pytest.approx((p.reuse_fraction - 15 / 16) / (1 / 16))


def test_reuse_zero_means_no_l1_hits():
    i = vecadd_input(accesses=6, reusable=3)  # per thread would say h1 = 0.5
    i.block_reuse = 0.0
    p = predict(i, logger=MagicMock())

    assert p.l1_hit_ratio == 0.0
    # the per-thread repeats L1 no longer catches go to L2 (resident here)
    assert p.l2_hit_ratio == pytest.approx(0.5)


def test_reuse_shows_in_trace():
    logger = MagicMock()
    i = vecadd_input(accesses=6, reusable=3)
    i.block_reuse = 0.25
    predict(i, logger=logger)

    cache_trace = next(
        c.args[1] for c in logger.trace.call_args_list if c.args[1].startswith("cache:")
    )
    assert "L1 from --reuse 0.25" in cache_trace
    assert "h1 0.250" in cache_trace


# ---------------------------------------------------------------------------
# L1 size: what the shared-memory carveout leaves of the array
# ---------------------------------------------------------------------------


def test_l1_size_without_shared_memory_keeps_smallest_carveout():
    p = predict(prediction_input(), logger=MagicMock())  # 96 KB, splits 32/64 KB

    assert p.shared_mem_carveout_bytes == 32 * 1024
    assert p.l1_cache_size_bytes == 64 * 1024


def test_l1_size_shrinks_when_blocks_need_more_shared_memory():
    i = prediction_input()  # 4 blocks per SM
    i.launch = replace(i.launch, shared_mem_per_block_bytes=12 * 1024)  # 48 KB
    p = predict(i, logger=MagicMock())

    assert p.shared_mem_carveout_bytes == 64 * 1024
    assert p.l1_cache_size_bytes == 32 * 1024


def test_l1_size_shows_in_trace():
    logger = MagicMock()
    predict(prediction_input(), logger=logger)

    cache_trace = next(
        c.args[1] for c in logger.trace.call_args_list if c.args[1].startswith("cache:")
    )
    assert "L1 65,536 B (carveout 32,768 B)" in cache_trace


# overrides


def overridden(**values: float):
    base = prediction_input()
    base.overrides = ModelOverrides(**values)
    return predict(base, logger=MagicMock())


def test_empty_overrides_change_nothing():
    plain = predict(prediction_input(), logger=MagicMock())

    assert overridden() == plain


def test_cache_override_replaces_only_the_hit_ratios():
    plain = predict(prediction_input(), logger=MagicMock())
    p = overridden(l1_hit_ratio=0.25, l2_hit_ratio=0.5)

    assert (p.l1_hit_ratio, p.l2_hit_ratio) == (0.25, 0.5)
    # h1 x L1 + (1 - h1) x (h2 x L2 + (1 - h2) x DRAM), coalesced
    assert p.amat == pytest.approx(0.25 * 30 + 0.75 * (0.5 * 200 + 0.5 * 300))
    assert p.t_comp == plain.t_comp
    assert (p.itilp, p.cwp, p.itmlp) == (plain.itilp, plain.cwp, plain.itmlp)
    assert p.average_trans_warp == plain.average_trans_warp


def test_cache_override_wins_over_reuse():
    base = prediction_input()
    base.block_reuse = 0.9
    base.overrides = ModelOverrides(l1_hit_ratio=0.25, l2_hit_ratio=0.5)

    assert predict(base, logger=MagicMock()).l1_hit_ratio == 0.25


def test_coalescing_override_replaces_only_average_trans_warp():
    plain = predict(prediction_input(), logger=MagicMock())
    p = overridden(average_trans_warp=8.0)

    assert p.average_trans_warp == 8.0
    assert p.t_comp == plain.t_comp
    assert (p.l1_hit_ratio, p.l2_hit_ratio) == (plain.l1_hit_ratio, plain.l2_hit_ratio)


def test_bank_override_adds_only_bank_overhead():
    plain = predict(prediction_input(), logger=MagicMock())
    p = overridden(bank_conflict_count=1.5)

    # 1.5 extra accesses per warp x 320 warps / 10 SMs x penalty 2
    assert p.t_comp - plain.t_comp == pytest.approx(1.5 * 320 / 10 * 2)
    assert p.t_mem == plain.t_mem


def test_counts_override_keeps_the_static_instruction_latency():
    plain = predict(prediction_input(), logger=MagicMock())
    # twice the static total of 10, with no serial overheads in this input
    p = overridden(total_inst_count=20.0)

    assert p.itilp == plain.itilp
    assert p.t_comp == pytest.approx(2 * plain.t_comp)


def test_crude_overrides_match_the_pico_fallback():
    uncoalesced = prediction_input()
    uncoalesced.pico = replace(pico(96, 3), bank_conflict_count=31)
    fallback = prediction_input()
    fallback.pico = replace(uncoalesced.pico, error=True)
    uncoalesced.overrides = build_overrides(
        {"coalescing": "crude", "bank": "crude"}, None, total_warps=320
    )

    assert predict(uncoalesced, logger=MagicMock()) == predict(
        fallback, logger=MagicMock()
    )


def test_each_override_is_traced():
    base = prediction_input()
    base.pico = pico(96, 3)
    base.overrides = ModelOverrides(average_trans_warp=1.0)
    logger = MagicMock()

    predict(base, logger=logger)

    logger.trace.assert_any_call("predict", "override average_trans_warp: static 8 → 1")


# ---------------------------------------------------------------------------
# SFU instructions: left out of #insts, charged only in O_SFU (Eq. 9)
# ---------------------------------------------------------------------------

FP_LAT, SFU_LAT = 4, 15


def sfu_input(total: int, sfu: int, mem: int = 2) -> PredictionInput:
    base = prediction_input(mem_inst_count=mem)
    base.dependency = replace(
        base.dependency,
        total_inst_count=total,
        sfu_inst_count=sfu,
        total_latency=(total - sfu) * FP_LAT + sfu * SFU_LAT,
        sfu_latency=sfu * SFU_LAT,
    )
    return base


def test_average_inst_latency_leaves_sfu_out():
    d = sfu_input(total=10, sfu=2).dependency

    assert get_average_inst_latency(d) == FP_LAT


def test_average_inst_latency_of_sfu_only_kernel_keeps_all():
    d = sfu_input(total=2, sfu=2, mem=0).dependency

    assert get_average_inst_latency(d) == SFU_LAT


def test_sfu_instructions_are_not_in_w_parallel():
    # 8 non-SFU instructions per warp; F_SFU = max(2/8 - 16/32, 0) = 0
    p = predict(sfu_input(total=10, sfu=2), logger=MagicMock())

    # ITILP = min(2 x 32, 4 / (32 / 32)) = 4, so each instruction costs 1
    assert p.itilp == FP_LAT
    assert p.t_comp == pytest.approx(8 * 320 / 10)


def test_f_sfu_uses_the_non_sfu_count():
    # 6 non-SFU, 4 SFU: F_SFU = 4/6 - 16/32 = 1/6
    p = predict(sfu_input(total=10, sfu=4), logger=MagicMock())

    w_parallel = 6 * 320 / 10
    o_sfu = 4 * 320 / 10 * (32 / 16) * (1 / 6)
    assert p.t_comp == pytest.approx(w_parallel + o_sfu)


def test_cwp_uses_the_non_sfu_count():
    # 40 non-SFU: Comp_cycles = 40 x 4 / 2, CWP = 300 / (80 / 2) = 7.5, under N
    with_sfu = predict(sfu_input(total=50, sfu=10), logger=MagicMock())
    without = predict(sfu_input(total=40, sfu=0), logger=MagicMock())

    assert with_sfu.cwp == pytest.approx(7.5)
    assert without.cwp == pytest.approx(7.5)


def test_sfu_only_kernel_is_all_o_sfu():
    # no other instructions: W_parallel 0, F_SFU = min(max(1 - 0.5, 0), 1)
    p = predict(sfu_input(total=2, sfu=2, mem=0), logger=MagicMock())

    assert p.t_comp == pytest.approx(2 * 320 / 10 * (32 / 16) * 0.5)


def test_counts_override_subtracts_sfu_inside_the_model():
    # ncu's inst_executed counts SFU instructions too
    base = sfu_input(total=8, sfu=0)
    base.overrides = ModelOverrides(total_inst_count=10.0, sfu_inst_count=2.0)
    p = predict(base, logger=MagicMock())

    assert p.t_comp == pytest.approx(8 * 320 / 10)


# ---------------------------------------------------------------------------
# models: GASPerf starts as a copy of GPUPerf
# ---------------------------------------------------------------------------


def test_models_are_gpuperf_then_gasperf():
    assert list(MODELS) == ["GPUPerf", "GASPerf"]


def test_prediction_names_its_model():
    base = prediction_input()
    base.model = "GASPerf"

    assert predict(base, logger=MagicMock()).model == "GASPerf"
    assert predict(prediction_input(), logger=MagicMock()).model == "GPUPerf"


@pytest.mark.parametrize(
    "build",
    [
        prediction_input,
        lambda: sfu_input(total=10, sfu=4),
        lambda: vecadd_input(accesses=3, reusable=1),
    ],
)
def test_gasperf_matches_gpuperf(build):
    """Holds for kernels without shared-memory accesses, whose L1 time is under
    T_mem and whose data fits in L2 either way. GASPerf reports its L1 time,
    which GPUPerf leaves at 0."""
    gpuperf = build()
    gasperf = build()
    gasperf.model = "GASPerf"

    expected = predict(gpuperf, logger=MagicMock())
    got = predict(gasperf, logger=MagicMock())
    assert got.l1_time > 0
    assert got == replace(expected, model="GASPerf", l1_time=got.l1_time)


# ---------------------------------------------------------------------------
# GASPerf: shared-memory pipe limit (O_shmem)
# ---------------------------------------------------------------------------


def shared_input(loads: int, stores: int = 0, model: str = "GASPerf"):
    # prediction_input: 10 instructions per warp, each costing 1 cycle of
    # W_parallel (latency 4 / ITILP 4), 320 warps on 10 SMs: issue time 320
    base = prediction_input()
    base.dependency = replace(
        base.dependency,
        shared_load_inst_count=loads,
        shared_store_inst_count=stores,
    )
    base.model = model
    return base


def test_shared_pipe_under_issue_time_adds_nothing():
    # 4 shared per warp x 32 warps/SM x 2 cycles = 256, under issue time 320
    p = predict(shared_input(loads=4), logger=MagicMock())
    gpuperf = predict(shared_input(loads=4, model="GPUPerf"), logger=MagicMock())

    assert p.shared_pipe_time == pytest.approx(4 * 320 / 10 * SHARED_THROUGHPUT)
    assert p.o_shmem == 0
    assert p.t_comp == pytest.approx(gpuperf.t_comp)


def test_shared_pipe_over_issue_time_sets_t_comp():
    # 8 loads + 1 store per warp x 32 x 2 = 576 cycles, issue time 320
    p = predict(shared_input(loads=8, stores=1), logger=MagicMock())
    gpuperf = predict(
        shared_input(loads=8, stores=1, model="GPUPerf"), logger=MagicMock()
    )

    issue = 10 * 320 / 10
    assert p.shared_pipe_time == pytest.approx(9 * 320 / 10 * SHARED_THROUGHPUT)
    assert p.o_shmem == pytest.approx(p.shared_pipe_time - issue)
    # no other serial overheads here, so T_comp is the pipe time
    assert p.t_comp == pytest.approx(p.shared_pipe_time)
    assert gpuperf.t_comp == pytest.approx(issue)


def test_shared_pipe_counts_o_cfdiv_as_issue_time():
    # divergent instructions still take issue slots: moving 4 of the 10 into
    # O_CFdiv leaves the comparison, and so O_shmem, unchanged
    clean = predict(shared_input(loads=8, stores=1), logger=MagicMock())
    divergent = shared_input(loads=8, stores=1)
    divergent.dependency = replace(divergent.dependency, divergent_inst_count=4)
    split = predict(divergent, logger=MagicMock())

    assert split.o_cfdiv > 0
    assert split.o_shmem == pytest.approx(clean.o_shmem)


def test_gpuperf_ignores_shared_counts():
    p = predict(shared_input(loads=8, stores=1, model="GPUPerf"), logger=MagicMock())

    assert (p.shared_pipe_time, p.o_shmem) == (0, 0)


# ---------------------------------------------------------------------------
# GASPerf: L1 throughput limit on T_mem
# ---------------------------------------------------------------------------


def l1_input(
    cycles_per_request: float, sectors_per_cycle: float = 2.0, model="GASPerf"
):
    # prediction_input: 2 global requests per warp x 320 warps / 10 SMs = 64 per
    # SM; T_mem from latency = 64 / ITMLP 29 x AMAT 300 = 662
    base = prediction_input()
    base.hardware = replace(
        base.hardware,
        l1_cycles_per_request=cycles_per_request,
        l1_sectors_per_cycle=sectors_per_cycle,
    )
    base.model = model
    return base


def test_l1_limit_under_latency_time_changes_nothing():
    gasperf = predict(l1_input(cycles_per_request=2.48), logger=MagicMock())
    gpuperf = predict(
        l1_input(cycles_per_request=2.48, model="GPUPerf"), logger=MagicMock()
    )

    assert gasperf.l1_time == pytest.approx(64 * 2.48)
    assert gasperf.t_mem == pytest.approx(gpuperf.t_mem)


def test_l1_limit_over_latency_time_sets_t_mem():
    gasperf = predict(l1_input(cycles_per_request=50.0), logger=MagicMock())
    gpuperf = predict(
        l1_input(cycles_per_request=50.0, model="GPUPerf"), logger=MagicMock()
    )

    assert gasperf.l1_time == pytest.approx(64 * 50.0)
    assert gasperf.t_mem == pytest.approx(64 * 50.0)
    assert gpuperf.t_mem == pytest.approx(64 * 300 / 29)
    assert gpuperf.l1_time == 0


def test_l1_wide_requests_are_charged_their_sectors():
    # 4 lines per request = 16 sectors at 2 per cycle = 8 cycles, over the
    # fixed 2.48 a request costs
    base = l1_input(cycles_per_request=2.48)
    base.overrides = ModelOverrides(average_trans_warp=4.0)
    p = predict(base, logger=MagicMock())

    assert p.l1_time == pytest.approx(64 * 8.0)


# ---------------------------------------------------------------------------
# GASPerf: L2 hit ratio against the blocks resident at the same time
# ---------------------------------------------------------------------------

L2_BYTES = 4 * 1024 * 1024


def concurrent_input(model: str = "GASPerf") -> PredictionInput:
    # 16 MB footprint, 4 MB L2: GPUPerf's capacity term is 4 / 16 = 0.25. The
    # fixture's 4 blocks x 10 SMs = 40 resident blocks of a 400-block grid use
    # 1.6 MB, which fits, so GASPerf's term is 1
    i = footprint_input(1024 * 1024, 4 * L2_BYTES)
    i.launch = replace(i.launch, grid_size=400)
    i.model = model
    return i


def test_l2_hit_ratio_from_concurrent_blocks():
    gasperf = predict(concurrent_input(), logger=MagicMock())
    gpuperf = predict(concurrent_input(model="GPUPerf"), logger=MagicMock())

    assert gasperf.l1_hit_ratio == gpuperf.l1_hit_ratio
    assert gpuperf.l2_hit_ratio > 0
    assert gasperf.l2_hit_ratio == pytest.approx(gpuperf.l2_hit_ratio / 0.25)
    assert gasperf.amat < gpuperf.amat


def test_l2_concurrent_shows_in_trace():
    logger = MagicMock()
    predict(concurrent_input(), logger=logger)

    cache_trace = next(
        c.args[1] for c in logger.trace.call_args_list if c.args[1].startswith("cache:")
    )
    assert "concurrent 1,677,722 B cap 1.000 (resident)" in cache_trace


def test_cache_override_replaces_the_concurrent_l2_hit_ratio_too():
    base = concurrent_input()
    base.overrides = ModelOverrides(l1_hit_ratio=0.25, l2_hit_ratio=0.5)
    p = predict(base, logger=MagicMock())

    assert p.l2_hit_ratio == 0.5
