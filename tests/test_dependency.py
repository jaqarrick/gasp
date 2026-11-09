"""Tests for gasp.dependency — register DAG, ILP/MLP, loop weighting, counts."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gasp.cfg import CFGEdge, CFGNode, ControlFlowGraph, filter_cfg, gen_cfg_from_raw
from gasp.dependency import (
    LatencyTable,
    LoopCountMismatchError,
    LoopIterationsRequiredError,
    analyze,
    count_arrays,
    describe_loops,
    find_loops,
    get_address_groups,
    get_ilp_mlp_for_node,
)
from gasp.logger import Logger
from gasp.parser import get_parser
from gasp.parser.types import OpCategory, SASSInstruction

CFG_DIR = Path(__file__).parent / "fixtures" / "cfg"
SGEMM_KERNEL = "_Z9mysgemmNTPKfiS0_iPfiiff"

FP32_LAT = 4
SFU_LAT = 20
LATENCY = LatencyTable(by_category={OpCategory.SFU: SFU_LAT}, default=FP32_LAT)


@pytest.fixture
def logger():
    return MagicMock(spec=Logger)


def instr(opcode, category, src=(), dst=()) -> SASSInstruction:
    return SASSInstruction(
        opcode=opcode,
        category=category,
        predicate=None,
        src_registers=set(src),
        dest_registers=set(dst),
    )


def ffma(dst, *src):
    return instr("FFMA", OpCategory.FP32, src, [dst])


def ldg(dst, addr):
    return instr("LDG", OpCategory.GLOBAL_LOAD, [addr], [dst])


def make_cfg(node_ids, edges) -> ControlFlowGraph:
    """First node is the entry; edges are (source, target) pairs."""
    return ControlFlowGraph(
        nodes={n: CFGNode(id=n, label="") for n in node_ids},
        edges=[CFGEdge(s, None, None, t, None, None, None) for s, t in edges],
    )


def one_inst_per_node(node_ids):
    return {n: [ffma(f"R{i}")] for i, n in enumerate(node_ids)}


# ---------------------------------------------------------------------------
# get_ilp_mlp_for_node
# ---------------------------------------------------------------------------


def test_ilp_independent_instructions():
    block = [ffma("R0"), ffma("R1"), ffma("R2"), ffma("R3")]
    ilp, _, total_latency = get_ilp_mlp_for_node(block, LATENCY)
    assert ilp == 4
    assert total_latency == 4 * FP32_LAT


def test_ilp_dependent_chain():
    block = [ffma("R0"), ffma("R1", "R0"), ffma("R2", "R1")]
    ilp, _, _ = get_ilp_mlp_for_node(block, LATENCY)
    assert ilp == 1


def test_ilp_single_instruction():
    # no edges: the critical path is the instruction's own latency, not 0
    ilp, _, _ = get_ilp_mlp_for_node([ffma("R0")], LATENCY)
    assert ilp == 1


def test_edge_weighted_by_producer_latency():
    # MUFU (20) -> FADD (4), plus an independent FADD
    # critical path = 20 + 4 = 24; total = 28
    block = [
        instr("MUFU", OpCategory.SFU, ["R0"], ["R1"]),
        ffma("R2", "R1"),
        ffma("R3"),
    ]
    ilp, _, _ = get_ilp_mlp_for_node(block, LATENCY)
    assert ilp == pytest.approx(28 / 24)


def test_dependency_uses_last_writer():
    # R0 is rewritten by an independent instruction; the reader depends on the
    # second write only, so the chain is 2 long, not 3
    block = [ffma("R0"), ffma("R1", "R0"), ffma("R0"), ffma("R2", "R0")]
    ilp, _, _ = get_ilp_mlp_for_node(block, LATENCY)
    assert ilp == 2


def test_mlp_independent_loads():
    block = [ldg("R4", "R2"), ldg("R5", "R3"), ffma("R6", "R4", "R5")]
    _, mlp, _ = get_ilp_mlp_for_node(block, LATENCY)
    assert mlp == 2


def test_mlp_pointer_chase():
    block = [ldg("R2", "R2"), ldg("R2", "R2"), ldg("R2", "R2")]
    _, mlp, _ = get_ilp_mlp_for_node(block, LATENCY)
    assert mlp == 1


def test_mlp_no_loads():
    _, mlp, _ = get_ilp_mlp_for_node([ffma("R0")], LATENCY)
    assert mlp == -1


# ---------------------------------------------------------------------------
# analyze: loop weighting
# ---------------------------------------------------------------------------


def test_no_loops(logger):
    nodes = ["entry", "a", "exit"]
    cfg = make_cfg(nodes, [("entry", "a"), ("a", "exit")])
    out = analyze(
        cfg,
        one_inst_per_node(nodes),
        loop_iterations=10,
        latency=LATENCY,
        logger=logger,
    )
    assert out.total_inst_count == 3


def test_self_loop(logger):
    nodes = ["entry", "body", "exit"]
    cfg = make_cfg(nodes, [("entry", "body"), ("body", "body"), ("body", "exit")])
    out = analyze(
        cfg,
        one_inst_per_node(nodes),
        loop_iterations=10,
        latency=LATENCY,
        logger=logger,
    )
    assert out.total_inst_count == 1 + 10 + 1


def test_loop_without_loop_iterations_raises(logger):
    nodes = ["entry", "body", "exit"]
    cfg = make_cfg(nodes, [("entry", "body"), ("body", "body"), ("body", "exit")])
    with pytest.raises(LoopIterationsRequiredError) as exc:
        analyze(
            cfg,
            one_inst_per_node(nodes),
            loop_iterations=None,
            latency=LATENCY,
            logger=logger,
        )
    assert exc.value.loop_headers == ["body"]


def test_no_loops_without_loop_iterations(logger):
    nodes = ["entry", "a", "exit"]
    cfg = make_cfg(nodes, [("entry", "a"), ("a", "exit")])
    out = analyze(
        cfg,
        one_inst_per_node(nodes),
        loop_iterations=None,
        latency=LATENCY,
        logger=logger,
    )
    assert out.total_inst_count == 3


def test_loop_body_spans_blocks(logger):
    # head -> body -> head: both blocks repeat
    nodes = ["entry", "head", "body", "exit"]
    cfg = make_cfg(
        nodes,
        [("entry", "head"), ("head", "body"), ("body", "head"), ("head", "exit")],
    )
    out = analyze(
        cfg,
        one_inst_per_node(nodes),
        loop_iterations=10,
        latency=LATENCY,
        logger=logger,
    )
    assert out.total_inst_count == 1 + 10 + 10 + 1


def test_nested_loops_multiply(logger):
    nodes = ["entry", "outer", "inner", "latch", "exit"]
    cfg = make_cfg(
        nodes,
        [
            ("entry", "outer"),
            ("outer", "inner"),
            ("inner", "inner"),
            ("inner", "latch"),
            ("latch", "outer"),
            ("latch", "exit"),
        ],
    )
    out = analyze(
        cfg,
        one_inst_per_node(nodes),
        loop_iterations=10,
        latency=LATENCY,
        logger=logger,
    )
    assert out.total_inst_count == 1 + 10 + 100 + 10 + 1


def test_two_back_edges_same_header_count_once(logger):
    # a `continue` gives the loop a second latch; it is still one loop
    nodes = ["entry", "head", "a", "b", "exit"]
    cfg = make_cfg(
        nodes,
        [
            ("entry", "head"),
            ("head", "a"),
            ("a", "head"),
            ("a", "b"),
            ("b", "head"),
            ("head", "exit"),
        ],
    )
    out = analyze(
        cfg,
        one_inst_per_node(nodes),
        loop_iterations=10,
        latency=LATENCY,
        logger=logger,
    )
    assert out.total_inst_count == 1 + 10 + 10 + 10 + 1


# ---------------------------------------------------------------------------
# analyze: per-loop counts
# ---------------------------------------------------------------------------

# three self-loops, written so DFS meets the last one first
THREE_LOOPS = ["entry", "a", "mid", "b", "c", "exit"]
THREE_LOOP_EDGES = [
    ("entry", "c"),
    ("entry", "a"),
    ("a", "a"),
    ("a", "mid"),
    ("mid", "b"),
    ("b", "b"),
    ("b", "c"),
    ("c", "c"),
    ("c", "exit"),
]


def test_loops_are_listed_in_code_order():
    cfg = make_cfg(THREE_LOOPS, THREE_LOOP_EDGES)
    loops, _ = find_loops(cfg)
    assert list(loops) == ["a", "b", "c"]


def test_loop_counts_apply_in_code_order(logger):
    cfg = make_cfg(THREE_LOOPS, THREE_LOOP_EDGES)
    out = analyze(
        cfg,
        one_inst_per_node(THREE_LOOPS),
        loop_iterations=None,
        latency=LATENCY,
        logger=logger,
        loop_counts=[64, 0, 3],
    )
    # entry, mid and exit once; a 64 times, b never, c 3 times
    assert out.total_inst_count == 3 + 64 + 0 + 3


def test_loop_counts_win_over_loop_iterations(logger):
    cfg = make_cfg(THREE_LOOPS, THREE_LOOP_EDGES)
    out = analyze(
        cfg,
        one_inst_per_node(THREE_LOOPS),
        loop_iterations=10,
        latency=LATENCY,
        logger=logger,
        loop_counts=[1, 1, 1],
    )
    assert out.total_inst_count == 6


def test_nested_loop_counts_multiply(logger):
    nodes = ["entry", "outer", "inner", "latch", "exit"]
    cfg = make_cfg(
        nodes,
        [
            ("entry", "outer"),
            ("outer", "inner"),
            ("inner", "inner"),
            ("inner", "latch"),
            ("latch", "outer"),
            ("latch", "exit"),
        ],
    )
    out = analyze(
        cfg,
        one_inst_per_node(nodes),
        loop_iterations=None,
        latency=LATENCY,
        logger=logger,
        loop_counts=[3, 4],
    )
    # outer and latch 3 times, inner 3 * 4 times
    assert out.total_inst_count == 1 + 3 + 12 + 3 + 1


@pytest.mark.parametrize("counts", [[1, 2], [1, 2, 3, 4], []])
def test_wrong_number_of_loop_counts_raises(counts, logger):
    cfg = make_cfg(THREE_LOOPS, THREE_LOOP_EDGES)
    with pytest.raises(LoopCountMismatchError) as e:
        analyze(
            cfg,
            one_inst_per_node(THREE_LOOPS),
            loop_iterations=10,
            latency=LATENCY,
            logger=logger,
            loop_counts=counts,
        )
    assert e.value.loop_headers == ["a", "b", "c"]
    assert e.value.given == len(counts)


def test_loop_counts_on_a_loop_free_kernel_are_ignored(logger):
    nodes = ["entry", "exit"]
    cfg = make_cfg(nodes, [("entry", "exit")])
    out = analyze(
        cfg,
        one_inst_per_node(nodes),
        loop_iterations=None,
        latency=LATENCY,
        logger=logger,
        loop_counts=[5],
    )
    assert out.total_inst_count == 2


def test_describe_loops_mm_basic(logger):
    cfg, insts = parse_fixture(
        CFG_DIR / "mm_basic.sm_75.bbcfg.dot", "_Z8mm_basicPfS_S_iii", "sm_75", logger
    )
    loops = describe_loops(cfg, insts)

    # 16-step main loop, 4-step and 1-step remainders, in code order
    assert [loop.header for loop in loops] == [".L_x_4", ".L_x_2", ".L_x_6"]
    assert [loop.inst_count for loop in loops] == [70, 24, 10]
    assert [loop.global_loads for loop in loops] == [32, 8, 2]
    assert all(loop.parent is None for loop in loops)


def test_mm_basic_loop_counts(logger):
    cfg, insts = parse_fixture(
        CFG_DIR / "mm_basic.sm_75.bbcfg.dot", "_Z8mm_basicPfS_S_iii", "sm_75", logger
    )
    out = analyze(cfg, insts, None, LATENCY, logger, loop_counts=[64, 0, 0])
    # 89 instructions outside the loops, the 70-instruction main loop 64 times
    assert out.total_inst_count == 89 + 64 * 70


def test_describe_loops_nested(logger):
    nodes = ["entry", "outer", "inner", "latch", "exit"]
    cfg = make_cfg(
        nodes,
        [
            ("entry", "outer"),
            ("outer", "inner"),
            ("inner", "inner"),
            ("inner", "latch"),
            ("latch", "outer"),
            ("latch", "exit"),
        ],
    )
    loops = describe_loops(cfg, one_inst_per_node(nodes))
    assert [(loop.header, loop.parent) for loop in loops] == [
        ("outer", None),
        ("inner", "outer"),
    ]
    assert loops[0].inst_count == 3


def test_empty_block_skipped(logger):
    nodes = ["entry", "empty", "exit"]
    cfg = make_cfg(nodes, [("entry", "empty"), ("empty", "exit")])
    insts = {"entry": [ffma("R0")], "empty": [], "exit": [ffma("R1")]}
    out = analyze(cfg, insts, loop_iterations=1, latency=LATENCY, logger=logger)
    assert out.total_inst_count == 2


# ---------------------------------------------------------------------------
# analyze: counts and kernel-wide ILP/MLP
# ---------------------------------------------------------------------------


def test_category_counts(logger):
    block = [
        ldg("R0", "R2"),
        instr("STG", OpCategory.GLOBAL_STORE, ["R2", "R0"]),
        instr("LDS", OpCategory.SHARED_LOAD, ["R4"], ["R5"]),
        instr("MUFU", OpCategory.SFU, ["R5"], ["R6"]),
        instr("BAR", OpCategory.BARRIER),
    ]
    cfg = make_cfg(["entry"], [])
    out = analyze(
        cfg, {"entry": block}, loop_iterations=1, latency=LATENCY, logger=logger
    )
    assert out.total_inst_count == 5
    # shared memory is on-chip: not in #mem_insts
    assert out.mem_inst_count == 2
    assert out.sfu_inst_count == 1
    assert out.sync_inst_count == 1


def test_counts_and_latency_weighted_by_iterations(logger):
    nodes = ["entry", "body"]
    cfg = make_cfg(nodes, [("entry", "body"), ("body", "body")])
    insts = {
        "entry": [ffma("R0")],
        "body": [ldg("R1", "R2"), instr("MUFU", OpCategory.SFU, ["R1"], ["R3"])],
    }
    out = analyze(cfg, insts, loop_iterations=5, latency=LATENCY, logger=logger)
    assert out.total_inst_count == 1 + 2 * 5
    assert out.mem_inst_count == 5
    assert out.sfu_inst_count == 5
    assert out.total_latency == FP32_LAT + (FP32_LAT + SFU_LAT) * 5


def test_kernel_ilp_weighted_toward_loop(logger):
    # entry ILP 1 (chain), loop body ILP 4 (independent); loop_iterations=100
    # pulls the kernel ILP toward 4
    nodes = ["entry", "body"]
    cfg = make_cfg(nodes, [("entry", "body"), ("body", "body")])
    insts = {
        "entry": [ffma("R0"), ffma("R1", "R0"), ffma("R2", "R1"), ffma("R3", "R2")],
        "body": [ffma("R4"), ffma("R5"), ffma("R6"), ffma("R7")],
    }
    out = analyze(cfg, insts, loop_iterations=100, latency=LATENCY, logger=logger)
    assert out.ilp == pytest.approx((1 * 1 + 4 * 100) / 101)


def test_kernel_mlp_defaults_to_1_without_loads(logger):
    cfg = make_cfg(["entry"], [])
    out = analyze(
        cfg, {"entry": [ffma("R0")]}, loop_iterations=1, latency=LATENCY, logger=logger
    )
    assert out.mlp == 1


# ---------------------------------------------------------------------------
# analyze on nvdisasm fixtures
# ---------------------------------------------------------------------------


def parse_fixture(path, kernel, arch, logger):
    cfg = gen_cfg_from_raw(path.read_text())
    parser = get_parser(cfg=cfg, kernel_name=kernel, arch=arch, logger=logger)
    return cfg, parser.parse()


def test_vecadd_fixture(logger):
    cfg, insts = parse_fixture(
        CFG_DIR / "vecadd.bbcfg.dot", "_Z6vecaddPfS_S_i", "sm_80", logger
    )
    out = analyze(cfg, insts, loop_iterations=10, latency=LATENCY, logger=logger)

    # straight-line kernel: loop_iterations has no effect
    assert out.total_inst_count == 16
    # 2 LDG + 1 STG
    assert out.mem_inst_count == 3
    assert out.sfu_inst_count == 0
    assert out.sync_inst_count == 0
    # the two LDGs are independent
    assert out.mlp == 2
    assert out.ilp > 1


# sm_80: one self-loop on .L_x_1
# sm_90/sm_100: .L_x_1 self-loop nested in .L_x_2 -> .L_x_1 -> BB1 -> .L_x_2
SGEMM_ITERATIONS = {
    "sm_80": {".L_x_1": 10},
    "sm_90": {".L_x_1": 100, ".L_x_2": 10, "BB1": 10},
    "sm_100": {".L_x_1": 100, ".L_x_2": 10, "BB1": 10},
}


@pytest.mark.parametrize("arch", ["sm_80", "sm_90", "sm_100"])
def test_sgemm_fixture_loop_weighting(arch, logger):
    cfg, insts = parse_fixture(
        CFG_DIR / "parboil" / "sgemm" / f"main.{arch}.bbcfg.dot",
        SGEMM_KERNEL,
        arch,
        logger,
    )
    out = analyze(cfg, insts, loop_iterations=10, latency=LATENCY, logger=logger)

    iterations = SGEMM_ITERATIONS[arch]
    expected = sum(len(block) * iterations.get(n, 1) for n, block in insts.items())
    assert out.total_inst_count == expected
    assert out.ilp >= 1
    assert out.mlp >= 1


TWO_KERNELS = """\
digraph f {
subgraph "cluster_first" {
"first"
[label="first"]
}
subgraph "cluster_second" {
"second"
[label="second"]
"second":exit0:s -> "loop":entry:n [style=solid];
"loop"
[label="loop"]
"loop":exit0:e -> "loop":entry:n [style=solid];
}
}
"""


def test_loop_in_a_later_kernel_is_found_once_filtered(logger):
    cfg = filter_cfg(gen_cfg_from_raw(TWO_KERNELS), "second")

    with pytest.raises(LoopIterationsRequiredError) as e:
        analyze(cfg, {}, None, LATENCY, logger)
    assert e.value.loop_headers == ["loop"]


def test_other_kernels_loop_does_not_require_iterations(logger):
    cfg = filter_cfg(gen_cfg_from_raw(TWO_KERNELS), "first")

    analyze(cfg, {}, None, LATENCY, logger)


# ---------------------------------------------------------------------------
# address groups and working set
# ---------------------------------------------------------------------------


def gmem(opcode, category, base, width=4, dst=(), offset=0):
    return SASSInstruction(
        opcode=opcode,
        category=category,
        predicate=None,
        src_registers={base},
        dest_registers=set(dst),
        access_width_bytes=width,
        address_base_register=base,
        address_offset=offset,
    )


def g_load(dst, base, width=4, offset=0):
    return gmem("LDG", OpCategory.GLOBAL_LOAD, base, width, [dst], offset)


def g_store(base, width=4):
    return gmem("STG", OpCategory.GLOBAL_STORE, base, width)


def test_same_base_same_definition_is_one_group():
    # vecadd_poor-style repeat: [R2] twice with R2 unchanged
    groups = get_address_groups([g_load("R8", "R2"), g_load("R9", "R2")])
    assert groups == {(None, "R2", 0): 4}


def test_redefined_base_starts_a_new_group():
    insts = [
        g_load("R8", "R2"),
        instr("IADD3", OpCategory.INT, ["R2"], ["R2"]),
        g_load("R9", "R2"),
    ]
    assert get_address_groups(insts) == {(None, "R2", 0): 4, (1, "R2", 0): 4}


def test_group_width_is_the_widest_access():
    groups = get_address_groups([g_load("R8", "R2"), g_load("R10", "R2", width=16)])
    assert groups == {(None, "R2", 0): 16}


def test_non_global_accesses_form_no_groups():
    insts = [instr("LDS", OpCategory.SHARED_LOAD, ["R2"], ["R8"]), ffma("R0", "R1")]
    assert get_address_groups(insts) == {}


def test_working_set_vecadd_shape(logger):
    nodes = ["entry"]
    insts = {"entry": [g_load("R8", "R2"), g_load("R9", "R4"), g_store("R6")]}
    out = analyze(
        make_cfg(nodes, []), insts, loop_iterations=None, latency=LATENCY, logger=logger
    )
    assert out.num_address_groups == 3
    assert out.global_bytes_per_thread == 12


def test_working_set_scales_with_trip_count(logger):
    nodes = ["entry", "body", "exit"]
    cfg = make_cfg(nodes, [("entry", "body"), ("body", "body"), ("body", "exit")])
    insts = {
        "entry": [ffma("R0", "R1")],
        "body": [g_load("R8", "R2"), g_load("R9", "R4", width=8)],
        "exit": [g_store("R6")],
    }
    out = analyze(cfg, insts, loop_iterations=10, latency=LATENCY, logger=logger)
    assert out.global_bytes_per_thread == (4 + 8) * 10 + 4


def test_offsets_from_one_base_are_distinct_elements_of_one_array():
    # mm_basic's unrolled k loop: [R2+0x4], [R2+0x8], ... from one unchanged R2
    insts = [g_load(f"R{10 + k}", "R2", offset=4 * k) for k in range(16)]
    groups = get_address_groups(insts)
    assert len(groups) == 16
    assert sum(groups.values()) == 64
    assert count_arrays(groups) == 1


def test_working_set_counts_unrolled_offsets(logger):
    insts = {"entry": [g_load("R8", "R2"), g_load("R9", "R2", offset=4)]}
    out = analyze(
        make_cfg(["entry"], []),
        insts,
        loop_iterations=None,
        latency=LATENCY,
        logger=logger,
    )
    assert out.global_bytes_per_thread == 8
    assert out.num_address_groups == 1


# ---------------------------------------------------------------------------
# reuse counts (GitHub #63)
# ---------------------------------------------------------------------------


def test_straight_line_first_touches_are_not_reusable(logger):
    # vecadd good: three addresses, each touched once
    insts = {"entry": [g_load("R8", "R2"), g_load("R9", "R4"), g_store("R6")]}
    out = analyze(
        make_cfg(["entry"], []),
        insts,
        loop_iterations=None,
        latency=LATENCY,
        logger=logger,
    )
    assert out.global_access_count == 3
    assert out.reusable_access_count == 0


def test_straight_line_repeats_are_reusable(logger):
    insts = {
        "entry": [
            g_load("R9", "R2"),
            g_load("R0", "R4"),
            g_store("R6"),
            g_load("R0", "R4"),
            g_load("R11", "R2"),
            g_store("R6"),
        ]
    }
    out = analyze(
        make_cfg(["entry"], []),
        insts,
        loop_iterations=None,
        latency=LATENCY,
        logger=logger,
    )
    assert out.global_access_count == 6
    assert out.reusable_access_count == 3


def test_loop_accesses_are_all_reusable(logger):
    nodes = ["entry", "body", "exit"]
    cfg = make_cfg(nodes, [("entry", "body"), ("body", "body"), ("body", "exit")])
    insts = {
        "entry": [ffma("R0", "R1")],
        "body": [g_load("R8", "R2"), g_load("R9", "R4")],
        "exit": [g_store("R6")],
    }
    out = analyze(cfg, insts, loop_iterations=10, latency=LATENCY, logger=logger)
    assert out.global_access_count == 2 * 10 + 1
    assert (
        out.reusable_access_count == 2 * 10
    )  # the store after the loop is a first touch
