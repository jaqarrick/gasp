"""Tests for gasp.dependency — register DAG, ILP/MLP, loop weighting, counts."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gasp.cfg import CFGEdge, CFGNode, ControlFlowGraph, gen_cfg_from_raw
from gasp.dependency import (
    LatencyTable,
    LoopIterationsRequiredError,
    analyze,
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
    out = analyze(cfg, one_inst_per_node(nodes), ni=10, latency=LATENCY, logger=logger)
    assert out.total_inst_count == 3


def test_self_loop(logger):
    nodes = ["entry", "body", "exit"]
    cfg = make_cfg(nodes, [("entry", "body"), ("body", "body"), ("body", "exit")])
    out = analyze(cfg, one_inst_per_node(nodes), ni=10, latency=LATENCY, logger=logger)
    assert out.total_inst_count == 1 + 10 + 1


def test_loop_without_ni_raises(logger):
    nodes = ["entry", "body", "exit"]
    cfg = make_cfg(nodes, [("entry", "body"), ("body", "body"), ("body", "exit")])
    with pytest.raises(LoopIterationsRequiredError) as exc:
        analyze(cfg, one_inst_per_node(nodes), ni=None, latency=LATENCY, logger=logger)
    assert exc.value.loop_headers == ["body"]


def test_no_loops_without_ni(logger):
    nodes = ["entry", "a", "exit"]
    cfg = make_cfg(nodes, [("entry", "a"), ("a", "exit")])
    out = analyze(
        cfg, one_inst_per_node(nodes), ni=None, latency=LATENCY, logger=logger
    )
    assert out.total_inst_count == 3


def test_loop_body_spans_blocks(logger):
    # head -> body -> head: both blocks repeat
    nodes = ["entry", "head", "body", "exit"]
    cfg = make_cfg(
        nodes,
        [("entry", "head"), ("head", "body"), ("body", "head"), ("head", "exit")],
    )
    out = analyze(cfg, one_inst_per_node(nodes), ni=10, latency=LATENCY, logger=logger)
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
    out = analyze(cfg, one_inst_per_node(nodes), ni=10, latency=LATENCY, logger=logger)
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
    out = analyze(cfg, one_inst_per_node(nodes), ni=10, latency=LATENCY, logger=logger)
    assert out.total_inst_count == 1 + 10 + 10 + 10 + 1


def test_empty_block_skipped(logger):
    nodes = ["entry", "empty", "exit"]
    cfg = make_cfg(nodes, [("entry", "empty"), ("empty", "exit")])
    insts = {"entry": [ffma("R0")], "empty": [], "exit": [ffma("R1")]}
    out = analyze(cfg, insts, ni=1, latency=LATENCY, logger=logger)
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
    out = analyze(cfg, {"entry": block}, ni=1, latency=LATENCY, logger=logger)
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
    out = analyze(cfg, insts, ni=5, latency=LATENCY, logger=logger)
    assert out.total_inst_count == 1 + 2 * 5
    assert out.mem_inst_count == 5
    assert out.sfu_inst_count == 5
    assert out.total_latency == FP32_LAT + (FP32_LAT + SFU_LAT) * 5


def test_kernel_ilp_weighted_toward_loop(logger):
    # entry ILP 1 (chain), loop body ILP 4 (independent); ni=100 pulls the
    # kernel ILP toward 4
    nodes = ["entry", "body"]
    cfg = make_cfg(nodes, [("entry", "body"), ("body", "body")])
    insts = {
        "entry": [ffma("R0"), ffma("R1", "R0"), ffma("R2", "R1"), ffma("R3", "R2")],
        "body": [ffma("R4"), ffma("R5"), ffma("R6"), ffma("R7")],
    }
    out = analyze(cfg, insts, ni=100, latency=LATENCY, logger=logger)
    assert out.ilp == pytest.approx((1 * 1 + 4 * 100) / 101)


def test_kernel_mlp_defaults_to_1_without_loads(logger):
    cfg = make_cfg(["entry"], [])
    out = analyze(cfg, {"entry": [ffma("R0")]}, ni=1, latency=LATENCY, logger=logger)
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
    out = analyze(cfg, insts, ni=10, latency=LATENCY, logger=logger)

    # straight-line kernel: ni has no effect
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
    out = analyze(cfg, insts, ni=10, latency=LATENCY, logger=logger)

    iterations = SGEMM_ITERATIONS[arch]
    expected = sum(len(block) * iterations.get(n, 1) for n, block in insts.items())
    assert out.total_inst_count == expected
    assert out.ilp >= 1
    assert out.mlp >= 1
