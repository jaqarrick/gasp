"""Tests for gasp.divergence — per-thread registers and branch kinds."""

from unittest.mock import MagicMock

import pytest

from gasp.cfg import CFGEdge, CFGNode, ControlFlowGraph
from gasp.dependency import find_loops
from gasp.divergence import (
    VIRTUAL_EXIT,
    BranchKind,
    analyze_divergence,
    find_branches,
    immediate_post_dominator,
    is_register_uniform,
    post_dominators,
)
from gasp.logger import Logger
from gasp.parser import get_parser
from gasp.parser.types import SASSInstruction


@pytest.fixture
def logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def parse(logger):
    """SASS lines -> parsed instructions, with the sm_80 parser."""
    empty = ControlFlowGraph(nodes={}, edges=[])
    parser = get_parser(cfg=empty, kernel_name="kernel", arch="sm_80", logger=logger)

    def _parse(*lines: str) -> list[SASSInstruction]:
        return [parser._parse_instruction(line) for line in lines]

    return _parse


def make_cfg(node_ids, edges) -> ControlFlowGraph:
    """First node is the entry; edges are (source, target) pairs."""
    return ControlFlowGraph(
        nodes={n: CFGNode(id=n, label="") for n in node_ids},
        edges=[CFGEdge(s, None, None, t, None, None, None) for s, t in edges],
    )


# ---------------------------------------------------------------------------
# is_register_uniform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reg, uniform",
    [
        ("SR_TID.X", False),
        ("SR_TID.Y", False),
        ("SR_LANEID", False),
        ("SR_LTMASK", False),
        ("SR_CLOCKLO", False),
        ("SR_CTAID.X", True),
        ("SR_CTAID.Z", True),
        ("SR_NCTAID.X", True),
        ("SR_CgaCtaId", True),
        ("SR_SMID", True),
    ],
)
def test_is_register_uniform(reg, uniform, logger):
    assert is_register_uniform(reg, logger) is uniform
    logger.warning.assert_not_called()


def test_unknown_special_register_is_per_thread_with_warning(logger):
    assert is_register_uniform("SR_SOMETHING_NEW", logger) is False
    logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# analyze_divergence — small graphs
# ---------------------------------------------------------------------------


def test_block_index_is_uniform(parse, logger):
    cfg = make_cfg(["A"], [])
    insts = {
        "A": parse(
            "S2R R0, SR_CTAID.X",
            "ISETP.GE.AND P0, PT, R0, 0x1, PT",
        )
    }
    assert analyze_divergence(cfg, insts, logger)["A"] == set()


def test_thread_index_spreads_through_registers(parse, logger):
    cfg = make_cfg(["A"], [])
    insts = {
        "A": parse(
            "S2R R3, SR_TID.X",
            "LOP3.LUT R2, R3, 0x1, RZ, 0xc0, !PT",
            "ISETP.NE.U32.AND P0, PT, R2, 0x1, PT",
        )
    }
    assert analyze_divergence(cfg, insts, logger)["A"] == {"R3", "R2", "P0"}


def test_uniform_write_clears_register(parse, logger):
    cfg = make_cfg(["A"], [])
    insts = {
        "A": parse(
            "S2R R2, SR_TID.X",
            "MOV R2, c[0x0][0x170]",
        )
    }
    assert analyze_divergence(cfg, insts, logger)["A"] == set()


def test_per_thread_value_reaches_later_block(parse, logger):
    # B never reads SR_TID, but its predicate comes from A's R3
    cfg = make_cfg(["A", "B"], [("A", "B")])
    insts = {
        "A": parse("S2R R3, SR_TID.X"),
        "B": parse(
            "IADD3 R2, R3, 0x1, RZ",
            "ISETP.NE.AND P0, PT, R2, RZ, PT",
        ),
    }
    end = analyze_divergence(cfg, insts, logger)
    assert {"R2", "P0"} <= end["B"]


def test_merge_takes_union_of_predecessors(parse, logger):
    # R4 is per-thread on one side of the diamond only
    cfg = make_cfg(
        ["A", "B", "C", "D"], [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    )
    insts = {
        "A": [],
        "B": parse("S2R R4, SR_LANEID"),
        "C": parse("MOV R4, 0x0"),
        "D": parse("IADD3 R5, R4, 0x1, RZ"),
    }
    end = analyze_divergence(cfg, insts, logger)
    assert "R4" not in end["C"]
    assert {"R4", "R5"} <= end["D"]


def test_loop_back_edge_carries_per_thread_value(parse, logger):
    # R6 becomes per-thread in the body, after the header first read it:
    # only a second pass over the header sees it
    cfg = make_cfg(
        ["E", "H", "L", "X"], [("E", "H"), ("H", "L"), ("L", "H"), ("H", "X")]
    )
    insts = {
        "E": [],
        "H": parse("IADD3 R7, R6, 0x1, RZ"),
        "L": parse("S2R R6, SR_TID.X"),
        "X": [],
    }
    end = analyze_divergence(cfg, insts, logger)
    assert "R7" in end["H"]
    assert "R7" in end["X"]


# ---------------------------------------------------------------------------
# analyze_divergence — the `divergent` kernel of
# tests/fixtures/probes/07_branches.cu, sm_80 SASS, loops left out
# ---------------------------------------------------------------------------


def divergent_probe(parse):
    """BB0 ends in the divergent branch; BB1 and .L_x_22 are the starts of the
    two sides (each skips its loop here); .L_x_21 is the rejoin point."""
    cfg = make_cfg(
        ["BB0", "BB1", ".L_x_22", ".L_x_23", ".L_x_21"],
        [
            ("BB0", ".L_x_22"),
            ("BB0", "BB1"),
            ("BB1", ".L_x_23"),
            (".L_x_22", ".L_x_23"),
            (".L_x_23", ".L_x_21"),
        ],
    )
    insts = {
        "BB0": parse(
            "MOV R1, c[0x0][0x28]",
            "S2R R0, SR_CTAID.X",
            "HFMA2.MMA R5, -RZ, RZ, 0, 2.384185791015625e-07",
            "ULDC.64 UR4, c[0x0][0x118]",
            "S2R R3, SR_TID.X",
            "IMAD R0, R0, c[0x0][0x0], R3",
            "IMAD.WIDE R4, R0, R5, c[0x0][0x168]",
            "LDG.E R5, [R4.64]",
            "LOP3.LUT R2, R3, 0x1, RZ, 0xc0, !PT",
            "BSSY B0, `(.L_x_21)",
            "SHF.R.S32.HI R3, RZ, 0x1f, R0",
            "ISETP.NE.U32.AND P0, PT, R2, 0x1, PT",
            "@P0 BRA `(.L_x_22)",
        ),
        "BB1": parse(
            "MOV R2, c[0x0][0x170]",
            "ISETP.GE.AND P0, PT, R2, 0x1, PT",
            "@!P0 BRA `(.L_x_23)",
        ),
        ".L_x_22": parse(
            "MOV R2, c[0x0][0x170]",
            "ISETP.GE.AND P0, PT, R2, 0x1, PT",
            "@!P0 BRA `(.L_x_23)",
        ),
        ".L_x_23": parse("BSYNC B0"),
        ".L_x_21": parse(
            "LEA R2, P0, R0, c[0x0][0x160], 0x2",
            "LEA.HI.X R3, R0, c[0x0][0x164], R3, 0x2, P0",
            "STG.E [R2.64], R5",
            "EXIT",
        ),
    }
    return cfg, insts


def test_probe_divergent_branch_predicate_is_per_thread(parse, logger):
    cfg, insts = divergent_probe(parse)
    end = analyze_divergence(cfg, insts, logger)
    # P0 <- R2 <- R3 <- SR_TID.X: the branch at the end of BB0 diverges
    assert {"R3", "R2", "P0"} <= end["BB0"]
    # global index and the value loaded with it
    assert {"R0", "R5"} <= end["BB0"]
    # stack pointer, kernel-argument copies: uniform
    assert "R1" not in end["BB0"]
    assert not {"UR4", "UR5"} & end["BB0"]


@pytest.mark.parametrize("side", ["BB1", ".L_x_22"])
def test_probe_loop_guard_predicate_is_uniform(side, parse, logger):
    cfg, insts = divergent_probe(parse)
    end = analyze_divergence(cfg, insts, logger)
    # R2 and P0 are overwritten from n (c[0x0][0x170]): the guard is uniform
    assert not {"R2", "P0"} & end[side]
    # per-thread values from BB0 are still live
    assert {"R0", "R5"} <= end[side]


def test_probe_rejoin_recomputes_per_thread_address(parse, logger):
    cfg, insts = divergent_probe(parse)
    end = analyze_divergence(cfg, insts, logger)
    # R2 is uniform on both sides, then the store address makes it per-thread again
    assert {"R2", "R3", "R5"} <= end[".L_x_21"]


# ---------------------------------------------------------------------------
# find_branches
# ---------------------------------------------------------------------------


def branches_of(cfg, insts, logger):
    end_state = analyze_divergence(cfg, insts, logger)
    loops, _ = find_loops(cfg)
    return find_branches(cfg, insts, end_state, loops, logger)


def two_way(predicate_source: str, branch: str = "@P0 BRA `(T)"):
    """A -> T (taken), A -> F (fall-through); P0 computed from predicate_source."""
    cfg = make_cfg(["A", "T", "F"], [("A", "T"), ("A", "F")])
    lines = (
        f"S2R R0, {predicate_source}",
        "ISETP.NE.AND P0, PT, R0, RZ, PT",
        branch,
    )
    return cfg, lines


def test_divergent_branch(parse, logger):
    cfg, lines = two_way("SR_TID.X")
    insts = {"A": parse(*lines), "T": [], "F": []}

    [b] = branches_of(cfg, insts, logger)

    assert b.node == "A"
    assert b.kind is BranchKind.DIVERGENT
    assert b.divergent is True
    assert b.predicate == "P0"
    assert (b.taken_succ, b.fall_succ) == ("T", "F")
    logger.warning.assert_not_called()


def test_uniform_branch(parse, logger):
    cfg, lines = two_way("SR_CTAID.X")
    insts = {"A": parse(*lines), "T": [], "F": []}

    [b] = branches_of(cfg, insts, logger)

    assert b.kind is BranchKind.UNIFORM
    assert b.divergent is False
    assert (b.taken_succ, b.fall_succ) == ("T", "F")


def test_negated_branch_taken_side_is_still_the_label(parse, logger):
    cfg, lines = two_way("SR_TID.X", branch="@!P0 BRA `(T)")
    insts = {"A": parse(*lines), "T": [], "F": []}

    [b] = branches_of(cfg, insts, logger)

    assert (b.taken_succ, b.fall_succ) == ("T", "F")


def test_straight_code_has_no_branches(parse, logger):
    cfg = make_cfg(["A", "B"], [("A", "B")])
    insts = {"A": parse("IADD3 R2, R2, 0x1, RZ"), "B": parse("EXIT")}

    assert branches_of(cfg, insts, logger) == []


def test_empty_block_is_skipped(parse, logger):
    cfg = make_cfg(["A", "T", "F"], [("A", "T"), ("A", "F")])
    insts = {"A": [], "T": [], "F": []}

    assert branches_of(cfg, insts, logger) == []


def test_early_exit(parse, logger):
    cfg = make_cfg(["A", "B"], [("A", "B")])
    insts = {
        "A": parse(
            "S2R R0, SR_TID.X",
            "ISETP.GE.AND P0, PT, R0, 0x10, PT",
            "@P0 EXIT",
        ),
        "B": parse("EXIT"),
    }

    [b] = branches_of(cfg, insts, logger)

    assert b.node == "A"
    assert b.kind is BranchKind.EARLY_EXIT
    assert b.divergent is True
    assert (b.taken_succ, b.fall_succ) == (None, None)


def test_target_not_a_successor_is_skipped_with_warning(parse, logger):
    cfg = make_cfg(["A", "T", "F"], [("A", "T"), ("A", "F")])
    insts = {
        "A": parse("ISETP.NE.AND P0, PT, R0, RZ, PT", "@P0 BRA `(ELSEWHERE)"),
        "T": [],
        "F": [],
    }

    assert branches_of(cfg, insts, logger) == []
    logger.warning.assert_called_once()


def loop_graph(parse, *body_lines):
    """E -> H -> B, B -> H (back edge), B -> X; the branch ends B."""
    cfg = make_cfg(
        ["E", "H", "B", "X"], [("E", "H"), ("H", "B"), ("B", "H"), ("B", "X")]
    )
    insts = {
        "E": [],
        "H": parse("IADD3 R6, R6, -0x1, RZ"),
        "B": parse(*body_lines, "ISETP.GT.AND P1, PT, R6, RZ, PT", "@P1 BRA `(H)"),
        "X": parse("EXIT"),
    }
    return cfg, insts


def test_loop_back_edge_is_loop_exit(parse, logger):
    cfg, insts = loop_graph(parse)

    [b] = branches_of(cfg, insts, logger)

    assert b.node == "B"
    assert b.kind is BranchKind.LOOP_EXIT
    assert b.divergent is False
    assert (b.taken_succ, b.fall_succ) == ("H", "X")


def test_per_thread_loop_exit_is_loop_exit_not_divergent(parse, logger):
    # trip count from threadIdx: still a loop exit, but marked divergent
    cfg, insts = loop_graph(parse, "S2R R6, SR_TID.X")

    [b] = branches_of(cfg, insts, logger)

    assert b.kind is BranchKind.LOOP_EXIT
    assert b.divergent is True


def test_if_inside_a_loop_is_not_a_loop_exit(parse, logger):
    # E -> H; H -> T, F; T, F -> J; J -> H (back edge), J -> X
    cfg = make_cfg(
        ["E", "H", "T", "F", "J", "X"],
        [
            ("E", "H"),
            ("H", "T"),
            ("H", "F"),
            ("T", "J"),
            ("F", "J"),
            ("J", "H"),
            ("J", "X"),
        ],
    )
    insts = {
        "E": [],
        "H": parse(
            "S2R R0, SR_TID.X",
            "ISETP.NE.AND P0, PT, R0, RZ, PT",
            "@P0 BRA `(T)",
        ),
        "T": parse("FADD R5, R5, 1"),
        "F": parse("FADD R5, R5, 2"),
        "J": parse(
            "IADD3 R6, R6, -0x1, RZ",
            "ISETP.GT.AND P1, PT, R6, RZ, PT",
            "@P1 BRA `(H)",
        ),
        "X": parse("EXIT"),
    }

    by_node = {b.node: b for b in branches_of(cfg, insts, logger)}

    assert set(by_node) == {"H", "J"}
    assert by_node["H"].kind is BranchKind.DIVERGENT
    assert by_node["J"].kind is BranchKind.LOOP_EXIT


def test_loop_guard_before_the_loop_is_not_a_loop_exit(parse, logger):
    # G -> H (enter the loop) or G -> X (skip it); H -> H is the loop
    cfg = make_cfg(["G", "H", "X"], [("G", "H"), ("G", "X"), ("H", "H"), ("H", "X")])
    insts = {
        "G": parse("ISETP.GE.AND P0, PT, R2, 0x1, PT", "@!P0 BRA `(X)"),
        "H": parse("ISETP.GT.AND P1, PT, R6, RZ, PT", "@P1 BRA `(H)"),
        "X": parse("EXIT"),
    }

    by_node = {b.node: b for b in branches_of(cfg, insts, logger)}

    assert by_node["G"].kind is BranchKind.UNIFORM
    assert (by_node["G"].taken_succ, by_node["G"].fall_succ) == ("X", "H")
    assert by_node["H"].kind is BranchKind.LOOP_EXIT


def divergent_probe_with_bodies(parse):
    """divergent_probe with a stand-in loop body on each side, so each side's
    loop guard (skip the loop if n < 1) is a two-way branch."""
    cfg, insts = divergent_probe(parse)
    for side, body in [("BB1", "BODY1"), (".L_x_22", "BODY2")]:
        cfg.nodes[body] = CFGNode(id=body, label="")
        cfg.edges.append(CFGEdge(side, None, None, body, None, None, None))
        cfg.edges.append(CFGEdge(body, None, None, ".L_x_23", None, None, None))
        insts[body] = parse("FFMA R5, R5, R4, 0.5")
    return cfg, insts


def test_probe_branch_kinds(parse, logger):
    cfg, insts = divergent_probe_with_bodies(parse)

    by_node = {b.node: b for b in branches_of(cfg, insts, logger)}

    assert set(by_node) == {"BB0", "BB1", ".L_x_22"}

    # if (threadIdx.x & 1): P0 <- R2 <- R3 <- SR_TID.X
    assert by_node["BB0"].kind is BranchKind.DIVERGENT
    assert (by_node["BB0"].taken_succ, by_node["BB0"].fall_succ) == (".L_x_22", "BB1")

    # each side's "skip the loop if n < 1": P0 from n, uniform
    assert by_node["BB1"].kind is BranchKind.UNIFORM
    assert (by_node["BB1"].taken_succ, by_node["BB1"].fall_succ) == (".L_x_23", "BODY1")
    assert by_node[".L_x_22"].kind is BranchKind.UNIFORM
    assert (by_node[".L_x_22"].taken_succ, by_node[".L_x_22"].fall_succ) == (
        ".L_x_23",
        "BODY2",
    )


# ---------------------------------------------------------------------------
# post_dominators / immediate_post_dominator
# ---------------------------------------------------------------------------


def test_post_dominators_diamond():
    cfg = make_cfg(
        ["A", "T", "F", "J"], [("A", "T"), ("A", "F"), ("T", "J"), ("F", "J")]
    )

    pdom = post_dominators(cfg)

    assert pdom["A"] == {"A", "J", VIRTUAL_EXIT}
    assert pdom["T"] == {"T", "J", VIRTUAL_EXIT}
    assert pdom["F"] == {"F", "J", VIRTUAL_EXIT}
    assert pdom["J"] == {"J", VIRTUAL_EXIT}
    assert immediate_post_dominator("A", pdom) == "J"


def test_post_dominators_if_without_else():
    cfg = make_cfg(["A", "B", "J"], [("A", "B"), ("B", "J"), ("A", "J")])

    pdom = post_dominators(cfg)

    assert pdom["A"] == {"A", "J", VIRTUAL_EXIT}
    assert immediate_post_dominator("A", pdom) == "J"


def test_post_dominators_loop():
    cfg = make_cfg(
        ["E", "H", "B", "X"], [("E", "H"), ("H", "B"), ("B", "H"), ("B", "X")]
    )

    pdom = post_dominators(cfg)

    assert pdom["B"] == {"B", "X", VIRTUAL_EXIT}
    assert pdom["H"] == {"H", "B", "X", VIRTUAL_EXIT}
    assert immediate_post_dominator("E", pdom) == "H"
    assert immediate_post_dominator("H", pdom) == "B"
    assert immediate_post_dominator("B", pdom) == "X"


def test_sides_that_only_meet_at_the_end_rejoin_at_virtual_exit():
    # T returns early; F carries on to X; both are EXIT blocks
    cfg = make_cfg(["A", "T", "F", "X"], [("A", "T"), ("A", "F"), ("F", "X")])

    pdom = post_dominators(cfg)

    assert immediate_post_dominator("A", pdom) == VIRTUAL_EXIT


def test_unreachable_block_is_left_out():
    # X is the padding loop after EXIT: nothing leads to it
    cfg = make_cfg(["A", "B", "X"], [("A", "B"), ("X", "X")])

    pdom = post_dominators(cfg)

    assert "X" not in pdom
    assert immediate_post_dominator("X", pdom) is None
    assert immediate_post_dominator("A", pdom) == "B"


def test_loop_with_no_way_out_does_not_break_the_others():
    # H loops forever; A's other side reaches the end
    cfg = make_cfg(["A", "H", "X"], [("A", "H"), ("A", "X"), ("H", "H")])

    pdom = post_dominators(cfg)

    assert pdom["H"] == {"H", VIRTUAL_EXIT}
    assert pdom["X"] == {"X", VIRTUAL_EXIT}
    assert immediate_post_dominator("A", pdom) == VIRTUAL_EXIT


# ---------------------------------------------------------------------------
# find_branches — rejoin points
# ---------------------------------------------------------------------------


def test_divergent_branch_rejoin(parse, logger):
    cfg = make_cfg(
        ["A", "T", "F", "J"], [("A", "T"), ("A", "F"), ("T", "J"), ("F", "J")]
    )
    insts = {
        "A": parse(
            "S2R R0, SR_TID.X",
            "ISETP.NE.AND P0, PT, R0, RZ, PT",
            "@P0 BRA `(T)",
        ),
        "T": [],
        "F": [],
        "J": parse("EXIT"),
    }

    [b] = branches_of(cfg, insts, logger)

    assert b.kind is BranchKind.DIVERGENT
    assert b.rejoin == "J"


def test_if_without_else_rejoins_at_its_jump_target(parse, logger):
    # A jumps straight to J (skipping B): J is both the taken side and the rejoin
    cfg = make_cfg(["A", "B", "J"], [("A", "B"), ("B", "J"), ("A", "J")])
    insts = {
        "A": parse("ISETP.GE.AND P0, PT, R2, 0x1, PT", "@!P0 BRA `(J)"),
        "B": parse("FADD R5, R5, 1"),
        "J": parse("EXIT"),
    }

    [b] = branches_of(cfg, insts, logger)

    assert b.kind is BranchKind.UNIFORM
    assert (b.taken_succ, b.fall_succ) == ("J", "B")
    assert b.rejoin == "J"


def test_loop_and_early_exits_have_no_rejoin(parse, logger):
    cfg, insts = loop_graph(parse)
    [loop_exit] = branches_of(cfg, insts, logger)
    assert loop_exit.kind is BranchKind.LOOP_EXIT
    assert loop_exit.rejoin is None

    cfg = make_cfg(["A", "B"], [("A", "B")])
    insts = {
        "A": parse("ISETP.GE.AND P0, PT, R0, 0x10, PT", "@P0 EXIT"),
        "B": parse("EXIT"),
    }
    [early_exit] = branches_of(cfg, insts, logger)
    assert early_exit.kind is BranchKind.EARLY_EXIT
    assert early_exit.rejoin is None


def test_probe_rejoin_is_the_bsync_block_not_the_bssy_label(parse, logger):
    cfg, insts = divergent_probe_with_bodies(parse)

    by_node = {b.node: b for b in branches_of(cfg, insts, logger)}

    # BSSY B0 names .L_x_21, but both sides reach .L_x_23 (BSYNC B0) first
    assert by_node["BB0"].rejoin == ".L_x_23"
    # each loop guard skips to .L_x_23, where its body also ends
    assert by_node["BB1"].rejoin == ".L_x_23"
    assert by_node[".L_x_22"].rejoin == ".L_x_23"


# ---------------------------------------------------------------------------
# find_branches — sides
# ---------------------------------------------------------------------------


def branch_to(target: str, source: str = "SR_TID.X") -> list[str]:
    return [
        f"S2R R0, {source}",
        "ISETP.NE.AND P0, PT, R0, RZ, PT",
        f"@P0 BRA `({target})",
    ]


def test_sides_of_if_else(parse, logger):
    cfg = make_cfg(
        ["A", "T", "F", "J"], [("A", "T"), ("A", "F"), ("T", "J"), ("F", "J")]
    )
    insts = {"A": parse(*branch_to("T")), "T": [], "F": [], "J": parse("EXIT")}

    [b] = branches_of(cfg, insts, logger)

    assert (b.taken, b.fall_through) == ({"T"}, {"F"})
    logger.warning.assert_not_called()


def test_if_without_else_has_an_empty_side(parse, logger):
    cfg = make_cfg(["A", "B", "J"], [("A", "B"), ("B", "J"), ("A", "J")])
    insts = {
        "A": parse("ISETP.GE.AND P0, PT, R2, 0x1, PT", "@!P0 BRA `(J)"),
        "B": parse("FADD R5, R5, 1"),
        "J": parse("EXIT"),
    }

    [b] = branches_of(cfg, insts, logger)

    assert (b.taken, b.fall_through) == (set(), {"B"})


def test_loop_inside_a_side_belongs_to_that_side(parse, logger):
    # A -> T -> L (loops on itself) -> J; A -> F -> J
    cfg = make_cfg(
        ["A", "T", "L", "F", "J"],
        [("A", "T"), ("T", "L"), ("L", "L"), ("L", "J"), ("A", "F"), ("F", "J")],
    )
    insts = {
        "A": parse(*branch_to("T")),
        "T": parse("FADD R5, R5, 1"),
        "L": parse("ISETP.GT.AND P1, PT, R6, RZ, PT", "@P1 BRA `(L)"),
        "F": parse("FADD R5, R5, 2"),
        "J": parse("EXIT"),
    }

    by_node = {b.node: b for b in branches_of(cfg, insts, logger)}

    assert (by_node["A"].taken, by_node["A"].fall_through) == ({"T", "L"}, {"F"})
    assert by_node["L"].kind is BranchKind.LOOP_EXIT


def test_early_return_side_and_rest_of_kernel(parse, logger):
    # T returns; F carries on to X: the sides only meet at the end
    cfg = make_cfg(["A", "T", "F", "X"], [("A", "T"), ("A", "F"), ("F", "X")])
    insts = {
        "A": parse(*branch_to("T")),
        "T": parse("EXIT"),
        "F": parse("FADD R5, R5, 1"),
        "X": parse("EXIT"),
    }

    [b] = branches_of(cfg, insts, logger)

    assert b.rejoin == VIRTUAL_EXIT
    assert (b.taken, b.fall_through) == ({"T"}, {"F", "X"})


def test_sides_of_if_inside_a_loop_stop_at_the_rejoin(parse, logger):
    # E -> H; H -> T, F; T, F -> J; J -> H (back edge), J -> X
    cfg = make_cfg(
        ["E", "H", "T", "F", "J", "X"],
        [
            ("E", "H"),
            ("H", "T"),
            ("H", "F"),
            ("T", "J"),
            ("F", "J"),
            ("J", "H"),
            ("J", "X"),
        ],
    )
    insts = {
        "E": [],
        "H": parse(*branch_to("T")),
        "T": parse("FADD R5, R5, 1"),
        "F": parse("FADD R5, R5, 2"),
        "J": parse("ISETP.GT.AND P1, PT, R6, RZ, PT", "@P1 BRA `(H)"),
        "X": parse("EXIT"),
    }

    by_node = {b.node: b for b in branches_of(cfg, insts, logger)}

    assert by_node["H"].rejoin == "J"
    assert (by_node["H"].taken, by_node["H"].fall_through) == ({"T"}, {"F"})


def test_nested_branch_is_inside_the_outer_side(parse, logger):
    # A -> O -> (I1 | I2) -> M; A -> F -> M
    cfg = make_cfg(
        ["A", "O", "I1", "I2", "F", "M"],
        [
            ("A", "O"),
            ("A", "F"),
            ("O", "I1"),
            ("O", "I2"),
            ("I1", "M"),
            ("I2", "M"),
            ("F", "M"),
        ],
    )
    insts = {
        "A": parse(*branch_to("O")),
        "O": parse(*branch_to("I1")),
        "I1": parse("FADD R5, R5, 1"),
        "I2": parse("FADD R5, R5, 2"),
        "F": parse("FADD R5, R5, 3"),
        "M": parse("EXIT"),
    }

    by_node = {b.node: b for b in branches_of(cfg, insts, logger)}

    assert (by_node["A"].taken, by_node["A"].fall_through) == ({"O", "I1", "I2"}, {"F"})
    assert (by_node["O"].taken, by_node["O"].fall_through) == ({"I1"}, {"I2"})


def test_loop_and_early_exits_have_no_sides(parse, logger):
    cfg, insts = loop_graph(parse)
    [loop_exit] = branches_of(cfg, insts, logger)
    assert (loop_exit.taken, loop_exit.fall_through) == (set(), set())

    cfg = make_cfg(["A", "B"], [("A", "B")])
    insts = {
        "A": parse("ISETP.GE.AND P0, PT, R0, 0x10, PT", "@P0 EXIT"),
        "B": parse("EXIT"),
    }
    [early_exit] = branches_of(cfg, insts, logger)
    assert (early_exit.taken, early_exit.fall_through) == (set(), set())


def test_probe_sides(parse, logger):
    cfg, insts = divergent_probe_with_bodies(parse)

    by_node = {b.node: b for b in branches_of(cfg, insts, logger)}

    assert by_node["BB0"].taken == {".L_x_22", "BODY2"}
    assert by_node["BB0"].fall_through == {"BB1", "BODY1"}
    # each guard jumps straight to the rejoin point when n < 1
    assert (by_node["BB1"].taken, by_node["BB1"].fall_through) == (set(), {"BODY1"})
    assert (by_node[".L_x_22"].taken, by_node[".L_x_22"].fall_through) == (
        set(),
        {"BODY2"},
    )
