from dataclasses import dataclass, field
from enum import Enum, auto

from gasp.cfg import ControlFlowGraph, neighbors
from gasp.logger import Logger
from gasp.parser.constants import (
    DIVERGENT_SPECIAL_REGS,
    EXIT_OPCODES,
    UNIFORM_SPECIAL_REGS,
)
from gasp.parser.types import SASSInstruction


class BranchKind(Enum):
    UNIFORM = auto()
    DIVERGENT = auto()
    LOOP_EXIT = auto()
    EARLY_EXIT = auto()


@dataclass
class Branch:
    node: str
    kind: BranchKind
    predicate: str | None
    divergent: bool
    taken_succ: str | None = None
    fall_succ: str | None = None
    rejoin: str | None = None
    taken: set[str] = field(default_factory=set)
    fall_through: set[str] = field(default_factory=set)


VIRTUAL_EXIT = "__exit__"


def reachable_from_entry(cfg: ControlFlowGraph) -> set[str]:
    """Basic DFS to find all reachable nodes in kernel"""
    successors, _ = neighbors(cfg)
    entry = next(iter(cfg.nodes))
    visited = {entry}
    stack = [entry]
    while stack:
        for s in successors[stack.pop()]:
            if s not in visited:
                visited.add(s)
                stack.append(s)
    return visited


def post_dominators(cfg: ControlFlowGraph) -> dict[str, set[str]]:
    successors, predecessors = neighbors(cfg)
    reachable = reachable_from_entry(cfg)
    next_blocks: dict[str, list[str]] = {}
    for node in reachable:
        unique = list(set(successors[node]))
        # an empty list means there are no successors so just exit
        if unique:
            next_blocks[node] = unique
        else:
            next_blocks[node] = [VIRTUAL_EXIT]

    # a block stuck in a loop with no way out never reaches the end, so its
    # guess would never shrink: give it an edge to the virtual exit too
    reaches_end = {n for n in reachable if next_blocks[n] == [VIRTUAL_EXIT]}
    stack = list(reaches_end)
    while stack:
        for p in predecessors[stack.pop()]:
            if p in reachable and p not in reaches_end:
                reaches_end.add(p)
                stack.append(p)
    for node in reachable - reaches_end:
        next_blocks[node].append(VIRTUAL_EXIT)

    # known: nothing comes after the end
    pdom: dict[str, set[str]] = {VIRTUAL_EXIT: {VIRTUAL_EXIT}}

    # first assume all nodes are next
    all_blocks_with_exit = reachable | {VIRTUAL_EXIT}
    for node in reachable:
        pdom[node] = set(all_blocks_with_exit)

    # shrink the guesses until none changes
    changed = True
    while changed:
        changed = False
        for node in reachable:
            first, *others = next_blocks[node]
            common = set(pdom[first])
            for other in others:
                common = common & pdom[other]

            new = {node} | common
            if new != pdom[node]:
                pdom[node] = new
                changed = True

    return pdom


def immediate_post_dominator(node: str, pdom: dict[str, set[str]]) -> str | None:
    if node not in pdom:
        return None
    strict = pdom[node] - {node}
    return max(strict, key=lambda d: len(pdom[d]))


def side_blocks(
    start: str, rejoin: str, branch_node: str, successors: dict[str, list[str]]
) -> set[str]:
    """Blocks reachable from start before the rejoin point."""
    if start == rejoin:
        return set()
    visited = {start}
    stack = [start]
    while stack:
        for s in successors[stack.pop()]:
            if s in (rejoin, branch_node) or s in visited:
                continue
            visited.add(s)
            stack.append(s)
    return visited


def leaves_loop(node: str, succs: set[str], loops: dict[str, set[str]]) -> bool:
    """True if a branch at node has one successor inside a loop containing
    node and one outside it."""
    for loop_nodes in loops.values():
        if node in loop_nodes and len(succs & loop_nodes) == 1:
            return True
    return False


def find_branches(
    cfg: ControlFlowGraph,
    all_instructions: dict[str, list[SASSInstruction]],
    end_state: dict[str, set[str]],
    loops: dict[str, set[str]],
    logger: Logger,
) -> list[Branch]:
    successors, _ = neighbors(cfg)
    pdom = post_dominators(cfg)
    branches: list[Branch] = []

    for node_id in cfg.nodes:
        instructions = all_instructions.get(node_id, [])
        if not instructions:
            continue
        last = instructions[-1]  # get last instruction (where branches live)
        successors_node = set(successors[node_id])

        # check if the last instruction's predicate is in the node's registers
        is_divergent = (
            last.predicate is not None and last.predicate in end_state[node_id]
        )

        if last.opcode in EXIT_OPCODES and last.predicate is not None:
            # early exit
            branches.append(
                Branch(
                    node_id,
                    BranchKind.EARLY_EXIT,
                    last.predicate,
                    divergent=is_divergent,
                )
            )
            continue

        # for now, we just track 2-way branches
        num_successors = len(successors_node)
        if num_successors != 2:
            if num_successors > 2:
                logger.trace(
                    "divergence",
                    f"node {node_id} has {num_successors}-way branch. Skipping for now",
                )
            continue

        if last.branch_target not in successors_node:
            logger.warning(
                f"divergence: branch at {node_id} targets {last.branch_target}, "
                f"not one of its successors {sorted(successors_node)}; skipped"
            )
            continue

        taken_succ = last.branch_target
        for s in successors_node:
            if s != taken_succ:
                fall_succ = s

        rejoin = None
        taken: set[str] = set()
        fall_through: set[str] = set()
        if leaves_loop(node_id, successors_node, loops):
            kind = BranchKind.LOOP_EXIT
        else:
            kind = BranchKind.DIVERGENT if is_divergent else BranchKind.UNIFORM
            rejoin = immediate_post_dominator(node_id, pdom)
            if rejoin is not None:
                taken = side_blocks(taken_succ, rejoin, node_id, successors)
                fall_through = side_blocks(fall_succ, rejoin, node_id, successors)
                if taken & fall_through:
                    logger.warning(
                        f"divergence: branch at {node_id}: sides share "
                        f"{sorted(taken & fall_through)}"
                    )

        branches.append(
            Branch(
                node_id,
                kind,
                last.predicate,
                is_divergent,
                taken_succ,
                fall_succ,
                rejoin,
                taken,
                fall_through,
            )
        )

    return branches


def is_register_uniform(reg: str, logger: Logger) -> bool:
    base = reg.split(".")[0]
    if base in UNIFORM_SPECIAL_REGS:
        return True
    if base in DIVERGENT_SPECIAL_REGS:
        return False

    logger.warning(
        f"divergence: special register {reg} is not recognized; "
        "treating it as per-thread"
    )
    return False


def analyze_divergence(
    cfg: ControlFlowGraph,
    all_instructions: dict[str, list[SASSInstruction]],
    logger: Logger,
) -> dict[str, set[str]]:
    """Per-thread registers at the end of each block."""

    successors, predecessors = neighbors(cfg)

    # track per-thread registers at the end of each block
    end_state: dict[str, set[str]] = {n: set() for n in cfg.nodes}

    queue = list(cfg.nodes)

    while queue:
        node_id = queue.pop(0)

        active_state = set()
        # inherit prev state from pred.
        for p in predecessors[node_id]:
            active_state |= end_state[p]

        for inst in all_instructions.get(node_id, []):
            # does this instruction write a per-thread value?
            inst_is_uniform = True

            # check special register dependency
            for reg in inst.special_registers:
                if not is_register_uniform(reg, logger):
                    inst_is_uniform = False

            # check if we read a per-thread register
            if any(src in active_state for src in inst.src_registers):
                inst_is_uniform = False

            # per-thread value: track all dest. regs
            if not inst_is_uniform:
                active_state |= inst.dest_registers
            else:
                active_state -= inst.dest_registers

        # if this block now passes on something new, its successors must redo
        if active_state != end_state[node_id]:
            end_state[node_id] = active_state
            queue.extend(successors[node_id])

    return end_state
