from dataclasses import dataclass

from gasp.cfg import ControlFlowGraph
from gasp.logger import Logger
from gasp.parser.constants import (
    GLOBAL_MEMORY_CATEGORIES,
    MEMORY_LOAD_CATEGORIES,
    OpCategory,
)
from gasp.parser.types import SASSInstruction


@dataclass
class DependencyOutput:
    ilp: float
    mlp: float
    total_inst_count: int
    mem_inst_count: int
    sfu_inst_count: int
    sync_inst_count: int
    # sum of instruction latencies weighted by execution frequency (one thread, no overlap)
    total_latency: float
    # global memory bytes one thread touches
    # assume one access width per distinct (definition, base, offset) per execution;
    # assume every execution touches new data (no reuse)
    # may need to change this model
    global_bytes_per_thread: int = 0
    num_address_groups: int = 0


@dataclass
class LatencyTable:
    by_category: dict[OpCategory, float]
    default: float

    def get(self, inst: SASSInstruction) -> float:
        return self.by_category.get(inst.category, self.default)


def is_memory_load(inst: SASSInstruction) -> bool:
    return inst.category in MEMORY_LOAD_CATEGORIES


AddressKey = tuple[int | None, str | None, int]


def get_address_groups(instructions: list[SASSInstruction]) -> dict[AddressKey, int]:
    """Distinct global addresses in one node, keyed by (index of the
    instruction that last wrote the base, base register, immediate offset).

    Accesses with the same (definition, base) read the same base value, so they
    are taken as the same array; different offsets from it are different
    elements, e.g. an unrolled loop's [R2+0x4], [R2+0x8], ... A base with no
    definition in the node gets None as its index.
    Value: the largest access width at that address, in bytes.
    """
    last_written: dict[str, int] = {}
    groups: dict[AddressKey, int] = {}
    for i, inst in enumerate(instructions):
        if inst.category in GLOBAL_MEMORY_CATEGORIES:
            base = inst.address_base_register
            definition = last_written.get(base) if base else None
            key = (definition, base, inst.address_offset or 0)
            width = inst.access_width_bytes or 0
            groups[key] = max(groups.get(key, 0), width)
        for reg in inst.dest_registers:
            last_written[reg] = i
    return groups


def count_arrays(groups: dict[AddressKey, int]) -> int:
    """Distinct (definition, base) pairs: the arrays a node accesses."""
    return len({(definition, base) for definition, base, _ in groups})


def get_ilp_mlp_for_node(instructions: list[SASSInstruction], latency: LatencyTable):
    last_written = {}
    # src, target, weight
    edges: set[tuple[int, int, float]] = set()
    for i, inst in enumerate(instructions):
        for reg in inst.src_registers:
            last = last_written.get(reg)
            if last is not None:
                # the consumer waits for the producer's result
                weight = latency.get(instructions[last])
                edges.add((last, i, weight))

        for reg in inst.dest_registers:
            last_written[reg] = i

    # find most expensive path in graph
    # dist[i] = earliest issue time of instruction i
    dist = [0.0] * len(instructions)

    adj = [[] for _ in range(len(instructions))]
    for src, tgt, weight in edges:
        adj[src].append((tgt, weight))

    for i in range(len(instructions)):
        for tgt, weight in adj[i]:
            dist[tgt] = max(dist[tgt], dist[i] + weight)

    total_latency = sum(latency.get(inst) for inst in instructions)
    # the path ends when its last instruction completes
    critical_path = max(d + latency.get(inst) for d, inst in zip(dist, instructions))
    ilp: float = total_latency / critical_path

    # mlp
    # memory loads on longest path from start of block to instruction i, inclusive
    load_count = [0] * len(instructions)

    for i, inst in enumerate(instructions):
        if is_memory_load(inst):
            load_count[i] += 1
        for j, _ in adj[i]:
            # propagate load count to dependent instructions
            load_count[j] = max(load_count[j], load_count[i])

    longest_mem_chain = max(load_count)
    total_loads = sum(1 for inst in instructions if is_memory_load(inst))
    if longest_mem_chain < 1:
        return ilp, -1, total_latency

    mlp: float = total_loads / longest_mem_chain

    return ilp, mlp, total_latency


class LoopIterationsRequiredError(ValueError):
    def __init__(self, loop_headers: list[str]):
        self.loop_headers = loop_headers
        super().__init__(
            f"kernel contains {len(loop_headers)} loop(s) "
            f"(headers: {', '.join(loop_headers)}) "
            "but no loop iteration count was supplied"
        )


class LoopCountMismatchError(ValueError):
    def __init__(self, loop_headers: list[str], given: int):
        self.loop_headers = loop_headers
        self.given = given
        super().__init__(
            f"loop_counts has {given} number(s), but the kernel has "
            f"{len(loop_headers)} loop(s)"
        )


@dataclass
class LoopInfo:
    """A loop in the compiled code, for telling loops apart."""

    header: str
    nodes: set[str]
    inst_count: int  # instructions in the loop's blocks, nested loops included
    global_loads: int
    shared_loads: int
    parent: str | None  # header of the innermost enclosing loop


UNVISITED, IN_PROGRESS, DONE = 0, 1, 2


def dfs(u, state, back_edges, successors):
    state[u] = IN_PROGRESS
    for v in successors[u]:
        if state[v] == UNVISITED:
            dfs(v, state, back_edges, successors)
        elif state[v] == IN_PROGRESS:
            back_edges.append((u, v))
    state[u] = DONE


def dfs_cfg(cfg, successors):
    state = {node_id: UNVISITED for node_id in cfg.nodes}
    back_edges = []
    entry_node = next(iter(cfg.nodes))
    dfs(entry_node, state, back_edges, successors)
    return back_edges


def find_loops(cfg: ControlFlowGraph) -> tuple[dict[str, set[str]], list]:
    """Loops by header, in code order, and the back edges (tail, header)."""
    # loop detection: perform DFS on the CFG to discover back edges
    successors = {node_id: [] for node_id in cfg.nodes}
    predecessors = {node_id: [] for node_id in cfg.nodes}

    for edge in cfg.edges:
        predecessors[edge.target].append(edge.source)
        successors[edge.source].append(edge.target)

    back_edges = dfs_cfg(cfg, successors)

    # one loop per header: several back edges to the same head (e.g. `continue`)
    # are the same loop and must only multiply by its count once
    loops: dict[str, set[str]] = {}
    for tail, head in back_edges:
        loop_nodes = loops.setdefault(head, {head})
        queue = [tail]
        while len(queue) > 0:
            node = queue.pop()
            if node not in loop_nodes:
                loop_nodes.add(node)
                queue.extend(predecessors[node])

    # DFS finds back edges in no useful order; nvdisasm lists nodes by address,
    # so this is the order loop_counts follows
    position = {node_id: i for i, node_id in enumerate(cfg.nodes)}
    ordered = dict(sorted(loops.items(), key=lambda item: position[item[0]]))
    return ordered, back_edges


def describe_loops(
    cfg: ControlFlowGraph, all_instructions: dict[str, list[SASSInstruction]]
) -> list[LoopInfo]:
    """The kernel's loops in code order, with what helps match them to the source."""
    loops, _ = find_loops(cfg)
    infos = []
    for head, nodes in loops.items():
        insts = [i for n in nodes for i in all_instructions.get(n, [])]
        enclosing = [
            other
            for other, other_nodes in loops.items()
            if other != head and nodes < other_nodes
        ]
        parent = min(enclosing, key=lambda o: len(loops[o])) if enclosing else None
        infos.append(
            LoopInfo(
                header=head,
                nodes=nodes,
                inst_count=len(insts),
                global_loads=sum(i.category is OpCategory.GLOBAL_LOAD for i in insts),
                shared_loads=sum(i.category is OpCategory.SHARED_LOAD for i in insts),
                parent=parent,
            )
        )
    return infos


def analyze(
    cfg: ControlFlowGraph,
    all_instructions: dict[str, list[SASSInstruction]],
    loop_iterations: int | None,
    latency: LatencyTable,
    logger: Logger,
    loop_counts: list[int] | None = None,
) -> DependencyOutput:
    """loop_counts: trips per entry for each loop, in code order. It wins over
    loop_iterations, which gives every loop the same count."""

    loops, back_edges = find_loops(cfg)

    if loop_counts is not None and loops:
        if len(loop_counts) != len(loops):
            raise LoopCountMismatchError(list(loops), len(loop_counts))
        counts = dict(zip(loops, loop_counts, strict=True))
    elif loops:
        if loop_iterations is None:
            raise LoopIterationsRequiredError(list(loops))
        counts = dict.fromkeys(loops, loop_iterations)
    else:
        counts = {}

    # from loop we can determine how many times a node repeats
    node_iterations = {node_id: 1 for node_id in cfg.nodes}
    for head, loop_nodes in loops.items():
        for node in loop_nodes:
            node_iterations[node] *= counts[head]

    logger.info(f"Dependency analysis: {len(loops)} loop(s) detected")
    logger.trace("dependency", f"loop_iterations: {loop_iterations}")
    logger.trace("dependency", f"loop_counts: {loop_counts}")
    logger.trace("dependency", f"back edges: {back_edges or '-'}")
    for head, loop_nodes in loops.items():
        logger.trace(
            "dependency", f"loop {head}: {sorted(loop_nodes)} count={counts[head]}"
        )

    ilps = {}
    mlps = {}
    total_inst_count = 0
    mem_inst_count = 0
    sfu_inst_count = 0
    sync_inst_count = 0
    global_bytes_per_thread = 0
    num_address_groups = 0

    latencies = {}
    for node_id, node in all_instructions.items():
        if not node:
            continue
        groups = get_address_groups(node)
        num_address_groups += count_arrays(groups)
        global_bytes_per_thread += sum(groups.values()) * node_iterations[node_id]
        if groups:
            logger.trace(
                "dependency",
                f"node {node_id}: {count_arrays(groups)} array(s), addresses "
                + ", ".join(
                    f"{base}@{d}{off:+#x}={w}B" for (d, base, off), w in groups.items()
                ),
            )
        # ilp/mlp per node
        ilp, mlp, total_latency = get_ilp_mlp_for_node(node, latency)
        ilps[node_id] = ilp
        mlps[node_id] = mlp
        latencies[node_id] = total_latency
        logger.trace(
            "dependency",
            f"node {node_id}: {len(node)} inst(s) x{node_iterations[node_id]} "
            f"ilp={ilp:.2f} mlp={mlp:.2f} latency={total_latency:.0f}",
        )

        # instructions by category
        node_iter_count = node_iterations[node_id]

        for inst in node:
            total_inst_count += node_iter_count
            # global loads + stores feed T_mem (Eq. 11)
            if inst.category in GLOBAL_MEMORY_CATEGORIES:
                mem_inst_count += node_iter_count
            if inst.category is OpCategory.SFU:
                sfu_inst_count += node_iter_count
            if inst.category is OpCategory.BARRIER:
                sync_inst_count += node_iter_count

    weighted_ilp_sum = 0
    weight_sum = 0

    for node_id, ilp in ilps.items():
        w = latencies[node_id] * node_iterations[node_id]
        weighted_ilp_sum += ilp * w
        weight_sum += w

    weighted_mlp_sum = 0
    mlp_weight_sum = 0

    for node_id, mlp in mlps.items():
        if mlp > 0:
            w = latencies[node_id] * node_iterations[node_id]
            weighted_mlp_sum += mlp * w
            mlp_weight_sum += w

    kernel_mlp = weighted_mlp_sum / mlp_weight_sum if mlp_weight_sum > 0 else 1
    kernel_ilp = weighted_ilp_sum / weight_sum if weight_sum > 0 else 1

    return DependencyOutput(
        ilp=kernel_ilp,
        mlp=kernel_mlp,
        total_inst_count=total_inst_count,
        mem_inst_count=mem_inst_count,
        sfu_inst_count=sfu_inst_count,
        sync_inst_count=sync_inst_count,
        total_latency=weight_sum,
        global_bytes_per_thread=global_bytes_per_thread,
        num_address_groups=num_address_groups,
    )
