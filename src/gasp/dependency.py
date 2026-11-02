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


@dataclass
class LatencyTable:
    by_category: dict[OpCategory, float]
    default: float

    def get(self, inst: SASSInstruction) -> float:
        return self.by_category.get(inst.category, self.default)


def is_memory_load(inst: SASSInstruction) -> bool:
    return inst.category in MEMORY_LOAD_CATEGORIES


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


def analyze(
    cfg: ControlFlowGraph,
    all_instructions: dict[str, list[SASSInstruction]],
    ni: int,
    latency: LatencyTable,
    logger: Logger,
) -> DependencyOutput:

    # loop detection: perform DFS on the CFG to discover back edges
    successors = {node_id: [] for node_id in cfg.nodes}
    predecessors = {node_id: [] for node_id in cfg.nodes}

    for edge in cfg.edges:
        predecessors[edge.target].append(edge.source)
        successors[edge.source].append(edge.target)

    back_edges = dfs_cfg(cfg, successors)

    # one loop per header: several back edges to the same head (e.g. `continue`)
    # are the same loop and must only multiply by ni once
    loops: dict[str, set[str]] = {}
    for tail, head in back_edges:
        loop_nodes = loops.setdefault(head, {head})
        queue = [tail]
        while len(queue) > 0:
            node = queue.pop()
            if node not in loop_nodes:
                loop_nodes.add(node)
                queue.extend(predecessors[node])

    # from loop we can determine how many times a node repeats
    node_iterations = {node_id: 1 for node_id in cfg.nodes}
    for loop_nodes in loops.values():
        for node in loop_nodes:
            node_iterations[node] *= ni

    logger.info(f"Dependency analysis: {len(loops)} loop(s) detected")
    logger.trace("dependency", f"back edges: {back_edges or '-'}")
    for head, loop_nodes in loops.items():
        logger.trace("dependency", f"loop {head}: {sorted(loop_nodes)}")

    ilps = {}
    mlps = {}
    total_inst_count = 0
    mem_inst_count = 0
    sfu_inst_count = 0
    sync_inst_count = 0

    latencies = {}
    for node_id, node in all_instructions.items():
        if not node:
            continue
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
    )
