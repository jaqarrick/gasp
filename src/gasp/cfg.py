import re

# organize CFG into Nodes and Edges (based on the structure of the DOT file
# in tests/fixtures/cfg/vecadd.bbcfg.dot).
from dataclasses import dataclass


@dataclass
class CFGNode:
    id: str
    label: str
    # name of the DOT cluster the node is in: a kernel, or $kernel$callee for a subroutine
    function: str | None = None


@dataclass
class CFGEdge:
    source: str
    source_port: str | None
    source_compass: str | None
    target: str
    target_port: str | None
    target_compass: str | None
    attributes: str | None


@dataclass
class ControlFlowGraph:
    nodes: dict[str, CFGNode]
    edges: list[CFGEdge]


# this parser is scoped to the raw cfg input in tests/fixtures/cfg/vecadd.bbcfg.dot
# structure per node (specifically bbcfgs): NodeID \l label \l edges

# regexes the raw bbcfg input
_GRAPH_START = re.compile(r"^\s*digraph\b.*\{\s*$")
_CLUSTER_START = re.compile(
    r'^\s*subgraph\s+"cluster_(?P<name>(?:\\.|[^"\\])*)"\s*\{\s*$'
)
_NODE_ID = re.compile(r'^\s*"(?P<id>(?:\\.|[^"\\])*)"\s*;?\s*$')
_NODE_LABEL = re.compile(r'^\s*\[\s*label="(?P<label>(?:\\.|[^"\\])*)"\s*\]\s*;?\s*$')
_EDGE = re.compile(
    r'^\s*"(?P<source>(?:\\.|[^"\\])*)"'
    r"(?::(?P<source_port>[A-Za-z_][\w.-]*))?"
    r"(?::(?P<source_compass>n|ne|e|se|s|sw|w|nw|c|_))?"
    r"\s*->\s*"
    r'"(?P<target>(?:\\.|[^"\\])*)"'
    r"(?::(?P<target_port>[A-Za-z_][\w.-]*))?"
    r"(?::(?P<target_compass>n|ne|e|se|s|sw|w|nw|c|_))?"
    r"(?:\s*\[(?P<attributes>.*?)\])?\s*;\s*$"
)


def gen_cfg_from_raw(raw_cfg: str) -> ControlFlowGraph:
    # extract lines and prune to verify input shape
    lines = raw_cfg.splitlines()
    trimmed_lines = [line.strip() for line in lines if line.strip()]

    # error handling for missing cfg, invalid start or finish
    if (
        not trimmed_lines
        or not _GRAPH_START.fullmatch(trimmed_lines[0])
        or trimmed_lines[-1] != "}"
    ):
        raise ValueError("expected nvdisasm -bbcfg DOT graph")

    nodes: dict[str, CFGNode] = {}
    edges: list[CFGEdge] = []
    pending_node: str | None = None
    current_function: str | None = None

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        # reads nodes into nodes dict
        if pending_node is not None:
            label_match = _NODE_LABEL.fullmatch(stripped)
            if label_match is None:
                raise ValueError(
                    f"expected a label for node {pending_node!r} on line {line_number}"
                )

            if pending_node in nodes:
                raise ValueError(f"duplicate node {pending_node!r}")

            nodes[pending_node] = CFGNode(
                id=pending_node,
                label=label_match.group("label"),
                function=current_function,
            )
            pending_node = None
            continue

        # read edges into list
        edge_match = _EDGE.fullmatch(stripped)
        if edge_match is not None:
            values = edge_match.groupdict()
            edges.append(
                CFGEdge(
                    source=values["source"],
                    source_port=values["source_port"],
                    source_compass=values["source_compass"],
                    target=values["target"],
                    target_port=values["target_port"],
                    target_compass=values["target_compass"],
                    attributes=values["attributes"],
                )
            )
            continue

        cluster_match = _CLUSTER_START.fullmatch(stripped)
        if cluster_match is not None:
            current_function = cluster_match.group("name")
            continue

        if stripped == "}":
            current_function = None
            continue

        if stripped == "{" or stripped.startswith(
            ("digraph", "subgraph", "node [", "edge [", "graph [")
        ):
            continue

        # makes sure that current line is only the node ID and updates pending_node as such
        node_match = _NODE_ID.fullmatch(stripped)
        if node_match is not None:
            pending_node = node_match.group("id")
            continue
        raise ValueError(f"unrecognized nvdisasm -bbcfg line {line_number}: {stripped}")

    if pending_node is not None:
        raise ValueError(f"node {pending_node!r} is missing its label")

    if not nodes:
        raise ValueError("nvdisasm -bbcfg graph contains no nodes")

    for edge in edges:
        if edge.source not in nodes:
            raise ValueError(f"edge references undefined source node {edge.source!r}")

        if edge.target not in nodes:
            raise ValueError(f"edge references undefined target node {edge.target!r}")

    return ControlFlowGraph(nodes=nodes, edges=edges)


def filter_cfg(cfg: ControlFlowGraph, kernel: str) -> ControlFlowGraph:
    """The part of a multi-function graph that belongs to one kernel.

    The kernel's own nodes come first, so the first node is its entry. Subroutines
    of the kernel stay in the graph; nvdisasm names their clusters $kernel$callee.
    """
    callee_prefix = f"${kernel}$"
    own = {k: n for k, n in cfg.nodes.items() if n.function == kernel}
    if not own:
        raise ValueError(f"no CFG nodes found for kernel {kernel!r}")
    callees = {
        k: n
        for k, n in cfg.nodes.items()
        if n.function is not None and n.function.startswith(callee_prefix)
    }
    nodes = {**own, **callees}
    edges = [e for e in cfg.edges if e.source in nodes and e.target in nodes]
    return ControlFlowGraph(nodes=nodes, edges=edges)


def format_cfg(cfg: ControlFlowGraph) -> str:
    """Render a ControlFlowGraph as readable text."""
    out = [f"=== CFG: {len(cfg.nodes)} nodes, {len(cfg.edges)} edges ===", ""]

    for node_id, node in cfg.nodes.items():
        raw = node.label.replace("\\l", "\n")
        raw = raw.replace("\\\\", "\x00")
        raw = re.sub(r"\\(.)", r"\1", raw)
        raw = raw.replace("\x00", "\\")
        lines = [l.strip() for l in raw.splitlines() if l.strip()]

        out.append(f"[{node_id}] ({len(lines)} lines)")
        for line in lines:
            if line in ("{", "}"):
                out.append(line)
            else:
                out.append(f"    {line}")
        out.append("")

    for edge in cfg.edges:
        attr = f"  [{edge.attributes}]" if edge.attributes else ""
        port_src = f":{edge.source_port}" if edge.source_port else ""
        port_tgt = f":{edge.target_port}" if edge.target_port else ""
        out.append(f"  {edge.source}{port_src} -> {edge.target}{port_tgt}{attr}")

    out.append("")
    return "\n".join(out)


def print_cfg(cfg: ControlFlowGraph) -> None:
    """Print a ControlFlowGraph for debugging."""
    print(format_cfg(cfg))
