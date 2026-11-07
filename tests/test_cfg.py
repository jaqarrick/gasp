from pathlib import Path

import pytest

from gasp.cfg import CFGEdge, filter_cfg, format_cfg, gen_cfg_from_raw, print_cfg

# raw cfg file to compare against
FIXTURE = Path(__file__).parent / "fixtures" / "cfg" / "vecadd.bbcfg.dot"


def test_parse_vecadd_bbcfg_fixture():
    graph = gen_cfg_from_raw(FIXTURE.read_text())

    assert set(graph.nodes) == {"_Z6vecaddPfS_S_i", "BB0"}
    assert "ISETP.GE.AND" in graph.nodes["_Z6vecaddPfS_S_i"].label
    assert "LDG.E" in graph.nodes["BB0"].label

    assert graph.edges == [
        CFGEdge(
            source="_Z6vecaddPfS_S_i",
            source_port="exit0",
            source_compass="s",
            target="BB0",
            target_port="entry",
            target_compass="n",
            attributes="style=solid",
        )
    ]


def test_parse_edge_without_ports_or_attributes():
    raw_cfg = """\
digraph f {
"kernel"
[label="function label"]
"BB0"
[label="block label"]
"kernel" -> "BB0";
}
"""

    graph = gen_cfg_from_raw(raw_cfg)

    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.source == "kernel"
    assert edge.source_port is None
    assert edge.source_compass is None
    assert edge.target == "BB0"
    assert edge.target_port is None
    assert edge.target_compass is None
    assert edge.attributes is None


@pytest.mark.parametrize(
    "raw_cfg",
    [
        "",
        "not a DOT graph",
        'digraph f {\n"kernel"\n}',
    ],
)
def test_parse_invalid_graph_raises(raw_cfg):
    with pytest.raises(ValueError):
        gen_cfg_from_raw(raw_cfg)


def test_parse_edge_to_undefined_node_raises():
    raw_cfg = """\
digraph f {
"kernel"
[label="function label"]
"kernel" -> "BB0";
}
"""

    with pytest.raises(ValueError, match="undefined target node 'BB0'"):
        gen_cfg_from_raw(raw_cfg)


def test_parse_unrecognized_line_raises():
    raw_cfg = """\
digraph f {
this is not valid nvdisasm output
}
"""

    with pytest.raises(ValueError, match="unrecognized"):
        gen_cfg_from_raw(raw_cfg)


def test_format_cfg_lists_nodes_and_edges(capsys):
    graph = gen_cfg_from_raw(FIXTURE.read_text())
    text = format_cfg(graph)

    assert text.startswith("=== CFG: 2 nodes, 1 edges ===\n")
    assert "[_Z6vecaddPfS_S_i]" in text
    assert "[BB0]" in text
    assert "  _Z6vecaddPfS_S_i:exit0 -> BB0:entry  [style=solid]" in text

    print_cfg(graph)
    assert capsys.readouterr().out == text + "\n"


# same shape as nvdisasm -bbcfg on a file with several kernels: one cluster per
# function, and $kernel$callee clusters for subroutines
MULTI_KERNEL = """\
digraph f {
subgraph "cluster__Z1av" {
node [fontname="Courier",fontsize=10,shape=Mrecord];
"_Z1av"
[label="{<entry>a entry\\l|<exit0>}"];
"_Z1av":exit0:s -> "BB0":entry:n [style=solid];
"BB0"
[label="{<entry>a block\\l}"];
}
subgraph "cluster_$_Z1av$_Z6helperf" {
node [fontname="Courier",fontsize=10,shape=Mrecord];
"$_Z1av$_Z6helperf"
[label="{<entry>a helper\\l}"];
}
subgraph "cluster__Z1bv" {
node [fontname="Courier",fontsize=10,shape=Mrecord];
"_Z1bv"
[label="{<entry>b entry\\l|<exit0>}"];
"_Z1bv":exit0:s -> ".L_x_1":entry:n [style=solid];
".L_x_1"
[label="{<entry>b loop\\l|<exit0>}"];
".L_x_1":exit0:e -> ".L_x_1":entry:n [style=solid];
}
}
"""


def test_parse_records_each_nodes_function():
    graph = gen_cfg_from_raw(MULTI_KERNEL)

    assert graph.nodes["_Z1av"].function == "_Z1av"
    assert graph.nodes["BB0"].function == "_Z1av"
    assert graph.nodes["$_Z1av$_Z6helperf"].function == "$_Z1av$_Z6helperf"
    assert graph.nodes[".L_x_1"].function == "_Z1bv"


def test_parse_node_outside_a_cluster_has_no_function():
    raw_cfg = 'digraph f {\n"kernel"\n[label="function label"]\n}\n'

    assert gen_cfg_from_raw(raw_cfg).nodes["kernel"].function is None


def test_filter_cfg_keeps_only_the_selected_kernel():
    graph = filter_cfg(gen_cfg_from_raw(MULTI_KERNEL), "_Z1bv")

    assert list(graph.nodes) == ["_Z1bv", ".L_x_1"]
    assert {(e.source, e.target) for e in graph.edges} == {
        ("_Z1bv", ".L_x_1"),
        (".L_x_1", ".L_x_1"),
    }


def test_filter_cfg_keeps_the_kernels_subroutines():
    graph = filter_cfg(gen_cfg_from_raw(MULTI_KERNEL), "_Z1av")

    assert list(graph.nodes) == ["_Z1av", "BB0", "$_Z1av$_Z6helperf"]
    assert [(e.source, e.target) for e in graph.edges] == [("_Z1av", "BB0")]


def test_filter_cfg_puts_the_kernel_entry_first():
    # a subroutine's cluster listed before its kernel must not become the entry
    raw_cfg = """\
digraph f {
subgraph "cluster_$k$helper" {
"$k$helper"
[label="helper"]
}
subgraph "cluster_k" {
"k"
[label="entry"]
}
}
"""
    assert next(iter(filter_cfg(gen_cfg_from_raw(raw_cfg), "k").nodes)) == "k"


def test_filter_cfg_unknown_kernel_raises():
    with pytest.raises(ValueError, match="_Z7missingv"):
        filter_cfg(gen_cfg_from_raw(MULTI_KERNEL), "_Z7missingv")
