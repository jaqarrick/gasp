from pathlib import Path

import pytest

from gasp.cfg import CFGEdge, format_cfg, gen_cfg_from_raw, print_cfg

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
