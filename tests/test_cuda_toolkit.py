"""Tests for gasp.cuda_toolkit — reading nvdisasm output."""

from pathlib import Path

from gasp.cuda_toolkit import parse_label_lines

SASS_DIR = Path(__file__).parent / "fixtures" / "sass"
MM_BASIC_CU = "benchmarks/mm/mm_basic.cu"


def test_label_lines_from_mm_basic():
    text = (SASS_DIR / "mm_basic.sm_75.lineinfo.sass").read_text()
    lines = parse_label_lines(text)

    # the three loops all come from the k loop body
    for label in (".L_x_4", ".L_x_2", ".L_x_6"):
        assert lines[label] == (MM_BASIC_CU, 13)
    assert lines[".L_x_5"] == (MM_BASIC_CU, 12)
    assert lines[".L_x_0"] == (MM_BASIC_CU, 15)


def test_label_without_its_own_line_comment_keeps_the_last_line():
    # .L_x_3 has no //## File comment before its first instruction
    text = (SASS_DIR / "mm_basic.sm_75.lineinfo.sass").read_text()
    lines = parse_label_lines(text)
    assert ".L_x_3" in lines


def test_label_lines_inline():
    text = """\
.L_x_1:
\t//## File "k.cu", line 7
        /*0010*/                   MOV R1, RZ ;
\t//## File "k.cu", line 9
        /*0020*/                   MOV R2, RZ ;
.L_x_2:
        /*0030*/                   EXIT ;
"""
    assert parse_label_lines(text) == {".L_x_1": ("k.cu", 7), ".L_x_2": ("k.cu", 9)}


def test_no_line_info():
    assert parse_label_lines(".L_x_1:\n        /*0010*/   EXIT ;\n") == {}
