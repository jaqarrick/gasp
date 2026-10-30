"""Tests for gasp.parser — SASS instruction parsing against nvdisasm fixtures."""

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gasp.cfg import ControlFlowGraph, gen_cfg_from_raw
from gasp.logger import Logger
from gasp.parser import get_parser
from gasp.parser.bracket import BracketSASSParser
from gasp.parser.descriptor import DescriptorSASSParser
from gasp.parser.types import OpCategory, SASSInstruction

FIXTURES = Path(__file__).parent / "fixtures"
SASS_DIR = FIXTURES / "sass"
CFG_DIR = FIXTURES / "cfg"
SGEMM = "parboil/sgemm"

# one instruction per match: "/*0040*/   IMAD R6, R6, c[0x0][0x0], R3 ;"
# single-line only, and must start with a predicate or opcode, so data-section
# lines like "/*0010*/  .byte 0xff, ..." are skipped
_SASS_LINE = re.compile(r"/\*[0-9a-f]{4,}\*/[ \t]+(?P<instr>[@A-Z][^;\n]*?)[ \t]*;")


def sass_instructions(path: Path) -> list[str]:
    return [m.group("instr") for m in _SASS_LINE.finditer(path.read_text())]


def base_opcode(line: str) -> str:
    tokens = line.split()
    opcode = tokens[1] if tokens[0].startswith("@") else tokens[0]
    return opcode.split(".")[0]


def instr(opcode, category, src=(), dst=(), predicate=None) -> SASSInstruction:
    return SASSInstruction(
        opcode=opcode,
        category=category,
        predicate=predicate,
        src_registers=set(src),
        dest_registers=set(dst),
    )


@pytest.fixture
def logger():
    # spec=Logger so calling a method Logger doesn't have fails the test
    return MagicMock(spec=Logger)


def make_parser(arch, logger, cfg=None):
    cfg = cfg or ControlFlowGraph(nodes={}, edges=[])
    return get_parser(cfg=cfg, kernel_name="kernel", arch=arch, logger=logger)


# ---------------------------------------------------------------------------
# get_parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arch, parser_cls",
    [
        ("sm_75", BracketSASSParser),
        ("sm_80", BracketSASSParser),
        ("sm_86", BracketSASSParser),
        ("sm_89", BracketSASSParser),
        ("sm_90", DescriptorSASSParser),
        ("sm_90a", DescriptorSASSParser),
        ("sm_100", DescriptorSASSParser),
    ],
)
def test_get_parser_selects_addressing_style(arch, parser_cls, logger):
    assert isinstance(make_parser(arch, logger), parser_cls)


def test_ldcu_only_in_sm100_table(logger):
    assert make_parser("sm_100", logger).opcode_table["LDCU"] == OpCategory.CONST_LOAD
    assert "LDCU" not in make_parser("sm_90", logger).opcode_table


@pytest.mark.parametrize("arch", ["sm_70", "compute_80", "80"])
def test_get_parser_rejects_unsupported_arch(arch, logger):
    with pytest.raises(SystemExit):
        make_parser(arch, logger)
    logger.error.assert_called_once()


# ---------------------------------------------------------------------------
# _parse_instruction — lines taken from the fixtures
# ---------------------------------------------------------------------------

BRACKET_CASES = [
    pytest.param(
        "MOV R1, c[0x0][0x28]",
        instr("MOV", OpCategory.MOVEMENT, dst={"R1"}),
        id="constant-bank-operand",
    ),
    pytest.param(
        "S2R R6, SR_CTAID.X",
        instr("S2R", OpCategory.MISC, dst={"R6"}),
        id="special-register",
    ),
    pytest.param(
        "IMAD R6, R6, c[0x0][0x0], R3",
        instr("IMAD", OpCategory.INT, src={"R6", "R3"}, dst={"R6"}),
        id="dest-also-source",
    ),
    pytest.param(
        "IMAD.WIDE R2, R6.reuse, R7.reuse, c[0x0][0x160]",
        instr("IMAD", OpCategory.INT, src={"R6", "R7"}, dst={"R2", "R3"}),
        id="wide-dest-and-reuse",
    ),
    pytest.param(
        "ISETP.GE.AND P0, PT, R6, c[0x0][0x178], PT",
        instr("ISETP", OpCategory.INT, src={"R6"}, dst={"P0"}),
        id="setp-predicate-dest",
    ),
    pytest.param(
        "@P0 EXIT",
        instr("EXIT", OpCategory.CONTROL, src={"P0"}, predicate="P0"),
        id="guard-predicate",
    ),
    pytest.param(
        "@!P0 BRA `(.L_x_0)",
        instr("BRA", OpCategory.CONTROL, src={"P0"}, predicate="P0"),
        id="negated-guard-and-branch-target",
    ),
    pytest.param(
        "HFMA2.MMA R7, -RZ, RZ, 0, 2.384185791015625e-07",
        instr("HFMA2", OpCategory.FP32, dst={"R7"}),
        id="float-literal",
    ),
    pytest.param(
        "ULDC.64 UR4, c[0x0][0x118]",
        instr("ULDC", OpCategory.UNIFORM, dst={"UR4", "UR5"}),
        id="uniform-64-dest",
    ),
    pytest.param(
        "LDG.E R4, [R4.64]",
        instr("LDG", OpCategory.GLOBAL_LOAD, src={"R4", "R5"}, dst={"R4"}),
        id="global-load",
    ),
    pytest.param(
        "STG.E [R6.64], R9",
        instr("STG", OpCategory.GLOBAL_STORE, src={"R6", "R7", "R9"}),
        id="global-store",
    ),
    pytest.param(
        "STS [R3.X4], R18",
        instr("STS", OpCategory.SHARED_STORE, src={"R3", "R18"}),
        id="scaled-address",
    ),
    pytest.param(
        "LDS.128 R12, [RZ]",
        instr("LDS", OpCategory.SHARED_LOAD, dst={"R12", "R13", "R14", "R15"}),
        id="128-dest-zero-address",
    ),
    pytest.param(
        "LDS.128 R8, [0x10]",
        instr("LDS", OpCategory.SHARED_LOAD, dst={"R8", "R9", "R10", "R11"}),
        id="immediate-address",
    ),
    pytest.param(
        "UISETP.GE.AND UP0, UPT, UR4, UR6, UPT",
        instr("UISETP", OpCategory.UNIFORM, src={"UR4", "UR6"}, dst={"UP0"}),
        id="uniform-predicate-dest",
    ),
    pytest.param(
        "PLOP3.LUT P0, PT, PT, PT, UP0, 0x80, 0x0",
        instr("PLOP3", OpCategory.PREDICATE, src={"UP0"}, dst={"P0"}),
        id="predicate-logic",
    ),
    pytest.param(
        "BAR.SYNC.DEFER_BLOCKING 0x0",
        instr("BAR", OpCategory.BARRIER),
        id="barrier",
    ),
]

DESCRIPTOR_CASES = [
    pytest.param(
        "LDC R1, c[0x0][0x28]",
        instr("LDC", OpCategory.CONST_LOAD, dst={"R1"}),
        id="constant-load",
    ),
    pytest.param(
        "LDC.64 R2, c[0x0][0x210]",
        instr("LDC", OpCategory.CONST_LOAD, dst={"R2", "R3"}),
        id="constant-load-64",
    ),
    pytest.param(
        "IMAD R9, R9, UR4, R0",
        instr("IMAD", OpCategory.INT, src={"R9", "UR4", "R0"}, dst={"R9"}),
        id="uniform-source",
    ),
    pytest.param(
        "LDG.E R3, desc[UR4][R2.64]",
        instr("LDG", OpCategory.GLOBAL_LOAD, src={"UR4", "R2", "R3"}, dst={"R3"}),
        id="descriptor-load",
    ),
    pytest.param(
        "STG.E desc[UR4][R6.64], R9",
        instr("STG", OpCategory.GLOBAL_STORE, src={"UR4", "R6", "R7", "R9"}),
        id="descriptor-store",
    ),
    pytest.param(
        "LDS.128 R12, [R0+UR6+-0x40]",
        instr(
            "LDS",
            OpCategory.SHARED_LOAD,
            src={"R0", "UR6"},
            dst={"R12", "R13", "R14", "R15"},
        ),
        id="register-plus-uniform-offset",
    ),
    pytest.param(
        "ULEA UR6, UR6, UR4, 0x18",
        instr("ULEA", OpCategory.UNIFORM, src={"UR6", "UR4"}, dst={"UR6"}),
        id="uniform-lea",
    ),
    pytest.param(
        "IADD3.X R19, PT, PT, R0, R0, RZ, P0, !PT",
        instr("IADD3", OpCategory.INT, src={"R0", "P0"}, dst={"R19"}),
        id="carry-in-and-negated-pt",
    ),
]

SM100_CASES = [
    pytest.param(
        "LDCU UR4, c[0x0][0x360]",
        instr("LDCU", OpCategory.CONST_LOAD, dst={"UR4"}),
        id="ldcu",
    ),
    pytest.param(
        "LDCU.64 UR8, c[0x0][0x358]",
        instr("LDCU", OpCategory.CONST_LOAD, dst={"UR8", "UR9"}),
        id="ldcu-64",
    ),
]


@pytest.mark.parametrize("line, expected", BRACKET_CASES)
def test_bracket_parse_instruction(line, expected, logger):
    parser = make_parser("sm_80", logger)

    assert parser._parse_instruction(line) == expected
    logger.warning.assert_not_called()


@pytest.mark.parametrize("arch", ["sm_90", "sm_100"])
@pytest.mark.parametrize("line, expected", DESCRIPTOR_CASES)
def test_descriptor_parse_instruction(arch, line, expected, logger):
    parser = make_parser(arch, logger)

    assert parser._parse_instruction(line) == expected
    logger.warning.assert_not_called()


@pytest.mark.parametrize("line, expected", SM100_CASES)
def test_sm100_parse_instruction(line, expected, logger):
    parser = make_parser("sm_100", logger)

    assert parser._parse_instruction(line) == expected
    logger.warning.assert_not_called()


def test_srz_is_not_a_register(logger):
    parsed = make_parser("sm_80", logger)._parse_instruction("CS2R R16, SRZ")

    assert parsed.category == OpCategory.MISC
    assert "SRZ" not in parsed.src_registers
    assert "R16" in parsed.dest_registers
    logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# register widths — operands that cover more than one 32-bit register
# ---------------------------------------------------------------------------

WIDTH_CASES = [
    pytest.param(
        "IMAD.WIDE R36, R2, 0x4, R40",
        instr("IMAD", OpCategory.INT, src={"R2", "R40", "R41"}, dst={"R36", "R37"}),
        id="wide-addend",
    ),
    pytest.param(
        "IMAD.WIDE.U32 R4, R2, R3, R6",
        instr("IMAD", OpCategory.INT, src={"R2", "R3", "R6", "R7"}, dst={"R4", "R5"}),
        id="wide-u32-32-bit-factors",
    ),
    pytest.param(
        "UIMAD.WIDE.U32 UR4, UR6, UR8, UR10",
        instr(
            "UIMAD",
            OpCategory.UNIFORM,
            src={"UR6", "UR8", "UR10", "UR11"},
            dst={"UR4", "UR5"},
        ),
        id="uniform-wide-addend",
    ),
    pytest.param(
        "STG.E.128 [R2.64], R4",
        instr(
            "STG",
            OpCategory.GLOBAL_STORE,
            src={"R2", "R3", "R4", "R5", "R6", "R7"},
        ),
        id="store-128-data",
    ),
    pytest.param(
        "STS.64 [R3+0x10], R4",
        instr("STS", OpCategory.SHARED_STORE, src={"R3", "R4", "R5"}),
        id="store-64-data-32-bit-address",
    ),
    pytest.param(
        "LDG.E.64 R4, [R2.64]",
        instr("LDG", OpCategory.GLOBAL_LOAD, src={"R2", "R3"}, dst={"R4", "R5"}),
        id="load-64",
    ),
    pytest.param(
        "DADD R4, R2, R6",
        instr(
            "DADD",
            OpCategory.FP64,
            src={"R2", "R3", "R6", "R7"},
            dst={"R4", "R5"},
        ),
        id="fp64-pairs",
    ),
    pytest.param(
        "DFMA R4, R2, c[0x0][0x160], R6",
        instr(
            "DFMA",
            OpCategory.FP64,
            src={"R2", "R3", "R6", "R7"},
            dst={"R4", "R5"},
        ),
        id="fp64-constant-operand",
    ),
    pytest.param(
        "DSETP.GT.AND P0, PT, R2, R4, PT",
        instr("DSETP", OpCategory.FP64, src={"R2", "R3", "R4", "R5"}, dst={"P0"}),
        id="fp64-setp-predicate-not-widened",
    ),
    pytest.param(
        "MUFU.RCP64H R9, R5",
        instr("MUFU", OpCategory.SFU, src={"R5"}, dst={"R9"}),
        id="rcp64h-is-32-bit",
    ),
    pytest.param(
        "F2F.F64.F32 R2, R0",
        instr("F2F", OpCategory.CONVERSION, src={"R0"}, dst={"R2", "R3"}),
        id="f2f-wide-dest",
    ),
    pytest.param(
        "F2F.F32.F64 R0, R2",
        instr("F2F", OpCategory.CONVERSION, src={"R2", "R3"}, dst={"R0"}),
        id="f2f-wide-source",
    ),
    pytest.param(
        "F2I.F64.TRUNC R0, R2",
        instr("F2I", OpCategory.CONVERSION, src={"R2", "R3"}, dst={"R0"}),
        id="f2i-single-type-wide-source",
    ),
    pytest.param(
        "F2I.U64.TRUNC R2, R0",
        instr("F2I", OpCategory.CONVERSION, src={"R0"}, dst={"R2", "R3"}),
        id="f2i-single-type-wide-dest",
    ),
    pytest.param(
        "F2I.S64.F64.TRUNC R2, R4",
        instr("F2I", OpCategory.CONVERSION, src={"R4", "R5"}, dst={"R2", "R3"}),
        id="f2i-two-types",
    ),
    pytest.param(
        "I2F.F64 R2, R0",
        instr("I2F", OpCategory.CONVERSION, src={"R0"}, dst={"R2", "R3"}),
        id="i2f-single-type-wide-dest",
    ),
    pytest.param(
        "I2F.U32.RP R0, R5",
        instr("I2F", OpCategory.CONVERSION, src={"R5"}, dst={"R0"}),
        id="i2f-32-bit",
    ),
    pytest.param(
        "I2F.F64.S64 R2, R4",
        instr("I2F", OpCategory.CONVERSION, src={"R4", "R5"}, dst={"R2", "R3"}),
        id="i2f-two-types",
    ),
    pytest.param(
        "FRND.F64.TRUNC R2, R4",
        instr("FRND", OpCategory.CONVERSION, src={"R4", "R5"}, dst={"R2", "R3"}),
        id="frnd-f64",
    ),
    pytest.param(
        "F2F.F16.F32 R0, R2",
        instr("F2F", OpCategory.CONVERSION, src={"R2"}, dst={"R0"}),
        id="narrow-conversion-unchanged",
    ),
    pytest.param(
        "CS2R R16, SRZ",
        instr("CS2R", OpCategory.MISC, dst={"R16", "R17"}),
        id="cs2r-pair",
    ),
    pytest.param(
        "CS2R.32 R5, SR_CLOCKLO",
        instr("CS2R", OpCategory.MISC, dst={"R5"}),
        id="cs2r-32",
    ),
]


@pytest.mark.parametrize("line, expected", WIDTH_CASES)
def test_get_register_widths(line, expected, logger):
    parser = make_parser("sm_80", logger)

    assert parser._parse_instruction(line) == expected
    logger.warning.assert_not_called()


def test_wide_addend_skips_carry_predicates(logger):
    # carry predicates don't count as positions, so the addend is still R8 + R9
    parsed = make_parser("sm_80", logger)._parse_instruction(
        "IMAD.WIDE.U32.X R8, P0, R5, R7, R8, P0"
    )

    assert {"R8", "R9"} <= parsed.src_registers
    assert {"R5", "R7"} <= parsed.src_registers
    assert not {"R6", "R10"} & parsed.src_registers


# ---------------------------------------------------------------------------
# carry predicates — carry-out right after the destination is written,
# carry-in at the end is read (sm_100 sgemm epilogue address chains)
# ---------------------------------------------------------------------------

CARRY_CASES = [
    pytest.param(
        "IADD3 R17, P0, PT, R11, R11, RZ",
        instr("IADD3", OpCategory.INT, src={"R11"}, dst={"R17", "P0"}),
        id="iadd3-carry-out",
    ),
    pytest.param(
        "IADD3.X R19, PT, PT, R0, R0, RZ, P0, !PT",
        instr("IADD3", OpCategory.INT, src={"R0", "P0"}, dst={"R19"}),
        id="iadd3-x-carry-in",
    ),
    pytest.param(
        "IADD3 R2, PT, PT, R2, R5, R5",
        instr("IADD3", OpCategory.INT, src={"R2", "R5"}, dst={"R2"}),
        id="iadd3-carry-discarded",
    ),
    pytest.param(
        "IADD3 R5, R4, 0x1, RZ",
        instr("IADD3", OpCategory.INT, src={"R4"}, dst={"R5"}),
        id="iadd3-no-carry-slots",
    ),
    pytest.param(
        "LEA R4, P0, R17, R2, 0x2",
        instr("LEA", OpCategory.INT, src={"R17", "R2"}, dst={"R4", "P0"}),
        id="lea-carry-out",
    ),
    pytest.param(
        "LEA.HI.X R5, R17, R3, R19, 0x2, P0",
        instr("LEA", OpCategory.INT, src={"R17", "R3", "R19", "P0"}, dst={"R5"}),
        id="lea-hi-x-carry-in",
    ),
    pytest.param(
        "LEA R2, R3, R4, 0x2",
        instr("LEA", OpCategory.INT, src={"R3", "R4"}, dst={"R2"}),
        id="lea-no-carry",
    ),
    pytest.param(
        "UIADD3 UR4, UP0, UPT, UR6, UR8, URZ",
        instr("UIADD3", OpCategory.UNIFORM, src={"UR6", "UR8"}, dst={"UR4", "UP0"}),
        id="uiadd3-uniform-carry-out",
    ),
    pytest.param(
        "SEL R0, R1, R2, P0",
        instr("SEL", OpCategory.MOVEMENT, src={"R1", "R2", "P0"}, dst={"R0"}),
        id="non-carry-op-predicate-is-read",
    ),
]


@pytest.mark.parametrize("line, expected", CARRY_CASES)
def test_carry_predicates(line, expected, logger):
    parser = make_parser("sm_80", logger)

    assert parser._parse_instruction(line) == expected
    logger.warning.assert_not_called()


@pytest.mark.parametrize(
    "line",
    ["HMMA.16816.F32 R4, R8, R12, R4", "LDSM.16.M88.4 R4, [R2]"],
)
def test_tensor_op_warns_once(line, logger):
    parser = make_parser("sm_80", logger)

    parser._parse_instruction(line)
    parser._parse_instruction(line)

    logger.warning.assert_called_once()


def test_unknown_opcode_is_categorized_unknown(logger):
    parsed = make_parser("sm_80", logger)._parse_instruction("FOO R1, R2")

    assert parsed.category == OpCategory.UNKNOWN
    logger.warning.assert_called()


# ---------------------------------------------------------------------------
# every instruction in the .sass fixtures
# ---------------------------------------------------------------------------

SASS_FIXTURES = [
    pytest.param(arch, SASS_DIR / name, id=name)
    for arch, name in [
        ("sm_80", "vecadd.sm_80.sass"),
        ("sm_90", "vecadd.sm_90.sass"),
        ("sm_100", "vecadd.sm_100.sass"),
        ("sm_80", f"{SGEMM}/main.sm_80.sass"),
        ("sm_90", f"{SGEMM}/main.sm_90.sass"),
        ("sm_100", f"{SGEMM}/main.sm_100.sass"),
    ]
]


@pytest.mark.parametrize("arch, path", SASS_FIXTURES)
def test_fixture_instructions_parse_cleanly(arch, path, logger):
    parser = make_parser(arch, logger)
    lines = sass_instructions(path)
    assert lines, f"no instructions matched in {path}"

    for line in lines:
        parsed = parser._parse_instruction(line)
        assert parsed.category != OpCategory.UNKNOWN, line

    logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# parse() over a whole CFG
# ---------------------------------------------------------------------------


def test_parse_vecadd_cfg(logger):
    cfg = gen_cfg_from_raw((CFG_DIR / "vecadd.bbcfg.dot").read_text())
    parser = get_parser(
        cfg=cfg, kernel_name="_Z6vecaddPfS_S_i", arch="sm_80", logger=logger
    )

    result = parser.parse()

    assert set(result) == {"_Z6vecaddPfS_S_i", "BB0"}
    assert [i.opcode for i in result["_Z6vecaddPfS_S_i"]] == [
        "MOV",
        "S2R",
        "S2R",
        "IMAD",
        "ISETP",
        "EXIT",
    ]
    assert [i.opcode for i in result["BB0"]] == [
        "HFMA2",
        "ULDC",
        "IMAD",
        "IMAD",
        "LDG",
        "LDG",
        "IMAD",
        "FADD",
        "STG",
        "EXIT",
    ]
    assert result["_Z6vecaddPfS_S_i"][-1].predicate == "P0"


@pytest.mark.parametrize("arch", ["sm_80", "sm_90", "sm_100"])
def test_parse_sgemm_cfg(arch, logger):
    cfg = gen_cfg_from_raw((CFG_DIR / SGEMM / f"main.{arch}.bbcfg.dot").read_text())
    parser = get_parser(
        cfg=cfg, kernel_name="_Z9mysgemmNTPKfiS0_iPfiiff", arch=arch, logger=logger
    )

    result = parser.parse()

    assert set(result) == set(cfg.nodes)
    assert all(result.values()), "every basic block should have instructions"

    # the CFG blocks hold the same FFMAs as the flat disassembly
    flat = sass_instructions(SASS_DIR / SGEMM / f"main.{arch}.sass")
    expected_ffma = sum(base_opcode(line) == "FFMA" for line in flat)
    parsed_ffma = sum(i.opcode == "FFMA" for block in result.values() for i in block)
    assert parsed_ffma == expected_ffma > 0

    logger.warning.assert_not_called()
