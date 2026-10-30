import sys

from gasp.cfg import ControlFlowGraph
from gasp.logger import Logger

from .base import SASSParser
from .bracket import BracketSASSParser
from .constants import common_ops, sm80_ops, sm90_ops, sm100_ops
from .descriptor import DescriptorSASSParser


def get_parser(
    cfg: ControlFlowGraph, kernel_name: str, arch: str, logger: Logger
) -> SASSParser:
    if arch in ("sm_100", "sm_100a", "sm_103", "sm_103a"):
        opcode_table = {**common_ops, **sm100_ops}
        return DescriptorSASSParser(cfg, kernel_name, opcode_table, logger)
    elif arch in ("sm_90", "sm_90a"):
        opcode_table = {**common_ops, **sm90_ops}
        return DescriptorSASSParser(cfg, kernel_name, opcode_table, logger)
    elif arch in ("sm_80", "sm_86", "sm_87", "sm_88", "sm_89"):
        opcode_table = {**common_ops, **sm80_ops}
        return BracketSASSParser(cfg, kernel_name, opcode_table, logger)
    elif arch == "sm_75":
        # temporary, for sims runs on a Turing card
        return BracketSASSParser(cfg, kernel_name, common_ops, logger)
    elif arch.startswith("sm_"):
        logger.error(f"SASS Parser: {arch} is not supported at this time")
        sys.exit(1)
    else:
        logger.error(f"SASS Parser: invalid architecture ({arch})")
        sys.exit(1)
