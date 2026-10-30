import re
from abc import ABC, abstractmethod

from gasp.cfg import ControlFlowGraph
from gasp.logger import Logger

from .constants import (
    CARRY_OUT_SLOTS,
    CONVERSION_OPCODES,
    FLOAT_TYPE_MODIFIERS,
    INT_TYPE_MODIFIERS,
    MATRIX_MEMORY_OPCODES,
    SIZE_MODIFIER_TO_REG_COUNT,
    WIDE_TYPE_MODIFIERS,
)
from .types import OpCategory, SASSInstruction


class SASSParser(ABC):
    """Base parser for SASS instructions."""

    def __init__(
        self,
        cfg: ControlFlowGraph,
        kernel: str,
        opcode_table: dict[str, OpCategory],
        logger: Logger,
    ):
        self.cfg: ControlFlowGraph = cfg
        self.logger: Logger = logger
        self.kernel: str = kernel  # name of the kernel
        self.opcode_table = opcode_table

        self.STORE_OPS = {
            OpCategory.GLOBAL_STORE,
            OpCategory.LOCAL_STORE,
            OpCategory.RUNTIME_STORE,
            OpCategory.SHARED_STORE,
        }

        self.SETP_OPCODES = {"ISETP", "FSETP", "DSETP", "HSETP2", "FSET", "HSET2"}

        self._warned_opcodes: set[str] = set()

    def parse(self) -> dict[str, list[SASSInstruction]]:
        nodes = self.cfg.nodes
        result: dict[str, list[SASSInstruction]] = {}
        for node_id, node in nodes.items():
            raw_instructions = self._trim_raw_instruction(node.label)
            result[node_id] = [self._parse_instruction(raw) for raw in raw_instructions]
        return result

    def _trim_raw_instruction(self, label: str) -> list[str]:
        raw = label.replace("\\l", "\n")
        raw = raw.replace("\\\\", "\x00")
        raw = re.sub(r"\\(.)", r"\1", raw)
        raw = raw.replace("\x00", "\\")

        instructions = []
        for line in raw.splitlines():
            line = line.strip()
            if not line.endswith(";"):
                continue

            # strip DOT record port prefixes: {<entry>, |<exit0>, }
            line = re.sub(r"^[{|](<[^>]*>)?\s*", "", line)

            # trailing ;
            line = line.rstrip(";").strip()

            if line:
                instructions.append(line)

        return instructions

    def _parse_instruction(self, instruction: str) -> SASSInstruction:
        tokens = instruction.split()
        pos = 0

        predicate = None
        # predicates handling
        if tokens[0].startswith("@"):
            pred_token = tokens[0]
            pred_register = re.sub(r"^@!?", "", pred_token)
            predicate = pred_register
            pos = 1

        raw_opcode = tokens[pos]
        pos += 1

        base_opcode = raw_opcode.split(".")[0]

        op_category = self.opcode_table.get(base_opcode)

        if op_category is None:
            self.logger.warning(f"Unknown opcode: {base_opcode}")
            op_category = OpCategory.UNKNOWN

        raw_operands = " ".join(tokens[pos:])
        operands = [op.strip() for op in raw_operands.split(",") if op.strip()]
        src_registers = set()
        dest_registers = set()

        # PT and UPT are always true, so never are dependencies
        if predicate and predicate not in ("PT", "UPT"):
            src_registers.add(predicate)

        src, dst = self._parse_operands(operands, op_category, raw_opcode)

        dest_registers.update(dst)
        src_registers.update(src)

        # self.logger.info(
        #     f"opcode:         {base_opcode}\n"
        #     f"  category:     {op_category!s}\n"
        #     f"  predicate:    {predicate}\n"
        #     f"  src_registers:  {src_registers}\n"
        #     f"  dest_registers: {dest_registers}"
        # )

        return SASSInstruction(
            opcode=base_opcode,
            category=op_category,
            predicate=predicate,
            src_registers=src_registers,
            dest_registers=dest_registers,
        )

    def _is_op_needed(self, op: str) -> bool:
        if op in ("RZ", "URZ", "PT", "UPT", "SRZ"):
            return False
        # same goes for special registers
        if op.startswith("SR_"):
            return False
        # hex / decimal literals
        if op.startswith("0x") or op.lstrip("-").isdigit():
            return False
        # float literals
        if re.match(r"^-?\d+\.\d+", op):
            return False
        # branch targets '(.L_1)
        return not op.startswith("`")

    def _strip_constant_mem_address(self, raw: str) -> set[str]:
        regs = set()
        inner = re.search(r"\[(\w+)\]$", raw)
        if inner:
            reg = self._strip_operand(inner.group(1))
            if re.match(r"^U?R\d+$", reg) and self._is_op_needed(reg):
                regs.add(reg)
        return regs

    def _strip_operand(self, raw: str) -> str:
        op = raw.strip()
        op = op.strip("|")
        op = op.lstrip("-~!")
        op = re.sub(r"\.(reuse|64)$", "", op)
        return op

    def _strip_operand_brackets(self, raw: str) -> set[str]:
        regs = set()
        inner = raw.strip("[]")
        # account for [R2+0x16]
        for tok in re.split(r"\+", inner):
            regs.update(self._address_registers(tok))

        return regs

    def _address_registers(self, raw: str) -> set[str]:
        # a .64 address register is a pair: [R2.64] reads R2 + R3
        tok = raw.strip()
        width = 2 if tok.endswith(".64") else 1
        tok = re.sub(r"\.X?\d+$", "", tok)
        tok = self._strip_operand(tok)
        if re.match(r"^U?R\d+$", tok) and self._is_op_needed(tok):
            return self._register_span(tok, width)
        return set()

    def _parse_operands(
        self, operands: list[str], category: OpCategory, raw_opcode: str
    ) -> tuple[set[str], set[str]]:

        opcode = raw_opcode.split(".")[0]
        modifiers = raw_opcode.upper().split(".")[1:]

        is_store_op = category in self.STORE_OPS
        is_pred_op = opcode in self.SETP_OPCODES
        carry_out_slots = CARRY_OUT_SLOTS.get(opcode, 0)
        """
        store operations:     no destination register only
                              [address], R1

        predicate ops:        two pred destinations,
                              followed by two registers src
                              followed by one pred src
                              P0 PT R1 R2 PT
        carry operation       some add ops have sum + carry bit for overflow
                              the carry bit goes to a predicate (e.g. P0)
        always end with PT?
        other ops:            a single dest and at least 1 src operand
                              R1 c[0x01][RZ]
        """

        if (
            category == OpCategory.TENSOR or opcode in MATRIX_MEMORY_OPCODES
        ) and opcode not in self._warned_opcodes:
            self._warned_opcodes.add(opcode)
            self.logger.warning(
                f"Tensor operation ({opcode}) is not modeled in this version. "
                "Register dependencies may be inaccurate."
            )

        src_registers = set()
        dst_registers = set()
        value_index = 0
        for i, op in enumerate(operands):
            op = self._strip_operand(op)

            # position among non-predicate operands, so carry predicates
            # don't shift the IMAD.WIDE addend
            index = value_index
            if not re.match(r"^U?P(\d+|T)$", op):
                value_index += 1

            if not self._is_op_needed(op):
                continue
            if op.startswith("c["):
                src_registers.update(self._strip_constant_mem_address(op))
                continue

            # get registers from address (arch-dependent syntax)
            regs = self._parse_operand_address(op)
            if regs is not None:
                src_registers.update(regs)
                continue

            if re.match(r"^U?R\d+$|^U?P\d+$", op):
                # we tell if this is a carry op if the predicate falls in the first slot
                is_carry_out = 1 <= i <= carry_out_slots and bool(
                    re.match(r"^U?P\d+$", op)
                )
                is_dst = not is_store_op and (
                    is_pred_op and i <= 1 or i == 0 or is_carry_out
                )
                width = self._get_register_width(
                    opcode, modifiers, category, index, is_dst
                )
                regs = self._register_span(op, width)
                if is_dst:
                    dst_registers.update(regs)
                else:
                    src_registers.update(regs)
                continue

            self.logger.warning(f"Unrecognized operand: {op}")

        return src_registers, dst_registers

    def _get_register_width(
        self,
        opcode: str,
        modifiers: list[str],
        category: OpCategory,
        index: int,
        is_dst: bool,
    ) -> int:
        """Number of consecutive 32-bit registers a register operand covers."""
        if opcode in CONVERSION_OPCODES:
            dst_type, src_type = self._conversion_types(opcode, modifiers)
            operand_type = dst_type if is_dst else src_type
            return 2 if operand_type in WIDE_TYPE_MODIFIERS else 1

        # FP64 ops carry no .64 modifier, but every register operand is a pair
        # DSETP predicates are not registers, so _register_span ignores them
        if category == OpCategory.FP64:
            return 2

        # CS2R R16, SRZ writes R16 + R17, CS2R.32 writes a single register
        if opcode == "CS2R":
            return 1 if "32" in modifiers else 2

        # IMAD.WIDE R2, R4, R5, R6: 32-bit factors R4 and R5,
        # 64-bit addend R6 + R7 and result R2 + R3
        if "WIDE" in modifiers:
            return 2 if is_dst or index == 3 else 1

        for mod in modifiers:
            if mod in SIZE_MODIFIER_TO_REG_COUNT:
                return SIZE_MODIFIER_TO_REG_COUNT[mod]

        return 1

    def _conversion_types(
        self, opcode: str, modifiers: list[str]
    ) -> tuple[str | None, str | None]:
        """(dst type, src type) of a conversion, None where nvdisasm omits a 32-bit type."""
        floats = [m for m in modifiers if m in FLOAT_TYPE_MODIFIERS]
        ints = [m for m in modifiers if m in INT_TYPE_MODIFIERS]

        # F2I and I2F name each side by its kind, whichever is omitted
        if opcode == "F2I":
            return (ints[0] if ints else None, floats[0] if floats else None)
        if opcode == "I2F":
            return (floats[0] if floats else None, ints[0] if ints else None)

        # F2F, I2I and FRND: destination first, a single type covers both sides
        types = ints if opcode == "I2I" else floats
        if not types:
            return None, None
        return types[0], types[-1]

    def _register_span(self, reg: str, width: int) -> set[str]:
        # R4 with width 2 -> R4 + R5, UR4 -> UR4 + UR5
        match = re.match(r"^(U?R)(\d+)$", reg)
        if not match:
            return {reg}
        prefix, num = match.group(1), int(match.group(2))
        return {f"{prefix}{num + offset}" for offset in range(width)}

    @abstractmethod
    def _parse_operand_address(self, op: str) -> set[str] | None:
        pass
