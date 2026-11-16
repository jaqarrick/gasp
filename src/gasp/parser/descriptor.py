import re

from .base import SASSParser


class DescriptorSASSParser(SASSParser):
    """
    sm_90+ arch uses descriptor-based addressing:

    Regular descriptor (two registers): LDG.WIDE R1, desc[UR2][R2.64]
    Global descriptor  (one register) : LDG.WIDE R1, gdesc[UR2].
    """

    def _parse_operand_address(self, op) -> set[str] | None:
        if op.startswith("desc["):
            return self._strip_descriptor_address(op)
        if op.startswith("gdesc["):
            return self._strip_gdescriptor_address(op)
        if op.startswith("["):
            return self._strip_operand_brackets(op)

        return None

    def _address_inner(self, op) -> str | None:
        # desc[UR4][R2.64+0x10] -> R2.64+0x10: UR4 is the memory descriptor,
        # the same for every access, so the address is the last group
        if op.startswith("desc["):
            groups = re.findall(r"\[([^\]]+)\]", op)
            return groups[-1] if groups else None
        # gdesc[UR2] -> UR2
        if op.startswith("gdesc["):
            match = re.search(r"\[([^\]]+)\]", op)
            return match.group(1) if match else None
        if op.startswith("["):
            return op.strip("[]")
        return None

    def _strip_descriptor_address(self, raw: str) -> set[str]:
        # desc[UR2][R4.64] extract UR2, R4 + R5
        regs = set()
        matches = re.findall(r"\[([^\]]+)\]", raw)
        for m in matches:
            regs.update(self._address_registers(m))
        return regs

    def _strip_gdescriptor_address(self, raw: str) -> set[str]:
        # gdesc[UR2].foo extract UR2
        regs = set()
        match = re.search(r"\[([^\]]+)\]", raw)
        if match:
            m = self._strip_operand(match.group(1))
            if re.match(r"^U?R\d+$", m) and self._is_op_needed(m):
                regs.add(m)
        return regs
