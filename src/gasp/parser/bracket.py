from .base import SASSParser


class BracketSASSParser(SASSParser):
    """
    sm_70-89 arch uses bracket addressing e.g. LDG.WIDE R1, [R2]
    """

    def _parse_operand_address(self, op) -> set[str] | None:
        if op.startswith("["):
            return self._strip_operand_brackets(op)
        return None

    def _address_inner(self, op) -> str | None:
        # [R2.64+0x10] -> R2.64+0x10
        if op.startswith("["):
            return op.strip("[]")
        return None
