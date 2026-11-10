from dataclasses import dataclass, field
from enum import Enum, auto


class OpCategory(Enum):
    FP32 = auto()
    FP64 = auto()
    FP16 = auto()
    INT = auto()
    SFU = auto()
    CONVERSION = auto()
    TENSOR = auto()
    GLOBAL_LOAD = auto()
    GLOBAL_STORE = auto()
    SHARED_LOAD = auto()
    SHARED_STORE = auto()
    LOCAL_LOAD = auto()
    LOCAL_STORE = auto()
    RUNTIME_LOAD = auto()
    RUNTIME_STORE = auto()
    CONST_LOAD = auto()

    ASYNC_COPY = auto()
    OTHER_MEM = auto()
    BARRIER = auto()
    CONTROL = auto()
    MOVEMENT = auto()
    PREDICATE = auto()
    UNIFORM = auto()
    MISC = auto()

    UNKNOWN = auto()


@dataclass
class SASSInstruction:
    opcode: str
    category: OpCategory
    predicate: str | None
    src_registers: set[str]
    dest_registers: set[str]
    # global memory instructions only
    access_width_bytes: int | None = None
    # un-expanded base register of the address e.g. R2 from [R2.64+0x10]
    address_base_register: str | None = None
    # immediate byte offset of the address: 16 from [R2.64+0x10]
    address_offset: int | None = None
    # special registers read, e.g. SR_TID.X from S2R R7, SR_TID.X
    special_registers: set[str] = field(default_factory=set)
    # label of BRA, BRX or BSSY, e.g. .L_x_21 from BSSY B0, `(.L_x_21)
    branch_target: str | None = None
