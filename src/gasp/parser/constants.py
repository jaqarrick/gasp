from .types import OpCategory

# Source: see https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference

common_ops = {
    # FP Scalar Arithmetic
    "FADD": OpCategory.FP32,  # FP32 Add
    "FADD32I": OpCategory.FP32,  # FP32 Add
    "FCHK": OpCategory.FP32,  # Floating-point Range Check
    "FFMA": OpCategory.FP32,  # FP32 Fused Multiply and Add
    "FFMA32I": OpCategory.FP32,  # FP32 Fused Multiply and Add
    "FMNMX": OpCategory.FP32,  # FP32 Minimum/Maximum
    "FMUL": OpCategory.FP32,  # FP32 Multiply
    "FMUL32I": OpCategory.FP32,  # FP32 Multiply
    "FSEL": OpCategory.FP32,  # Floating Point Select
    "FSET": OpCategory.FP32,  # FP32 Compare And Set
    "FSETP": OpCategory.FP32,  # FP32 Compare And Set Predicate
    "FSWZADD": OpCategory.FP32,  # FP32 Swizzle Add
    # SFU operation
    "MUFU": OpCategory.SFU,  # FP32 Multi Function Operation
    # These are FP16, but treat as same timing as FP32 in model
    "HADD2": OpCategory.FP32,  # FP16 Add
    "HADD2_32I": OpCategory.FP32,  # FP16 Add
    "HFMA2": OpCategory.FP32,  # FP16 Fused Multiply Add
    "HFMA2_32I": OpCategory.FP32,  # FP16 Fused Multiply Add
    "HMNMX2": OpCategory.FP32,  # FP16 Minimum / Maximum
    "HMUL2": OpCategory.FP32,  # FP16 Multiply
    "HMUL2_32I": OpCategory.FP32,  # FP16 Multiply
    "HSET2": OpCategory.FP32,  # FP16 Compare And Set
    "HSETP2": OpCategory.FP32,  # FP16 Compare And Set Predicate
    # FP64 Scalar Arithmetic
    "DADD": OpCategory.FP64,  # FP64 Add
    "DFMA": OpCategory.FP64,  # FP64 Fused Multiply Add
    "DMUL": OpCategory.FP64,  # FP64 Multiply
    "DSETP": OpCategory.FP64,  # FP64 Compare And Set Predicate
    # Tensor: matrix
    "DMMA": OpCategory.TENSOR,  # Matrix Multiply and Accumulate
    "HMMA": OpCategory.TENSOR,  # Matrix Multiply and Accumulate
    "IMMA": OpCategory.TENSOR,  # Integer Matrix Multiply and Accumulate
    "BMMA": OpCategory.TENSOR,  # Bit Matrix Multiply and Accumulate
    # INT32 scalar
    "BMSK": OpCategory.INT,  # Bitfield Mask
    "BREV": OpCategory.INT,  # Bit Reverse
    "FLO": OpCategory.INT,  # Find Leading One
    "IABS": OpCategory.INT,  # Integer Absolute Value
    "IADD": OpCategory.INT,  # Integer Addition
    "IADD3": OpCategory.INT,  # 3-input Integer Addition
    "IADD32I": OpCategory.INT,  # Integer Addition
    "IDP": OpCategory.INT,  # Integer Dot Product and Accumulate
    "IDP4A": OpCategory.INT,  # Integer Dot Product and Accumulate
    "IMAD": OpCategory.INT,  # Integer Multiply And Add
    "IMNMX": OpCategory.INT,  # Integer Minimum/Maximum
    "IMUL": OpCategory.INT,  # Integer Multiply
    "IMUL32I": OpCategory.INT,  # Integer Multiply
    "ISCADD": OpCategory.INT,  # Scaled Integer Addition
    "ISCADD32I": OpCategory.INT,  # Scaled Integer Addition
    "ISETP": OpCategory.INT,  # Integer Compare And Set Predicate
    "LEA": OpCategory.INT,  # LOAD Effective Address
    "LOP": OpCategory.INT,  # Logic Operation
    "LOP3": OpCategory.INT,  # Logic Operation
    "LOP32I": OpCategory.INT,  # Logic Operation
    "POPC": OpCategory.INT,  # Population count
    "SHF": OpCategory.INT,  # Funnel Shift
    "SHL": OpCategory.INT,  # Shift Left
    "SHR": OpCategory.INT,  # Shift Right
    "VABSDIFF": OpCategory.INT,  # Absolute Difference
    "VABSDIFF4": OpCategory.INT,  # Absolute Difference
    # Conversion, use FP32 latency
    "F2F": OpCategory.CONVERSION,  # Floating Point To Floating Point Conversion
    "F2I": OpCategory.CONVERSION,  # Floating Point To Integer Conversion
    "I2F": OpCategory.CONVERSION,  # Integer To Floating Point Conversion
    "I2I": OpCategory.CONVERSION,  # Integer To Integer Conversion
    "I2IP": OpCategory.CONVERSION,  # Integer To Integer Conversion and Packing
    "I2FP": OpCategory.CONVERSION,  # Integer to FP32 Convert and Pack
    "F2IP": OpCategory.CONVERSION,  # FP32 Down-Convert to Integer and Pack
    "FRND": OpCategory.CONVERSION,  # Round To Integer
    # Movement within registers
    "MOV": OpCategory.MOVEMENT,  # Move
    "MOV32I": OpCategory.MOVEMENT,  # Move
    "MOVM": OpCategory.MOVEMENT,  # Move Matrix with Transposition or Expansion
    "PRMT": OpCategory.MOVEMENT,  # Permute Register Pair
    "SEL": OpCategory.MOVEMENT,  # Select Source with Predicate
    "SGXT": OpCategory.MOVEMENT,  # Sign Extend
    "SHFL": OpCategory.MOVEMENT,  # Warp Wide Register Shuffle
    # Predicate
    "PLOP3": OpCategory.PREDICATE,  # Predicate Logic Operation
    "PSETP": OpCategory.PREDICATE,  # Combine Predicates and Set Predicate
    "P2R": OpCategory.PREDICATE,  # Move Predicate Register To Register
    "R2P": OpCategory.PREDICATE,  # Move Register To Predicate Register
    # Memory operations
    "LD": OpCategory.RUNTIME_LOAD,  # Load from generic Memory (space resolved at runtime)
    "LDC": OpCategory.CONST_LOAD,  # Load Constant
    "LDG": OpCategory.GLOBAL_LOAD,  # Load from Global Memory
    "LDL": OpCategory.LOCAL_LOAD,  # Load within Local Memory Window
    "LDS": OpCategory.SHARED_LOAD,  # Load within Shared Memory Window
    "LDSM": OpCategory.SHARED_LOAD,  # Load Matrix from Shared Memory with Element Size Expansion
    "ST": OpCategory.RUNTIME_STORE,  # Store to Generic Memory (space resolved at runtime)
    "STG": OpCategory.GLOBAL_STORE,  # Store to Global Memory
    "STL": OpCategory.LOCAL_STORE,  # Store to Local Memory
    "STS": OpCategory.SHARED_STORE,  # Store to Shared Memory
    "MATCH": OpCategory.OTHER_MEM,  # Match Register Values Across Thread Group
    "QSPC": OpCategory.OTHER_MEM,  # Query Space
    "ATOM": OpCategory.OTHER_MEM,  # Atomic Operation on Generic Memory
    "ATOMS": OpCategory.OTHER_MEM,  # Atomic Operation on Shared Memory
    "ATOMG": OpCategory.OTHER_MEM,  # Atomic Operation on Global Memory
    "RED": OpCategory.OTHER_MEM,  # Reduction Operation on Generic Memory
    "CCTL": OpCategory.OTHER_MEM,  # Cache Control
    "CCTLL": OpCategory.OTHER_MEM,  # Cache Control
    "CCTLT": OpCategory.OTHER_MEM,  # Texture Cache Control
    "ERRBAR": OpCategory.OTHER_MEM,  # Error Barrier
    "MEMBAR": OpCategory.OTHER_MEM,  # Memory Barrier
    "BAR": OpCategory.BARRIER,  # Barrier Synchronization
    "BRA": OpCategory.CONTROL,  # Relative Branch
    "BRX": OpCategory.CONTROL,  # Relative Branch Indirect
    "BRXU": OpCategory.CONTROL,  # Relative Branch with Uniform Register Based Offset
    "BSSY": OpCategory.CONTROL,  # Barrier Set Convergence Synchronization Point
    "BSYNC": OpCategory.CONTROL,  # Synchronize Threads on a Convergence Barrier
    "BREAK": OpCategory.CONTROL,  # Break out of the Specified Convergence Barrier
    "BMOV": OpCategory.CONTROL,  # Move Convergence Barrier State
    "BPT": OpCategory.CONTROL,  # BreakPoint/Trap
    "CALL": OpCategory.CONTROL,  # Call Function
    "EXIT": OpCategory.CONTROL,  # Exit Program
    "JMP": OpCategory.CONTROL,  # Absolute Jump
    "JMX": OpCategory.CONTROL,  # Absolute Jump Indirect
    "JMXU": OpCategory.CONTROL,  # Absolute Jump with Uniform Register Based Offset
    "KILL": OpCategory.CONTROL,  # Kill Thread
    "RET": OpCategory.CONTROL,  # Return From Subroutine
    "RPCMOV": OpCategory.CONTROL,  # PC Register Move
    "NANOSLEEP": OpCategory.MISC,  # Suspend Execution
    "WARPSYNC": OpCategory.MISC,  # Synchronize Threads in Warp
    "YIELD": OpCategory.MISC,  # Yield Control
    "NOP": OpCategory.MISC,  # No Operation
    "S2R": OpCategory.MISC,  # Move Special Register to Register
    "CS2R": OpCategory.MISC,  # Move Special Register to Register
    "B2R": OpCategory.MISC,  # Move Barrier To Register
    "DEPBAR": OpCategory.MISC,  # Dependency Barrier
    "LEPC": OpCategory.MISC,  # Load Effective PC
    "GETLMEMBASE": OpCategory.MISC,  # Get Local Memory Base Address
    "SETLMEMBASE": OpCategory.MISC,  # Set Local Memory Base Address
    "SETCTAID": OpCategory.MISC,  # Set CTA ID
    "PMTRIG": OpCategory.MISC,  # Performance Monitor Trigger
    "VOTE": OpCategory.MISC,  # Vote Across SIMD Thread Group
    "R2UR": OpCategory.UNIFORM,  # Move from Vector Register to a Uniform Register
    "S2UR": OpCategory.UNIFORM,  # Move Special Register to Uniform Register
    "REDUX": OpCategory.UNIFORM,  # Reduction of a Vector Register into a Uniform Register
    "UBMSK": OpCategory.UNIFORM,  # Uniform Bitfield Mask
    "UBREV": OpCategory.UNIFORM,  # Uniform Bit Reverse
    "UCLEA": OpCategory.UNIFORM,  # Load Effective Address for a Constant
    "UF2FP": OpCategory.UNIFORM,  # Uniform FP32 Down-convert and Pack
    "UFLO": OpCategory.UNIFORM,  # Uniform Find Leading One
    "UIADD3": OpCategory.UNIFORM,  # Uniform Integer Addition
    "UIMAD": OpCategory.UNIFORM,  # Uniform Integer Multiplication
    "UISETP": OpCategory.UNIFORM,  # Integer Compare and Set Uniform Predicate
    "ULDC": OpCategory.UNIFORM,  # Load from Constant Memory into a Uniform Register
    "ULEA": OpCategory.UNIFORM,  # Uniform Load Effective Address
    "ULOP": OpCategory.UNIFORM,  # Logic Operation
    "ULOP3": OpCategory.UNIFORM,  # Logic Operation
    "ULOP32I": OpCategory.UNIFORM,  # Logic Operation
    "UMOV": OpCategory.UNIFORM,  # Uniform Move
    "UP2UR": OpCategory.UNIFORM,  # Uniform Predicate to Uniform Register
    "UPLOP3": OpCategory.UNIFORM,  # Uniform Predicate Logic Operation
    "UPOPC": OpCategory.UNIFORM,  # Uniform Population Count
    "UPRMT": OpCategory.UNIFORM,  # Uniform Byte Permute
    "UPSETP": OpCategory.UNIFORM,  # Uniform Predicate Logic Operation
    "UR2UP": OpCategory.UNIFORM,  # Uniform Register to Uniform Predicate
    "USEL": OpCategory.UNIFORM,  # Uniform Select
    "USGXT": OpCategory.UNIFORM,  # Uniform Sign Extend
    "USHF": OpCategory.UNIFORM,  # Uniform Funnel Shift
    "USHL": OpCategory.UNIFORM,  # Uniform Left Shift
    "USHR": OpCategory.UNIFORM,  # Uniform Right Shift
    "VOTEU": OpCategory.UNIFORM,  # Voting across SIMD Thread Group with Results in Uniform Destination
    "TEX": OpCategory.OTHER_MEM,  # Texture Fetch
    "TLD": OpCategory.OTHER_MEM,  # Texture Load
    "TLD4": OpCategory.OTHER_MEM,  # Texture Load 4
    "TMML": OpCategory.OTHER_MEM,  # Texture MipMap Level
    "TXD": OpCategory.OTHER_MEM,  # Texture Fetch With Derivatives
    "TXQ": OpCategory.OTHER_MEM,  # Texture Query
    "SUATOM": OpCategory.OTHER_MEM,  # Atomic Op on Surface Memory
    "SULD": OpCategory.OTHER_MEM,  # Surface Load
    "SURED": OpCategory.OTHER_MEM,  # Reduction Op on Surface Memory
    "SUST": OpCategory.OTHER_MEM,  # Surface Store
}

MEMORY_LOAD_CATEGORIES = {
    OpCategory.GLOBAL_LOAD,
    OpCategory.SHARED_LOAD,
    OpCategory.LOCAL_LOAD,
    OpCategory.RUNTIME_LOAD,
    OpCategory.CONST_LOAD,
}

# Counted as #mem_insts in T_mem (Eq. 11)
# TODO: recalibrate for cache (l1, l2), shared, local mem
GLOBAL_MEMORY_CATEGORIES = {
    OpCategory.GLOBAL_LOAD,
    OpCategory.GLOBAL_STORE,
}


# Registers per operand implied by a size modifier
# e.g. LDS.128 R12, [R0] writes R12-R15, STG.E.64 [R2.64], R4 reads R4 + R5
# .WIDE is handled separately, since only some of its operands are 64-bit
SIZE_MODIFIER_TO_REG_COUNT = {
    "64": 2,
    "128": 4,
}

CONVERSION_OPCODES = {"F2F", "F2I", "I2F", "I2I", "FRND"}

# In conversion operations the type modifiers specify the size of the dst/src
# registers. With two types the destination comes first:
# F2F.F64.F32 R2, R4 converts 32-bit R4 to 64-bit in R2 + R3
FLOAT_TYPE_MODIFIERS = {"F16", "BF16", "F32", "F64"}

INT_TYPE_MODIFIERS = {"S8", "U8", "S16", "U16", "S32", "U32", "S64", "U64"}

WIDE_TYPE_MODIFIERS = {"F64", "S64", "U64"}

# Number of carry-out predicate slots right after the destination
# A predicate there is written, not read: IADD3 R2, P0, PT, R4, R6, RZ writes P0
# The carry-in comes last and is read: IADD3.X R3, PT, PT, R5, R7, RZ, P0, !PT
CARRY_OUT_SLOTS = {
    "IADD3": 2,
    "UIADD3": 2,
    "LEA": 1,
    "ULEA": 1,
}

# TODO: Size conversion for matrix load/store and MMA operands
# We will not implement in version 1, since we are not targeting tensor cores
# For now, log a warning if they show up (along with OpCategory.TENSOR)
MATRIX_MEMORY_OPCODES = {"LDSM", "STSM"}


sm80_ops = {
    "LDGSTS": OpCategory.ASYNC_COPY,  # Asynchronous Global to Shared Memcopy
    "LDGDEPBAR": OpCategory.ASYNC_COPY,  # Global Load Dependency Barrier
}

sm90_ops = {
    "LDGSTS": OpCategory.ASYNC_COPY,  # Asynchronous Global to Shared Memcopy
    "LDGDEPBAR": OpCategory.ASYNC_COPY,  # Global Load Dependency Barrier
    "LDGMC": OpCategory.GLOBAL_LOAD,  # Reducing Load
    "STSM": OpCategory.SHARED_STORE,  # Store Matrix to Shared Memory
    "STAS": OpCategory.ASYNC_COPY,  # Asynchronous Store to Distributed Shared Memory With Explicit Synchronization
    "REDAS": OpCategory.ASYNC_COPY,  # Asynchronous Reduction on Distributed Shared Memory With Explicit Synchronization
    "REDG": OpCategory.OTHER_MEM,  # Reduction Operation on Generic Memory
    "SYNCS": OpCategory.BARRIER,  # Sync Unit
    "FENCE": OpCategory.OTHER_MEM,  # Memory Visibility Guarantee for Shared or Global Memory
    "HGMMA": OpCategory.TENSOR,  # Matrix Multiply and Accumulate Across a Warpgroup
    "IGMMA": OpCategory.TENSOR,  # Integer Matrix Multiply and Accumulate Across a Warpgroup
    "QGMMA": OpCategory.TENSOR,  # FP8 Matrix Multiply and Accumulate Across a Warpgroup
    "BGMMA": OpCategory.TENSOR,  # Bit Matrix Multiply and Accumulate Across Warps
    "WARPGROUP": OpCategory.MISC,  # Warpgroup Synchronization
    "WARPGROUPSET": OpCategory.MISC,  # Set Warpgroup Counters
    "UTMALDG": OpCategory.ASYNC_COPY,  # Tensor Load from Global to Shared Memory
    "UTMASTG": OpCategory.ASYNC_COPY,  # Tensor Store from Shared to Global Memory
    "UTMAREDG": OpCategory.ASYNC_COPY,  # Tensor Store from Shared to Global Memory with Reduction
    "UTMAPF": OpCategory.ASYNC_COPY,  # Tensor Prefetch
    "UTMACCTL": OpCategory.ASYNC_COPY,  # TMA Cache Control
    "UTMACMDFLUSH": OpCategory.ASYNC_COPY,  # TMA Command Flush
    "UBLKCP": OpCategory.ASYNC_COPY,  # Bulk Data Copy
    "UBLKPF": OpCategory.ASYNC_COPY,  # Bulk Data Prefetch
    "UBLKRED": OpCategory.ASYNC_COPY,  # Bulk Data Copy from Shared Memory with Reduction
    "UCGABAR_ARV": OpCategory.BARRIER,  # CGA Barrier Synchronization
    "UCGABAR_WAIT": OpCategory.BARRIER,  # CGA Barrier Synchronization
    "ULEPC": OpCategory.UNIFORM,  # Uniform Load Effective PC
    "USETMAXREG": OpCategory.UNIFORM,  # Release, Deallocate and Allocate Registers
    "ELECT": OpCategory.CONTROL,  # Elect a Leader Thread
    "PREEXIT": OpCategory.CONTROL,  # Dependent Task Launch Hint
    "ACQBULK": OpCategory.CONTROL,  # Wait for Bulk Release Status Warp State
    "CGAERRBAR": OpCategory.MISC,  # CGA Error Barrier
    "ENDCOLLECTIVE": OpCategory.MISC,  # Reset the MCOLLECTIVE mask
    "VHMNMX": OpCategory.INT,  # SIMD FP16 3-Input Minimum / Maximum
    "VIADD": OpCategory.INT,  # SIMD Integer Addition
    "VIADDMNMX": OpCategory.INT,  # SIMD Integer Addition and Fused Min/Max Comparison
    "VIMNMX": OpCategory.INT,  # SIMD Integer Minimum / Maximum
    "VIMNMX3": OpCategory.INT,  # SIMD Integer 3-Input Minimum / Maximum
}

sm100_ops = {
    **sm90_ops,
    "LDCU": OpCategory.CONST_LOAD,  # Load from Constant Memory into a Uniform Register
}
