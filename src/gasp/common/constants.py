GPU_TO_ARCH: dict[str, str] = {
    "b200": "sm_100",
    "b100": "sm_100",
    "h100": "sm_90",
    "h200": "sm_90",
    "a100": "sm_80",
    "a10": "sm_86",
    "l40": "sm_89",
    "l4": "sm_89",
    "v100": "sm_70",
    "t4": "sm_75",
    "rtx2080ti": "sm_75",
}

# 32-byte sectors in one 128-byte cache line: Pico and ncu count sectors,
# GPUPerf's average_trans_warp counts lines
SECTORS_PER_LINE = 4

# bytes per thread per memory instruction, for MWP_peak_bw and the bandwidth floors
WORD_BYTES = 4
