#!/bin/bash
# extract_sass.sh — compile a CUDA file and extract SASS, CFG, and resource usage

set -e

usage() {
    echo "Usage: $0 <kernel.cu> [options]"
    echo ""
    echo "Arguments:"
    echo "  kernel.cu    CUDA source file to compile"
    echo ""
    echo "Options:"
    echo "  --arch ARCH        Target architecture (default: sm_80)"
    echo "                     Options: sm_70, sm_75, sm_80, sm_86, sm_89, sm_90, sm_100"
    echo "  --out-dir, -od DIR Output directory (default: same as input file)"
    echo "  -I DIR             Add an include directory (repeatable)"
    echo "  --nvcc-flags FLAGS Extra flags passed to nvcc, e.g. \"--use_fast_math -Xptxas -dlcm=cg\""
    echo "  --sass             Generate SASS disassembly"
    echo "  --cfg              Generate basic block CFG (DOT format)"
    echo "  --res              Generate resource usage"
    echo "  --keep-cubin       Keep the intermediate .cubin file"
    echo "  -h, --help         Show this help"
    echo ""
    echo "If none of --sass, --cfg, --res are specified, all are generated."
    exit 0
}

INPUT=""
ARCH="sm_80"
OUT_DIR=""
DO_SASS=false
DO_CFG=false
DO_RES=false
KEEP_CUBIN=false
NVCC_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)     usage ;;
        --arch)        ARCH="$2"; shift 2 ;;
        --out-dir|-od) OUT_DIR="$2"; shift 2 ;;
        -I)            NVCC_ARGS+=("-I$2"); shift 2 ;;
        --nvcc-flags)  read -ra FLAGS <<< "$2"; NVCC_ARGS+=("${FLAGS[@]}"); shift 2 ;;
        --sass)        DO_SASS=true; shift ;;
        --cfg)         DO_CFG=true; shift ;;
        --res)         DO_RES=true; shift ;;
        --keep-cubin)  KEEP_CUBIN=true; shift ;;
        -*)            echo "error: unknown option $1"; exit 1 ;;
        *)
            if [[ -z "$INPUT" ]]; then
                INPUT="$1"; shift
            else
                echo "error: unexpected argument $1"; exit 1
            fi
            ;;
    esac
done

if [[ -z "$INPUT" ]]; then
    echo "error: no input file specified (use -h for help)"
    exit 1
fi

if ! $DO_SASS && ! $DO_CFG && ! $DO_RES; then
    DO_SASS=true
    DO_CFG=true
    DO_RES=true
fi

NAME="$(basename "${INPUT%.cu}")"
OUT_DIR="${OUT_DIR:-$(dirname "$INPUT")}"
mkdir -p "$OUT_DIR"
CUBIN="${OUT_DIR}/${NAME}.${ARCH}.cubin"

for tool in nvcc nvdisasm cuobjdump; do
    if ! command -v "$tool" &>/dev/null; then
        echo "error: $tool not found. Is the CUDA toolkit installed and on PATH?"
        exit 1
    fi
done

echo "Compiling ${INPUT} for ${ARCH}..."
nvcc -cubin -arch="${ARCH}" "${NVCC_ARGS[@]}" "${INPUT}" -o "${CUBIN}"

OUTPUTS=()

if $KEEP_CUBIN; then
    OUTPUTS+=("${CUBIN}")
fi

if $DO_SASS; then
    OUT="${OUT_DIR}/${NAME}.${ARCH}.sass"
    nvdisasm "${CUBIN}" > "$OUT"
    OUTPUTS+=("$OUT")
fi

if $DO_CFG; then
    OUT="${OUT_DIR}/${NAME}.${ARCH}.bbcfg.dot"
    nvdisasm -bbcfg "${CUBIN}" > "$OUT"
    OUTPUTS+=("$OUT")
fi

if $DO_RES; then
    OUT="${OUT_DIR}/${NAME}.${ARCH}.res"
    cuobjdump -res-usage "${CUBIN}" > "$OUT"
    OUTPUTS+=("$OUT")
fi

if ! $KEEP_CUBIN; then
    rm -f "${CUBIN}"
fi

echo "Done:"
for f in "${OUTPUTS[@]}"; do
    echo "  $f"
done