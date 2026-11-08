#!/bin/bash
# Post-installation check: verify everything gasp needs is available.
# Exits non-zero if any required check fails.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"
MIN_PYTHON="3.12"
SAMPLE_KERNEL="$REPO/benchmarks/custom/vecadd/vecadd.cu"

failures=0

ok() { echo "  [ok]   $*"; }
fail() {
  echo "  [FAIL] $*"
  failures=$((failures + 1))
}
hint() { echo "         -> $*"; }

echo "Python"
# gasp runs from the venv, so its interpreter is the one that matters, not the
# system python3 (which may be older, e.g. when the venv came from a module)
if [ -x "$VENV/bin/python" ]; then
  version="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if "$VENV/bin/python" -c "import sys; sys.exit(sys.version_info[:2] < tuple(map(int, '$MIN_PYTHON'.split('.'))))"; then
    ok "virtualenv at $VENV (python $version)"
  else
    fail "virtualenv at $VENV uses python $version (need >= $MIN_PYTHON)"
    hint "./scripts/init_python.sh"
  fi
  # requirements.txt lines look like "name>=x.y"; check each distribution is installed
  while read -r req; do
    name="$(echo "$req" | sed -E 's/[[:space:]]*#.*//; s/[<>=!~;[ ].*//')"
    [ -n "$name" ] || continue
    if v="$("$VENV/bin/python" -c "import importlib.metadata as m; print(m.version('$name'))" 2> /dev/null)"; then
      ok "$name $v"
    else
      fail "python package $name not installed (wants $req)"
      hint "make install"
    fi
  done < "$REPO/requirements.txt"
else
  fail "virtualenv not found at $VENV"
  hint "./scripts/init_python.sh && make install"
fi

echo "CUDA toolkit"
for tool in nvcc nvdisasm cuobjdump; do
  if path="$(command -v "$tool")"; then
    ok "$tool ($path)"
  else
    fail "$tool not found on PATH"
    hint "install the CUDA toolkit and add its bin/ to PATH (or use Docker)"
  fi
done
if command -v nvcc > /dev/null; then
  ok "$(nvcc --version | grep -o 'release [0-9.]*' | sed 's/release/CUDA/')"
fi

echo "faial (pico)"
if path="$(command -v faial-cost)"; then
  ok "faial-cost ($path)"
  if out="$(faial-cost --help 2>&1 > /dev/null)"; then
    ok "faial-cost runs"
    # Exercises cu-to-json and libz3, which --help does not touch
    if [ -f "$SAMPLE_KERNEL" ]; then
      if out="$(faial-cost -m count -b 32 -g 1 "$SAMPLE_KERNEL" 2>&1)"; then
        ok "faial-cost analyzes $(basename "$SAMPLE_KERNEL")"
      else
        fail "faial-cost failed on $(basename "$SAMPLE_KERNEL"):"
        echo "$out" | head -5 | sed 's/^/         /'
        hint "./scripts/install_faial.sh --force"
      fi
    fi
  else
    fail "faial-cost does not run:"
    echo "$out" | head -5 | sed 's/^/         /'
    if echo "$out" | grep -q libgmp; then
      hint "sudo apt install libgmp10"
    else
      hint "./scripts/install_faial.sh --force"
    fi
  fi
else
  fail "faial-cost not found on PATH"
  hint "./scripts/install_faial.sh (and make sure ~/.local/bin is on PATH)"
fi

echo
if [ "$failures" -eq 0 ]; then
  echo "All requirements satisfied."
else
  echo "$failures check(s) failed."
  exit 1
fi
