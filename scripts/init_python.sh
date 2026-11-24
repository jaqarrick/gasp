#!/bin/bash
# Find a Python >= 3.12 (PATH, then environment modules) and create .venv with it.
# If a module was needed, prints the `module load` line to run before using gasp.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"
MIN_MINOR=12

die() {
  >&2 echo "error: $*"
  exit 1
}

# Succeeds if $1 is a python3 with minor version >= MIN_MINOR
recent_enough() {
  "$1" -c "import sys; sys.exit(not (sys.version_info[0] == 3 and sys.version_info[1] >= $MIN_MINOR))" 2> /dev/null
}

python=""
loaded_module=""

for candidate in "${PYTHON:-python3}" python3.13 python3.12; do
  if command -v "$candidate" > /dev/null && recent_enough "$candidate"; then
    python="$(command -v "$candidate")"
    break
  fi
done

if [ -z "$python" ]; then
  # `module` is a shell function, so it isn't defined in non-interactive scripts
  if ! type module > /dev/null 2>&1; then
    for init in "${MODULESHOME:-}/init/bash" /etc/profile.d/modules.sh \
      /usr/share/Modules/init/bash /etc/profile.d/lmod.sh; do
      if [ -f "$init" ]; then
        # shellcheck disable=SC1090
        source "$init"
        break
      fi
    done
  fi

  if type module > /dev/null 2>&1; then
    # one module per line, e.g. python/3.12.1(default)
    best="$(module -t avail 2>&1 \
      | grep -iE '^python[-/]?3\.[0-9]+' \
      | sed -E 's/\(.*\)$//' \
      | sort -V | tail -n 1 || true)"
    minor="$(echo "$best" | grep -oE '3\.[0-9]+' | head -n 1 | cut -d. -f2)"
    if [ -n "$best" ] && [ "${minor:-0}" -ge "$MIN_MINOR" ]; then
      echo "Loading module $best"
      module load "$best"
      if recent_enough python3; then
        python="$(command -v python3)"
        loaded_module="$best"
      fi
    elif [ -n "$best" ]; then
      echo "Newest python module is $best (need 3.$MIN_MINOR+)"
    fi
  fi
fi

if [ -z "$python" ]; then
  die "no Python >= 3.$MIN_MINOR found. Install one without root via uv:
  curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12
then rerun with PYTHON=\"\$(uv python find 3.12)\" $0"
fi

echo "Using $python ($("$python" --version 2>&1))"

if [ -x "$VENV/bin/python" ]; then
  if recent_enough "$VENV/bin/python"; then
    echo "$VENV already uses $("$VENV/bin/python" --version 2>&1)"
    exit 0
  fi
  echo "Recreating $VENV (built with $("$VENV/bin/python" --version 2>&1))"
  rm -rf "$VENV"
fi

"$python" -m venv "$VENV"
echo "Created $VENV"

if [ -n "$loaded_module" ]; then
  echo
  echo "note: this Python comes from a module. Load it in each new shell before using gasp:"
  echo "  module load $loaded_module"
  echo "(or add that line to ~/.bashrc)"
fi
