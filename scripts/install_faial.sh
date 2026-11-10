#!/bin/bash
# Fetch faial (faial-cost aka pico, cu-to-json, libz3) and expose faial-cost /
# faial-drf on PATH without Docker.
#
# Two install modes:
#   native     the prebuilt bundle. Its binaries have no rpath and faial-cost
#              shells out to cu-to-json via PATH, so the wrappers set
#              LD_LIBRARY_PATH and PATH first. Needs glibc >= 2.38.
#   container  the faial-tools image run through apptainer/singularity, for
#              hosts whose glibc is too old for the bundle (e.g. RHEL 9).
# The default (auto) tries native and falls back to container.
#
# Env overrides:
#   FAIAL_PREFIX  where the bundle/image is kept (default ~/.local/share/faial)
#   BIN_DIR       where the wrappers go          (default ~/.local/bin)
#   FAIAL_MODE    auto | native | container      (default auto)
#
# If BIN_DIR is not on PATH, the script offers to add it to your shell's rc
# file. Declining is fine: set GASP_FAIAL_COST to the wrapper's path instead.
#
# Usage: scripts/install_faial.sh [--force]
#   --force  reinstall even if a working faial-cost is already available
set -euo pipefail

FAIAL_PREFIX="${FAIAL_PREFIX:-$HOME/.local/share/faial}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
FAIAL_MODE="${FAIAL_MODE:-auto}"
TOOLS=(faial-cost faial-drf)
IMAGE="docker://registry.gitlab.com/umb-svl/faial/faial-tools:latest"
SIF_NAME="faial-tools.sif"

die() {
  >&2 echo "error: $*"
  exit 1
}

force=0
case "${1:-}" in
  --force) force=1 ;;
  "") ;;
  *) die "unknown argument: $1 (usage: $0 [--force])" ;;
esac

case "$FAIAL_MODE" in
  auto | native | container) ;;
  *) die "FAIAL_MODE must be auto, native or container (got '$FAIAL_MODE')" ;;
esac

# gasp finds faial-cost on PATH, or at $GASP_FAIAL_COST. When BIN_DIR is not on
# PATH, offer to add it to the shell's rc file (only when run from a terminal),
# and otherwise print the GASP_FAIAL_COST alternative.
offer_path() {
  case ":$PATH:" in
    *":$BIN_DIR:"*) return ;;
  esac
  echo "$BIN_DIR is not on PATH, so gasp can't find faial-cost there yet."

  case "$(basename "${SHELL:-}")" in
    zsh) rc="$HOME/.zshrc" ;;
    bash) rc="$HOME/.bashrc" ;;
    *) rc="$HOME/.profile" ;;
  esac
  line="export PATH=\"$BIN_DIR:\$PATH\""

  if [ -t 0 ]; then
    read -r -p "Add $BIN_DIR to PATH in $rc? [y/N] " answer || answer=""
    case "$answer" in
      y | Y | yes | YES)
        if [ -f "$rc" ] && grep -qxF "$line" "$rc"; then
          echo "$rc already adds $BIN_DIR to PATH"
        else
          printf '\n# Added by gasp scripts/install_faial.sh\n%s\n' "$line" >> "$rc"
          echo "Added to $rc"
        fi
        echo "Run 'source $rc' or open a new shell for it to take effect."
        return
        ;;
    esac
  fi

  echo "Either add it to PATH yourself:"
  echo "  $line"
  echo "or point gasp at the wrapper directly:"
  echo "  export GASP_FAIAL_COST=\"$BIN_DIR/faial-cost\""
}

existing="$(command -v faial-cost || true)"
if [ -z "$existing" ] && [ -x "$BIN_DIR/faial-cost" ]; then
  existing="$BIN_DIR/faial-cost"
fi
if [ -n "$existing" ] && [ "$force" -eq 0 ]; then
  if "$existing" --help > /dev/null 2>&1; then
    echo "faial-cost is already installed at $existing; rerun with --force to reinstall"
    command -v faial-cost > /dev/null || offer_path
    exit 0
  fi
  echo "faial-cost at $existing does not run; reinstalling"
fi

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)
    job=bundle-lin
    libvar=LD_LIBRARY_PATH
    ;;
  Darwin-arm64)
    job=bundle-mac
    libvar=DYLD_LIBRARY_PATH
    [ "$FAIAL_MODE" != container ] || die "container mode is only supported on Linux x86-64"
    FAIAL_MODE=native
    ;;
  *)
    die "no prebuilt faial bundle for $(uname -s)/$(uname -m); use the Dockerfile instead"
    ;;
esac
url="https://gitlab.com/umb-svl/faial/-/jobs/artifacts/main/raw/bundle/faial.tar.bz2?job=$job"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

download() {
  if command -v curl > /dev/null; then
    curl -fL --progress-bar "$1" -o "$2"
  elif command -v wget > /dev/null; then
    wget -nv "$1" -O "$2"
  else
    die "need curl or wget"
  fi
}

install_prefix() {
  echo "Installing to $FAIAL_PREFIX"
  rm -rf "$FAIAL_PREFIX"
  mkdir -p "$(dirname "$FAIAL_PREFIX")"
  mv "$1" "$FAIAL_PREFIX"
}

mode=""

if [ "$FAIAL_MODE" != container ]; then
  echo "Downloading faial bundle ($job)..."
  download "$url" "$tmp/faial.tar.bz2"
  mkdir "$tmp/faial"
  tar xjf "$tmp/faial.tar.bz2" -C "$tmp/faial"
  [ -x "$tmp/faial/bin/faial-cost" ] || die "bundle is missing bin/faial-cost"

  if native_err="$(env PATH="$tmp/faial/bin:$PATH" "$libvar=$tmp/faial/lib" \
    "$tmp/faial/bin/faial-cost" --help 2>&1 > /dev/null)"; then
    mode=native
  elif echo "$native_err" | grep -q "GLIBC"; then
    echo "The prebuilt bundle needs a newer glibc than this host has ($(ldd --version | head -n 1))."
  else
    echo "The prebuilt bundle does not run here:"
    echo "$native_err" | head -n 5 | sed 's/^/  /'
    # libgmp is the one runtime dependency not shipped in the bundle
    if echo "$native_err" | grep -q libgmp; then
      echo "  libgmp.so.10 comes from e.g. 'sudo apt install libgmp10' / 'sudo dnf install gmp'"
    fi
  fi

  if [ -z "$mode" ] && [ "$FAIAL_MODE" = native ]; then
    die "native install failed (unset FAIAL_MODE to allow the container fallback)"
  fi
fi

if [ "$mode" = native ]; then
  install_prefix "$tmp/faial"
else
  runtime="$(command -v apptainer || command -v singularity || true)"
  [ -n "$runtime" ] || die "no apptainer or singularity found for the container fallback; use the Dockerfile instead"

  echo "Pulling $IMAGE with $(basename "$runtime")..."
  mkdir "$tmp/image"
  "$runtime" pull "$tmp/image/$SIF_NAME" "$IMAGE"
  install_prefix "$tmp/image"
  mode=container
fi

mkdir -p "$BIN_DIR"
for tool in "${TOOLS[@]}"; do
  if [ "$mode" = native ]; then
    cat > "$BIN_DIR/$tool" << EOF
#!/bin/bash
# Generated by gasp scripts/install_faial.sh (native)
export PATH="$FAIAL_PREFIX/bin:\$PATH"
export $libvar="$FAIAL_PREFIX/lib\${$libvar:+:\$$libvar}"
exec "$FAIAL_PREFIX/bin/$tool" "\$@"
EOF
  else
    # The image keeps the bundle at /opt/faial. --cleanenv stops host
    # LD_LIBRARY_PATH (e.g. from CUDA modules) leaking in. $HOME and the
    # current directory are bound automatically; set APPTAINER_BIND for others.
    cat > "$BIN_DIR/$tool" << EOF
#!/bin/bash
# Generated by gasp scripts/install_faial.sh (container)
exec "$runtime" exec --cleanenv \\
  --env PATH=/opt/faial/bin:/usr/bin:/bin \\
  --env LD_LIBRARY_PATH=/opt/faial/lib \\
  "$FAIAL_PREFIX/$SIF_NAME" /opt/faial/bin/$tool "\$@"
EOF
  fi
  chmod +x "$BIN_DIR/$tool"
  echo "Installed $BIN_DIR/$tool ($mode)"
done

if "$BIN_DIR/faial-cost" --help > /dev/null 2>&1; then
  echo "faial-cost OK"
else
  echo "warning: '$BIN_DIR/faial-cost --help' failed"
fi

offer_path