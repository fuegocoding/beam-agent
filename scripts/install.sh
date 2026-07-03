#!/bin/bash
# Beam Agent — one-line install for macOS / Linux
#
# Recommended one-liner (file-based — lets the user inspect the script before
# running it, and lets on-access AV scan a real file rather than a stream):
#   curl -fsSL https://raw.githubusercontent.com/fuegocoding/beam-agent/main/scripts/install.sh -o install.sh && bash install.sh && rm install.sh
#
# Legacy one-liner (still works, but the form above is preferred):
#   curl -fsSL https://raw.githubusercontent.com/fuegocoding/beam-agent/main/scripts/install.sh | bash
#
# Environment overrides:
#   BEAM_REPO    Git URL to clone (default: https://github.com/fuegocoding/beam-agent.git)
#   BEAM_BRANCH  Git ref to check out after clone (default: whatever the clone URL implies)
#   BEAM_HOME    Install location (default: $HOME/.beam-agent)
#   BEAM_SKIP_UV Set to 1 to skip the bundled uv install step (assumes uv is on PATH)
#   BEAM_USE_STREAM_UV Set to 1 to use the legacy `curl | sh` form for the uv install
#                      (default: file-based download — friendlier to on-access AV)
set -e

BEAM_REPO="${BEAM_REPO:-https://github.com/fuegocoding/beam-agent.git}"
BEAM_BRANCH="${BEAM_BRANCH:-}"
BEAM_DIR="${BEAM_HOME:-$HOME/.beam-agent}"
BEAM_SKIP_UV="${BEAM_SKIP_UV:-0}"
BEAM_USE_STREAM_UV="${BEAM_USE_STREAM_UV:-0}"

echo "☄ Installing Beam Agent..."
echo "   repo:  $BEAM_REPO"
[ -n "$BEAM_BRANCH" ] && echo "   branch: $BEAM_BRANCH"
echo "   dir:   $BEAM_DIR"

# Install uv if missing.
# File-based by default — same rationale as the Windows script: lets on-access
# AV (e.g. Windows Defender via WSL, or Linux EDR) scan a real file once
# instead of a pipe stream.
if [ "$BEAM_SKIP_UV" != "1" ] && ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    if [ "$BEAM_USE_STREAM_UV" = "1" ]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        uv_installer="$(mktemp -t uv-install-XXXXXX.sh)"
        # Trap fires on any exit path (success, error, signal) so the temp
        # file is cleaned up even if curl/sh fails mid-install.
        trap '[ -n "${uv_installer:-}" ] && rm -f "$uv_installer"' EXIT
        curl -fsSL https://astral.sh/uv/install.sh -o "$uv_installer"
        sh "$uv_installer"
        rm -f "$uv_installer"
        uv_installer=""
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clone or update
if [ -d "$BEAM_DIR" ]; then
    echo "Updating existing installation..."
    cd "$BEAM_DIR"
    git pull --quiet
    if [ -n "$BEAM_BRANCH" ]; then
        git checkout --quiet "$BEAM_BRANCH"
    fi
else
    echo "Cloning beam-agent..."
    if [ -n "$BEAM_BRANCH" ]; then
        git clone --quiet --branch "$BEAM_BRANCH" "$BEAM_REPO" "$BEAM_DIR"
    else
        git clone --quiet "$BEAM_REPO" "$BEAM_DIR"
    fi
    cd "$BEAM_DIR"
fi

# Install
echo "Installing dependencies..."
uv tool install -e "$BEAM_DIR" --python 3.12 --force 2>/dev/null || \
uv tool install -e "$BEAM_DIR" --force 2>/dev/null || \
uv pip install -e "$BEAM_DIR" --system 2>/dev/null || {
    echo "Trying with venv fallback..."
    uv venv .venv --python 3.12 2>/dev/null || uv venv .venv
    uv pip install -e "$BEAM_DIR" --python .venv
    echo ""
    echo "Add to your PATH: $BEAM_DIR/.venv/bin"
}

# Create beam directories
mkdir -p "$HOME/.beam/brain/default"
mkdir -p "$HOME/.beam/memory/default"

echo ""
echo "☄ Beam Agent installed!"
echo ""
echo "Run:  beam"
echo ""
echo "On first run, Beam will:"
echo "  1. Set up your LLM provider and API key"
echo "  2. Run the interview to build your digital clone"
echo ""
