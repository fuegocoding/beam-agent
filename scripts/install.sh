#!/bin/bash
# Beam Agent — one-line install
# Usage: curl -fsSL https://raw.githubusercontent.com/fuegocoding/beam-agent/main/scripts/install.sh | bash
set -e

BEAM_REPO="https://github.com/fuegocoding/beam-agent.git"
BEAM_DIR="${BEAM_HOME:-$HOME/.beam-agent}"

echo "☄ Installing Beam Agent..."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clone or update
if [ -d "$BEAM_DIR" ]; then
    echo "Updating existing installation..."
    cd "$BEAM_DIR"
    git pull --quiet
else
    echo "Cloning beam-agent..."
    git clone --quiet "$BEAM_REPO" "$BEAM_DIR"
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
