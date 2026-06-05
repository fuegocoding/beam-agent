# Beam Agent — one-line install for Windows PowerShell
# Usage: iex (irm https://raw.githubusercontent.com/fuegocoding/beam-agent/main/scripts/install.ps1)
$ErrorActionPreference = "Stop"

$BEAM_REPO = "https://github.com/fuegocoding/beam-agent.git"
$BEAM_DIR = if ($env:BEAM_HOME) { $env:BEAM_HOME } else { "$env:USERPROFILE\.beam-agent" }

Write-Host "☄ Installing Beam Agent..." -ForegroundColor Cyan

# Check for uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    $ErrorActionPreference = "Continue"
    irm https://astral.sh/uv/install.ps1 | iex
    $ErrorActionPreference = "Stop"
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
}

# Clone or update
if (Test-Path "$BEAM_DIR\.git") {
    Write-Host "Updating existing installation..."
    git -C $BEAM_DIR pull --quiet
} else {
    Write-Host "Cloning beam-agent..."
    git clone --quiet $BEAM_REPO $BEAM_DIR
}

# Install
Write-Host "Installing dependencies..."
Set-Location $BEAM_DIR
$ErrorActionPreference = "Continue"
try {
    uv tool install -e $BEAM_DIR --python 3.12 --force 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "uv tool install failed" }
} catch {
    try {
        uv tool install -e $BEAM_DIR --force 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "uv tool install failed" }
    } catch {
        Write-Host "Trying with venv fallback..."
        uv venv .venv --python 3.12 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { uv venv .venv 2>&1 | Out-Null }
        uv pip install -e $BEAM_DIR --python .venv 2>&1 | Out-Null
        Write-Host ""
        Write-Host "Add to your PATH: $BEAM_DIR\.venv\Scripts"
    }
}
$ErrorActionPreference = "Stop"

# Create beam directories
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.beam\brain\default" | Out-Null
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.beam\memory\default" | Out-Null

Write-Host ""
Write-Host "☄ Beam Agent installed!" -ForegroundColor Green
Write-Host ""
Write-Host "Run:  beam"
Write-Host ""
Write-Host "On first run, Beam will:"
Write-Host "  1. Set up your LLM provider and API key"
Write-Host "  2. Run the interview to build your digital clone"
Write-Host ""
