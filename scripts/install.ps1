# Beam Agent - one-line install for Windows PowerShell
#
# Recommended one-liner (file-based - AMSI scans the file once, not a stream):
#   irm https://raw.githubusercontent.com/fuegocoding/beam-agent/main/scripts/install.ps1 -OutFile install.ps1; .\install.ps1
#
# Legacy one-liner (works in non-managed environments; some Defender policies
# block the `irm | iex` stream pattern. The file-based form above is preferred):
#   iex (irm https://raw.githubusercontent.com/fuegocoding/beam-agent/main/scripts/install.ps1)
#
# Environment overrides:
#   $env:BEAM_REPO    Git URL to clone (default: https://github.com/fuegocoding/beam-agent.git)
#   $env:BEAM_BRANCH  Git ref to check out after clone (default: whatever the clone URL implies)
#   $env:BEAM_HOME    Install location (default: $env:USERPROFILE\.beam-agent)
#   $env:BEAM_SKIP_UV Set to 1 to skip the bundled uv install step (assumes uv is on PATH)
#
[CmdletBinding()]
param(
    [string]$Repo,
    [string]$Branch,
    [string]$InstallDir,
    [switch]$SkipUv,
    [switch]$UseInlineUv
)

$ErrorActionPreference = "Stop"

# Resolve config: param > env > default.
$BEAM_REPO = if ($Repo) { $Repo } elseif ($env:BEAM_REPO) { $env:BEAM_REPO } else { "https://github.com/fuegocoding/beam-agent.git" }
$BEAM_BRANCH = if ($Branch) { $Branch } elseif ($env:BEAM_BRANCH) { $env:BEAM_BRANCH } else { "" }
$BEAM_DIR = if ($InstallDir) { $InstallDir } elseif ($env:BEAM_HOME) { $env:BEAM_HOME } else { "$env:USERPROFILE\.beam-agent" }
$BEAM_SKIP_UV = $SkipUv -or ($env:BEAM_SKIP_UV -eq "1")

Write-Host "☄ Installing Beam Agent..." -ForegroundColor Cyan
Write-Host "   repo:  $BEAM_REPO"
if ($BEAM_BRANCH) { Write-Host "   branch: $BEAM_BRANCH" }
Write-Host "   dir:   $BEAM_DIR"

# Install uv if missing.
# The file-based form (irm -OutFile then & .\file.ps1) lets AMSI scan a real
# file on disk and lets the user inspect what they're about to run. The
# `irm | iex` stream form triggers AMSI per-tokens-in-flight and is blocked
# by stricter Defender policies / corporate GPOs.
if (-not $BEAM_SKIP_UV -and -not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    $uvInstaller = Join-Path $env:TEMP "uv-install-$([guid]::NewGuid()).ps1"
    try {
        if ($UseInlineUv) {
            # Backwards-compat: the historical one-liner form. Kept as an opt-in
            # because it works on stock Win10/11 and skips a disk write, but
            # it makes AMSI's job harder.
            $ErrorActionPreference = "Continue"
            irm https://astral.sh/uv/install.ps1 | iex
            $ErrorActionPreference = "Stop"
        } else {
            irm https://astral.sh/uv/install.ps1 -OutFile $uvInstaller -UseBasicParsing
            & $uvInstaller
        }
        $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
    } finally {
        Remove-Item -LiteralPath $uvInstaller -ErrorAction SilentlyContinue
    }
}

# Clone or update
if (Test-Path "$BEAM_DIR\.git") {
    Write-Host "Updating existing installation..."
    git -C $BEAM_DIR pull --quiet
    if ($BEAM_BRANCH) {
        git -C $BEAM_DIR checkout --quiet $BEAM_BRANCH
    }
} else {
    Write-Host "Cloning beam-agent..."
    $cloneArgs = @("--quiet", $BEAM_REPO, $BEAM_DIR)
    if ($BEAM_BRANCH) {
        $cloneArgs = @("--quiet", "--branch", $BEAM_BRANCH) + $cloneArgs
    }
    git clone @cloneArgs
}

# Install
Write-Host "Installing dependencies..."
Set-Location $BEAM_DIR
$ErrorActionPreference = "Continue"

# Drop any prior editable install before re-installing. `uv tool install -e`
# writes a `__editable___hermes_agent_*.pth` finder that hardcodes the source
# path; once installed, it survives subsequent `uv tool install --force` calls
# (the new install lands in the same venv but the .pth still points at the
# old path). If the install dir ever changes (recovery from a broken install,
# moving the checkout, running this script against a temp dir for testing),
# the venv keeps importing from the stale path and `beam update` then tries
# to operate on the wrong repo. Uninstalling first guarantees a clean
# re-link. Failures here are non-fatal -- a first-time install won't have
# anything to uninstall, and an uninstall against a missing tool is a no-op.
try { uv tool uninstall hermes-agent 2>&1 | Out-Null } catch {}

try {
    uv tool install $BEAM_DIR --python 3.12 --force 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "uv tool install failed" }
} catch {
    try {
        uv tool install $BEAM_DIR --force 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "uv tool install failed" }
    } catch {
        Write-Host "Trying with venv fallback..."
        uv venv .venv --python 3.12 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { uv venv .venv 2>&1 | Out-Null }
        uv pip install $BEAM_DIR --python .venv 2>&1 | Out-Null
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
