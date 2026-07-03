@echo off
REM ============================================================================
REM Beam Agent Installer for Windows (CMD wrapper)
REM ============================================================================
REM This batch file launches the PowerShell installer for users running CMD.
REM
REM Usage (one-liner):
REM   curl -fsSL https://raw.githubusercontent.com/fuegocoding/beam-agent/main/scripts/install.cmd -o install.cmd ^&^& install.cmd ^&^& del install.cmd
REM
REM Or if you're already in PowerShell, use the file-based form directly
REM (preferred for AMSI / Defender compatibility — see install.ps1 header):
REM   irm https://raw.githubusercontent.com/fuegocoding/beam-agent/main/scripts/install.ps1 -OutFile install.ps1 ^& .\install.ps1
REM
REM Environment overrides forwarded to install.ps1:
REM   set BEAM_REPO=<git-url>     Override the cloned repo URL
REM   set BEAM_BRANCH=<ref>       Pin a specific branch / tag / commit
REM   set BEAM_HOME=<path>        Override install location
REM   set BEAM_SKIP_UV=1          Skip the bundled uv install step
REM ============================================================================

setlocal

echo.
echo  Beam Agent Installer
echo  Launching PowerShell installer...
echo.

REM Pass through BEAM_* env vars so users can pin repo/branch/home from CMD.
set "PS_FORWARDED_ENV="
if defined BEAM_REPO    set "PS_FORWARDED_ENV=%PS_FORWARDED_ENV% $env:BEAM_REPO='%BEAM_REPO%';"
if defined BEAM_BRANCH  set "PS_FORWARDED_ENV=%PS_FORWARDED_ENV% $env:BEAM_BRANCH='%BEAM_BRANCH%';"
if defined BEAM_HOME    set "PS_FORWARDED_ENV=%PS_FORWARDED_ENV% $env:BEAM_HOME='%BEAM_HOME%';"
if defined BEAM_SKIP_UV set "PS_FORWARDED_ENV=%PS_FORWARDED_ENV% $env:BEAM_SKIP_UV='%BEAM_SKIP_UV%';"

REM File-based download (NOT irm | iex) — see install.ps1 for the rationale
REM (AMSI scans the file once instead of per-tokens-in-flight; the user can
REM inspect it before running).
powershell -ExecutionPolicy ByPass -NoProfile -Command ^
    "$ErrorActionPreference = 'Stop';" ^
    "$url = 'https://raw.githubusercontent.com/fuegocoding/beam-agent/main/scripts/install.ps1';" ^
    "$tmp = Join-Path $env:TEMP ('beam-install-' + [guid]::NewGuid() + '.ps1');" ^
    "try {" ^
    "  Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing;" ^
    "  %PS_FORWARDED_ENV%" ^
    "  & $tmp" ^
    "} finally {" ^
    "  Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue" ^
    "}"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  Installation failed. Please try running PowerShell directly with the
    echo  file-based form (preferred for AMSI / Defender compatibility):
    echo.
    echo    powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://raw.githubusercontent.com/fuegocoding/beam-agent/main/scripts/install.ps1 -OutFile install.ps1; .\install.ps1"
    echo.
    pause
    exit /b 1
)

endlocal
