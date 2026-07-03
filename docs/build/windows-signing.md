# Windows installer signing

Why this exists: `Hermes-Setup.exe` is the only Windows binary the project ships. Without an Authenticode signature, Windows SmartScreen blocks the first run with "Windows protected your PC — Microsoft Defender SmartScreen prevented an unrecognized app from starting" and corporate Defender policies may quarantine the file outright. The Tauri-based bootstrap installer is the entry point, so it has to be signed.

## Current state

- `apps/bootstrap-installer/src-tauri/tauri.conf.json` declares the bundle (`nsis` target, `currentUser` install mode, embedded WebView2 bootstrapper).
- `.github/workflows/build-windows-installer.yml` builds `Hermes-Setup.exe` on `windows-latest` and signs it via **Azure Artifact Signing** (`azure/artifact-signing-action@v2`) — see the "Sign Hermes-Setup.exe with Azure Artifact Signing" step.
- The signing endpoint, account, and certificate profile are sourced from GitHub repo vars: `AZURE_SIGNING_ENDPOINT`, `AZURE_SIGNING_ACCOUNT_NAME`, `AZURE_SIGNING_CERTIFICATE_PROFILE`. OIDC auth uses `AZURE_CLIENT_ID` / `AZURE_TENANT_ID` / `AZURE_SUBSCRIPTION_ID` secrets with `id-token: write` permission.
- The workflow is gated behind an explicit admin check (`authorize` job calls `gh api ... /collaborators/${ACTOR}/permission`) — only repo admins can trigger it via `workflow_dispatch`.

## Adding a new signing cert / rotating

1. **Azure side**: create a new certificate profile in the Azure Trusted Signing account. Copy the profile name into the `AZURE_SIGNING_CERTIFICATE_PROFILE` repo variable.
2. **GitHub side**: the existing `AZURE_CLIENT_ID` / `AZURE_TENANT_ID` / `AZURE_SUBSCRIPTION_ID` federated credentials should already be wired to the action. If you're rotating the *app registration*, update the secrets and the federated credential subject on the Azure app reg to match the new client ID.
3. **No code change needed** — the workflow reads the profile name from vars, not from source.

## Local signing (for dev builds)

Tauri's `bundle.windows.signtool` config is intentionally **not** in `tauri.conf.json` — it requires a local `.pfx` / `.p12` cert and Tauri fails the build if the cert path doesn't resolve, which would break local builds for everyone (most devs don't have a code-signing cert). The CI workflow handles production signing.

If you have a local cert and want to sign a dev build:

```powershell
# Option A: external signing after tauri build
npm run tauri:build
# tauri writes Hermes-Setup.exe under apps/bootstrap-installer/src-tauri/target/release/
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 `
  /f path\to\cert.pfx /p $env:CERT_PASSWORD `
  apps\bootstrap-installer\src-tauri\target\release\Hermes-Setup.exe

# Option B: enable Tauri's built-in signing for one build
# Set HERMES_WINDOWS_CERT_PATH, then uncomment the signtool block in
# tauri.conf.json (see git history for the snippet) and run npm run tauri:build.
# Remember to revert before committing.
```

## Why Tauri and not electron-builder?

The Electron desktop (`apps/desktop/`) has its own electron-builder pipeline with `signAndEditExecutable: false` (see `apps/desktop/scripts/set-exe-identity.cjs` for the why — winCodeSign symlinks fail on non-admin Windows). The bootstrap installer is the only Windows entry point the project ships today; the desktop is built from source post-install via `hermes desktop --build-only` (see `apps/bootstrap-installer/src-tauri/src/update.rs:241`). Users who need the desktop app on Windows currently build it themselves, so SmartScreen prompts there are expected.

## What this doesn't cover (and why)

- **node-pty / pywinpty helper binaries** (`OpenConsole.exe`, `winpty-agent.exe`): signed by upstream maintainers via their wheel signing chain. The Tauri installer doesn't bundle them, so Azure signing them is unnecessary for the install path. If a future Windows bundle ever ships the desktop, see `apps/desktop/scripts/stage-native-deps.cjs` for where to plug in a signing step.
- **MSIX/AppX**: out of scope; the project ships NSIS.
- **EV certificates**: not required for SmartScreen reputation once enough signed installs accrue, but an EV cert (vs the standard OV used by Azure Trusted Signing) gives immediate SmartScreen trust on first run. Cost-benefit depends on distribution volume; not currently justified.

## Verifying a build

After downloading `Hermes-Setup.exe` from a workflow run:

```powershell
# Confirm signature + cert chain
Get-AuthenticodeSignature .\Hermes-Setup.exe | Select-Object Status, SignerCertificate.Subject
# Confirm timestamp (proves the signature is valid past the cert's expiry)
(Get-AuthenticodeSignature .\Hermes-Setup.exe).TimeStamperCertificate
```

A correctly signed build shows `Status: Valid` and a subject of `CN=Nous Research, O=Nous Research, ...` (or whatever the cert profile is configured for).
