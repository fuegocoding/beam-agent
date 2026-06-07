"""beam install - Install brains from the Beam marketplace.

Usage:
    beam install <slug>                    # Install official brain (no @)
    beam install @user/<slug>              # Install community brain
    beam install @user/<slug> --token=xxx  # Install paid brain

Brains are NEVER downloaded as full JSON. Instead, a lightweight proxy
config is stored in ~/.beam/brains/<name>/brain_config.json. All queries
are forwarded to the Beam API so the full personality graph stays server-side
and cannot be stolen.
"""
import json
import sys
from pathlib import Path

import httpx


API_URL_DEFAULT = "https://api.openbeam.me"


def _get_api_url() -> str:
    """Get the API URL from environment or default."""
    import os
    return os.environ.get("BEAM_API_URL", API_URL_DEFAULT)


def _parse_slug(raw_slug: str) -> tuple[str, str]:
    """Parse a slug into (display_slug, install_name).

    Examples:
        "creative-writer" -> ("creative-writer", "creative-writer")
        "@alice/coach" -> ("@alice/coach", "alice-coach")
        "@beam/writer" -> ("@beam/writer", "writer")  # official brain
    """
    if raw_slug.startswith("@"):
        # Community brain: @user/slug
        parts = raw_slug[1:].split("/", 1)
        if len(parts) == 2:
            user, slug = parts
            if user == "beam":
                # Official brain: @beam/writer -> install as "writer"
                return raw_slug, slug
            else:
                # Community brain: @alice/coach -> install as "alice-coach"
                return raw_slug, f"{user}-{slug}"
        else:
            return raw_slug, raw_slug[1:]
    else:
        # Official brain without @ prefix
        return raw_slug, raw_slug


def _get_auth_token() -> str | None:
    """Get the user's auth token from ~/.beam/auth.yaml or env."""
    import os
    from pathlib import Path

    # Env override
    env_token = os.environ.get("BEAM_AUTH_TOKEN")
    if env_token:
        return env_token

    # Auth file
    auth_path = Path.home() / ".beam" / "auth.yaml"
    if auth_path.exists():
        try:
            import yaml
            data = yaml.safe_load(auth_path.read_text(encoding="utf-8"))
            return data.get("token")
        except Exception:
            pass
    return None


def _download_brain(slug: str, output_path: Path, token: str | None = None) -> dict:
    """Install a brain from the marketplace.

    For ALL brains (free, community, paid) we store a proxy config
    (brain_config.json) instead of downloading the full personality graph.
    The graph stays server-side and is queried via API.
    """
    api_url = _get_api_url()
    output_path.mkdir(parents=True, exist_ok=True)

    if token:
        # Paid brain — user provided a purchase token directly
        # Validate it by pinging the context endpoint
        url = f"{api_url}/api/v1/brain-proxy/{slug}/context"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                print("Error: Invalid or expired install token.", file=sys.stderr)
                sys.exit(1)
            elif e.response.status_code == 402:
                print("Error: Purchase is no longer active.", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"Error: API returned {e.response.status_code}", file=sys.stderr)
                sys.exit(1)
        except httpx.ConnectError:
            print("Error: Cannot connect to Beam API. Paid brains require internet.", file=sys.stderr)
            sys.exit(1)

        proxy_config = {
            "type": "proxy",
            "slug": slug,
            "token": token,
            "api_url": api_url,
        }
        with open(output_path / "brain_config.json", "w") as f:
            json.dump(proxy_config, f, indent=2)
        return proxy_config

    # Free / community brain — request an install token from the API
    auth_token = _get_auth_token()
    if not auth_token:
        print("Error: BEAM_AUTH_TOKEN environment variable required for installing marketplace brains.", file=sys.stderr)
        print("  1. Go to https://beammind.dev/dashboard/settings", file=sys.stderr)
        print("  2. Copy your CLI token from the 'CLI Access' section", file=sys.stderr)
        print("  3. Run: export BEAM_AUTH_TOKEN=<your-token>", file=sys.stderr)
        print("  4. Then: beam install creative-writer", file=sys.stderr)
        sys.exit(1)

    url = f"{api_url}/api/v1/marketplace/{slug}/install-token"
    headers = {"Authorization": f"Bearer {auth_token}"}
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers)
            resp.raise_for_status()
            token_data = resp.json()
            install_token = token_data["token"]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"Error: Brain '{slug}' not found on marketplace.", file=sys.stderr)
        elif e.response.status_code == 401:
            print("Error: Authentication failed. Please log in again.", file=sys.stderr)
        else:
            print(f"Error: API returned {e.response.status_code}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print("Error: Cannot connect to Beam API.", file=sys.stderr)
        sys.exit(1)

    proxy_config = {
        "type": "proxy",
        "slug": slug,
        "token": install_token,
        "api_url": api_url,
    }
    with open(output_path / "brain_config.json", "w") as f:
        json.dump(proxy_config, f, indent=2)
    return proxy_config


def cmd_install(args):
    """Handle 'beam install' command."""
    from brain.paths import (
        ensure_beam_dirs,
        get_brain_path,
        list_brains,
        register_brain,
        set_active_brain,
    )

    # Parse arguments
    raw_slug = getattr(args, "slug", None)
    token = getattr(args, "token", None)
    no_activate = getattr(args, "no_activate", False)

    if not raw_slug:
        print("Usage: beam install <slug> [--token=<token>] [--no-activate]", file=sys.stderr)
        print("\nExamples:")
        print("  beam install creative-writer          # Official brain")
        print("  beam install @alice/coach              # Community brain")
        print("  beam install @bob/writer --token=xxx   # Paid brain")
        sys.exit(1)

    # Parse slug
    display_slug, install_name = _parse_slug(raw_slug)

    # Check if already installed
    brains = list_brains()
    existing = [b for b in brains if b["name"] == install_name]
    if existing:
        print(f"Brain '{install_name}' is already installed.", file=sys.stderr)
        print(f"Use 'beam brain remove {install_name}' first to reinstall.", file=sys.stderr)
        sys.exit(1)

    # Ensure directories exist
    ensure_beam_dirs()

    # Download
    print(f"Installing brain '{display_slug}'...")
    brain_path = get_brain_path(install_name)

    brain_data = _download_brain(display_slug, brain_path, token=token)

    # Determine source type
    if token:
        source = "marketplace-paid"
    elif raw_slug.startswith("@"):
        source = "marketplace-community"
    else:
        source = "marketplace-official"

    # Register in config
    slug_for_config = display_slug if display_slug.startswith("@") else None
    register_brain(
        install_name,
        source=source,
        slug=slug_for_config,
        token=token,
    )

    # Set as active brain unless --no-activate
    if not no_activate:
        set_active_brain(install_name)
        print(f"Brain '{install_name}' installed and set as active.")
    else:
        print(f"Brain '{install_name}' installed.")

    # Show summary
    print(f"  Type: API proxy (full graph stays server-side)")
    print(f"  Note: This brain requires internet to use.")


def register_install_command(subparsers):
    """Register the 'install' subcommand with argparse."""
    parser = subparsers.add_parser(
        "install",
        help="Install a brain from the Beam marketplace",
        description="Install pre-built brains from the marketplace.",
    )
    parser.add_argument(
        "slug",
        help="Brain slug (e.g., 'creative-writer' or '@alice/coach')",
    )
    parser.add_argument(
        "--token",
        help="Install token for paid brains",
    )
    parser.add_argument(
        "--no-activate",
        action="store_true",
        help="Don't set as active brain after install",
    )
    parser.set_defaults(func=cmd_install)
