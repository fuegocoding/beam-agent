"""beam install - Install brains from the Beam marketplace.

Usage:
    beam install <slug>                # Install official brain
    beam install @user/<slug>          # Install community brain

All marketplace brains are free. The full personality_graph.json is
downloaded once and stored locally at ~/.beam/brains/<name>/personality_graph.json.
All queries run offline against the local file.
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


def _download_brain(slug: str, output_path: Path) -> dict:
    """Install a brain from the marketplace.

    Downloads the full personality_graph.json via the public download
    endpoint. No auth required — all marketplace brains are free.
    """
    api_url = _get_api_url()
    output_path.mkdir(parents=True, exist_ok=True)

    url = f"{api_url}/api/v1/marketplace/{slug}/download"
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            graph_data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"Error: Brain '{slug}' not found on marketplace.", file=sys.stderr)
        else:
            print(f"Error: API returned {e.response.status_code}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print("Error: Cannot connect to Beam API.", file=sys.stderr)
        sys.exit(1)

    graph_path = output_path / "personality_graph.json"
    graph_path.write_text(json.dumps(graph_data, indent=2), encoding="utf-8")
    return graph_data


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
    no_activate = getattr(args, "no_activate", False)

    if not raw_slug:
        print("Usage: beam install <slug> [--no-activate]", file=sys.stderr)
        print("\nExamples:")
        print("  beam install creative-writer          # Official brain")
        print("  beam install @alice/coach              # Community brain")
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

    _download_brain(display_slug, brain_path)

    # Determine source type
    if raw_slug.startswith("@"):
        source = "marketplace-community"
    else:
        source = "marketplace-official"

    # Register in config
    slug_for_config = display_slug if display_slug.startswith("@") else None
    register_brain(
        install_name,
        source=source,
        slug=slug_for_config,
    )

    # Set as active brain unless --no-activate
    if not no_activate:
        set_active_brain(install_name)
        print(f"Brain '{install_name}' installed and set as active.")
    else:
        print(f"Brain '{install_name}' installed.")

    # Show summary
    print(f"  Type: Local (full brain, works offline)")
    print(f"  Path: {brain_path / 'personality_graph.json'}")


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
        "--no-activate",
        action="store_true",
        help="Don't set as active brain after install",
    )
    parser.set_defaults(func=cmd_install)
