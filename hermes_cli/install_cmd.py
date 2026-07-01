"""beam install - Install brains from the Beam marketplace.

Usage:
    beam install <slug>                # Install official brain
    beam install @user/<slug>          # Install community brain

All marketplace brains are free. The full personality_graph.json is
downloaded once and stored locally at ~/.beam/brains/<name>/personality_graph.json.

If Neo4j is configured (via ``beam brain setup-neo4j``), the brain
is auto-ingested into Neo4j on install so ``beam brain platform-search``
and the agent's GraphBackedBrainRetriever work against the same brain
without a separate ingest step.
All queries run offline against the local file.
"""
import json
import os
import sys
from pathlib import Path

import httpx


def _is_neo4j_configured() -> bool:
    """True if the user has set up Neo4j creds (in process env or ~/.hermes/.env).

    Used by ``beam install`` to decide whether to auto-ingest the
    marketplace brain into Neo4j. If Neo4j isn't configured, the
    install just leaves the JSON on disk for offline use.
    """
    for var in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        if os.environ.get(var):
            return True
    # Fall back to ~/.hermes/.env (where setup-neo4j writes the creds)
    try:
        from hermes_cli.config import load_env
        env = load_env()
        if env.get("NEO4J_URI") and env.get("NEO4J_USER") and env.get("NEO4J_PASSWORD"):
            return True
    except Exception:
        pass
    return False


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


# The full marketplace catalog. Mirrors openbeam.me/marketplace.
# Used by `beam install` (no args) for the interactive picker and by
# `beam install --list` to show what's available.
MARKETPLACE_CATALOG = [
    ("bill-gates", "Bill Gates", "technologist, philanthropist (Microsoft)"),
    ("elon-musk", "Elon Musk", "engineer, entrepreneur (Tesla, SpaceX)"),
    ("marcus-aurelius", "Marcus Aurelius", "Roman emperor, Stoic philosopher"),
    ("seneca", "Seneca", "Roman Stoic philosopher, tutor to Nero"),
    ("terence-tao", "Terence Tao", "Fields Medalist mathematician"),
    ("albert-einstein", "Albert Einstein", "theoretical physicist"),
    ("benjamin-franklin", "Benjamin Franklin", "founding father, scientist"),
    ("virginia-woolf", "Virginia Woolf", "modernist novelist, feminist"),
    ("leonardo-da-vinci", "Leonardo da Vinci", "Renaissance polymath"),
]


def _print_catalog(installed: set = None) -> None:
    """Print the marketplace catalog, marking already-installed brains."""
    installed = installed or set()
    print("\nMarketplace brains:")
    for slug, name, desc in MARKETPLACE_CATALOG:
        marker = "  ✓ INSTALLED" if slug in installed else ""
        print(f"  {slug:20s} {name:18s} — {desc}{marker}")
    print(f"\nBrowse: https://openbeam.me/marketplace")


def _interactive_pick(installed: set) -> str:
    """Show catalog + ask the user to pick a slug. Returns the chosen slug."""
    print()
    _print_catalog(installed)
    available = [s for s, _, _ in MARKETPLACE_CATALOG if s not in installed]
    if not available:
        print("\nYou already have all marketplace brains installed.")
        return ""
    default = available[0]
    print()
    return input(f"Slug [{default}]: ").strip() or default


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
    list_only = getattr(args, "list_only", False)

    # `beam install --list` — just show the catalog
    if list_only:
        installed = {b["name"] for b in list_brains()}
        _print_catalog(installed)
        return 0

    # `beam install` (no args) — interactive picker
    if not raw_slug:
        try:
            installed = {b["name"] for b in list_brains()}
            raw_slug = _interactive_pick(installed)
        except KeyboardInterrupt:
            print("\nCancelled.")
            return 1
        if not raw_slug:
            # User has all brains, or cancelled
            return 1

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

    try:
        graph_data = _download_brain(display_slug, brain_path)
    except SystemExit:
        raise
    except Exception as exc:
        # _download_brain already prints a clear error and exits — but
        # if anything else goes wrong, surface it here without crashing.
        print(f"Install failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Try again, or install manually from https://openbeam.me/marketplace")
        sys.exit(1)

    # Show what was downloaded
    size_kb = (brain_path / "personality_graph.json").stat().st_size / 1024
    node_count = len(graph_data.get("knowledge_graph", {}).get("nodes", []))
    print(f"  Downloaded: {size_kb:.1f} KB, {node_count} nodes")

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

    # If Neo4j is configured, auto-ingest the brain into Neo4j so
    # `beam brain platform-search` works against the same brain
    # without a separate `platform-ingest` step. This is what makes
    # the marketplace brain immediately queryable via the agent's
    # GraphBackedBrainRetriever.
    if not no_activate and _is_neo4j_configured():
        try:
            from brain_platform.cli.integration import _ingest_brain_file_json
            graph_path = brain_path / "personality_graph.json"
            if graph_path.exists():
                _ingest_brain_file_json(graph_path, install_name)
                print(f"  Neo4j: ingested into group '{install_name}' (Neo4j-backed search is now live)")
        except Exception as exc:
            # Don't fail the install if Neo4j ingest fails — the
            # marketplace brain is still usable offline. Just log it.
            print(f"  Neo4j: auto-ingest failed ({type(exc).__name__}). "
                  f"Run 'beam brain platform-ingest {graph_path}' manually.")

    # Eagerly materialize ~/.hermes/SOUL.md from the newly-installed
    # brain so the next `beam` launch already has the new identity in
    # place. Without this the user has to wait for the
    # on_session_start hook to fire (which only handles a few edge
    # cases — see plugins/brain-tools/_on_session_start).
    if not no_activate:
        try:
            from hermes_cli.brain_cmds import _regenerate_soul
            _regenerate_soul(install_name)
            print(f"  SOUL.md regenerated for '{install_name}'.")
        except Exception as exc:
            print(f"  Warning: Could not regenerate SOUL.md: {exc}", file=sys.stderr)

    return 0


def register_install_command(subparsers):
    """Register the 'install' subcommand with argparse."""
    parser = subparsers.add_parser(
        "install",
        help="Install a brain from the Beam marketplace",
        description=(
            "Install pre-built brains from the marketplace. "
            "Run without arguments for an interactive picker, "
            "or use --list to see available brains."
        ),
    )
    parser.add_argument(
        "slug",
        nargs="?",  # optional — interactive picker if omitted
        help="Brain slug (e.g., 'creative-writer' or '@alice/coach')",
    )
    parser.add_argument(
        "--no-activate",
        action="store_true",
        help="Don't set as active brain after install",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="List available marketplace brains and exit",
    )
    parser.set_defaults(func=cmd_install)
