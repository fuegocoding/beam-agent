"""Multi-brain management commands for the CLI.

Adds: /brain list, /brain switch, /brain info, /brain remove, /brain install
"""
import json
import shutil
import sys
from pathlib import Path


def cmd_brain_list():
    """List all installed brains."""
    from brain.paths import list_brains

    brains = list_brains()
    if not brains:
        print("No brains installed.")
        print("Use 'beam install <slug>' to install a brain from the marketplace.")
        print("(Or run 'beam install' with no args for the interactive picker.)")
        return

    print("Installed brains:\n")
    for b in brains:
        active_marker = "●" if b["active"] else "○"
        source_tag = f"[{b['source']}]"
        token_tag = " 🔒" if b["has_token"] else ""
        print(f"  {active_marker} {b['name']:<20} {source_tag:<25}{token_tag}")

    print(f"\nActive: {next((b['name'] for b in brains if b['active']), 'none')}")
    print("Use 'beam brain install' (no args) to add a marketplace brain.")


def cmd_brain_install(slug: str | None = None, no_activate: bool = False):
    """Install a marketplace brain.

    Delegates to ``hermes_cli.install_cmd.cmd_install`` which handles
    the full download + register + auto-ingest-into-Neo4j +
    regenerate-SOUL.md flow.

    With no slug, ``cmd_install`` shows the interactive catalog picker
    (recommended default = first un-installed brain).
    """
    import argparse
    from hermes_cli.install_cmd import cmd_install as _install

    args = argparse.Namespace(
        brain=slug,
        no_activate=no_activate,
        list_only=False,
    )
    return _install(args)


def cmd_brain_switch(name: str):
    """Switch the active brain."""
    from brain.paths import (
        get_brain_path,
        get_brain_soul_path,
        list_brains,
        set_active_brain,
    )

    brains = list_brains()
    brain_names = [b["name"] for b in brains]

    if name not in brain_names:
        print(f"Brain '{name}' not found.", file=sys.stderr)
        print(f"Available brains: {', '.join(brain_names)}", file=sys.stderr)
        sys.exit(1)

    # Check the brain has data
    brain_path = get_brain_path(name)
    graph_path = brain_path / "personality_graph.json"

    if not graph_path.exists():
        print(f"Warning: Brain '{name}' has no data files.", file=sys.stderr)

    set_active_brain(name)
    print(f"Switched to brain '{name}'.")

    # Generate SOUL.md for the new brain
    try:
        _regenerate_soul(name)
        print(f"SOUL.md regenerated for '{name}'.")
    except Exception as e:
        print(f"Warning: Could not regenerate SOUL.md: {e}")


def cmd_brain_info(name: str | None = None):
    """Show info about a brain."""
    from brain.paths import (
        get_active_brain_name,
        get_brain_info,
        get_brain_path,
    )

    if not name:
        name = get_active_brain_name()

    info = get_brain_info(name)
    brain_path = get_brain_path(name)

    print(f"Brain: {name}")
    print(f"Path: {brain_path}")

    if info:
        print(f"Source: {info.get('source', 'unknown')}")
        print(f"Installed: {info.get('installed_at', 'unknown')}")
        if info.get("slug"):
            print(f"Marketplace slug: {info['slug']}")
        if info.get("token"):
            print(f"Has install token: yes")

    # Check for data
    graph_path = brain_path / "personality_graph.json"
    soul_path = brain_path / "soul.md"

    if graph_path.exists():
        try:
            with open(graph_path, encoding="utf-8") as f:
                graph = json.load(f)
            node_count = 0
            for key in ["traits", "beliefs", "values", "memories", "patterns",
                        "procedural_patterns", "work_loops", "expertise"]:
                node_count += len(graph.get(key, []))
            kg = graph.get("knowledge_graph", {})
            node_count += len(kg.get("nodes", []))
            edge_count = len(graph.get("edges", [])) + len(kg.get("edges", []))
            print(f"Type: Local (full brain, offline)")
            print(f"Nodes: {node_count}")
            print(f"Edges: {edge_count}")
        except Exception:
            print(f"Type: Local (could not parse graph)")
    else:
        print(f"Type: No data")

    if soul_path.exists():
        print(f"SOUL.md: present ({soul_path.stat().st_size} bytes)")
    else:
        print(f"SOUL.md: not generated")


def cmd_brain_remove(name: str):
    """Remove an installed brain."""
    from brain.paths import (
        get_active_brain_name,
        get_brain_path,
        list_brains,
        set_active_brain,
        unregister_brain,
    )

    brains = list_brains()
    brain_names = [b["name"] for b in brains]

    if name not in brain_names:
        print(f"Brain '{name}' not found.", file=sys.stderr)
        sys.exit(1)

    if name == "default":
        print("Cannot remove the default brain.", file=sys.stderr)
        sys.exit(1)

    # Confirm
    active = get_active_brain_name()
    if name == active:
        print(f"Warning: '{name}' is the active brain.")

    confirm = input(f"Remove brain '{name}'? [y/N] ")
    if confirm.lower() != "y":
        print("Cancelled.")
        return

    # If removing active brain, switch to default
    if name == active:
        set_active_brain("default")
        print("Switched to 'default' brain.")

    # Remove files
    brain_path = get_brain_path(name)
    if brain_path.exists():
        shutil.rmtree(brain_path)

    # Remove from config
    unregister_brain(name)
    print(f"Brain '{name}' removed.")


def cmd_brain_update(name: str | None = None):
    """Re-download an installed brain to pick up the latest version.

    This is the runtime equivalent of `beam install <slug>` — the brain
    subsystem is fully offline, so the only network call is the download
    itself. After the new graph is on disk, all queries run locally.
    """
    from brain.paths import (
        get_active_brain_name,
        get_brain_info,
        get_brain_path,
    )
    from hermes_cli.install_cmd import _download_brain

    if not name:
        name = get_active_brain_name()

    info = get_brain_info(name)
    if not info:
        print(f"Brain '{name}' is not installed.", file=sys.stderr)
        sys.exit(1)

    source = info.get("source", "local")
    slug = info.get("slug")

    if source == "local":
        print(f"Brain '{name}' is a local brain (no marketplace slug).", file=sys.stderr)
        print(f"To update a local brain, edit {get_brain_path(name) / 'personality_graph.json'} directly.", file=sys.stderr)
        sys.exit(1)

    if source == "marketplace-official":
        # Official brains are stored under the bare slug (e.g. "creative-writer").
        display_slug = name
    elif source == "marketplace-community":
        if not slug:
            print(f"Brain '{name}' has no marketplace slug recorded. Reinstall with:", file=sys.stderr)
            print(f"  beam brain remove {name} && beam install @{slug or '<user/slug>'}", file=sys.stderr)
            sys.exit(1)
        display_slug = slug
    else:
        print(f"Brain '{name}' has unknown source '{source}'; cannot update.", file=sys.stderr)
        sys.exit(1)

    print(f"Updating brain '{name}' from marketplace ({display_slug})...")
    brain_path = get_brain_path(name)
    graph_data = _download_brain(display_slug, brain_path)

    # Re-write the graph on disk (the downloader already does this, but
    # keep this defensive in case _download_brain's behavior changes).
    graph_path = brain_path / "personality_graph.json"
    graph_path.write_text(json.dumps(graph_data, indent=2), encoding="utf-8")

    nodes = 0
    if isinstance(graph_data, dict):
        for key in ["traits", "beliefs", "values", "memories", "patterns",
                    "procedural_patterns", "work_loops", "expertise"]:
            nodes += len(graph_data.get(key, []))
        kg = graph_data.get("knowledge_graph", {})
        nodes += len(kg.get("nodes", []))
    print(f"  Type: Local (downloaded, works offline)")
    print(f"  Nodes: {nodes}")

    # Refresh SOUL.md so it reflects the updated graph.
    try:
        _regenerate_soul(name)
        print(f"  SOUL.md regenerated for '{name}'.")
    except Exception as e:
        print(f"  Warning: Could not regenerate SOUL.md: {e}")


def _regenerate_soul(brain_name: str):
    """Regenerate SOUL.md for a brain (local or proxy)."""
    from brain.brain_resolver import resolve_brain
    from brain.paths import get_brain_soul_path
    from hermes_constants import get_hermes_home

    brain = resolve_brain(brain_name)
    soul_md = brain.get_soul()
    if not soul_md:
        return

    # Write to brain-specific path
    soul_path = get_brain_soul_path(brain_name)
    try:
        soul_path.write_text(soul_md, encoding="utf-8")
    except Exception:
        pass

    # Also update the active SOUL.md in Hermes home
    hermes_home = get_hermes_home()
    hermes_home.mkdir(parents=True, exist_ok=True)
    try:
        (hermes_home / "SOUL.md").write_text(soul_md, encoding="utf-8")
    except Exception:
        pass


def register_brain_subcommands(existing_subcommands: list[str] | None = None):
    """Return additional brain subcommand names for the command registry."""
    return ["list", "switch", "info", "remove", "update", "install"]
