"""Multi-brain management commands for the CLI.

Adds: /brain list, /brain switch, /brain info, /brain remove
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
        return

    print("Installed brains:\n")
    for b in brains:
        active_marker = "●" if b["active"] else "○"
        source_tag = f"[{b['source']}]"
        token_tag = " 🔒" if b["has_token"] else ""
        print(f"  {active_marker} {b['name']:<20} {source_tag:<25}{token_tag}")

    print(f"\nActive: {next((b['name'] for b in brains if b['active']), 'none')}")


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
    config_path = brain_path / "brain_config.json"

    if not graph_path.exists() and not config_path.exists():
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
    config_path = brain_path / "brain_config.json"
    soul_path = brain_path / "soul.md"

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if cfg.get("type") == "proxy":
            print(f"Type: API proxy (full graph stays server-side)")
            print(f"API slug: {cfg.get('slug')}")
    elif graph_path.exists():
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
            print(f"Type: Local (full brain)")
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
    return ["list", "switch", "info", "remove"]
