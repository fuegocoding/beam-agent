"""Brain tools for beam-agent.

Provides brain_search, brain_export, and brain_status tools
that work with both local and remote (proxy) brains via the
brain resolver abstraction.
"""

import json
import os
from pathlib import Path

from brain.brain_resolver import get_active_brain_interface

BEAM_HOME = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam"))


def brain_search(query: str, trust_level: str = "owner", brain_power: str = "standard") -> str:
    """Search your digital brain for relevant personality traits, beliefs, memories, and patterns.

    Args:
        query: What to search for (e.g., "beliefs about AI", "work style", "key relationships")
        trust_level: Privacy level - "visitor" (public only), "known" (public+personal), "owner" (all)
        brain_power: How much context to return - "light" (top 3), "standard" (top 10), "full" (all)

    Returns:
        JSON string with matching nodes and edges from your personality graph.
    """
    brain = get_active_brain_interface()
    result = brain.search(query, trust_level, brain_power)
    return json.dumps(result, indent=2)


def brain_export() -> str:
    """Export your digital brain as human-readable Markdown files.

    Returns:
        JSON string with the path to exported files.
    """
    from brain.md_memory import MDMemory

    brain = get_active_brain_interface()
    memory = MDMemory()

    soul_result = brain.export_soul()
    soul_md = soul_result.get("soul_md", "")

    # Write SOUL.md to Hermes home
    from hermes_constants import get_hermes_home
    hermes_home = get_hermes_home()
    hermes_home.mkdir(parents=True, exist_ok=True)
    soul_path = hermes_home / "SOUL.md"
    soul_path.write_text(soul_md, encoding="utf-8")

    # For local brains, also export the full graph breakdown
    # For proxy brains, we only have the SOUL.md
    export_path = memory.brain_export_dir / "brain-summary.md"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(soul_md, encoding="utf-8")

    return json.dumps({
        "status": "success",
        "brain_export": str(export_path),
        "soul_md": str(soul_path),
    }, indent=2)


def brain_status() -> str:
    """Get the current status of your digital brain.

    Returns:
        JSON string with brain statistics and coverage.
    """
    brain = get_active_brain_interface()
    stats = brain.get_stats()
    return json.dumps(stats, indent=2)
