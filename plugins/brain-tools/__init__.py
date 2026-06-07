"""brain-tools plugin — digital brain integration for beam-agent.

Registers brain_search, brain_export, and brain_status tools
that work with both local and remote (proxy) brains via the
brain resolver abstraction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from brain.brain_resolver import get_active_brain_interface, is_proxy_brain

logger = logging.getLogger(__name__)

BEAM_HOME = Path.home() / ".beam"


def _brain_search_handler(args: dict, **kw: Any) -> str:
    query = args.get("query", "")
    trust_level = args.get("trust_level", "owner")
    brain_power = args.get("brain_power", "standard")

    brain = get_active_brain_interface()
    result = brain.search(query, trust_level, brain_power)
    return json.dumps(result, indent=2)


def _brain_export_handler(args: dict, **kw: Any) -> str:
    from brain.md_memory import MDMemory
    from hermes_constants import get_hermes_home

    brain = get_active_brain_interface()
    memory = MDMemory()

    soul_result = brain.export_soul()
    soul_md = soul_result.get("soul_md", "")

    # Write SOUL.md in Hermes home
    hermes_home = get_hermes_home()
    hermes_home.mkdir(parents=True, exist_ok=True)
    soul_path = hermes_home / "SOUL.md"
    try:
        soul_path.write_text(soul_md, encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write SOUL.md: %s", exc)

    # Export summary to brain-export dir
    export_path = memory.brain_export_dir / "brain-summary.md"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_path.write_text(soul_md, encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write brain export: %s", exc)

    return json.dumps({
        "status": "success",
        "brain_export": str(export_path),
        "soul_md": str(soul_path),
    }, indent=2)


def _brain_status_handler(args: dict, **kw: Any) -> str:
    brain = get_active_brain_interface()
    stats = brain.get_stats()
    return json.dumps(stats, indent=2)


def _check_brain_available() -> bool:
    """Check if any brain (local or proxy) is configured."""
    from brain.paths import get_active_brain_name, get_brain_path
    brain_path = get_brain_path(get_active_brain_name())
    # Local brain graph or proxy config
    if (brain_path / "personality_graph.json").exists() or (brain_path / "brain_config.json").exists():
        return True
    # Fallback: old path (pre-migration)
    old_path = Path.home() / ".beam" / "brain" / get_active_brain_name() / "personality_graph.json"
    return old_path.exists()


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

BRAIN_SEARCH_SCHEMA = {
    "name": "brain_search",
    "description": "Search your digital brain for personality traits, beliefs, memories, and patterns.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for (e.g., 'beliefs about AI', 'work style', 'key relationships')",
            },
            "trust_level": {
                "type": "string",
                "enum": ["visitor", "known", "owner"],
                "description": "Privacy level — visitor (public only), known (public+personal), owner (all). Default: owner",
            },
            "brain_power": {
                "type": "string",
                "enum": ["light", "standard", "full"],
                "description": "How much context — light (top 3), standard (top 10), full (all). Default: standard",
            },
        },
        "required": ["query"],
    },
}

BRAIN_EXPORT_SCHEMA = {
    "name": "brain_export",
    "description": "Export your digital brain as human-readable Markdown files and SOUL.md.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

BRAIN_STATUS_SCHEMA = {
    "name": "brain_status",
    "description": "Get the current status and coverage statistics of your digital brain.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_tool(
        name="brain_search",
        toolset="brain",
        schema=BRAIN_SEARCH_SCHEMA,
        handler=_brain_search_handler,
        check_fn=_check_brain_available,
        description="Search your digital brain for personality traits, beliefs, memories, and patterns.",
        emoji="🧠",
    )
    ctx.register_tool(
        name="brain_export",
        toolset="brain",
        schema=BRAIN_EXPORT_SCHEMA,
        handler=_brain_export_handler,
        check_fn=_check_brain_available,
        description="Export your digital brain as human-readable Markdown files.",
        emoji="📄",
    )
    ctx.register_tool(
        name="brain_status",
        toolset="brain",
        schema=BRAIN_STATUS_SCHEMA,
        handler=_brain_status_handler,
        check_fn=_check_brain_available,
        description="Get the current status of your digital brain.",
        emoji="📊",
    )

    # Register gateway hook: auto-update SOUL.md when brain tools are used
    ctx.register_hook("post_tool_call", _on_post_tool_call)


# ---------------------------------------------------------------------------
# Gateway hook: auto-update SOUL.md on brain change
# ---------------------------------------------------------------------------

def _on_post_tool_call(
    tool_name: str = "",
    args: dict = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    **_: Any,
) -> None:
    """Regenerate SOUL.md after brain-building or export operations."""
    if tool_name not in ("brain_export", "continue_interview"):
        return

    # For continue_interview, only regenerate if brain was just built
    if tool_name == "continue_interview":
        try:
            data = json.loads(result) if isinstance(result, str) else result
            if not data.get("brain_built"):
                return
        except Exception:
            return

    try:
        brain = get_active_brain_interface()
        soul_result = brain.export_soul()
        soul_md = soul_result.get("soul_md", "")
        if soul_md:
            from hermes_constants import get_hermes_home
            hermes_home = get_hermes_home()
            hermes_home.mkdir(parents=True, exist_ok=True)
            (hermes_home / "SOUL.md").write_text(soul_md, encoding="utf-8")
            logger.info("Auto-updated SOUL.md after %s", tool_name)
    except Exception as exc:
        logger.debug("SOUL.md auto-update failed: %s", exc)
