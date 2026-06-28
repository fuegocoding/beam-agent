"""brain-tools plugin — digital brain integration for beam-agent.

Registers brain_search, brain_export, and brain_status tools backed by
the local personality graph (no network). The previous proxy mode has
been removed — brains are downloaded once at install time and queried
locally via the resolver.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from brain.brain_resolver import get_active_brain_interface

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
    """Check if any brain is configured locally."""
    from brain.paths import get_active_brain_name, get_brain_path
    brain_path = get_brain_path(get_active_brain_name())
    if (brain_path / "personality_graph.json").exists():
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

    # Register session-start hook: materialize ~/.hermes/SOUL.md from
    # the active brain the first time a session begins, and invalidate
    # the agent's cached system prompt so the regenerated identity is
    # picked up on the very next turn. Without this, a marketplace
    # brain installed in a previous session leaves SOUL.md stale
    # (or empty) and the agent has no personality until the user
    # explicitly runs `brain_export` or restarts.
    ctx.register_hook("on_session_start", _on_session_start)


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


# ---------------------------------------------------------------------------
# Session-start hook: materialize SOUL.md from the active brain
# ---------------------------------------------------------------------------

# Cached signal that the very first session-start of this process
# already materialized SOUL.md. Without this, every gateway turn
# would re-run the export (the gateway spins up a fresh AIAgent per
# message, so on_session_start fires for every message). The export
# is cheap but the noise in gateway.log isn't.
_session_start_seen: set[str] = set()


def _on_session_start(
    session_id: str = "",
    model: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    """Materialize ~/.hermes/SOUL.md from the active brain on first turn.

    Two triggers to handle:

    1. **First-ever session for this active brain** — SOUL.md doesn't
       exist or is the previous brain's. Regenerate from the active
       brain's graph.
    2. **Marketplace-schema brain installed** — the previous
       legacy-only soul_generator would have produced a 582-byte
       empty SOUL.md. Detect this via
       :func:`brain.schema_adapter.has_marketplace_payload` and
       rewrite using the schema-aware retriever.

    Gated by ``_session_start_seen`` so we don't re-export on every
    gateway turn (each gateway turn creates a new AIAgent, which
    re-fires on_session_start).
    """
    from brain.paths import (
        get_active_brain_graph_path,
        get_active_brain_name,
    )
    from brain.schema_adapter import has_marketplace_payload
    from hermes_constants import get_hermes_home

    cache_key = f"{platform}:{get_active_brain_name()}:{session_id}"
    if cache_key in _session_start_seen:
        return

    graph_path = get_active_brain_graph_path()
    if not graph_path.exists():
        # No brain installed — nothing to do. The agent will fall back
        # to DEFAULT_SOUL_MD via the prompt builder.
        _session_start_seen.add(cache_key)
        return

    # Decide whether the existing SOUL.md needs regenerating.
    hermes_home = get_hermes_home()
    hermes_home.mkdir(parents=True, exist_ok=True)
    soul_path = hermes_home / "SOUL.md"

    needs_regen = False
    if not soul_path.exists():
        needs_regen = True
    elif soul_path.stat().st_size < 800:
        # Legacy template produces ~582 bytes for marketplace brains
        # (only voice_dna populated). Anything shorter than the
        # rough minimum for a populated brain is treated as a stub.
        try:
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            if has_marketplace_payload(graph):
                needs_regen = True
        except Exception:
            pass

    if not needs_regen:
        _session_start_seen.add(cache_key)
        return

    try:
        brain = get_active_brain_interface()
        soul_result = brain.export_soul()
        soul_md = soul_result.get("soul_md", "")
        if soul_md:
            soul_path.write_text(soul_md, encoding="utf-8")
            logger.info(
                "Materialized SOUL.md from active brain '%s' (%d chars)",
                get_active_brain_name(),
                len(soul_md),
            )
    except Exception as exc:
        logger.debug("SOUL.md materialization failed: %s", exc)
    finally:
        _session_start_seen.add(cache_key)
