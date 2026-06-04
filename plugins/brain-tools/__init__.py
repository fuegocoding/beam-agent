"""brain-tools plugin — digital brain integration for beam-agent.

Registers brain_search, brain_export, and brain_status tools
that bridge to the Rust brain-runtime binary.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BEAM_HOME = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam"))


def _load_graph(user_id: str = "default") -> dict:
    graph_path = BEAM_HOME / "brain" / user_id / "personality_graph.json"
    if graph_path.exists():
        return json.loads(graph_path.read_text(encoding="utf-8"))
    return {}


def _brain_search_handler(args: dict, **kw: Any) -> str:
    query = args.get("query", "")
    trust_level = args.get("trust_level", "owner")
    brain_power = args.get("brain_power", "standard")

    graph = _load_graph()
    if not graph:
        return json.dumps({"error": "No brain data found. Run the interview first."})

    from brain.brain_retriever import BrainRetriever
    retriever = BrainRetriever()
    result = retriever.search(query, graph, trust_level, brain_power)
    return json.dumps(result, indent=2)


def _brain_export_handler(args: dict, **kw: Any) -> str:
    from brain.brain_retriever import BrainRetriever
    from brain.md_memory import MDMemory
    from brain.soul_generator import generate_soul_md
    from hermes_constants import get_hermes_home

    graph = _load_graph()
    if not graph:
        return json.dumps({"error": "No brain data found. Run the interview first."})

    memory = MDMemory()
    retriever = BrainRetriever()
    retriever.export_soul(graph)
    export_path = memory.write_brain_export(graph)

    if graph.get("voice_dna") or graph.get("work_dna"):
        memory.write_style(
            graph.get("voice_dna", {}),
            graph.get("work_dna", {}),
        )

    # Also update SOUL.md in Hermes home
    hermes_home = get_hermes_home()
    hermes_home.mkdir(parents=True, exist_ok=True)
    try:
        generate_soul_md(graph, hermes_home / "SOUL.md")
    except Exception as exc:
        logger.warning("Failed to generate SOUL.md: %s", exc)

    return json.dumps({
        "status": "success",
        "brain_export": export_path,
        "soul_md": str(hermes_home / "SOUL.md"),
        "style_md": str(memory.style_file),
    }, indent=2)


def _brain_status_handler(args: dict, **kw: Any) -> str:
    from brain.brain_retriever import BrainRetriever

    graph = _load_graph()
    if not graph:
        return json.dumps({"status": "empty", "message": "No brain data found. Run the interview first."})

    retriever = BrainRetriever()
    stats = retriever.get_stats(graph)
    return json.dumps(stats, indent=2)


def _check_brain_available() -> bool:
    return (BEAM_HOME / "brain").exists()


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
            import json
            data = json.loads(result) if isinstance(result, str) else result
            if not data.get("brain_built"):
                return
        except Exception:
            return

    try:
        from brain.soul_generator import generate_soul_md
        from hermes_constants import get_hermes_home

        graph = _load_graph()
        if graph:
            hermes_home = get_hermes_home()
            hermes_home.mkdir(parents=True, exist_ok=True)
            generate_soul_md(graph, hermes_home / "SOUL.md")
            logger.info("Auto-updated SOUL.md after %s", tool_name)
    except Exception as exc:
        logger.debug("SOUL.md auto-update failed: %s", exc)
