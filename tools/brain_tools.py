"""Brain tools for beam-agent.

Provides brain_search, brain_export, and brain_status tools
that bridge to the Rust brain-runtime binary.
"""

import json
import os
from pathlib import Path

BEAM_HOME = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam"))


def _load_graph(user_id: str = "default") -> dict:
    """Load the personality graph from disk."""
    graph_path = BEAM_HOME / "brain" / user_id / "personality_graph.json"
    if graph_path.exists():
        return json.loads(graph_path.read_text(encoding="utf-8"))
    return {}


def brain_search(query: str, trust_level: str = "owner", brain_power: str = "standard") -> str:
    """Search your digital brain for relevant personality traits, beliefs, memories, and patterns.

    Args:
        query: What to search for (e.g., "beliefs about AI", "work style", "key relationships")
        trust_level: Privacy level - "visitor" (public only), "known" (public+personal), "owner" (all)
        brain_power: How much context to return - "light" (top 3), "standard" (top 10), "full" (all)

    Returns:
        JSON string with matching nodes and edges from your personality graph.
    """
    from brain.brain_retriever import BrainRetriever

    graph = _load_graph()
    if not graph:
        return json.dumps({"error": "No brain data found. Run the interview first with 'build-my-brain' skill."})

    retriever = BrainRetriever()
    result = retriever.search(query, graph, trust_level, brain_power)
    return json.dumps(result, indent=2)


def brain_export() -> str:
    """Export your digital brain as human-readable Markdown files.

    Returns:
        JSON string with the path to exported files.
    """
    from brain.brain_retriever import BrainRetriever
    from brain.md_memory import MDMemory

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

    return json.dumps({
        "status": "success",
        "brain_export": export_path,
        "soul_md": str(BEAM_HOME / "SOUL.md"),
        "style_md": str(memory.style_file),
    }, indent=2)


def brain_status() -> str:
    """Get the current status of your digital brain.

    Returns:
        JSON string with brain statistics and coverage.
    """
    from brain.brain_retriever import BrainRetriever

    graph = _load_graph()
    if not graph:
        return json.dumps({"status": "empty", "message": "No brain data found. Run the interview first."})

    retriever = BrainRetriever()
    stats = retriever.get_stats(graph)
    return json.dumps(stats, indent=2)
