"""Brain retriever — Python-native graph search.

Searches the personality graph locally without Rust binary dependency.
Uses keyword matching and relevance scoring.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BEAM_HOME = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam"))


def _score_match(query: str, text: str) -> float:
    """Score how relevant a text is to a query (0-1)."""
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    if not query_words:
        return 0.0
    overlap = query_words & text_words
    return len(overlap) / len(query_words)


def _search_nodes(graph: dict, query: str, trust_level: str = "owner") -> list:
    """Search all nodes in the graph for query matches.

    Works against any schema the adapter supports: the legacy
    flat schema (``traits`` / ``beliefs`` / ``values`` / …) AND the
    marketplace schema (``personality_profile`` /
    ``knowledge_graph`` / ``knowledge_domains`` /
    ``episodic_memories``). Both are flattened to a uniform node
    shape by :mod:`brain.schema_adapter` before scoring.
    """
    # Lazy import keeps the brain module importable even if the
    # adapter ever needs to be torn out (e.g. for hot-reload in dev).
    from brain.schema_adapter import iter_nodes

    results: list = []
    query_lower = query.lower()

    def _score_node(node: dict) -> None:
        name = node.get("name", "")
        summary = node.get("summary", "")
        text = f"{name} {summary}"
        score = _score_match(query_lower, text)
        if score > 0 or query_lower in text.lower():
            node_relevance = max(score, 0.3 if query_lower in text.lower() else 0)
            results.append({**node, "relevance": node_relevance})

    # Owner-only node types (e.g. boundaries) are filtered per
    # trust_level, mirroring the legacy gating.
    owner_only_types = {"boundary"}

    for node in iter_nodes(graph):
        if trust_level != "owner" and node.get("type") in owner_only_types:
            continue
        _score_node(node)

    # Search raw_transcript (top-level field set by the offline builder)
    raw_transcript = graph.get("raw_transcript", "")
    if raw_transcript and query_lower:
        # Cheap match: just check if any query word is in the transcript.
        # We surface a single chunk around the best hit instead of dumping
        # the whole transcript into one result.
        transcript_lower = raw_transcript.lower()
        first_hit = -1
        for word in query_lower.split():
            idx = transcript_lower.find(word)
            if idx != -1 and (first_hit == -1 or idx < first_hit):
                first_hit = idx
        if first_hit != -1:
            window = 240
            start = max(0, first_hit - window // 2)
            end = min(len(raw_transcript), first_hit + window // 2)
            chunk = raw_transcript[start:end].strip()
            results.append({
                "type": "transcript_excerpt",
                "name": "raw_transcript",
                "summary": chunk,
                "relevance": 0.4,
            })

    # Sort by relevance
    results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return results


def _format_context(nodes: list, graph: dict, brain_power: str = "standard") -> str:
    """Format search results into a context string for the agent."""
    if not nodes:
        return "No relevant brain data found for this query."

    limits = {"light": 3, "standard": 10, "full": 100}
    limit = limits.get(brain_power, 10)
    nodes = nodes[:limit]

    parts = []
    for node in nodes:
        ntype = node.get("type", "unknown")
        if ntype == "trait":
            parts.append(f"Trait '{node['name']}' (strength {node.get('strength', 0.5):.0%}): {node.get('summary', '')}")
        elif ntype == "belief":
            parts.append(f"Belief '{node['name']}' (confidence {node.get('confidence', 0.5):.0%}): {node.get('summary', '')}")
        elif ntype == "value":
            parts.append(f"Value '{node['name']}' (importance {node.get('importance', 0.5):.0%}): {node.get('summary', '')}")
        elif ntype == "boundary":
            parts.append(f"Boundary on '{node.get('topic') or node.get('name', '?')}' (comfort {node.get('comfort_level', 0.5):.0%}): {node.get('summary', '')}")
        elif ntype == "life_event":
            parts.append(f"Life event ({node.get('year', '?')}): {node.get('event') or node.get('name', '?')} — {node.get('impact', '')}")
        elif ntype == "person":
            parts.append(f"{node.get('relationship', 'Person')}: {node['name']} — {node.get('significance', node.get('summary', ''))}")
        elif ntype == "memory":
            tone = node.get("emotional_tone", 0.0)
            tone_str = f" (tone {tone:+.2f})" if tone else ""
            parts.append(f"Memory '{node['name']}'{tone_str}: {node.get('summary', '')}")
        elif ntype == "expertise":
            depth = node.get("depth", node.get("confidence", 0.5))
            sources = node.get("source_count")
            suffix = f" ({sources} sources)" if isinstance(sources, int) and sources else ""
            parts.append(f"Expertise '{node['name']}' (depth {depth:.0%}){suffix}: {node.get('summary', '')}")
        elif ntype == "pattern":
            parts.append(f"Pattern '{node['name']}': {node.get('summary', '')}")
        elif ntype == "procedural_pattern":
            parts.append(f"Procedural pattern '{node['name']}': {node.get('summary', '')}")
        elif ntype == "work_loop":
            parts.append(f"Work loop '{node['name']}': {node.get('summary', '')}")
        elif ntype == "transcript_excerpt":
            parts.append(f"Transcript excerpt: …{node.get('summary', '')}…")
        elif ntype in ("knowledge_node", "concept", "place"):
            # Generic catch-all for marketplace knowledge_graph entries
            # that don't map to a more specific node type.
            parts.append(f"{ntype.replace('_', ' ').title()} '{node['name']}': {node.get('summary', '')}")
        else:
            parts.append(f"{ntype} '{node.get('name', '?')}': {node.get('summary', '')}")

    return "\n".join(parts)


class BrainRetriever:
    """Queries the personality graph at runtime."""

    def search(self, query: str, graph: dict, trust_level: str = "owner", brain_power: str = "standard") -> dict:
        """Search the brain for relevant nodes/edges."""
        nodes = _search_nodes(graph, query, trust_level)
        context = _format_context(nodes, graph, brain_power)

        return {
            "nodes": nodes,
            "edges": [],  # Edge search not yet implemented
            "context": context,
            "total_matches": len(nodes),
        }

    def build_context(self, graph: dict, trust_level: str = "owner", brain_power: str = "standard") -> dict:
        """Build structured context from the graph.

        Walks the schema-agnostic node list (legacy flat + marketplace
        ``personality_profile`` / ``knowledge_graph`` /
        ``knowledge_domains``) so marketplace brains produce a
        populated context instead of an empty string.
        """
        from brain.schema_adapter import iter_nodes

        parts = []

        if graph.get("user_summary"):
            parts.append(f"Who they are: {graph['user_summary']}")

        nodes = list(iter_nodes(graph))

        # Top 5 traits by strength
        traits = [n for n in nodes if n.get("type") == "trait" and n.get("summary")]
        if traits:
            def _score(t: dict) -> float:
                return t.get("strength", t.get("confidence", 0.5))
            top_traits = sorted(traits, key=_score, reverse=True)[:5]
            trait_str = ", ".join(
                f"{t['name']} ({_score(t):.0%})" for t in top_traits
            )
            parts.append(f"Top traits: {trait_str}")

        # Top 3 values by importance
        values = [n for n in nodes if n.get("type") == "value" and n.get("summary")]
        if values:
            def _score_v(v: dict) -> float:
                return v.get("importance", v.get("confidence", 0.5))
            top_values = sorted(values, key=_score_v, reverse=True)[:3]
            val_str = ", ".join(
                f"{v['name']} ({_score_v(v):.0%})" for v in top_values
            )
            parts.append(f"Core values: {val_str}")

        # Top 3 core beliefs by confidence (marketplace schema uses
        # core_beliefs inside personality_profile; the adapter maps
        # those to type="belief" so this just works).
        beliefs = [n for n in nodes if n.get("type") == "belief" and n.get("summary")]
        if beliefs:
            def _score_b(b: dict) -> float:
                return b.get("confidence", 0.5)
            top_beliefs = sorted(beliefs, key=_score_b, reverse=True)[:3]
            bel_str = ", ".join(
                f"{b['name']} ({_score_b(b):.0%})" for b in top_beliefs
            )
            parts.append(f"Core beliefs: {bel_str}")

        voice = graph.get("voice_dna", {})
        if voice.get("humor_style"):
            parts.append(f"Humor: {voice['humor_style']}")
        if voice.get("response_length_pattern"):
            parts.append(f"Response style: {voice['response_length_pattern']}")

        return {"context": "\n".join(parts)}

    def export_soul(self, graph: dict) -> dict:
        """Generate SOUL.md content from the graph."""
        from brain.soul_generator import _template_soul
        content = _template_soul(graph)
        return {"soul_md": content}

    def get_stats(self, graph: dict) -> dict:
        """Get graph statistics.

        Counts are derived via :mod:`brain.schema_adapter` so they
        include marketplace-schema nodes (``personality_profile.*``,
        ``knowledge_graph.nodes``, ``knowledge_domains``,
        ``episodic_memories``) in addition to the legacy flat keys.
        The ``coverage`` dict preserves the original legacy keys for
        backwards-compat (and adds ``knowledge_graph_nodes`` /
        ``knowledge_domains`` / ``episodic_memories`` so callers can
        tell which schema the graph actually used).
        """
        from brain.schema_adapter import coverage, iter_edges, iter_nodes

        node_counts: dict = coverage(graph)
        nodes = list(iter_nodes(graph))
        edges = list(iter_edges(graph))

        # ``knowledge_graph_nodes`` counts every node sourced from
        # the marketplace ``knowledge_graph.nodes`` array regardless
        # of what canonical type the adapter mapped it to. The
        # ``kg_type_map`` mirrors the adapter's own mapping so the
        # number reported here matches what the marketplace actually
        # published, not just the generic ``knowledge_node`` bucket.
        kg = graph.get("knowledge_graph") if isinstance(graph, dict) else None
        kg_node_count = len(kg.get("nodes", [])) if isinstance(kg, dict) else 0

        return {
            "user_summary": bool(graph.get("user_summary")),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "raw_transcript_chars": len(graph.get("raw_transcript", "")),
            "coverage": {
                # Legacy keys now derive from node_counts so marketplace-
                # schema brains (knowledge_graph.nodes, personality_profile,
                # episodic_memories) report real counts instead of 0.
                "traits": node_counts.get("trait", 0),
                "beliefs": node_counts.get("belief", 0),
                "values": node_counts.get("value", 0),
                "boundaries": node_counts.get("boundary", 0),
                "life_events": node_counts.get("life_event", 0),
                "people": node_counts.get("person", 0),
                "memories": node_counts.get("memory", 0),
                "raw_transcript": bool(graph.get("raw_transcript")),
                "voice_dna": bool(graph.get("voice_dna", {}).get("humor_style")),
                "work_dna": bool(graph.get("work_dna", {}).get("decomposition_style")),
                "emotional_profile": bool(graph.get("emotional_profile", {}).get("energy_sources")),
                # New keys — make the marketplace schema visible in
                # stats output and in `beam brain info` summaries.
                "knowledge_graph_nodes": kg_node_count,
                "knowledge_domains": node_counts.get("expertise", 0),
                "episodic_memories": node_counts.get("memory", 0),
            },
        }
