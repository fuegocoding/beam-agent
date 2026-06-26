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
    """Search all nodes in the graph for query matches."""
    results = []
    query_lower = query.lower()

    # Search traits
    for item in graph.get("traits", []):
        text = f"{item.get('name', '')} {item.get('summary', '')}"
        score = _score_match(query_lower, text)
        if score > 0 or query_lower in text.lower():
            results.append({
                "type": "trait",
                "name": item.get("name", ""),
                "summary": item.get("summary", ""),
                "strength": item.get("strength", 0.5),
                "relevance": max(score, 0.3 if query_lower in text.lower() else 0),
            })

    # Search beliefs
    for item in graph.get("beliefs", []):
        text = f"{item.get('name', '')} {item.get('summary', '')}"
        score = _score_match(query_lower, text)
        if score > 0 or query_lower in text.lower():
            results.append({
                "type": "belief",
                "name": item.get("name", ""),
                "summary": item.get("summary", ""),
                "confidence": item.get("confidence", 0.5),
                "relevance": max(score, 0.3 if query_lower in text.lower() else 0),
            })

    # Search values
    for item in graph.get("values", []):
        text = f"{item.get('name', '')} {item.get('summary', '')}"
        score = _score_match(query_lower, text)
        if score > 0 or query_lower in text.lower():
            results.append({
                "type": "value",
                "name": item.get("name", ""),
                "summary": item.get("summary", ""),
                "importance": item.get("importance", 0.5),
                "relevance": max(score, 0.3 if query_lower in text.lower() else 0),
            })

    # Search boundaries (owner only)
    if trust_level == "owner":
        for item in graph.get("boundaries", []):
            text = f"{item.get('topic', '')} {item.get('summary', '')}"
            score = _score_match(query_lower, text)
            if score > 0 or query_lower in text.lower():
                results.append({
                    "type": "boundary",
                    "topic": item.get("topic", ""),
                    "summary": item.get("summary", ""),
                    "comfort_level": item.get("comfort_level", 0.5),
                    "relevance": max(score, 0.3 if query_lower in text.lower() else 0),
                })

    # Search life events
    for item in graph.get("life_events", []):
        text = f"{item.get('event', '')} {item.get('impact', '')} {item.get('summary', '')}"
        score = _score_match(query_lower, text)
        if score > 0 or query_lower in text.lower():
            results.append({
                "type": "life_event",
                "event": item.get("event", ""),
                "year": item.get("year", ""),
                "impact": item.get("impact", ""),
                "relevance": max(score, 0.3 if query_lower in text.lower() else 0),
            })

    # Search people
    for item in graph.get("people", []):
        text = f"{item.get('name', '')} {item.get('relationship', '')} {item.get('significance', '')}"
        score = _score_match(query_lower, text)
        if score > 0 or query_lower in text.lower():
            results.append({
                "type": "person",
                "name": item.get("name", ""),
                "relationship": item.get("relationship", ""),
                "significance": item.get("significance", ""),
                "relevance": max(score, 0.3 if query_lower in text.lower() else 0),
            })

    # Search memories (episodic / interview chunks)
    for item in graph.get("memories", []):
        text = f"{item.get('name', '')} {item.get('summary', '')}"
        score = _score_match(query_lower, text)
        if score > 0 or query_lower in text.lower():
            results.append({
                "type": "memory",
                "name": item.get("name", ""),
                "summary": item.get("summary", ""),
                "emotional_tone": item.get("emotional_tone", 0.0),
                "relevance": max(score, 0.3 if query_lower in text.lower() else 0),
            })

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
            parts.append(f"Boundary on '{node['topic']}' (comfort {node.get('comfort_level', 0.5):.0%}): {node.get('summary', '')}")
        elif ntype == "life_event":
            parts.append(f"Life event ({node.get('year', '?')}): {node['event']} — {node.get('impact', '')}")
        elif ntype == "person":
            parts.append(f"{node['relationship']}: {node['name']} — {node.get('significance', '')}")
        elif ntype == "memory":
            tone = node.get("emotional_tone", 0.0)
            tone_str = f" (tone {tone:+.2f})" if tone else ""
            parts.append(f"Memory '{node['name']}'{tone_str}: {node.get('summary', '')}")
        elif ntype == "transcript_excerpt":
            parts.append(f"Transcript excerpt: …{node.get('summary', '')}…")

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
        """Build structured context from the graph."""
        parts = []

        if graph.get("user_summary"):
            parts.append(f"Who they are: {graph['user_summary']}")

        if graph.get("traits"):
            top_traits = sorted(graph["traits"], key=lambda t: t.get("strength", 0), reverse=True)[:5]
            trait_str = ", ".join(f"{t['name']} ({t.get('strength', 0):.0%})" for t in top_traits)
            parts.append(f"Top traits: {trait_str}")

        if graph.get("values"):
            top_values = sorted(graph["values"], key=lambda v: v.get("importance", 0), reverse=True)[:3]
            val_str = ", ".join(f"{v['name']} ({v.get('importance', 0):.0%})" for v in top_values)
            parts.append(f"Core values: {val_str}")

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
        """Get graph statistics."""
        return {
            "user_summary": bool(graph.get("user_summary")),
            "total_nodes": (
                len(graph.get("traits", []))
                + len(graph.get("beliefs", []))
                + len(graph.get("values", []))
                + len(graph.get("boundaries", []))
                + len(graph.get("life_events", []))
                + len(graph.get("people", []))
                + len(graph.get("memories", []))
            ),
            "total_edges": 0,  # Edge counting not yet implemented
            "raw_transcript_chars": len(graph.get("raw_transcript", "")),
            "coverage": {
                "traits": len(graph.get("traits", [])),
                "beliefs": len(graph.get("beliefs", [])),
                "values": len(graph.get("values", [])),
                "boundaries": len(graph.get("boundaries", [])),
                "life_events": len(graph.get("life_events", [])),
                "people": len(graph.get("people", [])),
                "memories": len(graph.get("memories", [])),
                "raw_transcript": bool(graph.get("raw_transcript")),
                "voice_dna": bool(graph.get("voice_dna", {}).get("humor_style")),
                "work_dna": bool(graph.get("work_dna", {}).get("decomposition_style")),
                "emotional_profile": bool(graph.get("emotional_profile", {}).get("energy_sources")),
            },
        }
