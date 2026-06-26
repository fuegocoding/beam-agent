"""Brain builder — offline personality graph construction.

Builds a PersonalityGraph from an interview transcript *without* calling
an LLM. The brain subsystem is fully offline once the data is on disk;
the heavy lifting of turning free-form answers into a structured graph
is now expected to happen either:

  1. At publish time (in beam_mind, where authors can use any model they
     want to build a graph from their own interview), or
  2. Manually by editing personality_graph.json.

What this module does offline:
  - Captures the interview transcript verbatim as memory chunks so the
    brain is still queryable (BrainRetriever indexes memories).
  - Fills in a short user_summary from the first ~320 chars of the
    transcript.
  - Merges new data into an existing graph using a deterministic,
    no-LLM dedupe pass.
"""

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _normalize_graph(graph: dict) -> dict:
    """Normalize graph output to the expected shape.

    Kept here as a safety net for graphs produced by older versions or
    external tooling that emit slightly different field names. Pure data
    manipulation, no model calls.
    """

    if not graph.get("user_summary"):
        for path in [
            ("core_identity",),
            ("core_identity", "description"),
            ("identity", "core_description"),
            ("identity", "self_description"),
            ("identity", "description"),
            ("summary",),
        ]:
            val = graph
            for key in path:
                val = val.get(key, {}) if isinstance(val, dict) else {}
            if isinstance(val, str) and val:
                graph["user_summary"] = val
                break

    if not graph.get("traits"):
        traits = []
        for source_key in ["identity", "core_identity"]:
            source = graph.get(source_key, {})
            if isinstance(source, dict):
                for list_key in ["core_values", "values", "traits"]:
                    items = source.get(list_key, [])
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, str):
                                traits.append({"name": item, "strength": 0.7, "summary": f"Core: {item}"})
        if traits:
            graph["traits"] = traits

    if not graph.get("values"):
        values = []
        for source_key in ["identity", "core_identity"]:
            source = graph.get(source_key, {})
            if isinstance(source, dict):
                for list_key in ["core_values", "values"]:
                    items = source.get(list_key, [])
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, str):
                                values.append({"name": item, "importance": 0.8, "summary": f"Value: {item}"})
        if values:
            graph["values"] = values

    beliefs_raw = graph.get("beliefs", [])
    if isinstance(beliefs_raw, str):
        graph["beliefs"] = [{"name": beliefs_raw, "confidence": 0.8, "summary": beliefs_raw}]
    elif isinstance(beliefs_raw, list):
        normalized = []
        for b in beliefs_raw:
            if isinstance(b, str):
                normalized.append({"name": b, "confidence": 0.8, "summary": b})
            elif isinstance(b, dict):
                normalized.append(b)
        graph["beliefs"] = normalized

    if not graph.get("emotional_profile") or not graph.get("emotional_profile", {}).get("energy_sources"):
        result = graph.setdefault("emotional_profile", {})
        ep = graph.get("emotional_triggers", {})
        if isinstance(ep, dict):
            drains = ep.get("drains", ep.get("energy_drains", []))
            if isinstance(drains, str):
                drains = [drains]
            if drains and not result.get("energy_drains"):
                result["energy_drains"] = drains
            recharges = ep.get("recharges", ep.get("energy_sources", []))
            if isinstance(recharges, str):
                recharges = [recharges]
            if recharges and not result.get("energy_sources"):
                result["energy_sources"] = recharges
        ip = graph.get("interaction_patterns", {})
        if isinstance(ip, dict) and not result.get("recovery_pattern"):
            result["recovery_pattern"] = str(ip)
        pi = graph.get("personality_indicators", {})
        if isinstance(pi, dict) and not result.get("triggers"):
            result["triggers"] = [{"stimulus": k, "reaction": v, "intensity": 0.5} for k, v in pi.items() if isinstance(v, str)]

    if not graph.get("work_dna") or not graph.get("work_dna", {}).get("decomposition_style"):
        for ws_key in ["work_style", "work_approach"]:
            ws = graph.get(ws_key, {})
            if ws:
                wd = graph.setdefault("work_dna", {})
                if ws.get("approach") and not wd.get("decomposition_style"):
                    wd["decomposition_style"] = ws["approach"]
                if ws.get("general") and not wd.get("decomposition_style"):
                    wd["decomposition_style"] = ws["general"]
                if ws.get("delegation") and not wd.get("delegation_style"):
                    wd["delegation_style"] = ws["delegation"]
                if ws.get("delegation_framework") and not wd.get("delegation_style"):
                    fw = ws["delegation_framework"]
                    if isinstance(fw, dict):
                        wd["delegation_style"] = f"Knowledge: {fw.get('knowledge_problems', '')}. Volume: {fw.get('volume_problems', '')}"
                if ws.get("validation") and not wd.get("debugging_approach"):
                    wd["debugging_approach"] = ws["validation"]
                if ws.get("bug_fixing") and not wd.get("debugging_approach"):
                    wd["debugging_approach"] = ws["bug_fixing"]

    if not graph.get("people"):
        people = []
        rels = graph.get("relationships", {})
        if isinstance(rels, dict):
            for person in rels.get("significant_people", []):
                if isinstance(person, dict):
                    name = person.get("role", person.get("name", "Unknown"))
                    significance = person.get("impact", person.get("description", person.get("lesson", "")))
                    people.append({"name": name, "relationship": name, "significance": significance})
            for key, val in rels.items():
                if key not in ("significant_people", "key_influences", "trust_approach", "relationship_learning") and isinstance(val, str):
                    people.append({"name": key, "relationship": key, "significance": val})
        if people:
            graph["people"] = people

    return graph


def _validate_graph(graph: dict) -> dict:
    """Ensure graph has required fields with defaults."""
    graph = _normalize_graph(graph)

    defaults = {
        "user_summary": "",
        "traits": [],
        "beliefs": [],
        "values": [],
        "boundaries": [],
        "life_events": [],
        "memories": [],
        "people": [],
        "voice_dna": {
            "characteristic_phrases": [],
            "phrases_to_avoid": [],
            "humor_style": "",
            "response_length_pattern": "",
            "formality_range": "",
            "storytelling_style": "",
        },
        "work_dna": {
            "decomposition_style": "",
            "debugging_approach": "",
            "risk_posture": "",
            "delegation_style": "",
            "documentation_habit": "",
        },
        "emotional_profile": {
            "triggers": [],
            "energy_sources": [],
            "energy_drains": [],
            "reaction_speed": "",
            "recovery_pattern": "",
        },
    }
    for key, default in defaults.items():
        if key not in graph:
            graph[key] = default
        elif isinstance(default, dict) and isinstance(graph[key], dict):
            for subkey, subdefault in default.items():
                if subkey not in graph[key]:
                    graph[key][subkey] = subdefault
    return graph


def _summary_from_transcript(transcript: str, max_chars: int = 320) -> str:
    """Pull a short summary from the first substantive sentence of a transcript."""
    if not transcript:
        return ""
    text = transcript.strip().replace("\n\n", " ").replace("\n", " ")
    # Skip past the first "Q:/A:" prefix if present
    for prefix in ("Q:", "A:", "Question:", "Answer:"):
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip(" :.-")
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if last_period > max_chars // 2:
        return cut[: last_period + 1]
    return cut.rstrip() + "..."


def _chunk_text(text: str, target_chars: int = 2000, max_chunks: int = 12) -> list[str]:
    """Split text into roughly target_chars-sized chunks, preferring paragraph boundaries."""
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining and len(chunks) < max_chunks:
        if len(remaining) <= target_chars:
            chunks.append(remaining)
            break
        window = remaining[: target_chars * 2]
        best = -1
        for sep in ("\n\n", "\n", ". ", "? ", "! "):
            idx = window.rfind(sep, target_chars // 2, target_chars * 2)
            if idx > best:
                best = idx
        if best <= 0:
            best = target_chars
        else:
            best += len("\n\n")
        chunks.append(remaining[:best].strip())
        remaining = remaining[best:].strip()
    return chunks


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BrainBuilder:
    """Builds a PersonalityGraph from interview data — offline only.

    The previous implementation called an LLM to extract structured
    fields from the free-form transcript. That required an API call for
    every interview, which violated the offline-only contract of the
    brain subsystem. The new approach stores the transcript as memory
    chunks and derives a short user_summary locally; structured
    extraction is expected to happen upstream (at publish time in
    beam_mind, or by hand-editing the JSON).
    """

    def extract(self, interview_data: dict, existing_graph: dict = None) -> dict:
        """Build a personality graph from interview data.

        Returns a graph with:
          - user_summary: short excerpt from the transcript
          - memories: one entry per transcript chunk
          - raw_transcript: the full transcript as a top-level string
            (BrainRetriever indexes this for keyword search)
          - all other structured fields left empty (or merged from
            existing_graph if provided)
        """
        transcript = interview_data.get("transcript", "") if isinstance(interview_data, dict) else ""
        if not transcript:
            answers = interview_data.get("answers", []) if isinstance(interview_data, dict) else []
            transcript = "\n\n".join(
                f"[{a.get('domain', 'unknown')}] Q: {a['question']}\nA: {a['answer']}"
                for a in answers
                if isinstance(a, dict) and a.get("answer")
            )

        if not transcript:
            return {"error": "No interview data to extract from"}

        graph: dict = {}
        if existing_graph:
            graph = json.loads(json.dumps(existing_graph))

        # user_summary: first ~320 chars of the transcript, lightly cleaned.
        existing_summary = graph.get("user_summary", "")
        graph["user_summary"] = existing_summary or _summary_from_transcript(transcript)

        # Store the full transcript as memory chunks so BrainRetriever can
        # search it. Drop any prior auto-generated transcript memories
        # (heuristic: name starts with "interview-").
        memories = [m for m in graph.get("memories", []) if isinstance(m, dict)]
        memories = [m for m in memories if not m.get("name", "").startswith("interview-")]
        chunks = _chunk_text(transcript, target_chars=2000, max_chunks=12)
        for idx, chunk in enumerate(chunks, start=1):
            memories.append({
                "name": f"interview-chunk-{idx}",
                "summary": chunk,
                "emotional_tone": 0.0,
                "sensitivity": "personal",
            })
        graph["memories"] = memories

        # Also keep the raw transcript as a top-level field so other
        # tools can index it without needing a structured MemoryNode.
        graph["raw_transcript"] = transcript
        graph["_generated_at"] = _now_iso()

        graph = _validate_graph(graph)
        return {"graph": graph}

    def merge(self, existing_graph: dict, new_graph: dict) -> dict:
        """Merge new extraction into existing graph (deterministic, no LLM)."""
        if not existing_graph:
            return new_graph
        if not new_graph:
            return existing_graph

        merged = json.loads(json.dumps(existing_graph))

        for list_key in ["traits", "beliefs", "values", "boundaries", "life_events", "people"]:
            existing_items = merged.get(list_key, [])
            new_items = new_graph.get(list_key, [])
            seen = {item.get("name", "") for item in existing_items if isinstance(item, dict)}
            for item in new_items:
                if isinstance(item, dict) and item.get("name", "") not in seen:
                    existing_items.append(item)
                    seen.add(item.get("name", ""))
            merged[list_key] = existing_items

        # memories: replace auto-generated chunks, keep hand-written ones
        existing_mems = [m for m in merged.get("memories", []) if isinstance(m, dict) and not m.get("name", "").startswith("interview-")]
        new_auto = [m for m in new_graph.get("memories", []) if isinstance(m, dict) and m.get("name", "").startswith("interview-")]
        merged["memories"] = existing_mems + new_auto

        for dict_key in ["voice_dna", "work_dna", "emotional_profile"]:
            if dict_key in new_graph and new_graph[dict_key]:
                if dict_key not in merged:
                    merged[dict_key] = new_graph[dict_key]
                else:
                    for k, v in new_graph[dict_key].items():
                        if v and (not merged[dict_key].get(k)):
                            merged[dict_key][k] = v

        if new_graph.get("user_summary"):
            merged["user_summary"] = new_graph["user_summary"]

        if new_graph.get("raw_transcript"):
            merged["raw_transcript"] = new_graph["raw_transcript"]

        return merged

    def validate(self, graph: dict) -> dict:
        """Validate graph completeness."""
        graph = _validate_graph(graph)
        issues = []

        if not graph.get("user_summary"):
            issues.append("Missing user_summary")
        if not graph.get("traits") and not graph.get("memories") and not graph.get("raw_transcript"):
            issues.append("No traits, memories, or transcript extracted")

        total_nodes = (
            len(graph.get("traits", []))
            + len(graph.get("beliefs", []))
            + len(graph.get("values", []))
            + len(graph.get("boundaries", []))
            + len(graph.get("life_events", []))
            + len(graph.get("people", []))
            + len(graph.get("memories", []))
        )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_nodes": total_nodes,
            "coverage": {
                "traits": len(graph.get("traits", [])),
                "beliefs": len(graph.get("beliefs", [])),
                "values": len(graph.get("values", [])),
                "boundaries": len(graph.get("boundaries", [])),
                "life_events": len(graph.get("life_events", [])),
                "people": len(graph.get("people", [])),
                "memories": len(graph.get("memories", [])),
            },
        }
