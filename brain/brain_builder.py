"""Brain builder — Python-native personality extraction using LLM.

Extracts a PersonalityGraph from interview transcript using call_llm.
No Rust binary dependency for extraction logic.
"""

import json
import logging

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a personality analyst. Extract a structured personality graph from this interview transcript.

OUTPUT FORMAT — follow this EXACTLY. Every field is required. Use empty arrays [] for missing data.

{
  "user_summary": "2-3 sentence summary of who this person is, in first person",
  "traits": [
    {"name": "trait name", "strength": 0.0 to 1.0, "summary": "one sentence"}
  ],
  "beliefs": [
    {"name": "belief name", "confidence": 0.0 to 1.0, "summary": "one sentence"}
  ],
  "values": [
    {"name": "value name", "importance": 0.0 to 1.0, "summary": "one sentence"}
  ],
  "boundaries": [],
  "life_events": [],
  "people": [
    {"name": "name", "relationship": "how they relate", "significance": "why they matter"}
  ],
  "voice_dna": {
    "characteristic_phrases": ["exact phrases they used"],
    "phrases_to_avoid": [],
    "humor_style": "description",
    "response_length_pattern": "description",
    "formality_range": "description",
    "storytelling_style": "description"
  },
  "work_dna": {
    "decomposition_style": "how they break down problems",
    "debugging_approach": "how they debug",
    "risk_posture": "relationship with risk",
    "delegation_style": "how they delegate",
    "documentation_habit": "how they document"
  },
  "emotional_profile": {
    "triggers": [{"stimulus": "what triggers", "reaction": "how they react", "intensity": 0.0 to 1.0}],
    "energy_sources": ["what gives energy"],
    "energy_drains": ["what drains energy"],
    "reaction_speed": "fast/slow/variable",
    "recovery_pattern": "how they recover from setbacks"
  }
}

RULES:
- traits: 3-8 personality traits with DIFFERENT names
- beliefs: 3-8 core beliefs, NOT traits. Beliefs are opinions/convictions about the world.
- values: 3-6 core values. Values are principles they live by.
- voice_dna: extract from HOW they spoke, not WHAT they believe
- emotional_profile: extract triggers, energy sources/drains from their answers
- Use their exact words where possible
- All arrays must contain objects with the specified fields
- Do NOT nest data — keep it flat as shown above"""


def _call_llm(system: str, user: str, temperature: float = 0.3, max_tokens: int = 8000) -> str:
    """Make a one-shot LLM call via Hermes auxiliary client."""
    from agent.auxiliary_client import call_llm
    response = call_llm(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=180.0,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _normalize_graph(graph: dict) -> dict:
    """Normalize LLM output to expected format."""

    # user_summary from various sources
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

    # traits from values/identity arrays
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

    # values from various sources
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

    # beliefs — handle string or list of strings
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

    # emotional_profile from multiple possible sources
    if not graph.get("emotional_profile") or not graph.get("emotional_profile", {}).get("energy_sources"):
        result = graph.setdefault("emotional_profile", {})
        # Try emotional_triggers
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
        # Try interaction_patterns
        ip = graph.get("interaction_patterns", {})
        if isinstance(ip, dict) and not result.get("recovery_pattern"):
            result["recovery_pattern"] = str(ip)
        # Try personality_indicators as triggers
        pi = graph.get("personality_indicators", {})
        if isinstance(pi, dict) and not result.get("triggers"):
            result["triggers"] = [{"stimulus": k, "reaction": v, "intensity": 0.5} for k, v in pi.items() if isinstance(v, str)]

    # work_dna from work_style or work_approach
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

    # people from relationships (handle nested significant_people array)
    if not graph.get("people"):
        people = []
        rels = graph.get("relationships", {})
        if isinstance(rels, dict):
            # Handle significant_people array
            for person in rels.get("significant_people", []):
                if isinstance(person, dict):
                    name = person.get("role", person.get("name", "Unknown"))
                    significance = person.get("impact", person.get("description", person.get("lesson", "")))
                    people.append({"name": name, "relationship": name, "significance": significance})
            # Handle flat relationships
            for key, val in rels.items():
                if key not in ("significant_people", "key_influences", "trust_approach", "relationship_learning") and isinstance(val, str):
                    people.append({"name": key, "relationship": key, "significance": val})
        if people:
            graph["people"] = people

    return graph


def _validate_graph(graph: dict) -> dict:
    """Ensure graph has required fields with defaults."""
    # Normalize: if LLM returned nested format, flatten it
    graph = _normalize_graph(graph)

    defaults = {
        "user_summary": "",
        "traits": [],
        "beliefs": [],
        "values": [],
        "boundaries": [],
        "life_events": [],
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


class BrainBuilder:
    """Builds a PersonalityGraph from interview data using LLM."""

    def extract(self, interview_data: dict, existing_graph: dict = None) -> dict:
        """Extract personality graph from interview data."""
        transcript = interview_data.get("transcript", "")
        if not transcript:
            answers = interview_data.get("answers", [])
            transcript = "\n\n".join(
                f"[{a.get('domain', 'unknown')}] Q: {a['question']}\nA: {a['answer']}"
                for a in answers
            )

        if not transcript:
            return {"error": "No interview data to extract from"}

        user_prompt = f"Interview transcript:\n\n{transcript}"
        if existing_graph:
            user_prompt += f"\n\nExisting personality graph (merge new data into this):\n{json.dumps(existing_graph, indent=2)}"

        try:
            result_text = _call_llm(
                "You are a personality extraction system. Output ONLY valid JSON.",
                user_prompt,
                temperature=0.2,
                max_tokens=8000,
            )
            graph = _parse_json(result_text)
            graph = _validate_graph(graph)
            return {"graph": graph}
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM extraction output: %s", e)
            # Try once more with a stricter prompt
            try:
                retry_text = _call_llm(
                    "Output ONLY valid JSON. No explanation, no markdown fences.",
                    f"Extract personality data from this transcript as JSON:\n\n{transcript[:6000]}",
                    temperature=0.1,
                    max_tokens=8000,
                )
                graph = _parse_json(retry_text)
                graph = _validate_graph(graph)
                return {"graph": graph}
            except Exception as e2:
                logger.error("Retry also failed: %s", e2)
                return {"error": f"Extraction failed: {e2}"}
        except Exception as e:
            logger.error("LLM extraction call failed: %s", e)
            return {"error": f"Extraction failed: {e}"}

    def merge(self, existing_graph: dict, new_graph: dict) -> dict:
        """Merge new extraction into existing graph."""
        if not existing_graph:
            return new_graph
        if not new_graph:
            return existing_graph

        # Simple merge: combine lists, prefer newer values for scalars
        merged = dict(existing_graph)

        for list_key in ["traits", "beliefs", "values", "boundaries", "life_events", "people"]:
            existing_items = merged.get(list_key, [])
            new_items = new_graph.get(list_key, [])
            # Deduplicate by name
            seen = {item.get("name", "") for item in existing_items}
            for item in new_items:
                if item.get("name", "") not in seen:
                    existing_items.append(item)
                    seen.add(item.get("name", ""))
            merged[list_key] = existing_items

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

        return merged

    def validate(self, graph: dict) -> dict:
        """Validate graph completeness."""
        graph = _validate_graph(graph)
        issues = []

        if not graph.get("user_summary"):
            issues.append("Missing user_summary")
        if not graph.get("traits"):
            issues.append("No traits extracted")
        if not graph.get("values"):
            issues.append("No values extracted")
        if not graph.get("voice_dna", {}).get("humor_style"):
            issues.append("Incomplete voice_dna")

        total_nodes = (
            len(graph.get("traits", []))
            + len(graph.get("beliefs", []))
            + len(graph.get("values", []))
            + len(graph.get("boundaries", []))
            + len(graph.get("life_events", []))
            + len(graph.get("people", []))
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
            },
        }
