"""Brain builder — Python-native personality extraction using LLM.

Extracts a PersonalityGraph from interview transcript using call_llm.
No Rust binary dependency for extraction logic.
"""

import json
import logging

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a personality analyst. Given a transcript of a deep interview, extract a structured personality graph.

Be thorough — extract EVERY trait, belief, value, boundary, event, and person mentioned. Use the person's own words where possible.

Output ONLY valid JSON with this exact structure (no markdown fences, no explanation):
{
  "user_summary": "2-3 sentence summary of who this person is",
  "traits": [
    {"name": "trait name", "strength": 0.0-1.0, "summary": "one sentence"}
  ],
  "beliefs": [
    {"name": "belief name", "confidence": 0.0-1.0, "summary": "one sentence"}
  ],
  "values": [
    {"name": "value name", "importance": 0.0-1.0, "summary": "one sentence"}
  ],
  "boundaries": [
    {"topic": "topic", "comfort_level": 0.0-1.0, "summary": "one sentence"}
  ],
  "life_events": [
    {"event": "what happened", "year": "approximate year or period", "impact": "how it shaped them", "summary": "one sentence"}
  ],
  "people": [
    {"name": "name or relation", "relationship": "how they relate", "significance": "why they matter"}
  ],
  "voice_dna": {
    "characteristic_phrases": ["phrase1", "phrase2"],
    "phrases_to_avoid": ["phrase1"],
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
    "triggers": [{"stimulus": "what triggers", "reaction": "how they react", "intensity": 0.0-1.0}],
    "energy_sources": ["what gives energy"],
    "energy_drains": ["what drains energy"],
    "reaction_speed": "fast/slow",
    "recovery_pattern": "how they recover"
  }
}"""


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


def _validate_graph(graph: dict) -> dict:
    """Ensure graph has required fields with defaults."""
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
