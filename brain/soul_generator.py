"""SOUL.md generator — Python-native, no Rust dependency.

Generates a SOUL.md from the personality graph using LLM.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

BEAM_HOME = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam"))

SOUL_PROMPT = """You are writing a SOUL.md file — a personal identity document that an AI agent loads to understand who it's talking to.

Given a personality graph, write a warm, natural SOUL.md. The agent will read this at the start of every conversation.

Rules:
- Start with "# Soul"
- Have a "## Who I Am" section (2-3 sentences, natural first-person voice)
- Have sections for core traits, values, beliefs, communication style, work style
- Write as if the person is describing themselves
- Be specific — use their actual phrases and patterns
- 200-400 words
- No clinical language, no bullet-point-heavy sections — make it feel human

Output ONLY the markdown content."""


def _call_llm(system: str, user: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """Make a one-shot LLM call via Hermes auxiliary client."""
    from agent.auxiliary_client import call_llm
    response = call_llm(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=60.0,
    )
    return (response.choices[0].message.content or "").strip()


def generate_soul_md(graph: dict, output_path: Path = None) -> str:
    """Generate SOUL.md from a PersonalityGraph.

    Uses LLM to write a natural, warm identity document.
    Falls back to a simple template if LLM is unavailable.
    """
    if output_path is None:
        output_path = BEAM_HOME / "SOUL.md"

    # Try LLM generation first
    try:
        graph_summary = _summarize_graph(graph)
        soul_content = _call_llm(
            "You are a personal identity writer. Write warm, natural prose.",
            f"{SOUL_PROMPT}\n\nPersonality data:\n{graph_summary}",
            temperature=0.7,
            max_tokens=2000,
        )
        # Clean up markdown fences if LLM wrapped them
        if soul_content.startswith("```"):
            lines = soul_content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            soul_content = "\n".join(lines).strip()
    except Exception as e:
        logger.warning("LLM SOUL.md generation failed, using template: %s", e)
        soul_content = _template_soul(graph)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(soul_content, encoding="utf-8")
    return soul_content


def _summarize_graph(graph: dict) -> str:
    """Create a text summary of the graph for the LLM."""
    parts = []

    if graph.get("user_summary"):
        parts.append(f"Summary: {graph['user_summary']}")

    if graph.get("traits"):
        traits = ", ".join(
            f"{t['name']} ({t.get('strength', 0.5):.0%}): {t.get('summary', '')}"
            for t in graph["traits"]
        )
        parts.append(f"Traits: {traits}")

    if graph.get("values"):
        values = ", ".join(
            f"{v['name']} ({v.get('importance', 0.5):.0%}): {v.get('summary', '')}"
            for v in graph["values"]
        )
        parts.append(f"Values: {values}")

    if graph.get("beliefs"):
        beliefs = ", ".join(
            f"{b['name']} ({b.get('confidence', 0.5):.0%}): {b.get('summary', '')}"
            for b in graph["beliefs"]
        )
        parts.append(f"Beliefs: {beliefs}")

    voice = graph.get("voice_dna", {})
    if voice:
        parts.append(f"Communication: humor={voice.get('humor_style', 'n/a')}, "
                      f"length={voice.get('response_length_pattern', 'n/a')}, "
                      f"formality={voice.get('formality_range', 'n/a')}")
        if voice.get("characteristic_phrases"):
            parts.append(f"Phrases they use: {', '.join(voice['characteristic_phrases'])}")

    work = graph.get("work_dna", {})
    if work:
        parts.append(f"Work style: decomposition={work.get('decomposition_style', 'n/a')}, "
                      f"debugging={work.get('debugging_approach', 'n/a')}, "
                      f"risk={work.get('risk_posture', 'n/a')}")

    emotional = graph.get("emotional_profile", {})
    if emotional:
        if emotional.get("energy_sources"):
            parts.append(f"Energy sources: {', '.join(emotional['energy_sources'])}")
        if emotional.get("energy_drains"):
            parts.append(f"Energy drains: {', '.join(emotional['energy_drains'])}")

    return "\n".join(parts)


def _template_soul(graph: dict) -> str:
    """Fallback: generate SOUL.md from template without LLM."""
    lines = ["# Soul\n"]

    if graph.get("user_summary"):
        lines.append(f"## Who I Am\n\n{graph['user_summary']}\n")

    if graph.get("traits"):
        lines.append("## Core Traits\n")
        for t in graph["traits"]:
            lines.append(f"- **{t['name']}** ({t.get('strength', 0.5):.0%}): {t.get('summary', '')}")
        lines.append("")

    if graph.get("values"):
        lines.append("## Values\n")
        for v in graph["values"]:
            lines.append(f"- **{v['name']}** ({v.get('importance', 0.5):.0%}): {v.get('summary', '')}")
        lines.append("")

    if graph.get("beliefs"):
        lines.append("## Beliefs\n")
        for b in graph["beliefs"]:
            lines.append(f"- **{b['name']}** ({b.get('confidence', 0.5):.0%}): {b.get('summary', '')}")
        lines.append("")

    voice = graph.get("voice_dna", {})
    if voice:
        lines.append("## Communication Style\n")
        if voice.get("humor_style"):
            lines.append(f"- Humor: {voice['humor_style']}")
        if voice.get("response_length_pattern"):
            lines.append(f"- Response length: {voice['response_length_pattern']}")
        if voice.get("formality_range"):
            lines.append(f"- Formality: {voice['formality_range']}")
        if voice.get("characteristic_phrases"):
            lines.append(f"- Phrases I use: {', '.join(voice['characteristic_phrases'])}")
        lines.append("")

    work = graph.get("work_dna", {})
    if work:
        lines.append("## Work Style\n")
        if work.get("decomposition_style"):
            lines.append(f"- Problem solving: {work['decomposition_style']}")
        if work.get("debugging_approach"):
            lines.append(f"- Debugging: {work['debugging_approach']}")
        if work.get("risk_posture"):
            lines.append(f"- Risk: {work['risk_posture']}")
        lines.append("")

    return "\n".join(lines)
