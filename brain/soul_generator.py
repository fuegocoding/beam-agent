"""SOUL.md generator — Python-native, template-only.

Generates a SOUL.md from the personality graph using a pure template
(no LLM). The brain subsystem is fully offline, so SOUL.md is built
locally from whatever structured data is in the graph.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

BEAM_HOME = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam"))


def generate_soul_md(graph: dict, output_path: Path = None) -> str:
    """Generate SOUL.md from a PersonalityGraph.

    Uses a deterministic template — no LLM call. The previous LLM-based
    generator was removed so the brain stays offline.
    """
    if output_path is None:
        output_path = BEAM_HOME / "SOUL.md"

    clone_name = graph.get("clone_name", "")
    soul_content = _template_soul(graph)

    if clone_name and "# Soul" in soul_content:
        soul_content = soul_content.replace("# Soul", f"# {clone_name}'s Soul", 1)
    elif clone_name and not soul_content.startswith(f"# {clone_name}"):
        soul_content = f"# {clone_name}'s Soul\n\n{soul_content}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(soul_content, encoding="utf-8")
    return soul_content


def _template_soul(graph: dict) -> str:
    """Generate SOUL.md from template — no LLM."""
    lines = ["# Soul\n"]

    if graph.get("user_summary"):
        lines.append(f"## Who I Am\n\n{graph['user_summary']}\n")

    if graph.get("traits"):
        lines.append("## Core Traits\n")
        for t in graph["traits"]:
            if isinstance(t, dict):
                lines.append(f"- **{t.get('name', '?')}** ({t.get('strength', 0.5):.0%}): {t.get('summary', '')}")
            else:
                lines.append(f"- {t}")
        lines.append("")

    if graph.get("values"):
        lines.append("## Values\n")
        for v in graph["values"]:
            if isinstance(v, dict):
                lines.append(f"- **{v.get('name', '?')}** ({v.get('importance', 0.5):.0%}): {v.get('summary', '')}")
            else:
                lines.append(f"- {v}")
        lines.append("")

    if graph.get("beliefs"):
        lines.append("## Beliefs\n")
        for b in graph["beliefs"]:
            if isinstance(b, dict):
                lines.append(f"- **{b.get('name', '?')}** ({b.get('confidence', 0.5):.0%}): {b.get('summary', '')}")
            else:
                lines.append(f"- {b}")
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
