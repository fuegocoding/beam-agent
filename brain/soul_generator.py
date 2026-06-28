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
    """Generate SOUL.md from template — no LLM.

    Walks the schema-agnostic node list from
    :mod:`brain.schema_adapter`, so marketplace brains
    (``personality_profile`` / ``knowledge_graph`` /
    ``knowledge_domains`` / ``episodic_memories``) produce a rich
    SOUL.md instead of an empty stub. Legacy-schema brains continue
    to render in the original section order.
    """
    from brain.schema_adapter import iter_nodes

    lines = ["# Soul\n"]

    if graph.get("user_summary"):
        lines.append(f"## Who I Am\n\n{graph['user_summary']}\n")

    # Cache nodes once — we use the same list to populate multiple
    # sections and to detect when a marketplace graph is the source.
    nodes = list(iter_nodes(graph))

    def _by_type(node_type: str):
        return [n for n in nodes if n.get("type") == node_type and n.get("summary")]

    def _render_list(items, score_attr: str, score_label: str | None = None) -> list[str]:
        out: list[str] = []
        for it in items:
            name = it.get("name", "?")
            summary = it.get("summary", "")
            score = it.get(score_attr) if score_attr else None
            if score is not None and score_label:
                out.append(f"- **{name}** ({score_label} {score:.0%}): {summary}")
            else:
                out.append(f"- **{name}**: {summary}")
        return out

    trait_items = _by_type("trait")
    if trait_items:
        lines.append("## Core Traits\n")
        # Prefer "strength" but fall back to "confidence" (marketplace
        # uses confidence for trait-like nodes in personality_profile).
        scored = [t for t in trait_items if "strength" in t]
        unscored = [t for t in trait_items if "strength" not in t]
        scored_sorted = sorted(
            scored,
            key=lambda t: t.get("strength", 0),
            reverse=True,
        )[:10]
        unscored_sorted = unscored[:5]
        lines.extend(_render_list(scored_sorted, "strength", "strength"))
        lines.extend(_render_list(unscored_sorted, "confidence", "strength"))
        lines.append("")

    value_items = _by_type("value")
    if value_items:
        lines.append("## Values\n")
        scored = [v for v in value_items if "importance" in v]
        unscored = [v for v in value_items if "importance" not in v]
        scored_sorted = sorted(
            scored,
            key=lambda v: v.get("importance", 0),
            reverse=True,
        )[:8]
        unscored_sorted = unscored[:5]
        lines.extend(_render_list(scored_sorted, "importance", "importance"))
        lines.extend(_render_list(unscored_sorted, "confidence", "importance"))
        lines.append("")

    belief_items = _by_type("belief")
    if belief_items:
        lines.append("## Beliefs\n")
        scored = [b for b in belief_items if "confidence" in b]
        unscored = [b for b in belief_items if "confidence" not in b]
        scored_sorted = sorted(
            scored,
            key=lambda b: b.get("confidence", 0),
            reverse=True,
        )[:8]
        unscored_sorted = unscored[:5]
        lines.extend(_render_list(scored_sorted, "confidence", "confidence"))
        lines.extend(_render_list(unscored_sorted, None))
        lines.append("")

    # Memory + transcript excerpts — these are usually long; cap to
    # the top 5 to keep SOUL.md from blowing past the agent's
    # 20,000-char context-file limit.
    memory_items = _by_type("memory")[:5]
    if memory_items:
        lines.append("## Memorable Context\n")
        for m in memory_items:
            tone = m.get("emotional_tone", 0.0)
            tone_str = f" (tone {tone:+.2f})" if tone else ""
            summary = m.get("summary", "")
            # Truncate to ~400 chars to keep this section bounded.
            if len(summary) > 400:
                summary = summary[:397] + "..."
            lines.append(f"- **{m.get('name', 'memory')}**{tone_str}: {summary}")
        lines.append("")

    # Knowledge domains from the marketplace schema — high-signal
    # summaries that describe the brain's areas of expertise.
    expertise_items = _by_type("expertise")
    if expertise_items and any(graph.get("knowledge_domains")):
        lines.append("## Areas of Expertise\n")
        scored = [e for e in expertise_items if "depth" in e]
        unscored = [e for e in expertise_items if "depth" not in e]
        scored_sorted = sorted(
            scored,
            key=lambda e: e.get("depth", 0),
            reverse=True,
        )[:8]
        unscored_sorted = unscored[:5]
        for e in scored_sorted:
            sources = e.get("source_count")
            suffix = f" ({sources} sources)" if isinstance(sources, int) and sources else ""
            lines.append(
                f"- **{e.get('name', '?')}** (depth {e.get('depth', 0):.0%}){suffix}: "
                f"{e.get('summary', '')}"
            )
        for e in unscored_sorted:
            lines.append(f"- **{e.get('name', '?')}**: {e.get('summary', '')}")
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
