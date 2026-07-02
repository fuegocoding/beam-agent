"""Export brain file as Claude Projects system prompt + knowledge pack."""

from brain_platform.pipeline.brain_file.schema import BrainFileSchema


def export_claude(brain_file: BrainFileSchema) -> dict:
    """Generate a Claude Projects-compatible export.

    Returns a dict with 'system_prompt' and 'knowledge_files' keys.
    """
    pp = brain_file.personality_profile
    ws = brain_file.writing_style

    # Build system prompt
    system_parts = [
        f"You are a digital replica of a specific person. Respond as they would — "
        f"with their knowledge, personality, and communication style.",
        "",
        f"## Communication Style",
        f"- Style: {pp.communication_style}",
        f"- Formality: {'formal' if pp.formality > 0.6 else 'casual' if pp.formality < 0.4 else 'moderate'}",
        f"- Tone: {ws.tone}",
        f"- Average sentence length: ~{ws.avg_sentence_length:.0f} words",
        f"- Vocabulary: {ws.vocabulary_level}",
    ]

    if ws.common_phrases:
        system_parts.append(f"- Characteristic phrases: {', '.join(ws.common_phrases[:10])}")

    if pp.values:
        system_parts.extend(["", "## Core Values", ""])
        for v in pp.values:
            system_parts.append(f"- {v}")

    if pp.core_beliefs:
        system_parts.extend(["", "## Core Beliefs", ""])
        for b in pp.core_beliefs:
            system_parts.append(f"- {b}")

    if pp.cognitive_patterns:
        system_parts.extend(["", "## Behavioral Patterns", ""])
        for cp in pp.cognitive_patterns:
            system_parts.append(f"- When facing {cp.situation_type}: {cp.typical_response}")

    system_prompt = "\n".join(system_parts)

    # Build knowledge file (key facts from graph)
    knowledge_lines = ["# Knowledge Base", ""]
    for domain in brain_file.knowledge_domains:
        knowledge_lines.append(f"## {domain.topic}")
        if domain.community_summary:
            knowledge_lines.append(domain.community_summary)
        for entity in domain.key_entities:
            knowledge_lines.append(f"- {entity}")
        knowledge_lines.append("")

    # Add top edge facts
    knowledge_lines.append("## Key Facts")
    for edge in brain_file.knowledge_graph.edges[:50]:
        if edge.fact:
            knowledge_lines.append(f"- {edge.fact}")

    knowledge = "\n".join(knowledge_lines)

    return {
        "system_prompt": system_prompt,
        "knowledge_files": [
            {"filename": "knowledge.md", "content": knowledge},
        ],
    }
