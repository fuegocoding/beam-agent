"""Export brain file as Obsidian-compatible vault .zip."""

import io
import zipfile

from brain_platform.pipeline.brain_file.schema import BrainFileSchema


def export_obsidian(brain_file: BrainFileSchema) -> bytes:
    """Generate a .zip of markdown files with wikilinks."""
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Index file
        index_lines = [
            f"# Brain File — {brain_file.metadata.user_id}",
            f"",
            f"Generated: {brain_file.metadata.created_at}",
            f"Schema: v{brain_file.metadata.schema_version}",
            f"Sources: {brain_file.metadata.source_count}",
            f"Nodes: {len(brain_file.knowledge_graph.nodes)}",
            f"Edges: {len(brain_file.knowledge_graph.edges)}",
            f"",
            f"## Knowledge Domains",
            "",
        ]
        for domain in brain_file.knowledge_domains:
            index_lines.append(f"- [[{domain.topic}]] (confidence: {domain.confidence:.0%})")
        index_lines.extend(["", "## Quick Links", "- [[_personality]]", "- [[_writing_style]]"])
        zf.writestr("_index.md", "\n".join(index_lines))

        # Personality file
        pp = brain_file.personality_profile
        personality_lines = [
            "# Personality Profile",
            "",
            f"**Communication Style:** {pp.communication_style}",
            f"**Formality:** {pp.formality:.0%}",
            f"**Humor Frequency:** {pp.humor_frequency:.0%}",
            f"**Empathy:** {pp.empathy_indicators:.0%}",
            "",
            "## Values",
            "",
        ]
        for v in pp.values:
            personality_lines.append(f"- {v}")
        personality_lines.extend(["", "## Core Beliefs", ""])
        for b in pp.core_beliefs:
            personality_lines.append(f"- {b}")
        if pp.cognitive_patterns:
            personality_lines.extend(["", "## Cognitive Patterns", ""])
            for cp in pp.cognitive_patterns:
                personality_lines.append(
                    f"- **{cp.situation_type}**: {cp.typical_response} ({cp.emotional_tendency})"
                )
        zf.writestr("_personality.md", "\n".join(personality_lines))

        # Writing style file
        ws = brain_file.writing_style
        style_lines = [
            "# Writing Style",
            "",
            f"**Avg Sentence Length:** {ws.avg_sentence_length} words",
            f"**Vocabulary Level:** {ws.vocabulary_level}",
            f"**Tone:** {ws.tone}",
            "",
            "## Common Phrases",
            "",
        ]
        for phrase in ws.common_phrases[:20]:
            style_lines.append(f"- \"{phrase}\"")
        zf.writestr("_writing_style.md", "\n".join(style_lines))

        # Knowledge domain files
        node_map = {n.id: n for n in brain_file.knowledge_graph.nodes}
        for domain in brain_file.knowledge_domains:
            lines = [
                f"# {domain.topic}",
                "",
                f"> {domain.community_summary}" if domain.community_summary else "",
                "",
                f"**Confidence:** {domain.confidence:.0%}",
                f"**Sources:** {domain.source_count}",
                "",
                "## Key Entities",
                "",
            ]
            for entity_name in domain.key_entities:
                lines.append(f"- [[{entity_name}]]")

            safe_name = domain.topic.replace("/", "-").replace("\\", "-")
            zf.writestr(f"domains/{safe_name}.md", "\n".join(lines))

        # Entity files (top 100 by summary length to avoid huge exports)
        sorted_nodes = sorted(
            brain_file.knowledge_graph.nodes, key=lambda n: len(n.summary), reverse=True
        )
        for node in sorted_nodes[:100]:
            lines = [
                f"# {node.label}",
                "",
                f"**Type:** {node.type}",
            ]
            if node.summary:
                lines.extend(["", node.summary])

            # Find edges involving this node
            related = []
            for edge in brain_file.knowledge_graph.edges:
                if edge.source == node.id:
                    target = node_map.get(edge.target)
                    if target:
                        related.append(f"- {edge.relation} → [[{target.label}]]: {edge.fact}")
                elif edge.target == node.id:
                    source = node_map.get(edge.source)
                    if source:
                        related.append(f"- [[{source.label}]] → {edge.relation}: {edge.fact}")

            if related:
                lines.extend(["", "## Relationships", ""])
                lines.extend(related[:20])

            safe_label = node.label.replace("/", "-").replace("\\", "-")[:80]
            zf.writestr(f"entities/{safe_label}.md", "\n".join(lines))

    return buf.getvalue()
