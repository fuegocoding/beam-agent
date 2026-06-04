"""Manages MD files for episodic, semantic, procedural memory."""

import os
from datetime import datetime
from pathlib import Path

BEAM_HOME = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam"))


class MDMemory:
    """Read/write memory as Markdown files."""

    def __init__(self, user_id: str = "default"):
        self.base = BEAM_HOME / "memory" / user_id
        self.episodic_dir = self.base / "episodic"
        self.semantic_dir = self.base / "semantic"
        self.procedural_dir = self.base / "procedural"
        self.style_file = self.base / "style.md"
        self.brain_export_dir = self.base / "brain-export"

        for d in [self.episodic_dir, self.semantic_dir, self.procedural_dir, self.brain_export_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def write_episodic(self, title: str, content: str, emotional_tone: float = 0.0, tags: list = None):
        """Write an episodic memory."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{title.lower().replace(' ', '-')}.md"
        filepath = self.episodic_dir / filename

        frontmatter = f"""---
title: {title}
date: {datetime.now().isoformat()}
emotional_tone: {emotional_tone}
tags: {tags or []}
---

"""
        filepath.write_text(frontmatter + content, encoding="utf-8")
        return str(filepath)

    def write_semantic(self, topic: str, content: str, confidence: float = 0.5, source: str = "conversation"):
        """Write a semantic memory."""
        filename = f"{topic.lower().replace(' ', '-')}.md"
        filepath = self.semantic_dir / filename

        frontmatter = f"""---
topic: {topic}
confidence: {confidence}
source: {source}
updated: {datetime.now().isoformat()}
---

"""
        filepath.write_text(frontmatter + content, encoding="utf-8")
        return str(filepath)

    def write_procedural(self, pattern_name: str, content: str, domain: str = "", source: str = "interview"):
        """Write a procedural memory."""
        filename = f"{pattern_name.lower().replace(' ', '-')}.md"
        filepath = self.procedural_dir / filename

        frontmatter = f"""---
pattern: {pattern_name}
domain: {domain}
source: {source}
updated: {datetime.now().isoformat()}
---

"""
        filepath.write_text(frontmatter + content, encoding="utf-8")
        return str(filepath)

    def write_style(self, voice_dna: dict, work_dna: dict):
        """Write the style profile."""
        content = f"""---
updated: {datetime.now().isoformat()}
---

# Communication Style (Voice DNA)

## Characteristic Phrases
{chr(10).join(f'- {p}' for p in voice_dna.get('characteristic_phrases', []))}

## Phrases to Avoid
{chr(10).join(f'- {p}' for p in voice_dna.get('phrases_to_avoid', []))}

## Humor Style
{voice_dna.get('humor_style', 'Not specified')}

## Response Length Pattern
{voice_dna.get('response_length_pattern', 'Not specified')}

## Formality Range
{voice_dna.get('formality_range', 'Not specified')}

## Storytelling Style
{voice_dna.get('storytelling_style', 'Not specified')}

# Work Style (Work DNA)

## Problem Decomposition
{work_dna.get('decomposition_style', 'Not specified')}

## Debugging Approach
{work_dna.get('debugging_approach', 'Not specified')}

## Risk Posture
{work_dna.get('risk_posture', 'Not specified')}

## Delegation Style
{work_dna.get('delegation_style', 'Not specified')}

## Documentation Habit
{work_dna.get('documentation_habit', 'Not specified')}
"""
        self.style_file.write_text(content, encoding="utf-8")
        return str(self.style_file)

    def write_brain_export(self, graph: dict):
        """Write human-readable brain export."""
        summary_file = self.brain_export_dir / "brain-summary.md"
        lines = ["# My Brain — Summary\n"]
        lines.append(f"**Generated:** {datetime.now().isoformat()}\n")

        if graph.get("user_summary"):
            lines.append(f"## Who I Am\n{graph['user_summary']}\n")

        if graph.get("traits"):
            lines.append("## Personality Traits")
            for t in graph["traits"]:
                if isinstance(t, dict):
                    lines.append(f"- **{t.get('name', '?')}** (strength: {t.get('strength', 0.5):.0%}): {t.get('summary', '')}")
                else:
                    lines.append(f"- {t}")
            lines.append("")

        if graph.get("beliefs"):
            lines.append("## Beliefs")
            for b in graph["beliefs"]:
                if isinstance(b, dict):
                    lines.append(f"- **{b.get('name', '?')}** (confidence: {b.get('confidence', 0.5):.0%}): {b.get('summary', '')}")
                else:
                    lines.append(f"- {b}")
            lines.append("")

        if graph.get("values"):
            lines.append("## Core Values")
            for v in graph["values"]:
                if isinstance(v, dict):
                    lines.append(f"- **{v.get('name', '?')}** (importance: {v.get('importance', 0.5):.0%}): {v.get('summary', '')}")
                else:
                    lines.append(f"- {v}")
            lines.append("")

        summary_file.write_text("\n".join(lines), encoding="utf-8")
        return str(summary_file)

    def read_all_episodic(self) -> list:
        """Read all episodic memories."""
        memories = []
        for f in sorted(self.episodic_dir.glob("*.md")):
            memories.append(f.read_text(encoding="utf-8"))
        return memories

    def read_all_semantic(self) -> list:
        """Read all semantic memories."""
        memories = []
        for f in sorted(self.semantic_dir.glob("*.md")):
            memories.append(f.read_text(encoding="utf-8"))
        return memories

    def read_style(self) -> str:
        """Read the style profile."""
        if self.style_file.exists():
            return self.style_file.read_text(encoding="utf-8")
        return ""
