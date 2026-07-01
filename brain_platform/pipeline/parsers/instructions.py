"""Parser for user-authored AI instructions files.

Users upload their custom AI instructions (Claude.md, system prompts, custom
rules files). Unlike prompts (which get split into individual prompts), an
instructions file is treated as a single coherent unit of directives.

The parser detects structural features that inform downstream extraction.
"""

import re

from brain_platform.pipeline.parsers.base import BaseParser, ParseResult


def _detect_instruction_features(text: str) -> dict:
    """Detect structural features relevant to instructions files."""
    text_lower = text.lower()
    word_count = len(text.split())

    has_rules = any(
        marker in text_lower
        for marker in [
            "must", "always", "never", "do not", "don't",
            "required", "rule:", "important:", "critical:",
        ]
    )
    has_persona = any(
        marker in text_lower
        for marker in [
            "you are", "act as", "your role", "as a",
            "your persona", "behave as",
        ]
    )
    has_boundaries = any(
        marker in text_lower
        for marker in [
            "refuse", "decline", "don't", "do not",
            "never", "forbidden", "prohibited", "off-limits",
        ]
    )
    has_workflow = any(
        marker in text_lower
        for marker in [
            "when", "if asked", "in this case", "step 1",
            "first,", "then,", "workflow:", "process:",
        ]
    )
    has_examples = any(
        marker in text_lower
        for marker in [
            "example:", "for example", "e.g.", "for instance",
            "such as:", "<example>", "sample:",
        ]
    )

    # Detect markdown structure
    has_sections = bool(re.search(r"^#{1,3}\s", text, re.MULTILINE))
    has_bullet_lists = bool(re.search(r"^[\s]*[-*]\s", text, re.MULTILINE))

    return {
        "word_count": word_count,
        "has_rules": has_rules,
        "has_persona": has_persona,
        "has_boundaries": has_boundaries,
        "has_workflow": has_workflow,
        "has_examples": has_examples,
        "has_sections": has_sections,
        "has_bullet_lists": has_bullet_lists,
    }


class InstructionsParser(BaseParser):
    """Parse instructions files into a single ParseResult.

    Unlike PromptParser, instructions are NOT split — the entire file is
    a coherent set of directives that should be processed as one unit.
    """

    def parse(self, data: bytes, filename: str) -> list[ParseResult]:
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            return []

        features = _detect_instruction_features(text)

        return [
            ParseResult(
                text=text,
                title=filename.rsplit(".", 1)[0],
                metadata={
                    "source_file": filename,
                    "source_type": "instructions",
                    **features,
                },
            )
        ]
