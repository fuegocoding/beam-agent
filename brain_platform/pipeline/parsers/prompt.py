"""Parser for user-authored LLM prompts.

Users upload prompts they've written — system prompts, task instructions,
chain-of-thought templates, etc. We extract prompting style patterns
(structure, constraint phrasing, verbosity, correction approach) and
feed them into the brain as PromptingStyleNode with OBSERVED_AS edges.

Multi-prompt files are split by separators (---, ===, blank line gaps).
"""

import re

from brain_platform.pipeline.parsers.base import BaseParser, ParseResult


def _detect_structure_features(text: str) -> dict:
    """Extract structural features from a single prompt."""
    lines = text.splitlines()
    total_lines = len(lines)
    word_count = len(text.split())

    text_lower = text.lower()

    # Structure detection
    has_examples = any(
        marker in text_lower
        for marker in [
            "example:", "for example", "e.g.", "here is an example",
            "sample:", "like this:", "such as:", "<example>",
        ]
    )
    has_constraints = any(
        marker in text_lower
        for marker in [
            "do not", "don't", "never", "must not", "avoid",
            "always", "must", "required", "constraint", "rule:",
            "important:", "critical:",
        ]
    )
    has_chain_of_thought = any(
        marker in text_lower
        for marker in [
            "step by step", "think through", "let's think",
            "step 1", "first,", "chain of thought", "reasoning",
            "walk through", "break down",
        ]
    )
    has_persona = any(
        marker in text_lower
        for marker in [
            "you are", "act as", "your role", "as a",
            "pretend you", "imagine you", "your persona",
        ]
    )
    has_output_format = any(
        marker in text_lower
        for marker in [
            "output format", "respond in", "format:", "json",
            "markdown", "return as", "structured as", "schema:",
            "```", "xml", "yaml",
        ]
    )
    has_context = any(
        marker in text_lower
        for marker in [
            "context:", "background:", "given that", "here is the",
            "below is", "the following", "attached",
        ]
    )

    # Length classification
    if word_count < 50:
        length_class = "terse"
    elif word_count < 200:
        length_class = "moderate"
    elif word_count < 500:
        length_class = "detailed"
    else:
        length_class = "comprehensive"

    return {
        "word_count": word_count,
        "line_count": total_lines,
        "length_class": length_class,
        "has_examples": has_examples,
        "has_constraints": has_constraints,
        "has_chain_of_thought": has_chain_of_thought,
        "has_persona": has_persona,
        "has_output_format": has_output_format,
        "has_context_setting": has_context,
    }


def _split_prompts(text: str) -> list[str]:
    """Split a multi-prompt file into individual prompts.

    Splits on:
    - Lines of --- or === (3+ chars)
    - Double blank lines
    """
    # Split on separator lines
    parts = re.split(r"\n\s*(?:-{3,}|={3,})\s*\n", text)

    # If only one part, try splitting on double blank lines
    if len(parts) == 1:
        parts = re.split(r"\n\s*\n\s*\n", text)

    # Filter out empty parts
    return [p.strip() for p in parts if p.strip()]


class PromptParser(BaseParser):
    """Parse prompt files into ParseResults with structure metadata."""

    def parse(self, data: bytes, filename: str) -> list[ParseResult]:
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            return []

        prompts = _split_prompts(text)

        results = []
        for i, prompt_text in enumerate(prompts):
            if len(prompt_text) < 10:
                continue

            features = _detect_structure_features(prompt_text)

            title = f"Prompt {i + 1}" if len(prompts) > 1 else filename.rsplit(".", 1)[0]

            results.append(
                ParseResult(
                    text=prompt_text,
                    title=title,
                    metadata={
                        "source_file": filename,
                        "source_type": "prompt",
                        "prompt_index": i,
                        "total_prompts": len(prompts),
                        **features,
                    },
                )
            )

        return results
