"""Personality profile extraction from typed graph data.

Faithful port of the cloud's
``beam_mind.pipeline.brain_file.personality_extractor.PersonalityExtractor``
with the same sync facade pattern. Two-phase approach:

1. STRUCTURED: Read typed nodes directly from the graph (Value, Belief,
   CognitivePattern, Boundary, etc.) — these are already structured data
   extracted by Graphiti using our custom entity types.
2. LLM SYNTHESIS: Use the LLM to fill gaps (formality, humor, empathy
   scores) and synthesize the structured data into a coherent profile.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from brain_platform.pipeline.brain_file.schema import (
    CognitivePattern,
    GraphNode,
    PersonalityProfile,
)

logger = logging.getLogger(__name__)


PERSONALITY_SYNTHESIS_PROMPT = """You are a personality psychologist reviewing structured personality data extracted from a person's knowledge graph. The data has already been categorized into values, beliefs, cognitive patterns, traits, etc.

Your job is to SYNTHESIZE this into a coherent profile, filling in the numeric scores based on the evidence.

Given the structured data below, determine:
1. **communication_style**: One of: analytical, narrative, directive, collaborative, metaphorical
2. **formality**: 0.0 (very casual) to 1.0 (very formal)
3. **humor_frequency**: 0.0 (never) to 1.0 (frequently)
4. **empathy_indicators**: 0.0 (low) to 1.0 (highly empathetic)

Return ONLY the JSON scores (values and beliefs are already extracted):
{
  "communication_style": "analytical",
  "formality": 0.65,
  "humor_frequency": 0.2,
  "empathy_indicators": 0.6
}

Base your analysis ONLY on the provided data."""


class PersonalityExtractor:
    """Extract a PersonalityProfile from typed graph data.

    Mirrors the cloud's PersonalityExtractor but with a sync API.
    """

    def __init__(self, llm: Any):
        """Args:
            llm: An :class:`LLMAdapter` (or any object with
                ``generate_response(messages=..., task=...)``).
        """
        self._llm = llm

    def extract(
        self,
        node_summaries: list[str],
        edge_facts: list[str],
        community_summaries: list[str],
        typed_nodes: Optional[list[GraphNode]] = None,
    ) -> PersonalityProfile:
        """Extract personality profile from structured graph data.

        Args:
            node_summaries: Plain-text summaries of graph nodes.
            edge_facts: Plain-text facts from graph edges.
            community_summaries: Plain-text summaries of communities/clusters.
            typed_nodes: Typed GraphNode objects (Value, Belief, etc.)
                for structured extraction.

        Returns:
            PersonalityProfile with values, beliefs, cognitive patterns,
            and LLM-synthesized numeric scores.
        """
        if not node_summaries and not edge_facts:
            return PersonalityProfile()

        # ── Phase 1: Extract structured data directly from typed nodes ──
        values = []
        core_beliefs = []
        cognitive_patterns = []

        if typed_nodes:
            for node in typed_nodes:
                if node.type == "Value":
                    name = node.attributes.get("value_name", node.label)
                    values.append(name)
                elif node.type == "Belief":
                    position = node.attributes.get(
                        "position", node.summary or node.label
                    )
                    core_beliefs.append(position)
                elif node.type == "CognitivePattern":
                    cognitive_patterns.append(
                        CognitivePattern(
                            situation_type=node.attributes.get(
                                "situation_type", node.label
                            ),
                            typical_response=node.attributes.get(
                                "typical_response", ""
                            ),
                            emotional_tendency=node.attributes.get(
                                "outcome", ""
                            ),
                        )
                    )
                elif node.type == "Boundary":
                    # Boundaries inform values — "won't do X" implies "values Y"
                    desc = node.attributes.get("description", node.label)
                    values.append(f"will not: {desc}")

        # Deduplicate
        values = list(dict.fromkeys(values))[:7]
        core_beliefs = list(dict.fromkeys(core_beliefs))[:7]
        cognitive_patterns = cognitive_patterns[:5]

        # ── Phase 2: LLM synthesis for numeric scores ──
        parts = []
        if values:
            parts.append("=== Values ===")
            parts.extend(f"- {v}" for v in values)
        if core_beliefs:
            parts.append("\n=== Core Beliefs ===")
            parts.extend(f"- {b}" for b in core_beliefs)
        if node_summaries:
            parts.append("\n=== Key Entity Summaries ===")
            parts.extend(node_summaries[:50])
        if edge_facts:
            parts.append("\n=== Relationship Facts ===")
            parts.extend(edge_facts[:50])

        context = "\n".join(parts)

        communication_style = "analytical"
        formality = 0.5
        humor_frequency = 0.1
        empathy_indicators = 0.5

        if context.strip():
            try:
                response = self._llm.generate_response(
                    messages=[
                        {"role": "system", "content": PERSONALITY_SYNTHESIS_PROMPT},
                        {"role": "user", "content": context},
                    ],
                    task="brain_personality_synthesis",
                )
                scores = self._parse_scores(response)
                communication_style = scores.get(
                    "communication_style", communication_style
                )
                formality = float(scores.get("formality", formality))
                humor_frequency = float(
                    scores.get("humor_frequency", humor_frequency)
                )
                empathy_indicators = float(
                    scores.get("empathy_indicators", empathy_indicators)
                )
            except Exception:
                logger.exception(
                    "Personality synthesis LLM call failed, using defaults"
                )

        return PersonalityProfile(
            communication_style=communication_style,
            formality=formality,
            humor_frequency=humor_frequency,
            empathy_indicators=empathy_indicators,
            values=values,
            core_beliefs=core_beliefs,
            cognitive_patterns=cognitive_patterns,
        )

    def _parse_scores(self, response: str) -> dict:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(
                r"\{[^{}]*\"communication_style\".*?\}", response, re.DOTALL
            )
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}


__all__ = ["PersonalityExtractor", "PERSONALITY_SYNTHESIS_PROMPT"]
