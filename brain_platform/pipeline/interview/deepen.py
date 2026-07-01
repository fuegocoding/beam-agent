"""Interview deepening — post-extraction analysis that identifies gaps.

Faithful port of the cloud's
``beam_mind.pipeline.interview.deepen.analyze_brain_gaps()`` and
``generate_probe_questions()``. Used after a brain is extracted from
an interview to find thin dimensions and generate targeted probe
questions.

Public API:

  result = analyze_brain_gaps(personality_graph)
  # result.completeness_pct, result.gaps, result.probe_questions, result.summary

  questions = generate_probe_questions(gaps, graph, covered_questions, llm)
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Dimension coverage targets ──────────────────────────────
# Two tiers: BASE (minimum viable brain) and ENRICHED (deep brain)

ENRICHED = True  # Toggle this to use higher targets


def _t(base: int, enriched: int) -> int:
    return enriched if ENRICHED else base


DIMENSION_TARGETS = {
    "traits": {
        "min_nodes": _t(5, 9),
        "description": "Personality traits and behavioral patterns",
        "probe_focus": "How they act in different contexts, what patterns others notice about them, their contradictions",
    },
    "beliefs": {
        "min_nodes": _t(4, 8),
        "description": "Beliefs, opinions, and worldview positions",
        "probe_focus": "What they think about the world, contrarian views, changed beliefs, beliefs about themselves",
    },
    "values": {
        "min_nodes": _t(3, 5),
        "description": "Core values that drive choices",
        "probe_focus": "What they sacrifice for, what makes life feel meaningful, how values were formed",
    },
    "boundaries": {
        "min_nodes": _t(2, 4),
        "description": "Lines they won't cross",
        "probe_focus": "Situations where they said no at personal cost, social boundaries, professional lines",
    },
    "life_events": {
        "min_nodes": _t(4, 7),
        "description": "Pivotal turning points",
        "probe_focus": "Decisions that changed direction, failures, health events, relationship ruptures",
    },
    "memories": {
        "min_nodes": _t(4, 7),
        "description": "Specific episodic memories",
        "probe_focus": "Vivid moments, emotionally charged scenes, small details they chose to share",
    },
    "patterns": {
        "min_nodes": _t(3, 5),
        "description": "Decision-making frameworks",
        "probe_focus": "How they approach hard choices, who they consult, what frameworks they use",
    },
    "social": {
        "min_nodes": _t(2, 4),
        "description": "Social and conflict patterns",
        "probe_focus": "How they handle disagreements, what triggers them, how they've evolved",
    },
    "expertise": {
        "min_nodes": _t(2, 3),
        "description": "Areas of deep knowledge",
        "probe_focus": "What they could teach, misconceptions they correct, what drew them in",
    },
    "style": {
        "min_nodes": _t(3, 5),
        "description": "Communication fingerprint",
        "probe_focus": "How they text, email, speak — the patterns that make them sound like themselves",
    },
    "people": {
        "min_nodes": _t(2, 4),
        "description": "Key people in their life",
        "probe_focus": "Who they rely on, who shaped them, important relationships and dynamics",
    },
}


class GapAnalysis(BaseModel):
    """Analysis of a single dimension gap in the brain."""
    dimension: str
    current_count: int
    target_count: int
    gap: int
    description: str


class ProbeQuestion(BaseModel):
    """A targeted question designed to fill a specific gap."""
    dimension: str
    question: str = Field(description="The probe question to ask")
    why: str = Field(description="What this question would reveal for the brain")
    priority: int = Field(description="1=critical gap, 2=would help, 3=nice to have")


class DeepenResult(BaseModel):
    """Complete deepening analysis."""
    completeness_pct: int
    gaps: List[GapAnalysis]
    probe_questions: List[ProbeQuestion]
    summary: str


PROBE_GENERATION_PROMPT = """\
You are designing targeted interview questions to fill gaps in a personality
knowledge graph. The brain was built from an initial interview but some
dimensions are thin.

CURRENT BRAIN STATE:
{brain_state}

GAPS TO FILL:
{gaps}

For each gap, generate 1-2 probe questions that would extract the missing
information. The questions should:
1. Be conversational and warm — not clinical or interrogating
2. Ask for specific stories, examples, or memories — not abstract answers
3. Target the exact gap (e.g., if missing boundaries, ask about a time they
   said no to something tempting)
4. Not repeat questions already answered in the interview

INTERVIEW ALREADY COVERED:
{covered_questions}

Generate probe questions with priority:
1 = critical gap (dimension has < 50% target coverage)
2 = would help (dimension exists but is thin)
3 = nice to have (would add richness but not essential)
"""


def analyze_brain_gaps(graph: Any) -> DeepenResult:
    """Analyze a PersonalityGraph for dimension gaps.

    Mirrors the cloud's analyze_brain_gaps but with a sync signature
    (no async needed — the analysis is pure Python).

    Args:
        graph: A PersonalityGraph instance from brain extraction.

    Returns:
        DeepenResult with gaps and probe_questions=[]. Call
        generate_probe_questions() to fill the probe_questions list.
    """
    from brain_platform.pipeline.brain_schema import PersonalityGraph

    if not isinstance(graph, PersonalityGraph):
        raise TypeError(f"Expected PersonalityGraph, got {type(graph).__name__}")

    # Count nodes per dimension
    counts = {
        "traits": len(graph.traits),
        "beliefs": len(graph.beliefs),
        "values": len(graph.values),
        "boundaries": len(graph.boundaries),
        "life_events": len(graph.life_events),
        "memories": len(graph.memories),
        "patterns": len(graph.patterns),
        "social": len(graph.social),
        "expertise": len(graph.expertise),
        "style": len(graph.style),
        "people": len(graph.people),
    }

    # Identify gaps
    gaps = []
    total_target = 0
    total_actual = 0

    for dim, target_info in DIMENSION_TARGETS.items():
        target = target_info["min_nodes"]
        actual = counts.get(dim, 0)
        total_target += target
        total_actual += min(actual, target)

        if actual < target:
            gaps.append(GapAnalysis(
                dimension=dim,
                current_count=actual,
                target_count=target,
                gap=target - actual,
                description=target_info["description"],
            ))

    completeness = (
        int((total_actual / total_target) * 100) if total_target > 0 else 0
    )

    # Sort gaps by severity (largest gap first)
    gaps.sort(key=lambda g: g.gap, reverse=True)

    # Generate summary
    if completeness >= 90:
        summary = "Brain is well-populated. Minor enrichment possible."
    elif completeness >= 70:
        summary = f"Brain is {completeness}% complete. {len(gaps)} dimensions could use more depth."
    elif completeness >= 50:
        summary = f"Brain is {completeness}% complete. {len(gaps)} significant gaps need attention."
    else:
        summary = f"Brain is only {completeness}% complete. More interview questions needed."

    return DeepenResult(
        completeness_pct=completeness,
        gaps=gaps,
        probe_questions=[],
        summary=summary,
    )


def generate_probe_questions(
    gaps: List[GapAnalysis],
    graph: Any,
    covered_questions: List[str],
    llm_client: Any,
) -> List[ProbeQuestion]:
    """Generate targeted probe questions to fill identified gaps.

    Mirrors the cloud's generate_probe_questions but with a sync
    signature. The LLM call is routed through the local LLMAdapter
    (same as the rest of brain_platform).

    Args:
        gaps: Gaps from analyze_brain_gaps().
        graph: The PersonalityGraph.
        covered_questions: List of questions already asked.
        llm_client: LLM client (LLMAdapter or Graphiti-compatible).

    Returns:
        List of ProbeQuestion objects, sorted by priority.
        Falls back to a simple per-gap probe if the LLM call fails.
    """
    if not gaps:
        return []

    from pydantic import BaseModel

    # Build brain state summary
    brain_state_parts = []
    for dim, target_info in DIMENSION_TARGETS.items():
        count = len(getattr(graph, dim, []))
        status = (
            "OK"
            if count >= target_info["min_nodes"]
            else f"THIN ({count}/{target_info['min_nodes']})"
        )
        brain_state_parts.append(f"  {dim}: {count} nodes — {status}")
    brain_state = "\n".join(brain_state_parts)

    # Build gaps description
    gaps_text = "\n".join(
        f"  {g.dimension}: need {g.gap} more (have {g.current_count}/{g.target_count}) — {g.description}\n"
        f"    Focus: {DIMENSION_TARGETS[g.dimension]['probe_focus']}"
        for g in gaps
    )

    # Build covered questions
    covered_text = "\n".join(f"  - {q}" for q in covered_questions[:20])

    prompt = PROBE_GENERATION_PROMPT.format(
        brain_state=brain_state,
        gaps=gaps_text,
        covered_questions=covered_text,
    )

    try:
        class ProbeQuestionsResponse(BaseModel):
            questions: List[ProbeQuestion]

        result = llm_client.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": "You are a personality interviewer. Generate warm, targeted probe questions.",
                },
                {"role": "user", "content": prompt},
            ],
            response_model=ProbeQuestionsResponse,
            task="brain_probe_generation",
        )
        probes = ProbeQuestionsResponse(**result)
        probes.questions.sort(key=lambda q: q.priority)
        return probes.questions
    except Exception:
        logger.exception("Failed to generate probe questions")
        # Fallback: simple per-gap probe
        fallback = []
        for gap in gaps[:5]:
            target = DIMENSION_TARGETS.get(gap.dimension, {})
            fallback.append(ProbeQuestion(
                dimension=gap.dimension,
                question=f"Can you tell me more about {target.get('probe_focus', gap.description)}?",
                why=f"Your {gap.description} dimension only has {gap.current_count} data points",
                priority=1 if gap.gap >= 2 else 2,
            ))
        return fallback


__all__ = [
    "DIMENSION_TARGETS",
    "GapAnalysis",
    "ProbeQuestion",
    "DeepenResult",
    "analyze_brain_gaps",
    "generate_probe_questions",
]
