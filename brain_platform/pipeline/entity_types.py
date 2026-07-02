"""Custom Graphiti entity types for the Beam Mind knowledge graph.

These Pydantic models define the structured attributes that Graphiti
extracts and stores on entity nodes. Docstrings are used by the LLM
for entity classification — make them descriptive.

10 personality types covering all 8 interview dimensions + general entities:
  PersonalityTrait, Belief, EpisodicMemory, StyleProfile (original 4)
  Value, Boundary, CognitivePattern, SocialPattern, KnowledgeDomain, LifeEvent (new 6)

4 procedural memory types (plan/10 §6.1):
  ProceduralPattern, WorkLoop, PromptingStyle, TechnicalGap
"""

from typing import Literal

from pydantic import BaseModel, Field


class PersonalityTrait(BaseModel):
    """A personality trait or behavioral pattern of a person. Includes both
    self-reported traits ("I'm analytical") and traits implied by how they
    communicate (e.g., a detailed, structured answer implies analytical thinking)."""

    trait_name: str = Field(description="Name of the trait (e.g., 'analytical', 'empathetic', 'conflict-avoidant')")
    strength: float = Field(description="How strongly this trait is expressed, 0-1", ge=0, le=1)
    evidence_count: int = Field(description="Number of data points supporting this trait", default=1)


class Belief(BaseModel):
    """A stated belief, opinion, or position on a topic. Includes contrarian
    beliefs, professional convictions, and worldview statements. Look for
    'I believe', 'I think', 'I'm convinced', and positions defended with reasoning."""

    topic: str = Field(description="The topic of the belief")
    position: str = Field(description="Their stated position or opinion")
    confidence: float = Field(description="How confidently held, 0-1", ge=0, le=1)
    reasoning: str = Field(description="The reasoning or evidence behind the belief", default="")


class EpisodicMemory(BaseModel):
    """A specific event, experience, or decision from a person's life. These are
    concrete stories with who/what/when/where, not abstract statements. Look for
    narrative markers: 'one time', 'I remember', 'there was this moment when'."""

    event: str = Field(description="Description of the event")
    emotional_valence: float = Field(
        description="Emotional tone, -1 (very negative) to 1 (very positive)", ge=-1, le=1
    )
    salience: float = Field(description="How important/memorable this is to the person, 0-1", ge=0, le=1)
    temporal_context: str = Field(description="When this happened (date, period, or age)", default="")
    spatial_context: str = Field(description="Where this happened", default="")


class StyleProfile(BaseModel):
    """Writing and communication style characteristics. How the person expresses
    themselves: their texting habits, email style, explanation patterns, use of
    humor, code-switching between contexts."""

    formality: float = Field(description="0=very casual, 1=very formal", ge=0, le=1)
    vocabulary_level: str = Field(
        description="Vocabulary complexity: basic, intermediate, advanced, specialized",
        default="intermediate",
    )
    communication_style: str = Field(
        description="Primary style: analytical/narrative/directive/collaborative/metaphorical",
        default="analytical",
    )


class Value(BaseModel):
    """A core value that drives the person's choices and priorities. These are
    the things they care about most deeply — what they'd sacrifice for, what
    makes them feel fulfilled, what they organize their life around. Look for:
    'I care about', 'what matters to me', 'what drives me'."""

    value_name: str = Field(description="The value (e.g., 'autonomy', 'honesty', 'family')")
    importance: float = Field(description="How central this value is to their identity, 0-1", ge=0, le=1)
    evidence: str = Field(description="A specific example of this value in action", default="")


class Boundary(BaseModel):
    """A line the person will not cross — an anti-value, a non-negotiable, something
    they refuse to do even at personal cost. These define personality as much as
    positive values. Look for: 'I won't', 'I refuse', 'that's my line'."""

    description: str = Field(description="What they won't do or tolerate")
    tested: bool = Field(description="Whether this boundary has been tested in a real situation", default=False)
    cost_paid: str = Field(description="What it cost them to maintain this boundary", default="")


class CognitivePattern(BaseModel):
    """A decision-making or thinking pattern — how the person approaches choices,
    processes information, or responds to situations. Includes analytical vs
    intuitive style, risk tolerance, and consultation habits."""

    situation_type: str = Field(description="The type of situation that triggers this pattern")
    typical_response: str = Field(description="How they typically respond or decide")
    outcome: str = Field(description="What usually results from this pattern", default="")


class SocialPattern(BaseModel):
    """How the person handles relationships, conflict, and social situations.
    Includes conflict avoidance/confrontation style, emotional regulation,
    trust-building patterns, and Theory of Mind indicators."""

    context: str = Field(description="The social context (conflict, disagreement, caregiving, etc.)")
    approach: str = Field(description="How they typically handle this situation")
    emotional_response: str = Field(description="Their emotional reaction pattern", default="")


class KnowledgeDomain(BaseModel):
    """An area of deep expertise or passionate knowledge. These are topics the
    person could discuss at length, where they correct common misconceptions,
    and where they have teaching-level understanding."""

    topic: str = Field(description="The domain or subject area")
    depth: float = Field(description="Depth of knowledge, 0-1 (0=surface, 1=expert)", ge=0, le=1)
    misconception_corrected: str = Field(
        description="A common misconception they've identified and can correct", default=""
    )


class LifeEvent(BaseModel):
    """A pivotal moment or turning point in the person's life — a decision that
    changed their direction, a formative experience, a loss or achievement that
    shaped who they became. More specific and impactful than general episodic memories."""

    description: str = Field(description="What happened")
    emotional_valence: float = Field(
        description="Emotional tone, -1 (very negative) to 1 (very positive)", ge=-1, le=1
    )
    life_impact: str = Field(description="How this event changed their life trajectory", default="")
    temporal_context: str = Field(description="When this happened", default="")


# ──────────────────────────────────────────────────────────────
# Procedural memory entity types (plan/10 §6.1)
#
# Capture *how* the person works — reasoning, prompting, action patterns —
# independent of any specific tech stack. Source field tags whether the
# pattern is self-narrated (interview) or observed (artifact).
# ──────────────────────────────────────────────────────────────


class ProceduralPattern(BaseModel):
    """A pattern in *how* the person works on a task — their meta-method, not
    technique. Role-agnostic: applies whether they are coding, designing, writing.
    Examples: 'hypothesis-first debugging', 'small-PR refactoring', 'spec-then-build'.
    Look for descriptions of approach, sequence, and what they explicitly avoid."""

    pattern_name: str = Field(description="Concise pattern name (e.g., 'hypothesis-first debugging')")
    domain: str = Field(description="Free-text role context (e.g., 'debugging', 'design critique', 'writing')", default="")
    approach: str = Field(description="The method — what they do", default="")
    anti_pattern: str = Field(description="What they explicitly do NOT do in this situation", default="")
    source: Literal["interview", "artifact", "synthesis"] = Field(
        description="interview = self-narrated; artifact = observed in PRs/prompts/etc.; synthesis = combined",
        default="interview",
    )
    sample_size: int = Field(description="Number of artifacts/answers this is aggregated from", default=1)


class WorkLoop(BaseModel):
    """An ordered work loop — trigger → steps → stop condition → recovery.
    Captures how someone iterates through a task. Look for narrations like
    'first I X, then I Y, then if it doesn't work I Z'."""

    loop_name: str = Field(description="Concise name for the loop")
    trigger: str = Field(description="What kicks off this loop", default="")
    steps_summary: str = Field(description="Ordered steps as the person describes them", default="")
    stop_condition: str = Field(description="When they stop — finished, frustrated, time-boxed", default="")
    recovery: str = Field(description="What they do when blocked or stuck", default="")
    source: Literal["interview", "artifact", "synthesis"] = Field(default="interview")


class PromptingStyle(BaseModel):
    """How the person directs an AI or delegates to another person. Captures
    structure, length, correction style, and constraint phrasing — NOT specific
    tools or models. Look for descriptions of how they brief, request, or correct."""

    style_label: str = Field(description="Concise style label")
    structure: str = Field(
        description="spec-first | example-first | chain-of-thought | iterative-refinement | other",
        default="",
    )
    correction_style: str = Field(
        description="How they course-correct: gentle / direct / restart-from-scratch / clarifying",
        default="",
    )
    constraint_phrasing: str = Field(description="How they phrase limits and requirements", default="")
    source: Literal["interview", "artifact", "synthesis"] = Field(default="interview")


class TechnicalGap(BaseModel):
    """A knowledge gap the person has — something they avoid, are unaware of,
    or hold an outdated view on. Aspirational gaps are flagged. Non-aspirational
    gaps are still preserved as personality data — they help the agent stay in
    character (e.g., 'this person doesn't engage with finance topics')."""

    topic: str = Field(description="What the gap is about")
    gap_type: Literal["avoids", "unaware", "outdated"] = Field(
        description="avoids = deliberate; unaware = unknown unknown; outdated = stale view",
        default="avoids",
    )
    aspirational: bool = Field(
        description="True if the person has indicated they want to close this gap",
        default=False,
    )
    evidence: str = Field(description="What revealed the gap", default="")


# ──────────────────────────────────────────────────────────────
# Registry passed to Graphiti add_episode()
# ──────────────────────────────────────────────────────────────

BEAM_MIND_ENTITY_TYPES = {
    "PersonalityTrait": PersonalityTrait,
    "Belief": Belief,
    "EpisodicMemory": EpisodicMemory,
    "StyleProfile": StyleProfile,
    "Value": Value,
    "Boundary": Boundary,
    "CognitivePattern": CognitivePattern,
    "SocialPattern": SocialPattern,
    "KnowledgeDomain": KnowledgeDomain,
    "LifeEvent": LifeEvent,
    # Procedural memory layer (plan/10)
    "ProceduralPattern": ProceduralPattern,
    "WorkLoop": WorkLoop,
    "PromptingStyle": PromptingStyle,
    "TechnicalGap": TechnicalGap,
}
