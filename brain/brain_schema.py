"""Structured brain schema for personality extraction.

The PersonalityGraph is the complete output of one LLM call over a full
interview. It defines every node and edge in the personality knowledge graph.

Design principles:
- EverMemOS: structured profile with typed categories + evidence
- Mem0: explicit lifecycle decisions (ADD/SKIP)
- Windsurf: semantic dedup built into the schema
- All edge types read left-to-right: "source VERB target"
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

SensitivityLevel = Literal["public", "personal", "private"]

ROLE_LABELS = frozenset({
    "wife", "husband", "spouse", "partner", "girlfriend", "boyfriend",
    "mother", "father", "mom", "dad", "son", "daughter", "daughters", "sons",
    "sister", "brother", "grandmother", "grandfather",
    "boss", "mentor", "therapist", "doctor",
})


def _coerce_to_dict(v: object, defaults: dict) -> dict:
    """If v is a plain string, turn it into a minimal valid node dict."""
    if isinstance(v, str):
        return {"name": v, "summary": v, **defaults}
    return v


class TraitNode(BaseModel):
    """A personality trait or behavioral pattern."""
    name: str = Field(description="Concise trait name under 50 chars, e.g. 'conflict-avoidant'")
    strength: float = Field(default=0.5, description="How strongly expressed, 0-1", ge=0, le=1)
    summary: str = Field(default="", description="2-3 sentences: how this trait manifests with evidence")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {"strength": 0.5})


class BeliefNode(BaseModel):
    """A stated belief, opinion, or position."""
    name: str = Field(description="Concise belief under 50 chars, e.g. 'rest is productive'")
    confidence: float = Field(default=0.5, description="How confidently held, 0-1", ge=0, le=1)
    sensitivity: SensitivityLevel = Field(default="public", description="Privacy tier: public, personal, or private")
    summary: str = Field(default="", description="The belief with its reasoning and evidence")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {"confidence": 0.5})


class ValueNode(BaseModel):
    """A core value that drives choices and priorities."""
    name: str = Field(description="The value word under 50 chars, e.g. 'autonomy'")
    importance: float = Field(default=0.5, description="How central to identity, 0-1", ge=0, le=1)
    sensitivity: SensitivityLevel = Field(default="public", description="Privacy tier: public, personal, or private")
    summary: str = Field(default="", description="How this value shows up in their life with examples")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {"importance": 0.5})


class BoundaryNode(BaseModel):
    """A non-negotiable line they won't cross."""
    name: str = Field(description="What they won't do, under 50 chars")
    tested: bool = Field(default=False, description="Has this boundary been tested?")
    cost_paid: str = Field(default="", description="What it cost them to maintain this boundary")
    sensitivity: SensitivityLevel = Field(default="public", description="Privacy tier: public, personal, or private")
    summary: str = Field(default="", description="The boundary with context and evidence")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {"tested": False, "cost_paid": ""})


class LifeEventNode(BaseModel):
    """A pivotal moment or turning point."""
    name: str = Field(description="Concise event name under 50 chars")
    impact: str = Field(default="", description="How this changed their life trajectory")
    sensitivity: SensitivityLevel = Field(default="public", description="Privacy tier: public, personal, or private")
    summary: str = Field(default="", description="What happened, when, and why it mattered")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {"impact": ""})


class MemoryNode(BaseModel):
    """A specific episodic memory or experience."""
    name: str = Field(description="Concise memory under 50 chars")
    emotional_tone: float = Field(default=0.0, description="-1 (very negative) to 1 (very positive)", ge=-1, le=1)
    sensitivity: SensitivityLevel = Field(default="public", description="Privacy tier: public, personal, or private")
    summary: str = Field(default="", description="The memory with emotional and temporal context")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {"emotional_tone": 0.0})


class PatternNode(BaseModel):
    """A decision-making or cognitive pattern."""
    name: str = Field(description="Concise pattern under 50 chars")
    sensitivity: SensitivityLevel = Field(default="public", description="Privacy tier: public, personal, or private")
    summary: str = Field(default="", description="How they use this pattern, when it's triggered, what results")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {})


class SocialNode(BaseModel):
    """A social interaction or conflict handling pattern."""
    name: str = Field(description="Concise pattern under 50 chars")
    sensitivity: SensitivityLevel = Field(default="public", description="Privacy tier: public, personal, or private")
    summary: str = Field(default="", description="How they handle this social situation with evidence")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {})


class ExpertiseNode(BaseModel):
    """An area of deep knowledge or expertise."""
    name: str = Field(description="The domain under 50 chars")
    depth: float = Field(default=0.5, description="0=surface, 1=expert", ge=0, le=1)
    summary: str = Field(default="", description="What they know, what misconceptions they correct")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {"depth": 0.5})


class StyleNode(BaseModel):
    """A communication style characteristic."""
    name: str = Field(description="The style marker under 50 chars")
    summary: str = Field(default="", description="How this communication pattern manifests")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {})


class PersonNode(BaseModel):
    """A named person in the interview subject's life."""
    name: str = Field(description="The person's name, e.g. 'Priya'")
    role: str = Field(default="", description="Their role: 'therapist friend', 'co-founder', 'mother'")
    sensitivity: SensitivityLevel = Field(default="private", description="Privacy tier: public, personal, or private (defaults to private)")
    summary: str = Field(default="", description="Who they are and their significance")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {"role": ""})


class PlaceNode(BaseModel):
    """A named place significant to the person."""
    name: str = Field(description="The place name, e.g. 'Portland'")
    significance: str = Field(default="", description="Why this place matters to them")
    sensitivity: SensitivityLevel = Field(default="public", description="Privacy tier: public, personal, or private")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {"significance": ""})


# ── Procedural memory layer ─────────────────────────────────

ProceduralSource = Literal["interview", "artifact", "synthesis"]


class CalibrationKnobs(BaseModel):
    """Per-node knobs that adjust runtime delivery without erasing source data."""
    tone_temperature: float = Field(
        default=1.0,
        description="0.0=neutral delivery, 1.0=raw, >1=amplified.",
        ge=0.0, le=2.0,
    )
    confidence_clip: float | None = Field(
        default=None,
        description="If set, cap stated confidence at this value (0-1).",
        ge=0.0, le=1.0,
    )
    contradiction_policy: Literal["aspirational", "observed", "both"] = Field(
        default="both",
        description="When ASPIRES_TO and OBSERVED_AS disagree, which to embody.",
    )
    factual_correction: bool = Field(
        default=True,
        description="Allow runtime to correct category-1 factual errors.",
    )
    harm_gate: Literal["off", "soften", "exclude"] = Field(
        default="soften",
        description="Category-5 harmful-pattern gating.",
    )


class ProceduralPatternNode(BaseModel):
    """A pattern in how the person works. Role-agnostic — captures meta-method."""
    name: str = Field(description="Concise pattern name under 50 chars")
    domain: str = Field(default="", description="Free-text role context")
    situation: str = Field(default="", description="When this pattern fires")
    approach: str = Field(default="", description="What they do — the method")
    tells: list[str] = Field(default_factory=list, description="3-6 observable markers of this pattern")
    anti_pattern: str = Field(default="", description="What they explicitly do NOT do")
    source: ProceduralSource = Field(default="interview", description="interview | artifact | synthesis")
    sample_size: int = Field(default=1, description="Number of artifacts/answers this is aggregated from")
    sensitivity: SensitivityLevel = Field(default="public", description="Privacy tier")
    summary: str = Field(default="", description="Concise summary with provenance context")
    calibration: CalibrationKnobs | None = Field(default=None, description="Optional runtime calibration knobs")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {"tells": []})


class WorkLoopNode(BaseModel):
    """An ordered work loop — trigger -> steps -> stop condition -> recovery."""
    name: str = Field(description="Concise loop name under 50 chars")
    trigger: str = Field(default="", description="What kicks off this loop")
    steps: list[str] = Field(default_factory=list, description="Ordered steps as the person describes them")
    stop_condition: str = Field(default="", description="When they stop")
    recovery: str = Field(default="", description="What they do when blocked or stuck")
    source: ProceduralSource = Field(default="interview", description="interview | artifact | synthesis")
    sensitivity: SensitivityLevel = Field(default="public", description="Privacy tier")
    summary: str = Field(default="", description="Loop description with context")
    calibration: CalibrationKnobs | None = Field(default=None, description="Optional runtime calibration knobs")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {"steps": []})


class PromptingStyleNode(BaseModel):
    """How the person directs an AI or delegates to a person."""
    name: str = Field(description="Concise style label under 50 chars")
    structure: str = Field(default="", description="spec-first | example-first | chain-of-thought | iterative-refinement | other")
    length_preference: str = Field(default="", description="Typical prompt length and density")
    correction_style: str = Field(default="", description="How they course-correct an agent")
    constraint_phrasing: str = Field(default="", description="How they phrase limits and requirements")
    examples_excerpts: list[str] = Field(default_factory=list, description="0-3 redacted excerpts of their actual prompts")
    source: ProceduralSource = Field(default="interview", description="interview | artifact | synthesis")
    sensitivity: SensitivityLevel = Field(default="public", description="Privacy tier")
    summary: str = Field(default="", description="Description with context")
    calibration: CalibrationKnobs | None = Field(default=None, description="Optional runtime calibration knobs")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {"examples_excerpts": []})


class TechnicalGapNode(BaseModel):
    """A knowledge gap the person has."""
    name: str = Field(description="Concise gap topic under 50 chars")
    gap_type: Literal["avoids", "unaware", "outdated"] = Field(
        default="avoids",
        description="avoids = deliberate; unaware = unknown unknown; outdated = held-but-stale view",
    )
    evidence: str = Field(default="", description="What revealed the gap")
    aspirational: bool = Field(default=False, description="True if they want to close this gap")
    sensitivity: SensitivityLevel = Field(default="personal", description="Privacy tier — gaps default to personal")
    summary: str = Field(default="", description="Description with context")
    calibration: CalibrationKnobs | None = Field(default=None, description="Optional runtime calibration knobs")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, v):
        return _coerce_to_dict(v, {"gap_type": "avoids", "aspirational": False})


# ── Behavioral specification (Clone DNA) ────────────────────


class BehavioralRule(BaseModel):
    """A trigger -> response pattern that defines how the person acts."""
    trigger: str = Field(description="The situation or stimulus")
    response: str = Field(description="What they typically do")
    exceptions: str = Field(default="", description="When they break this pattern")


class ContradictionPattern(BaseModel):
    """A topic where the person has a strong stance and will push back."""
    topic: str = Field(description="What they'll fight about")
    stance: str = Field(description="Their position")
    how_they_push_back: str = Field(default="", description="Their style when correcting")


class WorkDNA(BaseModel):
    """The procedural fingerprint — how this person works across tasks."""
    decomposition_style: str = Field(default="", description="How they break down problems")
    error_taxonomy: str = Field(default="", description="Which classes of error they assume vs. hunt for")
    debugging_approach: str = Field(default="", description="How they investigate problems")
    review_depth: str = Field(default="", description="How they review others' work")
    documentation_habit: str = Field(default="", description="What/when/how they document")
    abstraction_timing: str = Field(default="", description="When they introduce abstractions")
    risk_posture: str = Field(default="", description="Risk preference under uncertainty")
    delegation_style: str = Field(default="", description="How they hand work off to people or AI")
    stop_conditions: str = Field(default="", description="When they decide work is done")


class VoiceDNA(BaseModel):
    """The communication fingerprint that makes the clone SOUND like this person."""
    characteristic_phrases: list[str] = Field(default_factory=list, description="Actual phrases/expressions they use")
    phrases_to_avoid: list[str] = Field(default_factory=list, description="Things they would NEVER say")
    punctuation_and_formatting: str = Field(default="", description="How they text/write")
    emoji_usage: str = Field(default="", description="Emoji habits")
    humor_style: str = Field(default="", description="How they're funny")
    response_length_pattern: str = Field(default="", description="How long their messages typically are")
    formality_range: str = Field(default="", description="How formality shifts by context")
    filler_words: list[str] = Field(default_factory=list, description="Verbal tics and filler")
    storytelling_style: str = Field(default="", description="How they tell stories")
    listener_vs_talker: str = Field(default="", description="Conversation balance")


# ── Edge specification ──────────────────────────────────────


class EdgeSpec(BaseModel):
    """A directed edge between two named entities.

    Direction reads left-to-right: source VERB target.
    """
    source_name: str = Field(description="Name of source entity (must match a node name exactly)")
    target_name: str = Field(description="Name of target entity (must match a node name exactly)")
    edge_type: str = Field(default="INVOLVES", description="One of: SHAPED, EVOLVED_INTO, ENFORCES, GUIDES, TAUGHT, EXPRESSES, TESTED, INFORMS, INVOLVES, HAS_TRAIT, HOLDS, DRIVEN_BY, EXPERIENCED, EXPERT_IN, HANDLES_CONFLICT_BY, COMMUNICATES_VIA, OPERATES_IN, EXPLAINED_BY, ASPIRES_TO, CONTRADICTS, WORKS_BY")
    fact: str = Field(default="", description="One-sentence description of the relationship")

    @model_validator(mode="before")
    @classmethod
    def normalize_field_names(cls, v: dict) -> dict:
        """Accept alternate field names that some LLMs produce."""
        if not isinstance(v, dict):
            return v
        if "source" in v and "source_name" not in v:
            v["source_name"] = v.pop("source")
        if "target" in v and "target_name" not in v:
            v["target_name"] = v.pop("target")
        for key in ("relationship", "relation", "type"):
            if key in v and "edge_type" not in v:
                v["edge_type"] = v.pop(key)
                break
        for key in ("description", "label", "text"):
            if key in v and "fact" not in v:
                v["fact"] = v.pop(key)
                break
        return v


# ── Emotional dynamics ──────────────────────────────────────


class EmotionalTrigger(BaseModel):
    """Something that reliably produces an emotional response."""
    trigger: str = Field(description="Situation or stimulus, under 50 chars")
    emotion: str = Field(description="Primary emotion: joy, frustration, anxiety, excitement, sadness, anger, calm")
    intensity: float = Field(default=0.5, description="How strong, 0-1", ge=0, le=1)
    expression: str = Field(default="", description="How it manifests externally")
    context: str = Field(default="", description="When/where this trigger is strongest")


class EmotionalProfile(BaseModel):
    """The person's overall emotional dynamics."""
    baseline_mood: str = Field(default="", description="Default emotional state")
    processing_style: str = Field(default="", description="Internal vs external processor")
    reaction_speed: str = Field(default="", description="Fast vs slow to react emotionally")
    recovery_pattern: str = Field(default="", description="How they bounce back from strong emotions")
    energy_sources: list[str] = Field(default_factory=list, description="What recharges them")
    energy_drains: list[str] = Field(default_factory=list, description="What depletes them")
    emotional_tells: list[str] = Field(default_factory=list, description="Visible signs of emotional state")


class ContextualMood(BaseModel):
    """How emotional expression shifts by social context."""
    context: str = Field(description="Social setting, e.g. 'with close friends'")
    mood: str = Field(default="", description="Typical emotional tone in this context")
    guard_level: float = Field(default=0.5, description="0=open, 1=guarded", ge=0, le=1)
    energy_level: float = Field(default=0.5, description="0=drained, 1=energized", ge=0, le=1)


# ── Complete graph ──────────────────────────────────────────


class PersonalityGraph(BaseModel):
    """Complete personality graph extracted from one interview.

    Every node connects to THE_USER. Every edge reads left-to-right.
    """
    user_summary: str = Field(default="", description="2-3 sentence overview of who this person is")

    traits: list[TraitNode] = Field(default_factory=list)
    beliefs: list[BeliefNode] = Field(default_factory=list)
    values: list[ValueNode] = Field(default_factory=list)
    boundaries: list[BoundaryNode] = Field(default_factory=list)
    life_events: list[LifeEventNode] = Field(default_factory=list)
    memories: list[MemoryNode] = Field(default_factory=list)
    patterns: list[PatternNode] = Field(default_factory=list)
    social: list[SocialNode] = Field(default_factory=list)
    expertise: list[ExpertiseNode] = Field(default_factory=list)
    style: list[StyleNode] = Field(default_factory=list)
    people: list[PersonNode] = Field(default_factory=list)
    places: list[PlaceNode] = Field(default_factory=list)

    procedural_patterns: list[ProceduralPatternNode] = Field(default_factory=list)
    work_loops: list[WorkLoopNode] = Field(default_factory=list)
    prompting_styles: list[PromptingStyleNode] = Field(default_factory=list)
    technical_gaps: list[TechnicalGapNode] = Field(default_factory=list)

    edges: list[EdgeSpec] = Field(default_factory=list)

    voice_dna: VoiceDNA | None = Field(default=None, description="Communication fingerprint")
    work_dna: WorkDNA | None = Field(default=None, description="Procedural fingerprint")
    behavioral_rules: list[BehavioralRule] = Field(default_factory=list, description="Trigger -> response patterns")
    contradiction_patterns: list[ContradictionPattern] = Field(default_factory=list, description="Topics they push back on")

    emotional_triggers: list[EmotionalTrigger] = Field(default_factory=list, description="Situations that produce emotional responses")
    emotional_profile: EmotionalProfile | None = Field(default=None, description="Overall emotional wiring")
    contextual_moods: list[ContextualMood] = Field(default_factory=list, description="How mood shifts by social context")


# ── Neo4j label mappings ────────────────────────────────────

NODE_TYPE_TO_LABEL = {
    "traits": "PersonalityTrait",
    "beliefs": "Belief",
    "values": "Value",
    "boundaries": "Boundary",
    "life_events": "LifeEvent",
    "memories": "EpisodicMemory",
    "patterns": "CognitivePattern",
    "social": "SocialPattern",
    "expertise": "KnowledgeDomain",
    "style": "StyleProfile",
    "people": "Entity",
    "places": "Entity",
    "procedural_patterns": "ProceduralPattern",
    "work_loops": "WorkLoop",
    "prompting_styles": "PromptingStyle",
    "technical_gaps": "TechnicalGap",
}

HUB_EDGE_FOR_TYPE = {
    "PersonalityTrait": "HAS_TRAIT",
    "Belief": "HOLDS",
    "Value": "DRIVEN_BY",
    "Boundary": "DRIVEN_BY",
    "LifeEvent": "EXPERIENCED",
    "EpisodicMemory": "EXPERIENCED",
    "CognitivePattern": "HAS_TRAIT",
    "SocialPattern": "HANDLES_CONFLICT_BY",
    "KnowledgeDomain": "EXPERT_IN",
    "StyleProfile": "COMMUNICATES_VIA",
    "ProceduralPattern": "WORKS_BY",
    "WorkLoop": "WORKS_BY",
    "PromptingStyle": "WORKS_BY",
    "TechnicalGap": "WORKS_BY",
}
