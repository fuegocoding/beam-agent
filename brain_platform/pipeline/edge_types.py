"""Custom Graphiti edge types for the Beam Mind knowledge graph.

These define typed relationships that Graphiti's LLM will extract between
entity nodes. Combined with edge_type_map, they constrain which entity
types can connect via which relationship types — preventing nonsense
triples and guiding extraction toward personality-meaningful connections.

Docstrings are used by the LLM for relationship classification.
"""

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Edge type models — each describes a relationship kind
# ──────────────────────────────────────────────────────────────

class ShapedBy(BaseModel):
    """A causal relationship where a life event or experience shaped a belief,
    value, or personality trait. The source entity (event) causally influenced
    the formation or reinforcement of the target entity."""

    causal_strength: float = Field(
        description="How strongly the event shaped the target, 0-1",
        ge=0, le=1, default=0.5,
    )


class EnforcedAs(BaseModel):
    """A value that manifests as a specific boundary or non-negotiable rule.
    The person holds this value (source) and enforces it as this boundary (target)."""
    pass


class EvolvedInto(BaseModel):
    """A belief that changed over time — the old belief (source) evolved into
    the new belief (target). Captures intellectual growth and changed minds.
    Look for: 'I used to think X, now I think Y'."""

    trigger: str = Field(
        description="What caused the belief to change",
        default="",
    )


class Guides(BaseModel):
    """A value or belief that guides a decision-making pattern. The source
    (value/belief) informs how the person approaches the target (cognitive pattern)."""
    pass


class LearnedFrom(BaseModel):
    """A lesson or pattern learned from a specific life event or failure.
    The source (event) taught the person the target (cognitive pattern or belief)."""
    pass


class ExpressedThrough(BaseModel):
    """A personality trait that manifests in a specific communication style.
    The source (trait) is expressed through the target (style marker)."""
    pass


class TestedBy(BaseModel):
    """A boundary that was tested by a specific life event. The source (boundary)
    was put to the test by the target (event) — and the person either held
    the line or discovered the boundary."""

    held: bool = Field(
        description="Whether the person maintained the boundary when tested",
        default=True,
    )


class Involves(BaseModel):
    """A life event or memory that involves a specific person, place, or
    organization. Connects narrative entities to the events they participated in."""

    role: str = Field(
        description="The role this entity played in the event (e.g., 'advisor', 'antagonist', 'location')",
        default="",
    )


class InformedBy(BaseModel):
    """Deep knowledge in a domain that informs or supports a belief or value.
    The source (expertise) provides evidence for the target (belief)."""
    pass


# ── THE_USER hub edge types ─────────────────────────────────
# Connect THE_USER (generic Entity) to every personality construct.


class HasTrait(BaseModel):
    """THE_USER possesses this personality trait. Source is the person (Entity),
    target is the trait they exhibit (PersonalityTrait or CognitivePattern)."""
    pass


class Holds(BaseModel):
    """THE_USER holds this belief or opinion. Source is the person (Entity),
    target is the belief they maintain (Belief)."""
    pass


class DrivenBy(BaseModel):
    """THE_USER is driven by this core value or boundary. Source is the person
    (Entity), target is the value or boundary that motivates them."""
    pass


class Experienced(BaseModel):
    """THE_USER experienced this life event or memory. Source is the person
    (Entity), target is the event or memory (LifeEvent or EpisodicMemory)."""
    pass


class ExpertIn(BaseModel):
    """THE_USER has deep expertise in this knowledge domain. Source is the
    person (Entity), target is their area of expertise (KnowledgeDomain)."""
    pass


class HandlesConflictBy(BaseModel):
    """THE_USER handles social situations using this pattern. Source is the
    person (Entity), target is their social approach (SocialPattern)."""
    pass


class CommunicatesVia(BaseModel):
    """THE_USER communicates using this style. Source is the person (Entity),
    target is their communication pattern (StyleProfile)."""
    pass


# ── Procedural memory edges (plan/10 §6.2) ─────────────────


class OperatesIn(BaseModel):
    """A procedural pattern operates within a knowledge domain. Source is the
    pattern (ProceduralPattern/WorkLoop/PromptingStyle), target is the domain
    (KnowledgeDomain). E.g. hypothesis-first-debugging OPERATES_IN backend-systems."""
    pass


class ExplainedBy(BaseModel):
    """A procedural pattern is grounded in a personality node — a value,
    belief, or trait that explains *why* the person works this way. Source is
    the procedural pattern, target is the personality node."""
    pass


class AspiresTo(BaseModel):
    """A self-reported aspiration — a value or trait that points to a procedural
    pattern the person says they follow. Pairs with OBSERVED_AS for contradiction
    detection. Source is the personality node (Trait/Value), target is the
    self-reported procedural pattern (source: 'interview')."""
    pass


class Contradicts(BaseModel):
    """An observed-vs-aspirational gap between two procedural patterns. Source is
    the artifact-derived pattern (source: 'artifact'), target is the self-reported
    pattern (source: 'interview'). Stored as data per TinyStyler — not error."""

    discrepancy: str = Field(
        description="Brief description of how the observed pattern differs from the aspirational one",
        default="",
    )


class WorksBy(BaseModel):
    """THE_USER works using this procedural pattern. Source is the person (Entity),
    target is the procedural pattern (ProceduralPattern, WorkLoop, PromptingStyle,
    or TechnicalGap). Hub edge analogous to HAS_TRAIT for personality nodes."""
    pass


# ──────────────────────────────────────────────────────────────
# Registries passed to Graphiti add_episode()
# ──────────────────────────────────────────────────────────────

BEAM_MIND_EDGE_TYPES: dict[str, type[BaseModel]] = {
    "SHAPED_BY": ShapedBy,
    "ENFORCED_AS": EnforcedAs,
    "EVOLVED_INTO": EvolvedInto,
    "GUIDES": Guides,
    "LEARNED_FROM": LearnedFrom,
    "EXPRESSED_THROUGH": ExpressedThrough,
    "TESTED_BY": TestedBy,
    "INVOLVES": Involves,
    "INFORMED_BY": InformedBy,
    # THE_USER hub edges
    "HAS_TRAIT": HasTrait,
    "HOLDS": Holds,
    "DRIVEN_BY": DrivenBy,
    "EXPERIENCED": Experienced,
    "EXPERT_IN": ExpertIn,
    "HANDLES_CONFLICT_BY": HandlesConflictBy,
    "COMMUNICATES_VIA": CommunicatesVia,
    # Procedural memory edges (plan/10)
    "OPERATES_IN": OperatesIn,
    "EXPLAINED_BY": ExplainedBy,
    "ASPIRES_TO": AspiresTo,
    "CONTRADICTS": Contradicts,
    "WORKS_BY": WorksBy,
}

# Constrains which entity type pairs can have which edge types.
# Graphiti uses this to filter extraction — the LLM only sees
# relevant relationship options for each node pair signature.
BEAM_MIND_EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    # Life events shape beliefs, values, and traits
    ("LifeEvent", "Belief"): ["SHAPED_BY"],
    ("LifeEvent", "Value"): ["SHAPED_BY"],
    ("LifeEvent", "PersonalityTrait"): ["SHAPED_BY"],
    ("EpisodicMemory", "Belief"): ["SHAPED_BY"],
    ("EpisodicMemory", "Value"): ["SHAPED_BY"],

    # Values enforce boundaries
    ("Value", "Boundary"): ["ENFORCED_AS"],
    ("Belief", "Boundary"): ["ENFORCED_AS"],

    # Beliefs evolve over time
    ("Belief", "Belief"): ["EVOLVED_INTO"],

    # Values and beliefs guide decision patterns
    ("Value", "CognitivePattern"): ["GUIDES"],
    ("Belief", "CognitivePattern"): ["GUIDES"],

    # Events teach lessons / patterns
    ("LifeEvent", "CognitivePattern"): ["LEARNED_FROM"],
    ("EpisodicMemory", "CognitivePattern"): ["LEARNED_FROM"],

    # Traits show in communication style
    ("PersonalityTrait", "StyleProfile"): ["EXPRESSED_THROUGH"],
    ("SocialPattern", "StyleProfile"): ["EXPRESSED_THROUGH"],

    # Boundaries get tested by events
    ("Boundary", "LifeEvent"): ["TESTED_BY"],
    ("Boundary", "EpisodicMemory"): ["TESTED_BY"],

    # Events involve people and places
    ("LifeEvent", "Entity"): ["INVOLVES"],
    ("EpisodicMemory", "Entity"): ["INVOLVES"],

    # Expertise informs beliefs
    ("KnowledgeDomain", "Belief"): ["INFORMED_BY"],
    ("KnowledgeDomain", "Value"): ["INFORMED_BY"],
    ("KnowledgeDomain", "CognitivePattern"): ["INFORMED_BY"],

    # THE_USER hub — connect Entity (THE_USER) to personality constructs
    ("Entity", "PersonalityTrait"): ["HAS_TRAIT"],
    ("Entity", "Belief"): ["HOLDS"],
    ("Entity", "Value"): ["DRIVEN_BY"],
    ("Entity", "Boundary"): ["DRIVEN_BY"],
    ("Entity", "LifeEvent"): ["EXPERIENCED"],
    ("Entity", "EpisodicMemory"): ["EXPERIENCED"],
    ("Entity", "KnowledgeDomain"): ["EXPERT_IN"],
    ("Entity", "SocialPattern"): ["HANDLES_CONFLICT_BY"],
    ("Entity", "StyleProfile"): ["COMMUNICATES_VIA"],
    ("Entity", "CognitivePattern"): ["HAS_TRAIT"],

    # ── Procedural memory pairs (plan/10) ──
    # Procedural patterns operate within knowledge domains
    ("ProceduralPattern", "KnowledgeDomain"): ["OPERATES_IN"],
    ("WorkLoop", "KnowledgeDomain"): ["OPERATES_IN"],
    ("PromptingStyle", "KnowledgeDomain"): ["OPERATES_IN"],

    # Procedural patterns explained by personality nodes
    ("ProceduralPattern", "Belief"): ["EXPLAINED_BY"],
    ("ProceduralPattern", "Value"): ["EXPLAINED_BY"],
    ("ProceduralPattern", "PersonalityTrait"): ["EXPLAINED_BY"],
    ("WorkLoop", "Value"): ["EXPLAINED_BY"],
    ("WorkLoop", "PersonalityTrait"): ["EXPLAINED_BY"],

    # Self-reported aspirations from personality → procedural patterns
    ("PersonalityTrait", "ProceduralPattern"): ["ASPIRES_TO"],
    ("Value", "ProceduralPattern"): ["ASPIRES_TO"],
    ("Value", "WorkLoop"): ["ASPIRES_TO"],

    # Observed-vs-aspirational gap (artifact-derived ↔ self-reported)
    ("ProceduralPattern", "ProceduralPattern"): ["CONTRADICTS"],
    ("WorkLoop", "WorkLoop"): ["CONTRADICTS"],
    ("PromptingStyle", "PromptingStyle"): ["CONTRADICTS"],

    # THE_USER hub — connect Entity to procedural nodes
    ("Entity", "ProceduralPattern"): ["WORKS_BY"],
    ("Entity", "WorkLoop"): ["WORKS_BY"],
    ("Entity", "PromptingStyle"): ["WORKS_BY"],
    ("Entity", "TechnicalGap"): ["WORKS_BY"],
}
