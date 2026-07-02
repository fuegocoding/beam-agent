"""Brain File JSON-LD v2 Schema.

The canonical data structure for a person's portable cognitive profile.
Serializes to JSON-LD for interoperability with knowledge graph tools.

v2.1.0 (plan/10): adds procedural memory layer (ProceduralPattern, WorkLoop,
PromptingStyle, TechnicalGap) + WorkDNA + CalibrationKnobs. All new fields
default to empty/None so existing v2.0.0 brain files load unchanged.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# Current brain file schema version.
BRAIN_FILE_SCHEMA_VERSION = "2.2.0"


# === Sub-schemas ===


class PersonMetadata(BaseModel):
    """Optional metadata for persona-type brain files (historical figures,
    fictional characters, composite archetypes, or user clones)."""

    persona_type: Literal["real_person", "fictional_character", "composite_archetype", "user_clone"] = "composite_archetype"
    birth_date: str | None = None
    death_date: str | None = None
    era: str | None = None
    nationality: str | None = None
    occupations: list[str] = []
    notable_works: list[str] = []
    primary_sources: list[str] = []
    is_deceased: bool = True
    right_of_publicity_status: Literal["public_domain", "licensed", "unclear"] = "unclear"
    disclaimer_required: str = ""
    source_attribution: str = ""


class BrainFileMetadata(BaseModel):
    version: str = BRAIN_FILE_SCHEMA_VERSION
    created_at: datetime
    updated_at: datetime
    user_id: str
    schema_version: str = BRAIN_FILE_SCHEMA_VERSION
    source_count: int = 0
    graphiti_group_id: str = ""


class KnowledgeDomain(BaseModel):
    topic: str
    confidence: float = Field(ge=0, le=1)
    source_count: int = 0
    key_entities: list[str] = []
    community_summary: str = ""


class CognitivePattern(BaseModel):
    situation_type: str
    typical_response: str
    emotional_tendency: str = ""


class BehavioralRuleEntry(BaseModel):
    trigger: str
    response: str
    exceptions: str = ""


class ContradictionPatternEntry(BaseModel):
    topic: str
    stance: str
    how_they_push_back: str = ""


class VoiceDNAEntry(BaseModel):
    characteristic_phrases: list[str] = []
    phrases_to_avoid: list[str] = []
    punctuation_and_formatting: str = ""
    emoji_usage: str = ""
    humor_style: str = ""
    response_length_pattern: str = ""
    formality_range: str = ""
    filler_words: list[str] = []
    storytelling_style: str = ""
    listener_vs_talker: str = ""


class PersonalityProfile(BaseModel):
    communication_style: str = "analytical"
    formality: float = Field(0.5, ge=0, le=1)
    humor_frequency: float = Field(0.1, ge=0, le=1)
    empathy_indicators: float = Field(0.5, ge=0, le=1)
    values: list[str] = []
    core_beliefs: list[str] = []
    cognitive_patterns: list[CognitivePattern] = []


class WritingStyle(BaseModel):
    avg_sentence_length: float = 0
    vocabulary_level: str = "intermediate"
    common_phrases: list[str] = []
    tone: str = "neutral"


class StyleEmbedding(BaseModel):
    authorship_vector: list[float] = []
    model: str = "AnnaWegmann/Style-Embedding"
    sample_count: int = 0


class GraphNode(BaseModel):
    id: str
    type: str = "Entity"
    label: str
    attributes: dict = {}
    summary: str = ""
    labels: list[str] = []
    sensitivity: str = "public"


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    fact: str = ""
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    weight: float = 1.0


class GraphCluster(BaseModel):
    id: str
    name: str
    member_node_ids: list[str] = []
    summary: str = ""


class KnowledgeGraph(BaseModel):
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    clusters: list[GraphCluster] = []


class BeliefHistory(BaseModel):
    topic: str
    position: str
    confidence: float = Field(0.5, ge=0, le=1)
    event_time: datetime | None = None
    ingestion_time: datetime | None = None
    superseded_by: str | None = None


class TemporalMetadata(BaseModel):
    belief_history: list[BeliefHistory] = []


class CausalLink(BaseModel):
    type: str = "LEADS_TO"
    target_memory: str
    description: str = ""


class EpisodicMemoryEntry(BaseModel):
    id: str = ""  # Graph node UUID when known; empty for raw-text-derived memories
    event: str
    temporal_context: str = ""
    spatial_context: str = ""
    emotional_valence: float = Field(0, ge=-1, le=1)
    activation_level: float = Field(0.5, ge=0, le=1)
    linked_semantic_nodes: list[str] = []
    causal_links: list[CausalLink] = []
    sensitivity: str = "personal"


class EmotionalTriggerEntry(BaseModel):
    trigger: str
    emotion: str = ""
    intensity: float = Field(0.5, ge=0, le=1)
    expression: str = ""
    context: str = ""
    sensitivity: str = "personal"


class EmotionalProfileEntry(BaseModel):
    baseline_mood: str = ""
    processing_style: str = ""
    reaction_speed: str = ""
    recovery_pattern: str = ""
    energy_sources: list[str] = []
    energy_drains: list[str] = []
    emotional_tells: list[str] = []


class ContextualMoodEntry(BaseModel):
    context: str = ""
    mood: str = ""
    guard_level: float = Field(0.5, ge=0, le=1)
    energy_level: float = Field(0.5, ge=0, le=1)
    sensitivity: str = "personal"


class SourceManifestEntry(BaseModel):
    source_type: str
    source_id: str = ""          # UUID of the DataSource record
    title: str = ""              # Filename, URL, journal week, or user-given title
    import_date: datetime
    record_count: int = 0
    size_bytes: int = 0


# === Procedural memory layer (plan/10 §6) ===


class CalibrationKnobsEntry(BaseModel):
    """Per-node runtime calibration. Defaults give the 'improved' projection;
    setting tone_temperature=1.0, factual_correction=False, harm_gate='off'
    yields the raw research projection."""
    tone_temperature: float = Field(1.0, ge=0.0, le=2.0)
    confidence_clip: float | None = Field(None, ge=0.0, le=1.0)
    contradiction_policy: Literal["aspirational", "observed", "both"] = "both"
    factual_correction: bool = True
    harm_gate: Literal["off", "soften", "exclude"] = "soften"


class ProceduralPatternEntry(BaseModel):
    name: str
    domain: str = ""
    situation: str = ""
    approach: str = ""
    tells: list[str] = []
    anti_pattern: str = ""
    source: Literal["interview", "artifact", "synthesis"] = "interview"
    sample_size: int = 1
    summary: str = ""
    sensitivity: str = "personal"
    calibration: CalibrationKnobsEntry | None = None


class WorkLoopEntry(BaseModel):
    name: str
    trigger: str = ""
    steps: list[str] = []
    stop_condition: str = ""
    recovery: str = ""
    source: Literal["interview", "artifact", "synthesis"] = "interview"
    summary: str = ""
    sensitivity: str = "personal"
    calibration: CalibrationKnobsEntry | None = None


class PromptingStyleEntry(BaseModel):
    name: str
    structure: str = ""
    length_preference: str = ""
    correction_style: str = ""
    constraint_phrasing: str = ""
    examples_excerpts: list[str] = []
    source: Literal["interview", "artifact", "synthesis"] = "interview"
    summary: str = ""
    sensitivity: str = "personal"
    calibration: CalibrationKnobsEntry | None = None


class TechnicalGapEntry(BaseModel):
    name: str
    gap_type: Literal["avoids", "unaware", "outdated"] = "avoids"
    evidence: str = ""
    aspirational: bool = False
    summary: str = ""
    sensitivity: str = "personal"
    calibration: CalibrationKnobsEntry | None = None


class UserInstructionsEntry(BaseModel):
    """Structured directives from user's AI instructions files.

    Stored verbatim — these are injected directly into the agent system prompt.
    """
    communication_rules: list[str] = []
    behavioral_rules: list[str] = []
    domain_constraints: list[str] = []
    workflow_rules: list[str] = []
    boundaries: list[str] = []


class WorkDNAEntry(BaseModel):
    """Procedural fingerprint — parallel to VoiceDNAEntry. Empty fields are
    valid: the interview may not include work-session content."""
    decomposition_style: str = ""
    error_taxonomy: str = ""
    debugging_approach: str = ""
    review_depth: str = ""
    documentation_habit: str = ""
    abstraction_timing: str = ""
    risk_posture: str = ""
    delegation_style: str = ""
    stop_conditions: str = ""


# === Top-level Schema ===


class BrainEmbeddings(BaseModel):
    """Companion embeddings for conceptual brain filtering at runtime.

    Each list is aligned by index with the corresponding brain file section.
    Stored as a separate S3 object to avoid bloating the main brain file.
    """
    knowledge_nodes: list[list[float]] = []
    knowledge_domains: list[list[float]] = []
    episodic_memories: list[list[float]] = []


class BrainFileSchema(BaseModel):
    """Brain File JSON-LD v2 — the portable cognitive profile."""

    metadata: BrainFileMetadata
    knowledge_domains: list[KnowledgeDomain] = []
    personality_profile: PersonalityProfile = PersonalityProfile()
    writing_style: WritingStyle = WritingStyle()
    style_embedding: StyleEmbedding = StyleEmbedding()
    knowledge_graph: KnowledgeGraph = KnowledgeGraph()
    temporal_metadata: TemporalMetadata = TemporalMetadata()
    episodic_memories: list[EpisodicMemoryEntry] = []
    source_manifest: list[SourceManifestEntry] = []

    # Clone specification (Pass 3) — behavioral operating manual
    voice_dna: VoiceDNAEntry = VoiceDNAEntry()
    behavioral_rules: list[BehavioralRuleEntry] = []
    contradiction_patterns: list[ContradictionPatternEntry] = []

    # Emotional dynamics (Pass 4)
    emotional_triggers: list[EmotionalTriggerEntry] = []
    emotional_profile: EmotionalProfileEntry = EmotionalProfileEntry()
    contextual_moods: list[ContextualMoodEntry] = []

    # Procedural memory layer (plan/10) — empty by default; populated only when
    # the brain has work-session interview data or artifact-derived patterns.
    procedural_patterns: list[ProceduralPatternEntry] = []
    work_loops: list[WorkLoopEntry] = []
    prompting_styles: list[PromptingStyleEntry] = []
    technical_gaps: list[TechnicalGapEntry] = []
    work_dna: WorkDNAEntry = WorkDNAEntry()

    # User-authored directives (dual-layer instructions pipeline)
    user_instructions: UserInstructionsEntry = UserInstructionsEntry()
    person_metadata: PersonMetadata | None = None

    @field_validator("person_metadata", mode="after")
    @classmethod
    def validate_person_metadata(cls, v: PersonMetadata | None) -> PersonMetadata | None:
        """Enforce validation rules based on persona_type."""
        if v is None:
            return None
        if v.persona_type == "real_person":
            if not v.primary_sources:
                raise ValueError("real_person brain files must have primary_sources")
        if v.persona_type == "fictional_character":
            if not v.notable_works:
                raise ValueError("fictional_character brain files must have notable_works (source work)")
        return v

    def to_jsonld(self) -> dict:
        """Serialize to JSON-LD with @context."""
        data = self.model_dump(mode="json")
        data["@context"] = {
            "@vocab": "https://beammind.dev/schema/v2/",
            "schema": "http://schema.org/",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        }
        data["@type"] = "BrainFile"
        return data
