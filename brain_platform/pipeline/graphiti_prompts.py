"""Custom Graphiti extraction prompts for personality-domain extraction.

Overrides Graphiti's default extract_text() which hard-codes
"NEVER extract abstract concepts" — a rule that kills personality
trait, value, boundary, and cognitive pattern extraction.

Applied via monkey-patch in GraphitiService.initialize():
    prompt_library.extract_nodes.extract_text = VersionWrapper(custom_extract_text)
"""

from typing import Any

from graphiti_core.prompts.models import Message
from graphiti_core.prompts.prompt_helpers import to_prompt_json


def custom_extract_text(context: dict[str, Any]) -> list[Message]:
    """Personality-aware entity extraction for interview and document text.

    Key differences from Graphiti's default extract_text:
    1. Personality constructs (traits, values, beliefs, boundaries) ARE entities
    2. custom_extraction_instructions placed ABOVE default rules for priority
    3. Explicit examples for personality entity types
    4. Named people always classified as generic Entity (entity_type_id 0)
    5. Synthesis mode suppresses new entity creation
    """
    source_desc = context.get("source_description", "")
    is_synthesis = "synthesis" in source_desc.lower()

    sys_prompt = (
        "You are an entity extraction specialist for personality and life narrative text. "
        "You extract both concrete entities (people, places, organizations) AND personality "
        "constructs (traits, values, beliefs, boundaries, cognitive patterns). "
        "In this domain, a person's values and behavioral patterns are first-class entities — "
        "as real and extractable as people and places."
    )

    # Build synthesis preamble if needed
    synthesis_section = ""
    if is_synthesis:
        synthesis_section = """
SYNTHESIS MODE — IMPORTANT:
You are seeing ALL interview answers together. Individual entities have ALREADY been
extracted from each answer. Your primary job is to identify entities that were MISSED
in individual episodes — not to re-extract what already exists.
- Strongly prefer REFERENCING existing entities over creating duplicates.
- If an entity with the same meaning already exists in a previous episode, do NOT extract it again.
- Focus on personality constructs (PersonalityTrait, Value, Boundary) that only become
  visible when viewing the full interview holistically.
"""

    user_prompt = f"""
{synthesis_section}
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<DOMAIN-SPECIFIC INSTRUCTIONS>
{context['custom_extraction_instructions']}
</DOMAIN-SPECIFIC INSTRUCTIONS>

<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>

<TEXT>
{context['episode_content']}
</TEXT>

Your task: extract **entity nodes** that are **explicitly** mentioned or clearly implied
in the TEXT above. Use the ENTITY TYPES and DOMAIN-SPECIFIC INSTRUCTIONS to guide what
to look for.

PERSONALITY EXTRACTION — These are VALID entities in this domain:
- PersonalityTrait: Behavioral patterns like "conflict-avoidant", "analytical", "risk-averse",
  "empathetic". Both self-reported ("I'm terrible at conflict") and implied by how they
  communicate or tell stories.
- Value: Core drivers like "autonomy", "honesty", "rest as productivity". Look for what they
  sacrifice for, what makes them feel fulfilled, what they organize life around.
- Boundary: Non-negotiables like "won't lie about product impact", "won't take VC money at
  cost of control". Things they refuse to do even at personal cost.
- Belief: Stated positions like "sustainable brands are guilt marketing", "rest is productive".
  Includes contrarian views and changed minds.
- CognitivePattern: Decision frameworks like "two-column pros/cons analysis",
  "consult Priya before deciding", "test in worst conditions first".
- SocialPattern: Relationship styles like "goes silent during conflict",
  "needs 24 hours to process disagreements".
- KnowledgeDomain: Areas of deep expertise like "bioplastics and compostable polymers",
  "mycelium packaging".
- StyleProfile: Communication patterns like "texts in lowercase with dashes",
  "emails are either 3 words or 3 paragraphs".

CRITICAL NAMING RULES:
- Named people (Priya, Marcus, Maya, etc.) → ALWAYS use entity_type_id 0 (generic Entity).
  NEVER classify a person as PersonalityTrait, Belief, Value, CognitivePattern, or any
  personality construct type. If someone is mentioned in context of a pattern, create the
  PATTERN as a separate entity and connect it to the person via INVOLVES.
  WRONG: "Priya" as CognitivePattern (because consulting Priya is a pattern)
  RIGHT: "consults friend before deciding" as CognitivePattern + INVOLVES → "Priya" as Entity
- THE_USER (the person being interviewed) → entity_type_id 0 (generic Entity).
  First-person references (I, me, my) refer to THE_USER. ALWAYS extract THE_USER.
- Personality constructs → name them CONCISELY, UNDER 50 CHARACTERS:
  GOOD: "conflict-avoidant", "autonomy", "won't lie about product impact"
  BAD: "Maya's tendency to go silent during conflict situations" (too long, sentence)
  BAD: "feeling of having to perform two versions of self growing up" (too long)
  RIGHT: "dual-identity performer" (concept, under 50 chars)
- Values → use the value WORD: "autonomy" not "value of autonomy"
- Boundaries → describe the LINE concisely: "won't lie about product impact"
- Put full context in the entity summary, NOT in the entity name

PERSON DEDUP:
If a person's name (e.g., "Priya", "Marcus") appeared in a PREVIOUS MESSAGE,
a new mention is THE SAME entity. Do not create a new node for them.
People are ALWAYS entity_type_id 0 regardless of context.

THE_USER HUB — CRITICAL:
ALWAYS extract THE_USER as an entity (entity_type_id 0) in EVERY episode.
THE_USER is the subject of every interview answer and should connect to every
personality construct extracted via typed edges (HAS_TRAIT, HOLDS, DRIVEN_BY,
EXPERIENCED, EXPERT_IN, HANDLES_CONFLICT_BY, COMMUNICATES_VIA).

EXCLUSION RULES (still apply):
- NEVER extract pronouns (I, me, you, he, she, they, we, them, this, that)
- NEVER extract generic common nouns (day, life, people, work, stuff, things, time)
- NEVER extract bare kinship terms (dad, mom, friend, boss) — qualify them:
  "Priya" not "friend", "Marcus" not "co-founder" (unless unnamed)
- NEVER extract sentence fragments or clauses as entity names
- NEVER extract duplicate references to the same entity within this text
- When in doubt about a generic word, do NOT extract

SPECIFICITY:
- Use the **most specific form** from the text: "mycelium packaging" not "packaging"
- If extracting an event, include enough context to be identifiable:
  "quitting the agency job" not "quitting"
- Ask: "Is this a distinct, nameable facet of this person's personality, a specific
  entity they reference, or a concrete event in their life? If yes, extract it."

FORMATTING:
- Be explicit and unambiguous in naming entities
- Each entity gets exactly ONE entity_type_id from the ENTITY TYPES list
- If no custom type fits, use entity_type_id 0 (generic Entity)
"""
    return [
        Message(role="system", content=sys_prompt),
        Message(role="user", content=user_prompt),
    ]


def apply_prompt_overrides() -> None:
    """Apply Beam Mind's custom prompts to Graphiti's prompt library.

    Call this once before any Graphiti add_episode() calls.
    Safe to call multiple times (idempotent).
    """
    from graphiti_core.prompts.lib import VersionWrapper, prompt_library

    prompt_library.extract_nodes.extract_text = VersionWrapper(custom_extract_text)
