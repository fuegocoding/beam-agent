"""Dimension-specific extraction instructions for Graphiti.

Each interview question targets a personality dimension. These instructions
are passed as custom_extraction_instructions to Graphiti's add_episode(),
telling its internal LLM exactly which entity types and relationships to
look for in each answer.

The old INTERVIEW_FACT_EXTRACTION_PROMPT is replaced — Graphiti does its
own extraction internally. We just need to guide it.
"""

# Base instructions prepended to every dimension's specific instructions.
# Tells Graphiti about the interview context and THE_USER convention.
BASE_INTERVIEW_INSTRUCTIONS = """\
This text is from a guided interview used to build a digital personality replica.
The person answering is referred to as THE_USER — all first-person references
(I, me, my, we) should be attributed to THE_USER as an entity.

ALWAYS extract THE_USER as an entity in every episode. THE_USER is the person
being interviewed — they should connect to every personality construct extracted
via hub edges (HAS_TRAIT, HOLDS, DRIVEN_BY, EXPERIENCED, EXPERT_IN, etc.).

Extract entities with their full context — not just names but the reasoning,
emotions, and stories behind them. A belief without its reasoning is incomplete.
A memory without its emotional context is hollow.

When the person mentions other people by name, extract them as entities
(ALWAYS entity_type_id 0, generic Entity — never as personality construct types)
and connect them to the relevant events or patterns via INVOLVES relationships.
If "Priya" appeared in a previous message, a new mention is THE SAME entity.

Entity names must be under 50 characters — use the concept, not a sentence.
Put full descriptions in the entity summary, not the name.
"""

# Per-dimension instructions that target specific entity types and edge types.
DIMENSION_INSTRUCTIONS: dict[str, str] = {
    "episodic_memory": BASE_INTERVIEW_INSTRUCTIONS + """
DIMENSION: Life Story & Experiences

PRIMARY ENTITY TYPES TO EXTRACT:
- LifeEvent: Pivotal decisions, turning points, formative experiences.
  Include temporal_context (when), emotional_valence (-1 to 1), and life_impact.
- EpisodicMemory: Specific stories with who/what/when/where detail.

RELATIONSHIPS TO LOOK FOR:
- LifeEvent --SHAPED_BY--> Belief: When the person explains how an event
  changed what they believe or think about the world.
- LifeEvent --SHAPED_BY--> Value: When an experience led them to care about
  something deeply.
- LifeEvent --INVOLVES--> [Person/Place]: Named people and places in the story.

ALSO EXTRACT (if present):
- PersonalityTrait: Traits implied by how they tell the story (not just content).
- StyleProfile: Communication patterns visible in their narrative style.
""",

    "core_beliefs": BASE_INTERVIEW_INSTRUCTIONS + """
DIMENSION: Beliefs & Identity

PRIMARY ENTITY TYPES TO EXTRACT:
- Belief: Stated positions with reasoning chains. Include topic, position,
  confidence (0-1), and the reasoning/evidence behind it.
- If they mention a belief they USED TO hold and now reject, create TWO Belief
  entities and connect them with an EVOLVED_INTO edge. Include what triggered
  the change.

RELATIONSHIPS TO LOOK FOR:
- Belief --EVOLVED_INTO--> Belief: When they describe changing their mind.
  Include the trigger (what caused the shift).
- EpisodicMemory --SHAPED_BY--> Belief: The specific experience that formed
  or reinforced this belief.
- KnowledgeDomain --INFORMED_BY--> Belief: When expertise supports a belief.

ALSO EXTRACT (if present):
- PersonalityTrait: Intellectual flexibility, contrarianism, conviction strength.
""",

    "decision_making": BASE_INTERVIEW_INSTRUCTIONS + """
DIMENSION: How They Think & Decide

PRIMARY ENTITY TYPES TO EXTRACT:
- CognitivePattern: Decision-making templates — how they approach hard choices.
  Include situation_type, typical_response, and outcome.
- LifeEvent: The specific decisions or failures they describe.

RELATIONSHIPS TO LOOK FOR:
- Value --GUIDES--> CognitivePattern: When a value shapes how they decide.
- LifeEvent --LEARNED_FROM--> CognitivePattern: When a specific failure or
  experience taught them a new approach.
- Belief --GUIDES--> CognitivePattern: When a belief about the world informs
  their decision process.

ALSO EXTRACT (if present):
- PersonalityTrait: Analytical vs intuitive, risk tolerance, consultation habits.
""",

    "values": BASE_INTERVIEW_INSTRUCTIONS + """
DIMENSION: What Drives Them

PRIMARY ENTITY TYPES TO EXTRACT:
- Value: Core values with concrete evidence — not just abstract words like
  "honesty" but specific examples of the value in action. Include value_name,
  importance (0-1), and evidence.

RELATIONSHIPS TO LOOK FOR:
- Value --ENFORCED_AS--> Boundary: When a value manifests as a specific rule
  or line they won't cross.
- Value --GUIDES--> CognitivePattern: When a value shapes their decision-making.
- LifeEvent --SHAPED_BY--> Value: The experience that made them care about this.

ALSO EXTRACT (if present):
- Belief: Value statements that are also beliefs ("I believe X matters").
""",

    "boundaries": BASE_INTERVIEW_INSTRUCTIONS + """
DIMENSION: Lines They Won't Cross

PRIMARY ENTITY TYPES TO EXTRACT:
- Boundary: Non-negotiables, anti-values, refusals. Include description,
  whether it's been tested, and what it cost them.
- LifeEvent: The specific situation where the boundary was tested.

RELATIONSHIPS TO LOOK FOR:
- Boundary --TESTED_BY--> LifeEvent: When a specific event tested this boundary.
- Value --ENFORCED_AS--> Boundary: The underlying value behind the boundary.
- Belief --ENFORCED_AS--> Boundary: The belief that justifies the boundary.

ALSO EXTRACT (if present):
- PersonalityTrait: Stubbornness, principled behavior, moral courage.
""",

    "social_orientation": BASE_INTERVIEW_INSTRUCTIONS + """
DIMENSION: Relationships & Conflict

PRIMARY ENTITY TYPES TO EXTRACT:
- SocialPattern: Conflict style, emotional regulation approach, trust patterns.
  Include context (what kind of social situation), approach (what they do),
  and emotional_response.
- EpisodicMemory: The specific conflict or relationship event they describe.

RELATIONSHIPS TO LOOK FOR:
- LifeEvent --SHAPED_BY--> SocialPattern: When an experience shaped how they
  handle relationships.
- SocialPattern --EXPRESSED_THROUGH--> StyleProfile: When their social style
  shows in how they communicate.

ALSO EXTRACT (if present):
- PersonalityTrait: Avoidant, confrontational, mediating, empathetic patterns.
- Boundary: Social boundaries (what they won't tolerate in relationships).
""",

    "knowledge_domains": BASE_INTERVIEW_INSTRUCTIONS + """
DIMENSION: Expertise & Deep Knowledge

PRIMARY ENTITY TYPES TO EXTRACT:
- KnowledgeDomain: Areas of deep expertise. Include topic, depth (0-1), and
  a misconception they can correct.

RELATIONSHIPS TO LOOK FOR:
- KnowledgeDomain --INFORMED_BY--> Belief: When expertise supports or creates
  a belief about the world.
- KnowledgeDomain --INFORMED_BY--> Value: When deep knowledge shapes what
  they care about.
- LifeEvent --SHAPED_BY--> KnowledgeDomain: The experience that drew them
  into this domain.

ALSO EXTRACT (if present):
- PersonalityTrait: Intellectual curiosity, teaching ability, depth vs breadth.
""",

    "procedural_memory": BASE_INTERVIEW_INSTRUCTIONS + """
DIMENSION: How They Work (procedural memory, plan/10)

This dimension captures HOW the person reasons, prompts, debugs, and reviews
— not WHAT tech they use. The brain runs through AI agents that handle
tooling, so we need to capture the meta-method: decomposition style, error
taxonomy, prompting structure, review criteria, recovery loops.

CRITICAL CONSTRAINT: do NOT enumerate languages, libraries, frameworks, or
specific tools. If the only signal is "they use Python", drop it. We capture
HOW they think, not WHAT stack they use.

PRIMARY ENTITY TYPES TO EXTRACT:
- ProceduralPattern: Role-agnostic patterns in how they work.
  Examples: "hypothesis-first debugging", "small-PR refactoring",
  "spec-then-build". Include pattern_name, domain (free text like
  "debugging" or "design critique"), approach (the method),
  anti_pattern (what they explicitly do NOT do).
  Set source="interview" for self-narrated patterns.

- WorkLoop: Ordered iteration loops they describe. Look for "first I X,
  then Y, if it doesn't work I Z" narrations. Include loop_name, trigger,
  steps_summary (ordered), stop_condition, recovery (what they do when
  blocked). Set source="interview".

- PromptingStyle: How they direct AI or delegate to people. Capture
  meta-structure: spec-first vs example-first, length preference, correction
  style (gentle / direct / restart), constraint phrasing. NEVER capture
  specific models or tools. Set source="interview".

- TechnicalGap: Knowledge gaps they reveal. gap_type: "avoids" (deliberate),
  "unaware" (unknown unknown), "outdated" (stale view). Mark
  aspirational=true ONLY if they explicitly say they want to close the gap.
  Non-aspirational gaps are personality data — preserve them, don't filter.

RELATIONSHIPS TO LOOK FOR:
- ProceduralPattern --OPERATES_IN--> KnowledgeDomain: When the pattern is
  scoped to a particular domain (e.g. "small-PR refactoring" OPERATES_IN
  "frontend").
- ProceduralPattern --EXPLAINED_BY--> Belief / Value: When a pattern is
  grounded in a personality node ("I write small PRs because I value
  reversibility").
- PersonalityTrait / Value --ASPIRES_TO--> ProceduralPattern: Self-reported
  aspirations from personality nodes to interview-source procedural
  patterns. Pairs with OBSERVED_AS edges from the artifact ingester
  (Phase B) to detect aspirational/observed gaps via CONTRADICTS edges.

ALSO EXTRACT (if present):
- PersonalityTrait: Traits visible in HOW they describe their work
  (methodical, scrappy, perfectionist, pragmatic).
- CognitivePattern: Decision frameworks that show up in their work
  narration.
""",

    "communication_style": BASE_INTERVIEW_INSTRUCTIONS + """
DIMENSION: How They Communicate

PRIMARY ENTITY TYPES TO EXTRACT:
- StyleProfile: Communication characteristics — texting style, email patterns,
  explanation approach, humor, code-switching. Include formality (0-1),
  vocabulary_level, and communication_style category.
- PersonalityTrait: Meta-awareness of their own communication patterns.

RELATIONSHIPS TO LOOK FOR:
- PersonalityTrait --EXPRESSED_THROUGH--> StyleProfile: When a personality
  trait manifests as a specific communication habit.
- SocialPattern --EXPRESSED_THROUGH--> StyleProfile: When their social style
  (e.g., directness in conflict) shows in how they write/speak.

ALSO EXTRACT (if present):
- Belief: Beliefs about communication ("I think precision is a form of respect").
""",
}


SYNTHESIS_INSTRUCTIONS = BASE_INTERVIEW_INSTRUCTIONS + """
SYNTHESIS PASS — CROSS-DIMENSIONAL CONNECTIONS

You are seeing the COMPLETE interview — all questions and answers together.
Individual entities (beliefs, values, memories, etc.) have already been
extracted from each answer separately. Your job now is ONLY to find the
CONNECTIONS BETWEEN them that span across different answers.

Focus EXCLUSIVELY on cross-answer relationships:

1. **SHAPED_BY**: Which life events (from life story questions) shaped which
   beliefs or values (from belief/value questions)? The person often won't
   state this directly — you must infer it. Example: a childhood experience
   of being an outsider may have shaped an adult belief about authenticity.

2. **ENFORCED_AS**: Which values (from values questions) manifest as which
   boundaries (from boundaries questions)? Example: valuing honesty →
   refusing to lie about products.

3. **EVOLVED_INTO**: Did any belief mentioned in the "changed beliefs" answer
   connect to a current belief mentioned elsewhere? Create the temporal chain.

4. **GUIDES**: Which values or beliefs guide the decision-making patterns
   described in the decisions questions?

5. **LEARNED_FROM**: Which specific failures or events taught the cognitive
   patterns they describe?

6. **EXPRESSED_THROUGH**: Which personality traits show up in the communication
   style they describe?

7. **TESTED_BY**: Were any boundaries tested by events mentioned in the life
   story or conflict questions?

8. **INFORMED_BY**: Does their expertise area inform their beliefs or values?

9. **INVOLVES**: Connect named people to all the events and patterns they
   appear in across the full interview.

DO NOT re-extract entities that already exist — focus on the EDGES between them.
Look for causal chains: event → belief → value → boundary → tested by event.
Look for people who bridge multiple life areas.
Look for themes (honesty, fear, family, growth) that connect across dimensions.
"""


def get_dimension_instructions(dimension: str) -> str:
    """Get extraction instructions for a specific interview dimension.

    Falls back to base instructions if the dimension is unknown.
    """
    return DIMENSION_INSTRUCTIONS.get(dimension, BASE_INTERVIEW_INSTRUCTIONS)


def get_synthesis_instructions() -> str:
    """Get extraction instructions for the synthesis episode."""
    return SYNTHESIS_INSTRUCTIONS
