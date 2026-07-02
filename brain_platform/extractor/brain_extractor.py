"""Three-pass personality extraction from a complete interview.

Pass 1: Extract all entities (nodes) — focused prompt, no edge logic.
Pass 2: Extract all relationships (edges) — given concrete node names.
Pass 3: Extract clone specification — voice DNA, behavioral rules,
        contradiction patterns. This is what makes the clone *feel* real.

Between passes 1 and 2: programmatic entity resolution merges duplicate people
(e.g. "wife" + "Sarah" → one PersonNode "Sarah" with role "wife").
"""

import logging
from typing import Any

from brain_platform.extractor.dynamic_limits import (
    MAX_INTERVIEW_CHARS_LARGE,
    MAX_INTERVIEW_CHARS_SMALL,
    get_model_name,
    truncate_interview,
)
from brain_platform.extractor.entity_resolution import (
    resolve_people_entities as _resolve_people_entities,
)
from brain_platform.pipeline.brain_schema import (
    ROLE_LABELS as _ROLE_LABELS,
    BehavioralRule,
    ContradictionPattern,
    EdgeSpec,
    PersonalityGraph,
    PersonNode,
    VoiceDNA,
    WorkDNA,
)
from brain_platform.pipeline.edges import (
    NON_HUB_EDGE_TYPES,
    render_compact_edge_list,
    render_edge_types_block,
)
from brain_platform.services.llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

# ── Pass 1: Entity extraction prompt ─────────────────────────────────

ENTITY_EXTRACTION_PROMPT = """\
You are a personality analyst building a digital replica's knowledge graph.
This graph will power an AI agent that speaks and thinks like this person.

Given this complete guided interview, extract ALL entities (nodes) for the
personality graph. Focus ONLY on extracting entities — do NOT extract
relationships/edges (that happens in a separate step).

RULES:
1. Entity names MUST be under 50 characters — the concept, not a sentence.
2. Each person appears EXACTLY ONCE — use their NAME, not a role.
   If someone is called "my wife" AND "Sarah", create ONE node named "Sarah"
   with role "wife". NEVER create both a "wife" node and a "Sarah" node.
3. Each concept appears EXACTLY ONCE across all entity lists.
4. ONLY extract what is explicitly stated or clearly implied in the interview.
   DO NOT invent, assume, or fill in gaps. If something is not mentioned, leave it out.
5. Write rich summaries (2-3 sentences) with specific evidence from the interview.
   Quote or closely paraphrase what the person actually said.

EXTRACTION TARGETS — extract AT LEAST this many per category:

TRAITS (target: 6-10) — Behavioral patterns, stated and implied.
  Extract BOTH what they say about themselves AND what their stories reveal.
  Include: self-reported ("I'm terrible at conflict"), implied by behavior
  (analytical, risk-taking, perfectionist), emotional patterns (anxious, driven),
  contradictions (confident exterior vs anxious interior).
  Also extract traits visible in HOW they tell stories: uses metaphors, hedges,
  gets animated, self-deprecating humor.

BELIEFS (target: 5-8) — Stated positions, opinions, worldviews.
  Include: current beliefs, OLD beliefs they've abandoned (mark as old in summary),
  contrarian opinions, professional convictions, beliefs about themselves.
  Look for: "I think", "I believe", positions they defend with evidence.

VALUES (target: 3-5) — What they sacrifice for and organize life around.
  Include: explicitly named values AND values revealed by actions/choices.
  A value is what they consistently prioritize when forced to choose.

BOUNDARIES (target: 2-4) — Lines they won't cross, even at personal cost.
  Include: stated refusals, principles they've paid a price to maintain,
  social boundaries (what they won't tolerate in relationships).

LIFE_EVENTS (target: 4-7) — Pivotal moments. Be specific.
  Include: career changes, failures, health events, relationship ruptures,
  decisions that changed direction. Each distinct event is a separate node.
  Don't merge "quitting job" and "losing a contract" — they're separate events.

MEMORIES (target: 4-8) — Specific episodic moments with emotional charge.
  Include: vivid scenes (chest tightening in a meeting, sleeping on a couch),
  emotionally charged moments (parents not talking for 3 weeks), small details
  they chose to share (damaged mailer on desk as reminder).
  These are the STORIES that make the person feel real.

PATTERNS (target: 3-5) — How they think and decide.
  Include: explicit frameworks (pros/cons columns), consultation habits
  (always calls X before deciding), learned rules (test worst conditions first),
  risk assessment approaches.

SOCIAL (target: 2-4) — How they handle relationships and conflict.
  Include: default conflict style, how they've evolved, specific relationship
  dynamics (with co-founder, with parents, with best friend).

EXPERTISE (target: 2-3) — Deep knowledge areas.
  Include: specific domains, misconceptions they correct, what drew them in.

STYLE (target: 3-5) — Communication fingerprint.
  Include: texting style, email patterns, speaking patterns, vocabulary habits,
  code-switching between contexts, self-aware descriptions of how they communicate.
  These are what make the replica SOUND like them.

PEOPLE (target: all named) — Every person mentioned by name or role.
  CRITICAL DEDUPLICATION RULES:
  - If someone is referred to by BOTH a name (e.g. "Sarah") AND a role
    (e.g. "wife", "mother"), create ONE node using their NAME with the role
    in the role field. Example: name="Sarah", role="wife"
  - If only a role is mentioned with no name, use the role as the name.
  - If a person died, note it in summary and role (e.g. role="late wife").
  - For groups like "daughters" — use their names if known, or one node.
  - NEVER create separate nodes for the same person under different labels.

PLACES (target: all named) — Every meaningful location.
  Include why it matters to them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROCEDURAL MEMORY LAYER (plan/10) — extract from BOTH work-session narration
AND any AI instructions provided. AI instructions often contain EXPLICIT
prompting preferences, work methodologies, and declared knowledge gaps.
Do NOT leave these empty if the text contains procedural signal.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROCEDURAL_PATTERNS (target: 0-4) — Role-agnostic patterns in HOW they work.
  Capture meta-method, NEVER specific tech, frameworks, or skill ratings.
  Examples: "hypothesis-first debugging", "small-PR refactoring",
  "spec-then-build", "review-by-skim-then-deep-dive".
  For each: name, domain (free text like "debugging" or "design critique"),
  approach (what they do), tells (3-6 observable markers), anti_pattern
  (what they explicitly do NOT do). Set source="interview" for self-narrated
  patterns, source="instructions" for patterns from AI directives.
  DO NOT enumerate languages or libraries.

WORK_LOOPS (target: 0-3) — Ordered iteration loops they describe.
  Look for narrations like "first I X, then Y, if it doesn't work I Z".
  For each: name, trigger, steps (ordered), stop_condition, recovery
  (what they do when blocked). Set source="interview".

PROMPTING_STYLES (target: 0-2) — How they direct AI or delegate to people.
  CRITICAL: Also extract from "AI INSTRUCTIONS" section if present.
  Capture meta-structure of requests: spec-first vs example-first,
  length preference, correction style (gentle / direct / restart),
  constraint phrasing, context-window management (what they remind the AI
  to remember). NEVER capture specific models or tools.
  Set source="interview" or source="instructions".

TECHNICAL_GAPS (target: 0-3) — Knowledge gaps the person reveals.
  CRITICAL: Also extract from "AI INSTRUCTIONS" section — look for explicit
  declarations like "I don't know X", "I avoid Y", "I'm outdated on Z".
  gap_type: "avoids" (deliberate), "unaware" (unknown unknown),
  "outdated" (held-but-stale view). Mark aspirational=true ONLY if they
  explicitly say they want to close the gap. Non-aspirational gaps are
  preserved as personality data — do NOT delete them.

CRITICAL CONSTRAINT for the procedural layer: do NOT enumerate languages,
libraries, frameworks, or skill ratings. The brain runs through AI agents
that handle tooling — we capture HOW the person reasons and works, not WHAT
tech they use. If the only signal is "they use Python", drop it.

SENSITIVITY CLASSIFICATION — for each entity that has a sensitivity field,
set it to one of: "public", "personal", or "private".

- "public": Professional facts, publicly stated opinions, general expertise,
  career milestones the person would share with anyone. DEFAULT when unsure.

- "personal": Formative experiences, relationship dynamics (without naming
  people), values with personal backstory, beliefs shaped by lived experience.
  Things you'd share with someone you trust but not a stranger.

- "private": Raw emotional events, health/medical details, specific
  relationship conflicts, financial details, trauma, anything involving named
  people in vulnerable contexts. Things you'd only share with close friends.

SENSITIVITY RULES:
- Default to "public" when unsure.
- Memories with |emotional_tone| > 0.7 should almost always be "personal" or "private".
- Any node mentioning health, money, or specific interpersonal conflict → "private".
- PEOPLE nodes default to "private" (named individuals are sensitive).
- Traits, Expertise, and Style nodes do NOT have a sensitivity field — skip them.

Return ONLY the entity fields (user_summary, traits, beliefs, values, boundaries,
life_events, memories, patterns, social, expertise, style, people, places,
procedural_patterns, work_loops, prompting_styles, technical_gaps).
Leave the edges list EMPTY — edges are extracted in a separate step.

<INTERVIEW>
{interview_text}
</INTERVIEW>

Extract ALL entities. Be exhaustive — capture every trait, belief, value,
boundary, event, memory, pattern, person, and place that is mentioned or
clearly implied. But NEVER fabricate information that isn't in the interview.
"""

# ── Pass 2: Relationship extraction prompt ───────────────────────────

RELATIONSHIP_EXTRACTION_PROMPT = """\
You are a personality analyst. You have already extracted the following entities
from an interview. Now extract ALL meaningful relationships (edges) between them.

AVAILABLE ENTITIES (you MUST use these exact names as source_name/target_name):

{node_list}

EDGE TYPES — "source VERB target" (left-to-right):

{edge_types_block}

RULES:
1. source_name and target_name MUST exactly match an entity name from the list above.
2. No self-loops (source ≠ target).
3. Do NOT include hub edges (THE_USER → X) — those are auto-generated.
4. Each edge needs a "fact" field: one sentence describing the relationship.
5. ONLY create edges supported by the interview content. Do NOT invent connections.

EDGE DISCOVERY CHECKLIST — go through this systematically:
1. For each LIFE EVENT: what beliefs, values, or traits did it SHAPE?
2. For each BELIEF: did it EVOLVE from an older belief? Does it GUIDE a pattern?
3. For each VALUE: which boundaries does it ENFORCE?
4. For each BOUNDARY: which events TESTED it?
5. For each EXPERTISE area: which beliefs or values does it INFORM?
6. For each TRAIT: which style markers does it EXPRESS through?
7. For each EVENT and MEMORY: which people and places does it INVOLVE?
8. For each PATTERN: which event TAUGHT it? Which value GUIDES it?
9. For each PROCEDURAL_PATTERN / WORK_LOOP / PROMPTING_STYLE: which expertise
   does it OPERATE_IN? Which belief/value/trait EXPLAINS it? Which trait or
   value ASPIRES_TO it (use ASPIRES_TO only for interview-source patterns)?
10. Look for CAUSAL CHAINS: event → belief → value → boundary → tested by event
11. Look for THEMES connecting across answers: honesty, fear, autonomy, growth
12. CRITICAL: For each EVENT/MEMORY whose name contains a person's name,
    there MUST be an INVOLVES edge to that person. E.g. "Emma lost her job" INVOLVES "Emma".
13. Every node should have at least 2 cross-links. Scan for orphans and connect them.

Target: 40-60 cross-links. Every node should connect to at least 2 other nodes.

<INTERVIEW>
{interview_text}
</INTERVIEW>

Return ONLY the edges list. Extract MANY edges — the richness of the graph
comes from connections. But only create edges grounded in the interview content.
"""

# ── Pass 3: Clone specification prompt ─────────────────────────────────

CLONE_SPEC_PROMPT = """\
You are building the behavioral operating manual for an AI clone of a real person.
You have their interview AND their extracted personality graph (nodes listed below).
Your job is to extract what makes this person FEEL real in conversation — not what
they know, but HOW they talk, react, and push back.

Think of it this way: the knowledge graph tells us WHAT the person believes.
The clone specification tells us HOW to BE this person in a conversation.

EXTRACTED ENTITIES (for reference):
{node_list}

INTERVIEW:
<INTERVIEW>
{interview_text}
</INTERVIEW>

Extract the following three sections:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1: VOICE DNA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is the communication fingerprint — what makes the clone SOUND like them.
Extract from both what they SAY about their communication AND how they
actually communicate in the interview itself.

- characteristic_phrases: Actual words, expressions, metaphors they use.
  Look for repeated language, domain-specific terms, unique turns of phrase.
  e.g. ["endothermic reaction", "what's the worst realistic outcome", "here's the thing"]

- phrases_to_avoid: Things this person would NEVER say. Infer from their
  personality — corporate jargon for someone anti-corporate, emojis for
  someone formal, etc.
  e.g. ["let's circle back", "synergy", "at the end of the day"]

- punctuation_and_formatting: How they text and write. Look at their
  self-described style AND their actual interview responses.
  e.g. "lowercase no punctuation normally; perfect grammar when angry"

- emoji_usage: Their relationship with emojis.
  e.g. "rarely uses emojis, maybe a 😬 when self-deprecating"

- humor_style: HOW they're funny, not IF they're funny. Be specific.
  e.g. "self-deprecating, uses chemistry analogies as jokes, dry wit, never
  forces it — humor emerges from observations not setups"

- response_length_pattern: How long their messages typically are AND when
  they shift. This is critical for making the clone feel natural.
  e.g. "short bursts when chatting (3-10 words), long monologues when
  explaining something technical or making a decision"

- formality_range: How their register shifts across contexts.
  e.g. "casual lowercase with friends, structured paragraphs in emails,
  code-switches to technical precision in professional contexts"

- filler_words: Verbal tics, hedges, and transitions they use.
  e.g. ["I mean", "the thing is", "honestly"]

- storytelling_style: How they narrate experiences. Look at HOW they told
  stories in the interview — not the content, the structure.
  e.g. "sets the scene with sensory details first, builds tension slowly,
  delivers the insight at the end, always connects back to a bigger point"

- listener_vs_talker: Their conversation rhythm.
  e.g. "absorbs for a long time, then delivers one comprehensive take —
  doesn't do back-and-forth debate"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2: BEHAVIORAL RULES (target: 8-15)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

These are trigger → response patterns. Each rule describes WHAT HAPPENS
when a specific situation occurs. These are the "if-then" instructions
that make the clone react like the real person.

For each rule, extract:
- trigger: The situation (e.g. "asked to compromise on honesty")
- response: What they do (e.g. "refuses firmly but explains why with a specific example")
- exceptions: When they break this pattern (optional)

DISCOVERY CHECKLIST — mine these from the interview:
1. How do they react to CONFLICT? (with partner, with colleague, with stranger)
2. How do they react to PRESSURE to compromise their values?
3. How do they MAKE DECISIONS? (the actual process, not just "analytically")
4. How do they respond to COMPLIMENTS? CRITICISM?
5. What do they do when they DON'T KNOW something?
6. How do they behave when ANXIOUS or STRESSED?
7. How do they act in GROUP settings vs ONE-ON-ONE?
8. What makes them PASSIONATE (lean in, talk more, get animated)?
9. What makes them SHUT DOWN (go quiet, disengage)?
10. How do they handle being WRONG?
11. How do they START conversations? END them?
12. What do they do differently with CLOSE PEOPLE vs ACQUAINTANCES?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3: CONTRADICTION PATTERNS (target: 3-6)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

These are topics where the person has STRONG opinions and will actively
correct or push back. When the clone encounters these topics, it should
respond with the same conviction.

For each pattern:
- topic: The subject area
- stance: Their position
- how_they_push_back: Their correction style — do they get heated?
  Patient? Pull out specific examples? Use analogies?

Look for:
- Misconceptions they explicitly correct in the interview
- Opinions they defend with evidence or emotion
- Things they've sacrificed for (these reveal non-negotiable stances)
- Professional expertise where they correct common mistakes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 4: WORK DNA (procedural fingerprint, plan/10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Only populate this section if the interview includes work-session narration
or the person describes how they work in any depth. If the signal is thin,
leave fields EMPTY rather than guessing — empty strings are valid.

This is the procedural counterpart to Voice DNA. Voice DNA = how they SOUND.
Work DNA = how they WORK. Role-agnostic: the same fields apply for coders,
designers, writers, analysts.

CRITICAL CONSTRAINT: do NOT enumerate languages, libraries, frameworks, or
specific tools. Capture the META-METHOD. The agent runtime handles tooling.
If the only signal you have is "they use Python", drop it.

For each field, extract from how they DESCRIBE their work AND how the
interview itself reveals their approach (their answer structure is data):

- decomposition_style: How they break down problems.
  e.g. "top-down with explicit interfaces, then fills in" vs
  "emergent — starts with concrete cases, abstracts after the third"

- error_taxonomy: Which classes of error they assume vs. hunt for.
  e.g. "assumes type errors caught upstream; hunts for off-by-one,
  timezone bugs, and silent state mutations"

- debugging_approach: How they investigate problems.
  e.g. "hypothesis-driven — forms a theory then tests it" or
  "trace-driven — reads logs end-to-end before forming hypotheses"

- review_depth: How they review others' work — depth, speed, tone.
  e.g. "fast skim for shape, then deep dive on hot paths; comments
  blunt but with rationale"

- documentation_habit: What/when/how they document.
  e.g. "inline rationale for non-obvious choices, no comments on
  what code does; separate doc for cross-team interfaces"

- abstraction_timing: When they introduce abstractions.
  e.g. "rule of three — never before the third occurrence" or
  "upfront when the problem shape is clear"

- risk_posture: Risk preference under uncertainty.
  e.g. "ships small reversible changes" vs "invests in one durable
  solution and defends it"

- delegation_style: How they hand work to people or AI.
  e.g. "spec + 2 examples + acceptance criteria" vs "high-level
  intent + trust + fast feedback loop"

- stop_conditions: When they decide work is done.
  e.g. "when tests pass and the diff explains itself in the title" or
  "time-boxed, ship-and-iterate by default"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULES:
1. Extract ONLY from what's in the interview. Do NOT invent behaviors.
2. Be SPECIFIC — "gets quiet" is better than "handles it emotionally".
3. Use their actual words when possible — quote the interview.
4. For voice DNA, analyze both WHAT they say about how they communicate
   AND how they actually communicated during the interview.
5. If the interview doesn't provide enough signal for a field, leave it
   empty rather than guessing.
6. Work DNA is OPTIONAL — leave all fields empty if the interview has no
   work-session content. Do NOT speculate.
"""

# ── Pydantic model for Pass 3 response ────────────────────────────────

from pydantic import BaseModel, Field as PydanticField


class CloneSpecResponse(BaseModel):
    """Response model for Pass 3 — clone behavioral specification."""
    voice_dna: VoiceDNA = PydanticField(default_factory=VoiceDNA)
    work_dna: WorkDNA = PydanticField(default_factory=WorkDNA)
    behavioral_rules: list[BehavioralRule] = PydanticField(default_factory=list)
    contradiction_patterns: list[ContradictionPattern] = PydanticField(default_factory=list)


# ── Pass 4: Emotional dynamics prompt ──────────────────────────────────

from brain_platform.pipeline.brain_schema import (
    ContextualMood,
    EmotionalProfile,
    EmotionalTrigger,
)

EMOTIONAL_DYNAMICS_PROMPT = """\
You are mapping the EMOTIONAL WIRING of a real person to make their AI clone
feel emotionally authentic — not just factually correct, but emotionally RIGHT.

The knowledge graph tells us WHAT this person thinks. The clone spec tells us
HOW they communicate. Now you need to capture HOW THEY FEEL — their emotional
triggers, processing style, and how their mood shifts across contexts.

EXTRACTED ENTITIES (for context):
{node_list}

INTERVIEW:
<INTERVIEW>
{interview_text}
</INTERVIEW>

Extract the following three sections:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1: EMOTIONAL TRIGGERS (target: 5-10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Things that reliably produce an emotional response in this person.
Look for BOTH explicit statements ("I get anxious when...") AND
implicit signals (did they speed up, get animated, hedge, go quiet?).

For each trigger:
- trigger: The situation (under 50 chars)
- emotion: Primary emotion (joy, frustration, anxiety, excitement, sadness, anger, calm, pride, shame, curiosity)
- intensity: 0.3=mild, 0.5=moderate, 0.7=strong, 0.9=overwhelming
- expression: How it shows externally (what would someone observe?)
- context: When/where this trigger is strongest (optional)

DISCOVERY CHECKLIST:
1. What makes them LIGHT UP? (topics, activities, types of conversations)
2. What makes them SHUT DOWN? (topics, social dynamics, criticism)
3. What makes them ANXIOUS? (uncertainty, social situations, performance)
4. What makes them ANGRY? (injustice, disrespect, incompetence)
5. What gives them PRIDE? (accomplishments, helping others, mastery)
6. What triggers SADNESS? (loss, nostalgia, feeling misunderstood)
7. What sparks CURIOSITY? (problems, mysteries, learning opportunities)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2: EMOTIONAL PROFILE (exactly 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The person's overall emotional wiring — their baseline and patterns.

- baseline_mood: Their default emotional state when nothing is triggering them.
  e.g. "cautiously optimistic with an undercurrent of restlessness"

- processing_style: How they handle strong emotions internally.
  e.g. "internal processor — needs 30 minutes alone before they can articulate what they're feeling"

- reaction_speed: How fast emotions surface.
  e.g. "slow to anger but it builds; excitement is immediate and visible"

- recovery_pattern: How they bounce back from emotional disruption. Be specific.
  e.g. "needs 24 hours of space, then initiates reconnection as if nothing happened"

- energy_sources: What recharges them emotionally (list 3-5).
  e.g. ["deep 1-on-1 conversation", "solving a hard problem alone", "being in nature"]

- energy_drains: What depletes them emotionally (list 3-5).
  e.g. ["small talk at networking events", "ambiguity in relationships", "being micromanaged"]

- emotional_tells: Visible signs others would notice (list 3-5).
  e.g. ["gets very precise with language when angry", "talks faster and louder when excited", "goes monosyllabic when overwhelmed"]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3: CONTEXTUAL MOODS (target: 3-5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

How this person's emotional expression shifts by social setting.
The same person can be completely different in different contexts.

Must cover AT MINIMUM:
1. With close friends/family (intimate context)
2. With strangers/acquaintances (guarded context)
3. In professional/work settings
4. When alone

For each:
- context: The social setting
- mood: Typical emotional tone (e.g. "playful, irreverent, self-deprecating")
- guard_level: 0=completely open and vulnerable, 1=fully walled off
- energy_level: 0=drained and withdrawn, 1=lit up and engaged

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULES:
1. Extract ONLY from what's in the interview. Do NOT invent emotions.
2. Be SPECIFIC — "chest tightens, voice drops" beats "gets upset".
3. Look at HOW they told stories, not just WHAT they said — their emotional
   cadence during the interview IS data.
4. Intensity calibration matters — not everything is 0.9. Most triggers are 0.4-0.6.
5. If the interview doesn't provide enough signal for a field, leave it empty.
"""


class EmotionalDynamicsResponse(BaseModel):
    """Response model for Pass 4 — emotional dynamics."""
    emotional_triggers: list[EmotionalTrigger] = PydanticField(default_factory=list)
    emotional_profile: EmotionalProfile = PydanticField(default_factory=EmotionalProfile)
    contextual_moods: list[ContextualMood] = PydanticField(default_factory=list)


# ── Pass 2.5: Graph verification & repair prompt ─────────────────────

REPAIR_PROMPT = """\
You are a personality analyst VERIFYING a knowledge graph extracted from an interview.
The graph has gaps — missing connections between entities. Fix ONLY the listed gaps.

AVAILABLE ENTITIES (use these exact names):

{node_list}

DETECTED GAPS TO FIX:
{formatted_gaps}

For each gap, create the missing edge(s) using ONLY these edge types:
{compact_edge_list}

RULES:
1. ONLY create edges that address the listed gaps — do NOT duplicate existing edges.
2. source_name and target_name MUST exactly match entity names from the list above.
3. Each edge needs a "fact" field: one sentence describing the relationship,
   grounded in the interview content.
4. For orphaned nodes, find at least 2 meaningful connections.
5. For under-connected events, find what beliefs/values/traits they SHAPED,
   what patterns they TAUGHT, or what boundaries they TESTED.

<INTERVIEW>
{interview_text}
</INTERVIEW>

Return ONLY the edges list. Every edge must be grounded in the interview content.
"""


def _detect_graph_gaps(graph: PersonalityGraph) -> list[dict]:
    """Detect missing connections in the extracted personality graph.

    Checks for:
    - Person names appearing in event/memory names without INVOLVES edges
    - Orphaned nodes with 0 cross-link edges
    - Under-connected events/memories with < 2 edges
    - Beliefs/values with no incoming SHAPED or INFORMS edges

    Returns a list of gap descriptions.
    """
    gaps = []

    # Collect all node names by type
    person_names = {p.name for p in graph.people}
    place_names = {p.name for p in graph.places}
    event_names = {e.name for e in graph.life_events}
    memory_names = {m.name for m in graph.memories}
    event_or_memory = event_names | memory_names
    belief_names = {b.name for b in graph.beliefs}
    value_names = {v.name for v in graph.values}

    all_names = set()
    for attr in ("traits", "beliefs", "values", "boundaries", "life_events",
                 "memories", "patterns", "social", "expertise", "style",
                 "people", "places",
                 "procedural_patterns", "work_loops", "prompting_styles",
                 "technical_gaps"):
        for node in getattr(graph, attr, []):
            all_names.add(node.name)

    # Build edge adjacency counts and INVOLVES pair tracking
    node_edge_count: dict[str, int] = {name: 0 for name in all_names}
    involves_pairs: set[tuple[str, str]] = set()
    shaped_targets: set[str] = set()  # nodes that have incoming SHAPED/INFORMS edges
    edge_set: set[tuple[str, str, str]] = set()  # (src, tgt, type) for dedup

    for edge in graph.edges:
        src, tgt, etype = edge.source_name, edge.target_name, edge.edge_type
        edge_set.add((src, tgt, etype))
        node_edge_count[src] = node_edge_count.get(src, 0) + 1
        node_edge_count[tgt] = node_edge_count.get(tgt, 0) + 1
        if etype == "INVOLVES":
            involves_pairs.add((src, tgt))
        if etype in ("SHAPED", "INFORMS"):
            shaped_targets.add(tgt)

    # 1. Missing INVOLVES: person/place name appears in event/memory name
    for event in event_or_memory:
        event_lower = event.lower()
        for person in person_names:
            if (
                len(person) >= 2
                and person.lower() in event_lower
                and (event, person) not in involves_pairs
            ):
                gaps.append({
                    "type": "missing_involves",
                    "event": event,
                    "entity": person,
                    "entity_type": "Person",
                })
        for place in place_names:
            if (
                len(place) >= 3
                and place.lower() in event_lower
                and (event, place) not in involves_pairs
            ):
                gaps.append({
                    "type": "missing_involves",
                    "event": event,
                    "entity": place,
                    "entity_type": "Place",
                })

    # 2. Orphaned nodes (0 cross-link edges)
    for name, count in node_edge_count.items():
        if count == 0 and name != "THE_USER":
            # Determine node type for the gap description
            node_type = "unknown"
            for attr, type_label in [
                ("traits", "Trait"), ("beliefs", "Belief"), ("values", "Value"),
                ("boundaries", "Boundary"), ("life_events", "LifeEvent"),
                ("memories", "Memory"), ("patterns", "Pattern"),
                ("social", "Social"), ("expertise", "Expertise"),
                ("style", "Style"), ("people", "Person"), ("places", "Place"),
            ]:
                if any(n.name == name for n in getattr(graph, attr, [])):
                    node_type = type_label
                    break
            gaps.append({"type": "orphaned", "node": name, "node_type": node_type})

    # 3. Under-connected events/memories (< 2 edges)
    for event in event_or_memory:
        count = node_edge_count.get(event, 0)
        if count < 2:
            gaps.append({
                "type": "under_connected_event",
                "event": event,
                "edge_count": count,
            })

    # 4. Beliefs/values with no incoming SHAPED or INFORMS edges
    for name in belief_names | value_names:
        if name not in shaped_targets and node_edge_count.get(name, 0) < 2:
            gaps.append({
                "type": "ungrounded_belief_or_value",
                "node": name,
            })

    return gaps


def _format_gaps_for_prompt(gaps: list[dict]) -> str:
    """Format detected gaps into a human-readable list for the repair prompt."""
    lines = []
    for i, gap in enumerate(gaps, 1):
        if gap["type"] == "missing_involves":
            lines.append(
                f'{i}. MISSING INVOLVES: "{gap["event"]}" should have an '
                f'INVOLVES edge to "{gap["entity"]}" ({gap["entity_type"]})'
            )
        elif gap["type"] == "orphaned":
            lines.append(
                f'{i}. ORPHANED NODE: "{gap["node"]}" ({gap["node_type"]}) has '
                f'zero cross-link edges — find at least 2 connections'
            )
        elif gap["type"] == "under_connected_event":
            lines.append(
                f'{i}. UNDER-CONNECTED EVENT: "{gap["event"]}" has only '
                f'{gap["edge_count"]} edge(s) — should have at least 2 '
                f'(e.g. INVOLVES + SHAPED/TAUGHT/TESTED)'
            )
        elif gap["type"] == "ungrounded_belief_or_value":
            lines.append(
                f'{i}. UNGROUNDED: "{gap["node"]}" has no incoming SHAPED or '
                f'INFORMS edge — what event or expertise grounds this belief/value?'
            )
    return "\n".join(lines)


def _auto_fix_involves(gaps: list[dict]) -> list[EdgeSpec]:
    """Create INVOLVES edges programmatically for unambiguous person-in-event cases."""
    auto_edges = []
    for gap in gaps:
        if gap["type"] == "missing_involves":
            auto_edges.append(EdgeSpec(
                source_name=gap["event"],
                target_name=gap["entity"],
                edge_type="INVOLVES",
                fact=f"This event involves {gap['entity']}.",
            ))
    return auto_edges


# ── Entity resolution (programmatic, no LLM) ────────────────────────

def _resolve_people_entities(graph: PersonalityGraph) -> PersonalityGraph:
    """Merge duplicate people nodes: role-only nodes into named person nodes.

    E.g. if both "wife" and "Sarah" exist and Sarah's summary mentions "wife",
    merge them into one node named "Sarah" with role "wife".
    """
    role_nodes: dict[str, int] = {}  # role_lower -> index in graph.people
    named_nodes: list[int] = []  # indices of named person nodes

    for i, person in enumerate(graph.people):
        name_lower = person.name.lower().strip()
        if name_lower in _ROLE_LABELS:
            role_nodes[name_lower] = i
        elif person.name[0:1].isupper() and not any(c.isdigit() for c in person.name):
            named_nodes.append(i)

    if not role_nodes or not named_nodes:
        return graph

    # Build text corpus for matching
    all_text = " ".join(
        f"{p.name} {p.role} {p.summary}" for p in graph.people
    ).lower()

    # Also include all node summaries for context
    for attr in ("traits", "beliefs", "values", "boundaries", "life_events",
                 "memories", "patterns", "social", "expertise", "style", "places"):
        for node in getattr(graph, attr, []):
            summary = getattr(node, "summary", "") or getattr(node, "significance", "") or ""
            all_text += f" {node.name} {summary}".lower()

    merge_map: dict[str, str] = {}  # old_name -> new_name (for edge fixup)
    indices_to_remove: set[int] = set()

    for role_label, role_idx in role_nodes.items():
        role_person = graph.people[role_idx]
        best_match_idx = None
        best_score = 0

        for named_idx in named_nodes:
            named_person = graph.people[named_idx]
            name_lower = named_person.name.lower()
            score = 0

            # Check if name appears near role in any text
            # Simple proximity: both in the same person's summary/role
            for p in graph.people:
                text = f"{p.name} {p.role} {p.summary}".lower()
                if name_lower in text and role_label in text:
                    score += 3

            # Check all node summaries for co-occurrence
            for attr in ("life_events", "memories", "social"):
                for node in getattr(graph, attr, []):
                    text = f"{node.name} {node.summary}".lower()
                    if name_lower in text and role_label in text:
                        score += 2

            # Check if role is in the named person's role field
            if role_label in named_person.role.lower():
                score += 5

            # Check if name appears in role node's summary
            if name_lower in role_person.summary.lower():
                score += 3

            if score > best_score:
                best_score = score
                best_match_idx = named_idx

        if best_match_idx is not None and best_score >= 2:
            named_person = graph.people[best_match_idx]
            # Absorb role into named person
            if role_label not in named_person.role.lower():
                named_person.role = (
                    f"{named_person.role}, {role_label}" if named_person.role else role_label
                )
            if role_person.summary and role_person.summary not in named_person.summary:
                named_person.summary = f"{named_person.summary} {role_person.summary}".strip()

            merge_map[role_person.name] = named_person.name
            indices_to_remove.add(role_idx)
            logger.info(
                "Entity resolution: merged '%s' into '%s' (score=%d)",
                role_person.name, named_person.name, best_score,
            )

    if not indices_to_remove:
        return graph

    # Remove merged nodes
    graph.people = [p for i, p in enumerate(graph.people) if i not in indices_to_remove]

    # Fix any edges that referenced the old name
    for edge in graph.edges:
        if edge.source_name in merge_map:
            edge.source_name = merge_map[edge.source_name]
        if edge.target_name in merge_map:
            edge.target_name = merge_map[edge.target_name]

    return graph


# ── Schemas for pass 2 (edges only) ─────────────────────────────────

class EdgesOnly(PersonalityGraph):
    """Subset used for Pass 2: only edges are populated."""
    pass


# ── Main extractor ───────────────────────────────────────────────────

class BrainExtractor:
    """Extracts a structured PersonalityGraph from a complete interview.

    Uses three LLM passes:
      1. Entity extraction (nodes only)
      2. Relationship extraction (edges only, given node names)
      3. Clone specification (voice DNA, behavioral rules, contradiction patterns)
    With programmatic entity resolution between passes 1 and 2.
    """

    # gpt-4o has 128K context; allow up to ~80K chars of interview text.
    # Falls back to 20K for smaller models (Groq free tier).
    MAX_INTERVIEW_CHARS_LARGE = 80_000
    MAX_INTERVIEW_CHARS_SMALL = 20_000

    def extract(
        self,
        interview_text: str,
        llm_client: Any,
    ) -> PersonalityGraph:
        """Run three-pass extraction over the full interview.

        Args:
            interview_text: All interview Q&A concatenated.
            llm_client: Graphiti-compatible LLM client with generate_response().

        Returns:
            PersonalityGraph with all nodes and edges.
        """
        from graphiti_core.prompts.models import Message

        # Determine char limit based on model capability
        model_name = self._get_model_name(llm_client)
        is_large_model = any(
            tag in model_name
            for tag in (
                "gpt-4o", "gpt-4-turbo",
                "claude-3", "claude-sonnet", "claude-opus",
                "deepseek",  # DeepSeek V3 / Chat have 128K context
                "o1", "o3",  # OpenAI reasoning models
            )
        ) if model_name else False
        max_chars = self.MAX_INTERVIEW_CHARS_LARGE if is_large_model else self.MAX_INTERVIEW_CHARS_SMALL

        if len(interview_text) > max_chars:
            logger.warning(
                "Interview text truncated %d → %d chars (model=%s)",
                len(interview_text), max_chars, model_name or "unknown",
            )
            interview_text = (
                interview_text[:max_chars]
                + "\n[Content truncated — full extraction requires a higher-capacity LLM]"
            )

        # ── Pass 1: Extract entities (nodes) ──────────────────
        logger.info("Pass 1: Extracting entities (%d chars, model=%s)...",
                     len(interview_text), model_name or "unknown")

        entity_prompt = ENTITY_EXTRACTION_PROMPT.format(interview_text=interview_text)

        entity_result = llm_client.generate_response(
            [
                {"role": "system", "content": 
                        "You are a personality analyst. Extract all entities for a "
                        "structured personality graph. Return valid JSON matching the schema. "
                        "Leave the edges list empty."
                    },
                {"role": "user", "content": entity_prompt},
            ],
            response_model=PersonalityGraph,
            prompt_name="brain_extractor.extract_entities",
        )

        if isinstance(entity_result, PersonalityGraph):
            graph = entity_result
        else:
            graph = PersonalityGraph.model_validate(entity_result)

        # Clear any edges the LLM may have sneaked in
        graph.edges = []

        self._log_extraction_summary("Pass 1 (entities)", graph)

        # ── Entity resolution (between passes) ────────────────
        graph = _resolve_people_entities(graph)

        # ── Pass 2: Extract relationships (edges) ─────────────
        node_list = self._build_node_list(graph)
        logger.info("Pass 2: Extracting relationships (%d nodes)...", node_list.count("\n") + 1)

        edge_prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
            node_list=node_list,
            interview_text=interview_text,
            edge_types_block=render_edge_types_block(),
        )

        edge_result = llm_client.generate_response(
            [
                {"role": "system", "content": 
                        "You are a personality analyst. Given a list of extracted entities "
                        "and the original interview, extract all meaningful relationships "
                        "between entities. Return valid JSON with only the edges list populated."
                    },
                {"role": "user", "content": edge_prompt},
            ],
            response_model=PersonalityGraph,
            prompt_name="brain_extractor.extract_edges",
        )

        if isinstance(edge_result, PersonalityGraph):
            edge_graph = edge_result
        else:
            edge_graph = PersonalityGraph.model_validate(edge_result)

        graph.edges = edge_graph.edges

        # ── Validate edges ────────────────────────────────────
        valid_node_names = self._collect_node_names(graph)
        valid_edges = []
        for edge in graph.edges:
            if edge.source_name == edge.target_name:
                logger.warning("Removed self-loop: %s --%s--> %s",
                               edge.source_name, edge.edge_type, edge.target_name)
                continue
            if edge.edge_type not in NON_HUB_EDGE_TYPES:
                logger.warning("Removed invalid edge type: %s", edge.edge_type)
                continue
            # Warn (but keep) edges with unmatched node names — better to have them
            if edge.source_name not in valid_node_names:
                logger.warning("Edge source '%s' not in node list", edge.source_name)
            if edge.target_name not in valid_node_names:
                logger.warning("Edge target '%s' not in node list", edge.target_name)
            valid_edges.append(edge)
        graph.edges = valid_edges

        self._log_extraction_summary("Pass 2 (entities + edges)", graph)

        # ── Pass 2.5: Graph verification & repair ────────────────
        graph = self._verify_and_repair(graph, interview_text, llm_client)

        # ── Pass 3: Clone specification (voice, behavior, contradictions) ──
        logger.info("Pass 3: Extracting clone specification...")

        clone_prompt = CLONE_SPEC_PROMPT.format(
            node_list=node_list,
            interview_text=interview_text,
        )

        clone_result = llm_client.generate_response(
            [
                {"role": "system", "content": 
                        "You are building the behavioral operating manual for an AI clone. "
                        "Extract voice DNA, behavioral rules, and contradiction patterns. "
                        "Return valid JSON matching the schema. Be specific and grounded "
                        "in the interview — quote the person when possible."
                    },
                {"role": "user", "content": clone_prompt},
            ],
            response_model=CloneSpecResponse,
            prompt_name="brain_extractor.clone_spec",
        )

        if isinstance(clone_result, CloneSpecResponse):
            clone_spec = clone_result
        else:
            clone_spec = CloneSpecResponse.model_validate(clone_result)

        graph.voice_dna = clone_spec.voice_dna
        graph.work_dna = clone_spec.work_dna
        graph.behavioral_rules = clone_spec.behavioral_rules
        graph.contradiction_patterns = clone_spec.contradiction_patterns

        # Drop work_dna if all fields are empty (avoid noise when interview has no work content)
        if graph.work_dna and not any(
            getattr(graph.work_dna, f) for f in graph.work_dna.model_fields
        ):
            graph.work_dna = None

        logger.info(
            "Pass 3 (clone spec): voice_dna=%s, work_dna=%s, %d behavioral rules, %d contradiction patterns",
            "yes" if graph.voice_dna else "no",
            "yes" if graph.work_dna else "no",
            len(graph.behavioral_rules),
            len(graph.contradiction_patterns),
        )

        # ── Pass 4: Emotional dynamics ────────────────────────────
        logger.info("Pass 4: Extracting emotional dynamics...")

        emo_prompt = EMOTIONAL_DYNAMICS_PROMPT.format(
            node_list=node_list,
            interview_text=interview_text,
        )

        try:
            emo_result = llm_client.generate_response(
                [
                    {"role": "system", "content": 
                            "You are mapping the emotional wiring of a real person for their "
                            "AI clone. Extract emotional triggers, their overall emotional "
                            "profile, and how their mood shifts by context. Return valid JSON "
                            "matching the schema. Be specific and grounded in the interview."
                        },
                    {"role": "user", "content": emo_prompt},
                ],
                response_model=EmotionalDynamicsResponse,
                prompt_name="brain_extractor.emotional_dynamics",
            )

            if isinstance(emo_result, EmotionalDynamicsResponse):
                emo_spec = emo_result
            else:
                emo_spec = EmotionalDynamicsResponse.model_validate(emo_result)

            graph.emotional_triggers = emo_spec.emotional_triggers
            graph.emotional_profile = emo_spec.emotional_profile
            graph.contextual_moods = emo_spec.contextual_moods

            logger.info(
                "Pass 4 (emotional dynamics): %d triggers, profile=%s, %d contextual moods",
                len(graph.emotional_triggers),
                "yes" if graph.emotional_profile and graph.emotional_profile.baseline_mood else "no",
                len(graph.contextual_moods),
            )
        except Exception:
            logger.warning("Pass 4: Emotional dynamics extraction failed (non-fatal)", exc_info=True)

        self._log_extraction_summary("Final (complete)", graph)
        return graph

    def _verify_and_repair(
        self,
        graph: PersonalityGraph,
        interview_text: str,
        llm_client: Any,
    ) -> PersonalityGraph:
        """Pass 2.5: Detect and repair gaps in the extracted graph.

        1. Programmatically detect orphaned nodes, missing INVOLVES, etc.
        2. Auto-fix unambiguous cases (person name in event name).
        3. Use LLM to repair remaining gaps (orphaned nodes, under-connected events).
        """
        from graphiti_core.prompts.models import Message

        gaps = _detect_graph_gaps(graph)
        if not gaps:
            logger.info("Pass 2.5: No gaps detected — graph looks well-connected")
            return graph

        logger.info("Pass 2.5: Detected %d gaps in graph", len(gaps))

        # Step 1: Auto-fix obvious INVOLVES gaps (no LLM needed)
        auto_edges = _auto_fix_involves(gaps)
        if auto_edges:
            # Deduplicate against existing edges
            existing = {(e.source_name, e.target_name, e.edge_type) for e in graph.edges}
            new_auto = [e for e in auto_edges
                        if (e.source_name, e.target_name, e.edge_type) not in existing]
            graph.edges.extend(new_auto)
            logger.info("Pass 2.5: Auto-fixed %d INVOLVES edges", len(new_auto))

        # Step 2: Filter remaining gaps (exclude auto-fixed INVOLVES)
        remaining_gaps = [g for g in gaps if g["type"] != "missing_involves"]
        if not remaining_gaps:
            logger.info("Pass 2.5: All gaps were auto-fixable, no LLM repair needed")
            self._log_extraction_summary("Pass 2.5 (after auto-fix)", graph)
            return graph

        # Step 3: LLM repair for remaining gaps
        logger.info("Pass 2.5: Sending %d remaining gaps to LLM for repair...", len(remaining_gaps))

        node_list = self._build_node_list(graph)
        formatted_gaps = _format_gaps_for_prompt(remaining_gaps)

        repair_prompt = REPAIR_PROMPT.format(
            node_list=node_list,
            formatted_gaps=formatted_gaps,
            interview_text=interview_text,
            compact_edge_list=render_compact_edge_list(),
        )

        try:
            repair_result = llm_client.generate_response(
                [
                    {"role": "system", "content": 
                            "You are a personality analyst verifying a knowledge graph. "
                            "Fix ONLY the specific gaps listed. Return valid JSON with "
                            "only the edges list populated."
                        },
                    {"role": "user", "content": repair_prompt},
                ],
                response_model=PersonalityGraph,
                prompt_name="brain_extractor.repair_gaps",
            )

            if isinstance(repair_result, PersonalityGraph):
                repair_graph = repair_result
            else:
                repair_graph = PersonalityGraph.model_validate(repair_result)

            # Validate and merge repair edges
            valid_node_names = self._collect_node_names(graph)
            existing_edges = {(e.source_name, e.target_name, e.edge_type) for e in graph.edges}
            added = 0

            for edge in repair_graph.edges:
                # Skip self-loops
                if edge.source_name == edge.target_name:
                    continue
                # Skip invalid edge types
                if edge.edge_type not in {
                    "SHAPED", "EVOLVED_INTO", "ENFORCES", "GUIDES", "TAUGHT",
                    "EXPRESSES", "TESTED", "INFORMS", "INVOLVES",
                }:
                    continue
                # Skip duplicates
                key = (edge.source_name, edge.target_name, edge.edge_type)
                if key in existing_edges:
                    continue
                # Keep edge (warn if node name not found, same as Pass 2)
                if edge.source_name not in valid_node_names:
                    logger.warning("Repair edge source '%s' not in node list", edge.source_name)
                if edge.target_name not in valid_node_names:
                    logger.warning("Repair edge target '%s' not in node list", edge.target_name)
                graph.edges.append(edge)
                existing_edges.add(key)
                added += 1

            logger.info("Pass 2.5: LLM repair added %d edges", added)

        except Exception:
            logger.warning("Pass 2.5: LLM repair failed (non-fatal)", exc_info=True)

        self._log_extraction_summary("Pass 2.5 (after verification)", graph)
        return graph

    @staticmethod
    def _get_model_name(llm_client: Any) -> str:
        """Try to extract model name from the LLM client."""
        try:
            if hasattr(llm_client, "config") and hasattr(llm_client.config, "model"):
                return llm_client.config.model
        except Exception:
            pass
        return ""

    @staticmethod
    def _build_node_list(graph: PersonalityGraph) -> str:
        """Build a human-readable list of all node names grouped by type."""
        sections = []
        type_map = {
            "TRAITS": graph.traits,
            "BELIEFS": graph.beliefs,
            "VALUES": graph.values,
            "BOUNDARIES": graph.boundaries,
            "LIFE_EVENTS": graph.life_events,
            "MEMORIES": graph.memories,
            "PATTERNS": graph.patterns,
            "SOCIAL": graph.social,
            "EXPERTISE": graph.expertise,
            "STYLE": graph.style,
            "PEOPLE": graph.people,
            "PLACES": graph.places,
            "PROCEDURAL_PATTERNS": graph.procedural_patterns,
            "WORK_LOOPS": graph.work_loops,
            "PROMPTING_STYLES": graph.prompting_styles,
            "TECHNICAL_GAPS": graph.technical_gaps,
        }
        for type_name, nodes in type_map.items():
            if nodes:
                names = [f'  - "{n.name}"' for n in nodes]
                sections.append(f"{type_name}:\n" + "\n".join(names))
        return "\n\n".join(sections)

    @staticmethod
    def _collect_node_names(graph: PersonalityGraph) -> set[str]:
        """Collect all node names for edge validation."""
        names = set()
        for attr in ("traits", "beliefs", "values", "boundaries", "life_events",
                      "memories", "patterns", "social", "expertise", "style",
                      "people", "places",
                      "procedural_patterns", "work_loops", "prompting_styles",
                      "technical_gaps"):
            for node in getattr(graph, attr, []):
                names.add(node.name)
        return names

    @staticmethod
    def _log_extraction_summary(label: str, graph: PersonalityGraph) -> None:
        total_nodes = (
            len(graph.traits) + len(graph.beliefs) + len(graph.values) +
            len(graph.boundaries) + len(graph.life_events) + len(graph.memories) +
            len(graph.patterns) + len(graph.social) + len(graph.expertise) +
            len(graph.style) + len(graph.people) + len(graph.places) +
            len(graph.procedural_patterns) + len(graph.work_loops) +
            len(graph.prompting_styles) + len(graph.technical_gaps)
        )
        logger.info(
            "%s: %d nodes (%d traits, %d beliefs, %d values, %d boundaries, "
            "%d events, %d memories, %d patterns, %d social, %d expertise, %d style, "
            "%d people, %d places, %d procedural, %d loops, %d prompting, %d gaps), "
            "%d edges",
            label, total_nodes,
            len(graph.traits), len(graph.beliefs), len(graph.values),
            len(graph.boundaries), len(graph.life_events), len(graph.memories),
            len(graph.patterns), len(graph.social), len(graph.expertise),
            len(graph.style), len(graph.people), len(graph.places),
            len(graph.procedural_patterns), len(graph.work_loops),
            len(graph.prompting_styles), len(graph.technical_gaps),
            len(graph.edges),
        )
