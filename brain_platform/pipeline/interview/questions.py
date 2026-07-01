"""Guided Interview Protocol — Question Set & Dimension Mapping.

Grounded in:
- Stanford GenAgents (Park et al. 2024): 2-hour qualitative interview → 85% fidelity
- EmoAgent CCD: Core beliefs → intermediate beliefs → cognitive patterns
- PlugMem: Episodic / semantic / procedural memory separation
- MICrONS connectome: Inhibitory connections (anti-values, boundaries) are critical
- ACT-R: Context-sensitive activation across domains
- TinyStyler / Wang et al.: Natural voice samples needed for style transfer

Design principles:
1. Conversational, not quiz-like — open-ended questions that elicit stories
2. Story-driven — specific experiences > abstract self-descriptions
3. Multi-context — trigger memories across work, family, conflict, joy
4. Include boundaries/anti-values — what they WON'T do defines them
5. Ask about changed beliefs — belief evolution is deeply personal
6. Responses ARE the style data — let them write/speak naturally
7. Adaptive follow-ups — shallow answers get deepened by LLM probing
8. Age-adaptive — younger users get questions tuned to their life contexts
   (Persona-DB cold-start, PlugMem procedural memory, EmoAgent safety)
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InterviewQuestion:
    """A single question in the guided interview protocol."""

    id: str
    dimension: str
    question: str
    purpose: str
    follow_up_hint: str
    order: int
    min_words_for_depth: int = 40
    # Age-adaptive variants (None = use standard question)
    young_question: str | None = None
    emerging_question: str | None = None
    young_follow_up_hint: str | None = None
    emerging_follow_up_hint: str | None = None


# ──────────────────────────────────────────────────────────
# Age tier helpers
# ──────────────────────────────────────────────────────────

TIER_YOUNG = "young"        # Under 18
TIER_EMERGING = "emerging"  # 18–24
TIER_STANDARD = "standard"  # 25+


def age_to_tier(age: int) -> str:
    """Convert a numeric age to an interview tier."""
    if age < 18:
        return TIER_YOUNG
    if age < 25:
        return TIER_EMERGING
    return TIER_STANDARD


def get_question_for_tier(question: "InterviewQuestion", tier: str) -> str:
    """Return the appropriate question text for a given tier."""
    if tier == TIER_YOUNG and question.young_question:
        return question.young_question
    if tier == TIER_EMERGING and question.emerging_question:
        return question.emerging_question
    return question.question


def get_min_words_for_tier(question: "InterviewQuestion", tier: str) -> int:
    """Return the min word threshold adjusted for tier."""
    if tier == TIER_YOUNG:
        return max(1, int(question.min_words_for_depth * 0.70))
    if tier == TIER_EMERGING:
        return max(1, int(question.min_words_for_depth * 0.85))
    return question.min_words_for_depth


def get_follow_up_hint_for_tier(question: "InterviewQuestion", tier: str) -> str:
    """Return the appropriate follow-up hint for a given tier."""
    if tier == TIER_YOUNG and question.young_follow_up_hint:
        return question.young_follow_up_hint
    if tier == TIER_EMERGING and question.emerging_follow_up_hint:
        return question.emerging_follow_up_hint
    return question.follow_up_hint


# ──────────────────────────────────────────────────────────
# Age pre-question (asked before the 14 core questions)
# ──────────────────────────────────────────────────────────

AGE_QUESTION = InterviewQuestion(
    id="age",
    dimension="identity",
    question=(
        "Before we dive in — how old are you? This helps me ask questions "
        "that actually make sense for your stage of life. Just type your age "
        "as a number."
    ),
    purpose="Determine age tier for adaptive question selection",
    follow_up_hint="If unclear: ask them to just type a number",
    order=-1,  # Before all core questions
    min_words_for_depth=1,
)


# ──────────────────────────────────────────────────────────
# The 14 core questions, ordered for conversational flow:
#   Start warm/easy → build trust → go deep → close reflective
# ──────────────────────────────────────────────────────────

INTERVIEW_QUESTIONS: list[InterviewQuestion] = [
    # ── Identity anchor (must be first — establishes name for the graph) ──
    InterviewQuestion(
        id="identity_name",
        dimension="identity",
        question=(
            "Before we start — what's your name, or what would you like your "
            "AI replica to be called?"
        ),
        purpose="Establish the identity anchor for the brain graph — name is used as THE_USER label and throughout the system prompt",
        follow_up_hint="If they give a nickname: confirm that's what they want the replica to go by",
        order=0,
        min_words_for_depth=1,
    ),

    # ── Opening: Life Narrative (warm up, build comfort) ──
    InterviewQuestion(
        id="life_story_1",
        dimension="episodic_memory",
        question=(
            "Tell me about yourself — not a resume summary, but the version you'd "
            "share with someone you just met at a dinner party. What would you want "
            "them to know about who you are and how you got here?"
        ),
        purpose="Elicit self-narrative structure, salience hierarchy, and natural voice",
        follow_up_hint="If too brief: ask about a specific turning point or fork in the road",
        order=1,
        min_words_for_depth=60,
        young_question=(
            "If a new friend wanted to really get you — not just surface-level "
            "stuff — what would you want them to know about you and your story so far?"
        ),
        emerging_question=(
            "Tell me about yourself — not the LinkedIn version, but what you'd share "
            "with someone you clicked with. What would you want them to know about "
            "who you are and how you got here?"
        ),
        young_follow_up_hint="If too brief: ask about a specific moment — could be at school, with family, online, in a hobby or creative project, or with a close friend",
        emerging_follow_up_hint="If too brief: ask about a specific turning point or moment that shaped who they're becoming",
    ),
    InterviewQuestion(
        id="life_story_2",
        dimension="episodic_memory",
        question=(
            "What's a decision you made that fundamentally changed the direction "
            "of your life? Walk me through how you thought about it at the time."
        ),
        purpose="Capture pivotal episodic memory with decision-making reasoning",
        follow_up_hint="If abstract: ask what specifically they were weighing, who they talked to",
        order=2,
        min_words_for_depth=50,
        young_question=(
            "What's a choice you made that felt really important at the time — "
            "like picking a path when you weren't sure what would happen? Walk me "
            "through how you thought about it."
        ),
        emerging_question=(
            "What's a decision you made that really changed things for you — could "
            "be about school, moving, a relationship, anything. Walk me through how "
            "you thought about it at the time."
        ),
        young_follow_up_hint="If abstract: ask about the specific options they were weighing, who they talked to about it",
        emerging_follow_up_hint="If abstract: ask what specifically they were weighing, who they consulted",
    ),

    # ── Core Beliefs & Values ──
    InterviewQuestion(
        id="beliefs_1",
        dimension="core_beliefs",
        question=(
            "What's something you believe strongly that many people around you "
            "would disagree with? What convinced you of it?"
        ),
        purpose="Elicit contrarian beliefs with reasoning chains — high signal for uniqueness",
        follow_up_hint="If vague: ask for the specific experience or evidence that convinced them",
        order=3,
        min_words_for_depth=40,
        young_question=(
            "What's something you think or believe that a lot of people your age "
            "would disagree with? What made you start seeing it that way?"
        ),
        emerging_question=(
            "What's something you believe that most people around you — friends, "
            "classmates, coworkers — would push back on? What convinced you?"
        ),
        young_follow_up_hint="If vague: ask what experience or conversation first made them question it",
        emerging_follow_up_hint="If vague: ask for the specific experience or moment that convinced them",
    ),
    InterviewQuestion(
        id="beliefs_2",
        dimension="core_beliefs",
        question=(
            "What's a belief or opinion you held strongly in the past but have "
            "since changed your mind about? What caused the shift?"
        ),
        purpose="Capture belief evolution (EvoKG temporal tracking) and intellectual flexibility",
        follow_up_hint="If they say 'nothing': ask about political, career, or relationship views from 10 years ago",
        order=4,
        min_words_for_depth=40,
        young_question=(
            "What's something you used to believe — maybe even just a year or two "
            "ago — that you see really differently now? What changed?"
        ),
        emerging_question=(
            "What's an opinion or belief you held strongly a few years ago that "
            "you've since changed your mind about? What caused the shift?"
        ),
        young_follow_up_hint="If they say 'nothing': ask about how they saw friendships, school, or their parents a couple years ago vs now",
        emerging_follow_up_hint="If they say 'nothing': ask about views on relationships, career, or politics from a few years ago",
    ),

    # ── Decision-Making & Cognitive Patterns ──
    InterviewQuestion(
        id="decisions_1",
        dimension="decision_making",
        question=(
            "When you're facing a hard decision with no obvious right answer — "
            "something with real stakes — how do you actually work through it? "
            "Can you walk me through a recent example?"
        ),
        purpose="Map cognitive architecture: analytical vs intuitive, risk tolerance, consultation patterns",
        follow_up_hint="If too general: press for the specific steps, who they consulted, what they prioritized",
        order=5,
        min_words_for_depth=50,
        young_question=(
            "When you're stuck on a decision and there's no obvious right answer — "
            "maybe about school, a friendship, or something personal — how do you "
            "actually figure out what to do? Can you walk me through a time this happened?"
        ),
        emerging_question=(
            "When you're facing a tough call with real stakes — no clear right "
            "answer — how do you actually work through it? Can you walk me through "
            "a recent example?"
        ),
        young_follow_up_hint="If too general: ask about the specific options they were weighing — maybe about a team, a creative project, school, or a friendship — and who they talked to",
        emerging_follow_up_hint="If too general: press for the specific steps, who they consulted, what they prioritized",
    ),
    InterviewQuestion(
        id="decisions_2",
        dimension="decision_making",
        question=(
            "What's something you spent a lot of time on that turned out to "
            "be a mistake or a failure? How did you process that?"
        ),
        purpose="Failure narrative reveals resilience patterns, attribution style, and emotional processing",
        follow_up_hint="If surface-level: ask what they learned, whether they'd do it again, how it changed their approach",
        order=6,
        min_words_for_depth=40,
        young_question=(
            "What's something you put real effort into that didn't work out the "
            "way you hoped? How did you deal with that?"
        ),
        emerging_question=(
            "What's something you invested real time or energy in that turned out "
            "to be a mistake or didn't pan out? How did you process it?"
        ),
        young_follow_up_hint="If surface-level: ask what they took away from it, whether they'd try again, how it changed how they approach things",
        emerging_follow_up_hint="If surface-level: ask what they learned, whether they'd do it differently, how it changed their approach",
    ),

    # ── Values & Moral Framework ──
    InterviewQuestion(
        id="values_1",
        dimension="values",
        question=(
            "If you had to explain to someone what you care about most in life — "
            "the things that really drive your choices — what would you tell them?"
        ),
        purpose="Direct values elicitation with narrative context",
        follow_up_hint="If they list abstract words: ask for a time when one of those values was tested",
        order=7,
        min_words_for_depth=40,
        young_question=(
            "What matters most to you right now — the things that actually drive "
            "what you do and the choices you make? Not what you think you *should* "
            "say, but what's really true."
        ),
        emerging_question=(
            "If you had to explain to someone what really drives your choices — "
            "not the polished version, the real one — what would you say?"
        ),
        young_follow_up_hint="If they list abstract words: ask for a time one of those things was actually tested — peer pressure, a choice at school, a moment in a hobby or online, anything real",
        emerging_follow_up_hint="If they list abstract words: ask for a time when one of those values was tested or challenged",
    ),
    InterviewQuestion(
        id="values_2",
        dimension="boundaries",
        question=(
            "What's something you absolutely will not compromise on — a line "
            "you won't cross, even if it costs you? Has that ever been tested?"
        ),
        purpose="Inhibitory connections (MICrONS) — anti-values and boundaries define personality as much as values",
        follow_up_hint="If hypothetical: ask for a real situation where they said no at a cost",
        order=8,
        min_words_for_depth=40,
        young_question=(
            "What's something you'd refuse to do even if everyone around you was "
            "doing it or pressuring you? Has that ever actually been tested?"
        ),
        emerging_question=(
            "What's something you absolutely won't compromise on — a line you "
            "won't cross even if it costs you socially or professionally? Has that "
            "been tested?"
        ),
        young_follow_up_hint="If hypothetical: ask about a real time they said no when it was hard — even a small moment counts",
        emerging_follow_up_hint="If hypothetical: ask for a real situation where they held their ground at a cost",
    ),

    # ── Social Orientation & Relationships ──
    InterviewQuestion(
        id="social_1",
        dimension="social_orientation",
        question=(
            "How do you typically handle disagreements or conflict with people "
            "you care about? Think of a specific time — what did you do, and "
            "looking back, how do you feel about how you handled it?"
        ),
        purpose="Social cognition, conflict style, emotional regulation, Theory of Mind patterns",
        follow_up_hint="If avoidant: ask what they wish they'd said, or how the relationship changed",
        order=9,
        min_words_for_depth=50,
        young_question=(
            "When you have a disagreement or falling out with someone you care "
            "about — a close friend, a family member — how do you usually handle "
            "it? Can you think of a specific time?"
        ),
        emerging_question=(
            "How do you typically handle conflict or disagreements with people "
            "close to you? Think of a specific time — what did you do, and how "
            "do you feel about it now?"
        ),
        young_follow_up_hint="If avoidant: ask what they wish they'd done differently, or how things changed after",
        emerging_follow_up_hint="If avoidant: ask what they wish they'd said, or how the relationship changed after",
    ),

    # ── Emotional Dynamics ──
    InterviewQuestion(
        id="emotional_triggers",
        dimension="emotional_dynamics",
        question=(
            "What kinds of moments or situations genuinely light you up — where "
            "you feel most alive, energized, or in your element? And on the flip "
            "side, what drains you or makes you want to disengage?"
        ),
        purpose="Map emotional energy sources and drains — what the clone should get excited about vs withdraw from",
        follow_up_hint="Can you give me a specific recent example of each?",
        order=10,
        min_words_for_depth=50,
        young_question=(
            "What makes you feel most hyped or in your zone? And what totally "
            "drains your energy or makes you want to check out?"
        ),
        emerging_question=(
            "What kinds of situations make you feel most alive and energized? "
            "And what drains you — like really makes you want to disengage?"
        ),
        young_follow_up_hint="Can you think of a specific time for each — one where you were in your zone and one where you were completely drained?",
        emerging_follow_up_hint="Can you give me a specific recent example of each?",
    ),
    InterviewQuestion(
        id="emotional_processing",
        dimension="emotional_dynamics",
        question=(
            "When something really gets to you emotionally — anger, excitement, "
            "sadness, stress — how does it show? Are you someone who processes "
            "out loud or internally? Do you react fast or does it hit you later?"
        ),
        purpose="Capture emotional expression style, processing speed, and visibility",
        follow_up_hint="Think of the last time you were really angry or stressed — what did people around you see vs what was happening inside?",
        order=11,
        min_words_for_depth=50,
        young_question=(
            "When you're really stressed or upset or excited, how do people "
            "around you know? Or do they not know? Are you more of an "
            "in-your-head person or does it show?"
        ),
        young_follow_up_hint="Think of a recent time you were upset or stressed — what would someone watching you have noticed vs what was actually going on?",
        emerging_follow_up_hint="Think of the last time you were really angry or stressed — what did people see vs what was happening inside?",
    ),
    InterviewQuestion(
        id="emotional_contexts",
        dimension="emotional_dynamics",
        question=(
            "Are you a different person in different settings? Like, how you are "
            "with your closest friend versus a room full of strangers, or at work "
            "versus at home? What shifts?"
        ),
        purpose="Map emotional register across social contexts — when to be guarded vs playful vs professional",
        follow_up_hint="What would surprise someone who only knows you in one of those contexts?",
        order=12,
        min_words_for_depth=40,
        young_question=(
            "Are you a different person with your best friend vs in class vs "
            "with your family? What changes about how you act or feel?"
        ),
        young_follow_up_hint="What would surprise someone who only knows you from one of those settings?",
        emerging_follow_up_hint="What would surprise someone who only knows you in one of those contexts?",
    ),
    InterviewQuestion(
        id="emotional_recovery",
        dimension="emotional_dynamics",
        question=(
            "Think of a time you were really upset, stressed, or knocked off "
            "balance. How did you come back from it? What does your recovery "
            "process look like — do you talk it out, need space, distract yourself?"
        ),
        purpose="Capture recovery patterns and coping mechanisms — the aftermath of emotion",
        follow_up_hint="How long does it usually take? Is there a person or activity that speeds it up?",
        order=13,
        min_words_for_depth=40,
        young_question=(
            "When you've had a really bad day or something knocked you sideways, "
            "how do you get back to feeling okay? What helps?"
        ),
        young_follow_up_hint="How long does it usually take you to bounce back? Is there a person or thing that helps?",
        emerging_follow_up_hint="How long does it usually take? Is there a person or activity that speeds it up?",
    ),

    # ── Knowledge Domains & Expertise ──
    InterviewQuestion(
        id="expertise_1",
        dimension="knowledge_domains",
        question=(
            "What's a topic you know deeply — something where you could talk "
            "for an hour and still have more to say? What draws you to it, "
            "and what's a misconception most people have about it?"
        ),
        purpose="Map knowledge graph topology, expertise depth, and teaching/explanation style",
        follow_up_hint="If too brief: ask them to explain the misconception as if teaching it",
        order=14,
        min_words_for_depth=50,
        young_question=(
            "What's something you know a lot about — something you could talk "
            "about for ages and still have more to say? What got you into it, "
            "and what do most people get wrong about it?"
        ),
        young_follow_up_hint="If too brief: ask them to explain what most people get wrong, like they're teaching a friend",
    ),
    InterviewQuestion(
        id="expertise_2",
        dimension="knowledge_domains",
        question=(
            "What's something you've learned recently that changed how you "
            "think about your work or your field? How did you come across it?"
        ),
        purpose="Capture learning patterns, information sources, intellectual curiosity topology",
        follow_up_hint="If vague: ask what specifically shifted in their thinking",
        order=15,
        min_words_for_depth=40,
        young_question=(
            "What's something you've learned recently — from a class, a video, "
            "a conversation, anything — that actually changed how you think about "
            "something?"
        ),
        emerging_question=(
            "What's something you've come across recently that shifted how you "
            "think about your interests or your work? How did you find it?"
        ),
        young_follow_up_hint="If vague: ask what specifically changed in how they see things now vs before",
        emerging_follow_up_hint="If vague: ask what specifically shifted in their thinking",
    ),

    # ── Communication & Style ──
    InterviewQuestion(
        id="style_1",
        dimension="communication_style",
        question=(
            "If someone who knows you really well had to describe the way you "
            "communicate — your texting style, how you write emails, how you "
            "explain things — what would they say? Do you agree with that?"
        ),
        purpose="Meta-awareness of communication patterns + actual demonstration in the response",
        follow_up_hint="If too short: ask for an example of a message or explanation they're proud of",
        order=16,
        min_words_for_depth=40,
        young_question=(
            "If your closest friend had to describe how you text, talk, or "
            "explain things — what would they say? Do you think they'd be right?"
        ),
        young_follow_up_hint="If too short: ask for an example — maybe a text they sent or a way they explained something they're proud of",
    ),

    # ── Closing: Identity & Reflection ──
    InterviewQuestion(
        id="identity_1",
        dimension="core_beliefs",
        question=(
            "What do you think most people get wrong about you when they first "
            "meet you? What's the gap between how you come across and who you "
            "actually are?"
        ),
        purpose="Self-perception vs social perception — reveals hidden dimensions of personality",
        follow_up_hint="If deflects: ask what their closest friend would say that a coworker wouldn't",
        order=17,
        min_words_for_depth=40,
        young_follow_up_hint="If deflects: ask what their best friend would say about them that most people wouldn't guess",
    ),
    InterviewQuestion(
        id="identity_2",
        dimension="episodic_memory",
        question=(
            "Imagine your AI replica exists and someone who loves you is talking "
            "to it. What would it absolutely need to get right about you for "
            "them to feel like it's really you?"
        ),
        purpose="Elicit what the user considers most essential about their identity — directly calibrates the brain file",
        follow_up_hint="If too general ('my humor'): ask for a specific example of that quality in action",
        order=18,
        min_words_for_depth=40,
        young_question=(
            "Imagine your AI replica exists and your best friend is talking to "
            "it. What would it absolutely need to get right about you for them "
            "to feel like it's really you?"
        ),
        young_follow_up_hint="If too general ('my humor'): ask for a specific example — like something they'd say or do that's unmistakably them",
    ),
]


# Dimension labels for progress UI
DIMENSION_LABELS: dict[str, str] = {
    "identity": "Your Identity",
    "episodic_memory": "Life Story & Experiences",
    "core_beliefs": "Beliefs & Identity",
    "decision_making": "How You Think & Decide",
    "values": "What Drives You",
    "boundaries": "Your Lines in the Sand",
    "social_orientation": "Relationships & Conflict",
    "emotional_dynamics": "Your Emotional Landscape",
    "knowledge_domains": "What You Know Deeply",
    "communication_style": "How You Communicate",
    "procedural_memory": "How You Work",
}

# Quick lookup by question ID
QUESTION_MAP: dict[str, InterviewQuestion] = {q.id: q for q in INTERVIEW_QUESTIONS}

TOTAL_QUESTIONS = len(INTERVIEW_QUESTIONS)


def get_all_interview_questions(include_work_session: bool = True) -> list[tuple[str, str]]:
    """Return all interview questions as (question_id, question_text) pairs.

    Args:
        include_work_session: If True, append the 4 work-session questions.
    """
    questions = [(q.id, q.question) for q in INTERVIEW_QUESTIONS]
    if include_work_session:
        questions += [(q.id, q.question) for q in WORK_SESSION_QUESTIONS]
    return questions


# ──────────────────────────────────────────────────────────
# Work-Session Interview Track (plan/10 §9)
#
# A separate, optional 4-question set that captures procedural memory:
# how the person reasons, prompts, debugs, and reviews. Role-agnostic and
# narrative — not a skills checklist.
#
# Lives parallel to INTERVIEW_QUESTIONS so the personality interview is
# unchanged. Routed through the procedural_memory dimension.
# ──────────────────────────────────────────────────────────

WORK_SESSION_QUESTIONS: list[InterviewQuestion] = [
    InterviewQuestion(
        id="work_hard_problem",
        dimension="procedural_memory",
        question=(
            "Walk me through the last hard problem you solved at work — could be "
            "code, a design, a tricky decision, anything that took real thought. "
            "What did you try first? What did you rule out, and how? When did "
            "you decide you were done?"
        ),
        purpose="Map decomposition style, error taxonomy, abstraction timing, and stop conditions",
        follow_up_hint="If too abstract: ask for the specific first action they took, and what made them stop",
        order=100,
        min_words_for_depth=80,
        emerging_question=(
            "Walk me through the last hard problem you wrestled with — "
            "schoolwork, a side project, a tough call, anything. What did you "
            "try first? What did you rule out? When did you decide it was good enough?"
        ),
        emerging_follow_up_hint="If abstract: ask for the very first thing they actually did, and what told them they were finished",
    ),
    InterviewQuestion(
        id="work_when_stuck",
        dimension="procedural_memory",
        question=(
            "When you're stuck on something — really stuck — what does being "
            "stuck feel like for you, and what do you actually do about it? "
            "Talk me through a real recent example."
        ),
        purpose="Recovery loop, consultation patterns, frustration signals — captures what they do when the main approach fails",
        follow_up_hint="If too brief: ask whether they ask for help, take a break, switch tools, or push harder — and what made them choose that",
        order=101,
        min_words_for_depth=60,
    ),
    InterviewQuestion(
        id="work_prompting_style",
        dimension="procedural_memory",
        question=(
            "When you're directing an AI assistant, or delegating to a "
            "teammate, how do you set them up to do good work? Show me how "
            "you'd phrase a real request you've made recently — the actual "
            "words, as much as you remember."
        ),
        purpose="Prompting and delegation style — structure, length, constraint phrasing, correction approach",
        follow_up_hint="If they give a polished version: ask what they actually wrote, including any rough edges. If they describe instead of showing, ask for the literal phrasing.",
        order=102,
        min_words_for_depth=50,
        young_question=(
            "When you ask an AI for help — ChatGPT, anything like that — or "
            "when you're explaining something to a friend who's helping you, "
            "how do you set them up? What does a real recent message look like?"
        ),
        young_follow_up_hint="If polished: ask for what they actually typed, even the messy version",
    ),
    InterviewQuestion(
        id="work_review_style",
        dimension="procedural_memory",
        question=(
            "Describe how you typically review someone else's work — code, a "
            "design, a draft, a plan, whatever applies to what you do. What "
            "do you look at first? How do you decide what's worth flagging "
            "vs. letting go?"
        ),
        purpose="Review depth, evaluation criteria, feedback delivery — captures critique style",
        follow_up_hint="If general: ask for a recent specific review — what they said, and what they chose NOT to comment on",
        order=103,
        min_words_for_depth=50,
    ),
]


# Quick lookup by question ID for the work-session track
WORK_SESSION_QUESTION_MAP: dict[str, InterviewQuestion] = {
    q.id: q for q in WORK_SESSION_QUESTIONS
}

TOTAL_WORK_SESSION_QUESTIONS = len(WORK_SESSION_QUESTIONS)
