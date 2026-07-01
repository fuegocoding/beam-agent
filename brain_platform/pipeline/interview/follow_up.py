"""Adaptive follow-up engine — LLM-powered probing for shallow answers.

When a user's answer is too brief or surface-level, this engine generates
a natural follow-up question that deepens the response. Grounded in
Stanford GenAgents' AI interviewer approach.

Max 1 follow-up per core question to keep the interview ~20 minutes.

Age-adaptive: follow-ups are scaffolded differently for younger users
(Persona-DB cold-start principle — less experienced users need more
guidance to reach depth, not less).
"""

import logging

from brain_platform.pipeline.interview.questions import (
    InterviewQuestion,
    TIER_STANDARD,
    TIER_EMERGING,
    TIER_YOUNG,
    get_follow_up_hint_for_tier,
    get_min_words_for_tier,
)
from brain_platform.services.llm_adapter import (
    LLMAdapter,
    TASK_DEPTH_CHECK,
    TASK_FOLLOW_UP,
)

logger = logging.getLogger(__name__)

# ── Tier-specific context blocks for prompts ──

_TIER_CONTEXT = {
    TIER_YOUNG: (
        "The person being interviewed is a teenager (under 18). They may not have "
        "major life milestones yet, but they DO have meaningful experiences — school, "
        "friendships, family, hobbies, sports/teams, social media, creative projects, "
        "online communities, and part-time work. Their beliefs and identity are actively "
        "forming, and their procedural knowledge (skills, hobbies, things they've built "
        "or created) is often their richest data source. When probing deeper:\n"
        "- Suggest concrete, relatable directions across MULTIPLE contexts — not just "
        "school: hobbies, creative work, online life, teams, family, friends\n"
        "- Keep the tone warm and curious, like an older friend — not a therapist\n"
        "- Don't ask about career, marriage, or decade-spanning change\n"
        "- Smaller moments can be just as revealing — validate that\n"
        "- Skills and things they've made/built are high-signal — lean into those"
    ),
    TIER_EMERGING: (
        "The person being interviewed is a young adult (18-24). They're in a transitional "
        "phase — may have some college, early career, or relationship experiences but "
        "probably haven't had the major life pivots that come later. When probing deeper:\n"
        "- Reference both school/university and early work contexts\n"
        "- They may have recent, vivid belief changes — lean into those\n"
        "- Friendships and identity shifts are often very active at this age"
    ),
    TIER_STANDARD: "",
}

FOLLOW_UP_SYSTEM_PROMPT = """You are an empathetic interviewer helping someone create a digital replica of themselves. You've just asked them a question and their answer was too brief or surface-level to capture who they really are.

Your job: generate ONE natural follow-up question that gently probes deeper. The follow-up should:

1. Acknowledge what they said (don't ignore their answer)
2. Ask for a specific story, example, or experience related to their answer
3. Feel like a curious friend asking, not a therapist or interrogator
4. Be 1-2 sentences max
5. NOT repeat the original question in different words

{tier_context}

Context about why this question matters:
- Dimension: {dimension}
- Purpose: {purpose}
- Hint for deepening: {follow_up_hint}

Return ONLY the follow-up question, nothing else."""

DEPTH_CHECK_SYSTEM_PROMPT = """You are evaluating whether an interview answer provides enough depth and specificity to build a personality profile.

A SUFFICIENT answer:
- Contains at least one specific story, example, or concrete detail
- Reveals something about how the person thinks, feels, or makes decisions
- Goes beyond generic statements anyone could say

An INSUFFICIENT answer:
- Is vague, generic, or could apply to almost anyone
- Lists abstract qualities without concrete examples
- Is extremely brief (1-2 sentences) for a question asking for narrative

{tier_context}

Original question: {question}
Dimension being mapped: {dimension}

Respond with ONLY "sufficient" or "insufficient"."""

_DEPTH_TIER_CONTEXT = {
    TIER_YOUNG: (
        "Note: This person is a teenager. Their experiences are valid even if they "
        "reference school, friendships, hobbies, creative projects, online communities, "
        "sports/teams, or family rather than career or major life events. A specific "
        "story about a friendship conflict, a creative project, a skill they developed, "
        "or a school choice IS sufficient depth. Judge by specificity and personal "
        "insight, not by the weight of the life event."
    ),
    TIER_EMERGING: (
        "Note: This person is a young adult (18-24). Their experiences may reference "
        "university, early career, recent relationship changes, or identity shifts. "
        "These are valid depth — judge by specificity and personal insight."
    ),
    TIER_STANDARD: "",
}


def needs_follow_up(
    llm: LLMAdapter,
    question: InterviewQuestion,
    answer: str,
    tier: str = TIER_STANDARD,
) -> bool:
    """Determine if an answer needs a follow-up probe."""
    word_count = len(answer.split())
    min_words = get_min_words_for_tier(question, tier)

    # Hard floor: very short answers always need follow-up
    if word_count < min_words:
        return True

    # Long answers are generally sufficient
    if word_count > 120:
        return False

    # Medium-length: ask the LLM to judge depth
    try:
        prompt = DEPTH_CHECK_SYSTEM_PROMPT.format(
            question=question.question,
            dimension=question.dimension,
            tier_context=_DEPTH_TIER_CONTEXT.get(tier, ""),
        )
        # Sync call via the local LLMAdapter (the cloud's LLMService
        # was async; the LLMAdapter wraps call_llm which is sync).
        result = llm.generate_response(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": answer},
            ],
            task=TASK_DEPTH_CHECK,
        )
        return result.strip().lower() == "insufficient"
    except Exception:
        logger.warning("Depth check LLM call failed, skipping follow-up")
        return False


def generate_follow_up(
    llm: LLMAdapter,
    question: InterviewQuestion,
    answer: str,
    tier: str = TIER_STANDARD,
) -> str:
    """Generate an adaptive follow-up question based on the user's answer."""
    follow_up_hint = get_follow_up_hint_for_tier(question, tier)

    prompt = FOLLOW_UP_SYSTEM_PROMPT.format(
        dimension=question.dimension,
        purpose=question.purpose,
        follow_up_hint=follow_up_hint,
        tier_context=_TIER_CONTEXT.get(tier, ""),
    )

    user_content = (
        f"Original question: {question.question}\n\n"
        f"Their answer: {answer}"
    )

    try:
        result = llm.generate_response(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            task=TASK_FOLLOW_UP,
        )
        return result.strip().strip('"')
    except Exception:
        logger.warning("Follow-up generation failed, using hint fallback")
        return follow_up_hint.split(": ", 1)[-1] if ": " in follow_up_hint else follow_up_hint
