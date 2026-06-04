"""Interview orchestrator — Python-native multi-pass interview.

Uses Hermes' call_llm for question generation and answer analysis.
No Rust binary dependency for LLM logic.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DOMAINS = ["identity", "relationships", "work", "emotional", "beliefs", "procedural"]

DOMAIN_QUESTIONS = {
    "identity": [
        "Who are you at your core? Not your job title or roles — who are you when no one's watching?",
        "What are the 3-5 traits that define how you move through the world?",
        "What values would you defend no matter what?",
    ],
    "relationships": [
        "Who are the 3-5 most important people in your life, and what does each teach you?",
        "How do you show up in relationships — what's your default mode?",
        "What's a boundary you've learned to set the hard way?",
    ],
    "work": [
        "How do you break down a hard problem? Walk me through your actual process.",
        "What's your relationship with risk — when do you bet big vs. play safe?",
        "How do you decide when to delegate vs. do it yourself?",
    ],
    "emotional": [
        "What drains your energy fast? What recharges you?",
        "When you're frustrated, what's usually the root cause?",
        "How do you recover from a really bad day?",
    ],
    "beliefs": [
        "What's something you believe that most people around you would disagree with?",
        "What topic have you completely changed your mind about?",
        "What's a conviction you hold that you arrived at through experience, not teaching?",
    ],
    "procedural": [
        "When you're stuck on a problem, what's your actual debugging process?",
        "How do you make big decisions — gut feel, spreadsheet, sleep on it?",
        "What's your approach to learning something completely new?",
    ],
}

SYSTEM_PROMPT = """You are conducting a deep personality interview. Your goal is to understand this person deeply — their traits, beliefs, values, emotional patterns, work style, and how they think.

Rules:
- Ask ONE question at a time
- Be conversational, not clinical
- If an answer is vague, ask for a specific example
- If something contradicts an earlier answer, gently explore it
- If something is emotionally significant, acknowledge it before moving on
- Keep questions focused — don't ask compound questions
- After 3-4 answers in a domain, move to the next domain
- After covering all domains, signal completion

Output format (strict JSON):
{
  "action": "ask" | "followup" | "move_to_domain" | "complete",
  "question": "the next question to ask",
  "domain": "current domain name",
  "reason": "brief reason for this question",
  "coverage_note": "what we still need to learn"
}"""

EXTRACTION_PROMPT = """You are a personality analyst. Given a transcript of a deep interview, extract a structured personality graph.

Extract the following, each as a JSON array:
- traits: personality traits with name (string), strength (0-1 float), summary (1 sentence)
- beliefs: core beliefs with name, confidence (0-1 float), summary
- values: core values with name, importance (0-1 float), summary
- boundaries: personal boundaries with topic, comfort_level (0-1 float), summary
- life_events: significant life events with event, year, impact, summary
- people: key people with name, relationship, significance
- voice_dna: communication style with characteristic_phrases (list), phrases_to_avoid (list), humor_style, response_length_pattern, formal_range, storytelling_style
- work_dna: work style with decomposition_style, debugging_approach, risk_posture, delegation_style, documentation_habit
- emotional_profile: triggers (list of {stimulus, reaction, intensity}), energy_sources (list), energy_drains (list)
- user_summary: a 2-3 sentence summary of who this person is

Output ONLY valid JSON matching this structure:
{
  "user_summary": "...",
  "traits": [...],
  "beliefs": [...],
  "values": [...],
  "boundaries": [...],
  "life_events": [...],
  "people": [...],
  "voice_dna": {...},
  "work_dna": {...},
  "emotional_profile": {...}
}"""

SOUL_PROMPT = """You are writing a SOUL.md file — a personal identity document that an AI agent loads to understand who it's talking to. Given a personality graph, write a warm, natural SOUL.md.

The SOUL.md should:
- Start with "# Soul" 
- Have a "## Who I Am" section (2-3 sentences, natural voice)
- Have sections for traits, values, beliefs, communication style, work style
- Be written so the agent can reference it naturally in conversation
- Feel like the person wrote it about themselves, not like a clinical report
- Be 200-400 words

Output ONLY the markdown content, no wrapping."""


def _call_llm(system: str, user: str, temperature: float = 0.7, max_tokens: int = 4000) -> str:
    """Make a one-shot LLM call via Hermes auxiliary client."""
    from agent.auxiliary_client import call_llm
    response = call_llm(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=120.0,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


class InterviewOrchestrator:
    """Manages the multi-pass interview process using LLM."""

    def __init__(self):
        self.answers: list[dict] = []
        self.current_pass = 1
        self.current_domain_idx = 0
        self.questions_asked: dict[str, int] = {d: 0 for d in DOMAINS}
        self.graph_draft: Optional[dict] = None

    def start(self) -> dict:
        """Start a new interview — return the first question."""
        domain = DOMAINS[0]
        question = DOMAIN_QUESTIONS[domain][0]
        self.questions_asked[domain] = 1

        return {
            "question_id": f"{domain}_1",
            "question": question,
            "domain": domain,
            "pass": 1,
            "total_domains": len(DOMAINS),
            "domains_covered": 0,
        }

    def answer(self, question_id: str, question: str, answer_text: str, domain: str) -> dict:
        """Record an answer and decide what to ask next."""
        self.answers.append({
            "question_id": question_id,
            "question": question,
            "answer": answer_text,
            "domain": domain,
        })

        self.questions_asked[domain] = self.questions_asked.get(domain, 0) + 1

        # After enough answers, use LLM to decide next move
        if len(self.answers) >= 2:
            return self._llm_next_step(domain)

        # For early questions, use scripted progression
        return self._scripted_next(domain)

    def _scripted_next(self, domain: str) -> dict:
        """Scripted question progression for early interview."""
        domain_idx = DOMAINS.index(domain)
        questions_in_domain = self.questions_asked[domain]

        # Move to next domain after 3 questions
        if questions_in_domain >= 3:
            next_idx = (domain_idx + 1) % len(DOMAINS)
            if next_idx <= domain_idx and self.current_pass == 1:
                # Completed first pass through all domains
                self.current_pass = 2
            next_domain = DOMAINS[next_idx]
            self.current_domain_idx = next_idx

            if self.current_pass >= 2 and self.questions_asked[next_domain] >= 2:
                # Enough data — check if we should complete
                return self._check_completion()

            q_idx = min(self.questions_asked[next_domain], len(DOMAIN_QUESTIONS[next_domain]) - 1)
            question = DOMAIN_QUESTIONS[next_domain][q_idx]
            self.questions_asked[next_domain] += 1

            return {
                "question_id": f"{next_domain}_{self.questions_asked[next_domain]}",
                "question": question,
                "domain": next_domain,
                "pass": self.current_pass,
                "action": "move_to_domain",
            }

        # Continue in same domain
        q_idx = min(questions_in_domain, len(DOMAIN_QUESTIONS[domain]) - 1)
        question = DOMAIN_QUESTIONS[domain][q_idx]

        return {
            "question_id": f"{domain}_{questions_in_domain + 1}",
            "question": question,
            "domain": domain,
            "pass": self.current_pass,
            "action": "ask",
        }

    def _llm_next_step(self, domain: str) -> dict:
        """Use LLM to decide what to ask next based on answers so far."""
        # Build context from recent answers
        recent = self.answers[-5:]
        context = "\n".join(
            f"Q: {a['question']}\nA: {a['answer']}\n(domain: {a['domain']})"
            for a in recent
        )

        domain_coverage = json.dumps(self.questions_asked)

        prompt = f"""Current domain: {domain}
Pass: {self.current_pass}
Questions asked per domain: {domain_coverage}

Recent answers:
{context}

What should I ask next? Consider:
- Is the current domain sufficiently explored (3+ substantive answers)?
- Are there follow-ups needed for vague or interesting answers?
- Should I move to a new domain?
- After all domains have 2+ answers, consider completing.

Respond with JSON: {{"action": "ask|followup|move_to_domain|complete", "question": "...", "domain": "...", "reason": "..."}}"""

        try:
            result_text = _call_llm(SYSTEM_PROMPT, prompt, temperature=0.7, max_tokens=500)
            result = _parse_json(result_text)
        except Exception as e:
            logger.warning("LLM next-step failed, falling back to scripted: %s", e)
            return self._scripted_next(domain)

        action = result.get("action", "ask")
        new_domain = result.get("domain", domain)
        question = result.get("question", "")

        if action == "complete" or self._should_complete():
            return self._complete()

        if new_domain != domain:
            self.current_domain_idx = DOMAINS.index(new_domain) if new_domain in DOMAINS else 0

        q_id = f"{new_domain}_{self.questions_asked.get(new_domain, 0) + 1}"
        self.questions_asked[new_domain] = self.questions_asked.get(new_domain, 0) + 1

        return {
            "question_id": q_id,
            "question": question,
            "domain": new_domain,
            "pass": self.current_pass,
            "action": action,
            "reason": result.get("reason", ""),
        }

    def _should_complete(self) -> bool:
        """Check if we have enough data to complete."""
        total = sum(self.questions_asked.values())
        domains_with_2 = sum(1 for v in self.questions_asked.values() if v >= 2)
        return total >= 15 and domains_with_2 >= 4

    def _check_completion(self) -> dict:
        """Check if interview should complete."""
        if self._should_complete():
            return self._complete()
        # Continue with next domain
        next_domain = DOMAINS[self.current_domain_idx]
        return self._scripted_next(next_domain)

    def _complete(self) -> dict:
        """Signal interview completion."""
        return {
            "status": "complete",
            "total_answers": len(self.answers),
            "domains_covered": {d: c for d, c in self.questions_asked.items()},
            "summary": f"Interview complete. {len(self.answers)} answers across {sum(1 for v in self.questions_asked.values() if v > 0)} domains.",
        }

    def get_full_transcript(self) -> dict:
        """Get the full interview data for brain building."""
        return {
            "answers": self.answers,
            "transcript": "\n\n".join(
                f"[{a['domain']}] Q: {a['question']}\nA: {a['answer']}"
                for a in self.answers
            ),
        }
