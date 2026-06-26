"""Interview orchestrator — Python-native multi-pass interview.

Drives a deterministic question progression through a fixed set of
domains. Pure data — no LLM calls. The LLM-based next-step planner that
used to live in `_llm_next_step` was removed so the brain subsystem
stays fully offline; the scripted progression walks each domain for
~3 questions, runs a second pass, then signals completion.
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


class InterviewOrchestrator:
    """Manages the multi-pass interview process using a scripted progression."""

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
        """Record an answer and decide what to ask next.

        Always uses the scripted progression — there is no LLM next-step
        planner anymore. The interview walks each domain for ~3 questions
        in pass 1, repeats a shorter pass 2 to fill gaps, then signals
        completion once coverage thresholds are met.
        """
        self.answers.append({
            "question_id": question_id,
            "question": question,
            "answer": answer_text,
            "domain": domain,
        })

        self.questions_asked[domain] = self.questions_asked.get(domain, 0) + 1
        return self._scripted_next(domain)

    def _scripted_next(self, domain: str) -> dict:
        """Scripted question progression — deterministic, no LLM."""
        domain_idx = DOMAINS.index(domain)
        questions_in_domain = self.questions_asked[domain]

        if questions_in_domain >= 3:
            next_idx = (domain_idx + 1) % len(DOMAINS)
            if next_idx <= domain_idx and self.current_pass == 1:
                self.current_pass = 2
            next_domain = DOMAINS[next_idx]
            self.current_domain_idx = next_idx

            if self.current_pass >= 2 and self.questions_asked[next_domain] >= 2:
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

        q_idx = min(questions_in_domain, len(DOMAIN_QUESTIONS[domain]) - 1)
        question = DOMAIN_QUESTIONS[domain][q_idx]

        return {
            "question_id": f"{domain}_{questions_in_domain + 1}",
            "question": question,
            "domain": domain,
            "pass": self.current_pass,
            "action": "ask",
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
