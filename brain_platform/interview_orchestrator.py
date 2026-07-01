"""Adaptive interview orchestrator — ties the lifted interview components together.

Lifts the cloud's ``InterviewOrchestrator`` pattern (formerly in
``brain/interview_orchestrator.py`` for the simple 18-question script)
and wires it to the cloud-quality pipeline:

  - :func:`brain_platform.pipeline.interview.questions`: question bank
  - :func:`brain_platform.pipeline.interview.follow_up`: adaptive LLM
    depth check + follow-up generation
  - :func:`brain_platform.pipeline.interview.coverage_scorer`:
    per-dimension coverage scoring
  - :func:`brain_platform.pipeline.interview.gap_identifier`:
    4-rule termination
  - :func:`brain_platform.pipeline.interview.question_selector`:
    reactive next-question picker

The offline default ``brain/interview_orchestrator.py`` stays untouched
— that's the 18-question deterministic path. This is the
LLM-powered path behind ``beam interview --adaptive``.

Public API:

  orch = AdaptiveInterviewOrchestrator(
      llm=LLMAdapter(),
      user_age=40,
      max_questions=19,
  )
  question = orch.start()
  while not orch.is_complete():
      answer = input(f\"Q ({orch.question_count}): {question}\\nA: \")
      result = orch.answer(answer)
      if result.follow_up:
          fu = input(f\"  Follow-up: {result.follow_up}\\nA: \")
          result = orch.answer(fu)
      question = result.next_question
  transcript = orch.get_transcript()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from brain_platform.pipeline.interview.coverage_scorer import (
    DimensionCoverageScorer,
)
from brain_platform.pipeline.interview.gap_identifier import GapIdentifier
from brain_platform.pipeline.interview.questions import (
    INTERVIEW_QUESTIONS,
    InterviewQuestion,
    TIER_STANDARD,
    age_to_tier,
    get_question_for_tier,
    get_min_words_for_tier,
)
from brain_platform.pipeline.interview.question_selector import QuestionSelector
from brain_platform.pipeline.interview.follow_up import (
    needs_follow_up,
    generate_follow_up,
)
from brain_platform.services.llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """The result of answering one question in the interview."""

    next_question: Optional[InterviewQuestion]
    """The next question to ask. None if the interview is complete."""

    follow_up: Optional[str] = None
    """An LLM-generated follow-up question, if the answer was shallow."""

    is_complete: bool = False
    """True if the interview should terminate after this answer."""

    termination_reason: Optional[str] = None
    """If ``is_complete``, why the interview stopped."""

    question_count: int = 0
    """How many questions have been asked so far (including this one)."""


class AdaptiveInterviewOrchestrator:
    """Cloud-quality adaptive interview in 4 phases per answer:

    1. Record the answer
    2. Check depth (LLM call) — if shallow, generate a follow-up
    3. Update the in-memory graph with the answer
    4. Score coverage across the 10 personality dimensions
    5. Decide: terminate, or pick the next question (reactive)
    """

    def __init__(
        self,
        llm: LLMAdapter,
        *,
        user_age: int = 30,
        max_questions: int = 19,
        max_followups_per_question: int = 1,
    ):
        self.llm = llm
        self.tier = age_to_tier(user_age)
        self.max_questions = max_questions
        self.max_followups = max_followups_per_question

        self.answers: list[dict] = []
        self.questions_asked: list[tuple[InterviewQuestion, str]] = []
        self.questions_skipped: set[str] = set()
        self.followups_asked: dict[str, int] = {}  # question_id → count
        self.turn_count = 0

        self.scorer = DimensionCoverageScorer()
        self.gap_identifier = GapIdentifier()
        self.selector = QuestionSelector()

        # Start with an empty graph; the build pipeline writes to it
        # as answers come in. The schema adapter is the source of
        # truth for what nodes look like.
        self._graph: dict = {}

    def start(self) -> InterviewQuestion:
        """Return the first question to ask."""
        return self._next_question_to_ask()

    def answer(self, answer_text: str) -> AnswerResult:
        """Record an answer, generate a follow-up if shallow, return next question.

        Args:
            answer_text: The user's answer to the most recent question.

        Returns:
            ``AnswerResult`` with the next question (or None if complete),
            an optional follow-up question to ask first, and the
            termination status.
        """
        if not self.questions_asked and not self.questions_skipped:
            raise RuntimeError("start() must be called before answer()")
        current_q, current_a = self.questions_asked[-1]
        self.turn_count += 1

        # 1. Record the answer
        self.answers.append({
            "question_id": current_q.id,
            "question": current_q.question,
            "answer": answer_text,
            "domain": current_q.dimension,
            "tier": self.tier,
        })

        # 2. Update the in-memory graph from the answer
        #    (full ingestion is the Chunk 3 work; for now we keep a
        #    raw transcript that the coverage scorer can read).
        self._graph.setdefault("raw_transcript", "")
        self._graph["raw_transcript"] += (
            f"[{current_q.dimension}] Q: {current_q.question}\n"
            f"A: {answer_text}\n\n"
        )

        # 3. Check if a follow-up is needed. The AGE_QUESTION and
        # IDENTITY_NAME questions are metadata anchors (collect the
        # user's age and name for graph labeling) — they're not
        # part of the adaptive interview, so we never trigger a
        # follow-up or LLM depth check for them. Any answer
        # advances immediately.
        follow_up_text = None
        followups_for_this = self.followups_asked.get(current_q.id, 0)
        if (
            current_q.id not in ("age", "identity_name")
            and followups_for_this < self.max_followups
        ):
            if needs_follow_up(self.llm, current_q, answer_text, self.tier):
                follow_up_text = generate_follow_up(
                    self.llm, current_q, answer_text, self.tier,
                )
                self.followups_asked[current_q.id] = followups_for_this + 1
                logger.info(
                    "Adaptive interview: follow-up triggered for '%s' (depth-checked shallow)",
                    current_q.id,
                )

        if follow_up_text:
            # Don't advance the question counter or coverage yet — the
            # follow-up is a sub-question. Caller should call
            # answer(follow_up_text) to record the follow-up answer.
            return AnswerResult(
                next_question=current_q,  # caller calls .answer() again
                follow_up=follow_up_text,
                question_count=len(self.questions_asked),
            )

        # 4. Mark the question as done and update coverage
        self.questions_skipped.discard(current_q.id)
        self._update_coverage(current_q, answer_text)

        # 5. Decide: terminate or continue
        next_q = self._next_question_to_ask()
        is_complete = next_q is None
        return AnswerResult(
            next_question=next_q,
            is_complete=is_complete,
            question_count=len(self.questions_asked),
        )

    def is_complete(self) -> bool:
        """True if the interview has terminated."""
        return self.gap_identifier.analyze(
            self.scorer.score(self._graph),
            questions_asked=len(self.questions_asked),
        ).should_terminate

    def get_transcript(self) -> dict:
        """Return the full interview transcript + metadata."""
        return {
            "answers": self.answers,
            "transcript": self._graph.get("raw_transcript", ""),
            "questions_asked": [
                {"id": q.id, "dimension": q.dimension, "text": q.question}
                for q, _ in self.questions_asked
            ],
            "followups_asked": dict(self.followups_asked),
            "turn_count": self.turn_count,
            "tier": self.tier,
            "coverage": {
                dim: s.score
                for dim, s in self.scorer.score(self._graph).items()
            },
        }

    def _next_question_to_ask(self) -> Optional[InterviewQuestion]:
        """Pick the next question using the reactive QuestionSelector.

        Falls back to round-robin per dimension if the selector returns
        no result (e.g. all candidates exhausted).
        """
        if len(self.questions_asked) >= self.max_questions:
            return None

        coverage = self.scorer.score(self._graph)
        analysis = self.gap_identifier.analyze(
            coverage, questions_asked=len(self.questions_asked),
        )
        if analysis.should_terminate:
            return None

        # Use the reactive QuestionSelector. Pass the asked question IDs
        # so the selector doesn't pick the same question twice.
        asked_ids = {q.id for q, _ in self.questions_asked}
        try:
            selected = self.selector.select_next(
                coverage=coverage,
                analysis=analysis,
                asked_question_ids=asked_ids,
                questions_asked_count=len(self.questions_asked),
            )
            if selected is not None:
                # Apply the age-tier-appropriate text. The selector
                # returns the standard text; if the user is a young/
                # emerging tier, swap in the variant.
                original = selected.question
                age_text = get_question_for_tier(original, self.tier)
                if age_text != original.question:
                    # Frozen dataclass — construct a new one with the
                    # variant text. The variant only changes `.question`;
                    # all other fields are identical.
                    final_q = InterviewQuestion(
                        id=original.id,
                        dimension=original.dimension,
                        question=age_text,
                        purpose=original.purpose,
                        follow_up_hint=original.follow_up_hint,
                        order=original.order,
                        min_words_for_depth=original.min_words_for_depth,
                    )
                else:
                    final_q = original
                self.questions_asked.append((final_q, ""))
                return final_q
        except Exception as e:
            logger.warning("QuestionSelector failed, falling back to round-robin: %s", e)

        # Round-robin fallback
        asked_ids = {q.id for q, _ in self.questions_asked}
        for q in INTERVIEW_QUESTIONS:
            if q.id not in asked_ids:
                q_text = get_question_for_tier(q, self.tier)
                q = InterviewQuestion(
                    id=q.id, dimension=q.dimension, question=q_text,
                    purpose=q.purpose, follow_up_hint=q.follow_up_hint,
                    order=q.order, min_words_for_depth=q.min_words_for_depth,
                    young_question=q.young_question,
                    emerging_question=q.emerging_question,
                    young_follow_up_hint=q.young_follow_up_hint,
                    emerging_follow_up_hint=q.emerging_follow_up_hint,
                )
                self.questions_asked.append((q, ""))
                return q
        return None

    def _update_coverage(self, question: InterviewQuestion, answer: str) -> None:
        """Update the in-memory graph with the Q&A pair.

        The coverage scorer reads the graph via the schema adapter. We
        don't run the full 3-pass extraction per answer (too slow) —
        instead we count the question itself as a coverage signal. The
        full extraction runs after the interview ends.
        """
        # Add a placeholder memory node for this Q&A
        mems = self._graph.setdefault("memories", [])
        mems.append({
            "name": f"interview-{question.id}",
            "summary": f"Q: {question.question}\nA: {answer[:200]}",
            "emotional_tone": 0.0,
        })


__all__ = ["AdaptiveInterviewOrchestrator", "AnswerResult"]
