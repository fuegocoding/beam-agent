"""Question selector for the reactive interview.

Given a coverage gap map and a set of already-asked question IDs, selects
the highest-value next question from all available banks (core personality,
work session, and professional sector banks).
"""

import logging
from dataclasses import dataclass

from brain_platform.models.enums import InterviewMode, ProfessionalCategory
from brain_platform.pipeline.interview.coverage_types import DimensionScore
from brain_platform.pipeline.interview.gap_identifier import GapAnalysis
from brain_platform.pipeline.interview.questions import (
    INTERVIEW_QUESTIONS,
    WORK_SESSION_QUESTIONS,
    InterviewQuestion,
)

# Professional sector question bank is a Chunk 2+ lift (1,578 LOC).
# The question selector works without it; the professional-routing
# path is disabled until the bank is lifted. Lifting
# ``professional_questions.py`` is a follow-on — it has zero cross-
# imports beyond ``models.enums`` which is already lifted.
try:
    from brain_platform.pipeline.interview.professional_questions import (
        PROFESSIONAL_QUESTIONS,
        get_questions,
    )
    _PROFESSIONAL_QUESTIONS_AVAILABLE = True
except ImportError:
    PROFESSIONAL_QUESTIONS = {}  # type: ignore[misc]
    _PROFESSIONAL_QUESTIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Reactive interview limits ──
MAX_QUESTIONS = 19

# ── Professional dimension → canonical personality dimension mapping ──

_PROF_DIM_MAP: dict[str, str] = {
    "problem_decomposition": "decision_making",
    "decision_framework": "decision_making",
    "collaboration_mode": "social_orientation",
    "failure_handling": "emotional_dynamics",
    "stress_response": "emotional_dynamics",
    "domain_ethics": "values",
    "output_signature": "communication_style",
    "growth_orientation": "knowledge_domains",
    "stakeholder_navigation": "social_orientation",
    "stakeholder_communication": "social_orientation",
    "user_understanding": "knowledge_domains",
    "quality_standards": "values",
    "model_skepticism": "core_beliefs",
    "risk_framing": "decision_making",
    "knowledge_hierarchy": "knowledge_domains",
    "relationship_to_uncertainty": "decision_making",
    "student_modeling": "social_orientation",
    "curriculum_adaptation": "decision_making",
    "explanation_architecture": "communication_style",
    "assessment_design": "decision_making",
    "feedback_design": "communication_style",
    "group_dynamics": "social_orientation",
    "evaluation_design": "decision_making",
    "uncertainty_communication": "communication_style",
    "data_intuition": "decision_making",
    "hypothesis_formation": "decision_making",
    "evaluation_design": "decision_making",
}

# Questions that must be asked in fixed positions
_MANDATORY_FIRST = {"identity_name"}
_MANDATORY_SECOND = {"life_story_1"}
_MANDATORY_LAST = {"identity_2"}


@dataclass
class SelectedQuestion:
    """Result of question selection with scoring metadata."""

    question: InterviewQuestion
    gap_dimension: str       # the dimension this question targets
    score: float             # selection score (for debugging)
    bank: str                # which bank the question came from


class QuestionSelector:
    """Select the next question from all available banks."""

    def __init__(
        self,
        mode: InterviewMode = InterviewMode.REACTIVE,
        sector: ProfessionalCategory | None = None,
    ):
        self._mode = mode
        self._sector = sector

    def select_next(
        self,
        coverage: dict[str, DimensionScore],
        analysis: GapAnalysis,
        asked_question_ids: set[str],
        questions_asked_count: int,
    ) -> SelectedQuestion | None:
        """Select the highest-value next question.

        Returns None if no suitable question remains (should not happen
        before MAX_QUESTIONS because of mandatory questions).
        """
        # ── Mandatory position enforcement ──
        if questions_asked_count == 0:
            q = self._find_by_id("identity_name")
            if q and q.id not in asked_question_ids:
                return SelectedQuestion(question=q, gap_dimension="identity", score=1.0, bank="core")

        if questions_asked_count == 1:
            q = self._find_by_id("life_story_1")
            if q and q.id not in asked_question_ids:
                return SelectedQuestion(question=q, gap_dimension="episodic_memory", score=1.0, bank="core")

        # Build the candidate pool
        candidates = self._build_candidates(coverage, asked_question_ids)
        if not candidates:
            # Fallback: if we've asked enough core questions, try the closing question
            if questions_asked_count >= 7:
                q = self._find_by_id("identity_2")
                if q and q.id not in asked_question_ids:
                    return SelectedQuestion(question=q, gap_dimension="core_beliefs", score=0.5, bank="core")
            logger.warning("No candidate questions remaining (asked=%d)", questions_asked_count)
            return None

        # Score each candidate
        scored = []
        for cand in candidates:
            s = self._score_candidate(cand, coverage, analysis, asked_question_ids)
            scored.append((s, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        best_score = scored[0][0]

        # If we're near the end and identity_2 hasn't been asked, force it
        if questions_asked_count >= MAX_QUESTIONS - 1:
            q = self._find_by_id("identity_2")
            if q and q.id not in asked_question_ids:
                return SelectedQuestion(question=q, gap_dimension="core_beliefs", score=best_score, bank="core")

        return SelectedQuestion(
            question=best.question,
            gap_dimension=best.gap_dimension,
            score=round(best_score, 4),
            bank=best.bank,
        )

    def _build_candidates(
        self,
        coverage: dict[str, DimensionScore],
        asked_question_ids: set[str],
    ) -> list[SelectedQuestion]:
        """Build list of all questions that target under-covered dimensions."""
        candidates: list[SelectedQuestion] = []

        # Add core personality questions
        for q in INTERVIEW_QUESTIONS:
            if q.id in asked_question_ids:
                continue
            if q.id in _MANDATORY_FIRST or q.id in _MANDATORY_SECOND or q.id in _MANDATORY_LAST:
                continue  # handled separately
            candidates.append(SelectedQuestion(
                question=q,
                gap_dimension=q.dimension,
                score=0.0,
                bank="core",
            ))

        # Add work session questions
        for q in WORK_SESSION_QUESTIONS:
            if q.id in asked_question_ids:
                continue
            candidates.append(SelectedQuestion(
                question=q,
                gap_dimension="procedural_memory",
                score=0.0,
                bank="work_session",
            ))

        # Add professional bank questions (if sector is set)
        if self._sector and self._mode == InterviewMode.REACTIVE:
            prof_questions = get_questions(self._sector)
            for q in prof_questions:
                if q.id in asked_question_ids:
                    continue
                canonical_dim = _PROF_DIM_MAP.get(q.dimension, q.dimension)
                candidates.append(SelectedQuestion(
                    question=q,
                    gap_dimension=canonical_dim,
                    score=0.0,
                    bank=f"professional_{self._sector.value}",
                ))

        return candidates

    def _score_candidate(
        self,
        candidate: SelectedQuestion,
        coverage: dict[str, DimensionScore],
        analysis: GapAnalysis,
        asked_question_ids: set[str],
    ) -> float:
        """Compute a selection score for a candidate question."""

        gap_dim = candidate.gap_dimension
        gap_score = 1.0 - coverage.get(gap_dim, DimensionScore(gap_dim, 0.0, 0, 0, 0.0, 0.0, "")).score

        # Relevance: does this question directly target a gap dimension?
        # 1.0 if direct match, 0.7 if cross-bank mapped match
        relevance = 1.0
        if candidate.bank.startswith("professional_"):
            original_dim = candidate.question.dimension
            if original_dim != gap_dim:
                relevance = 0.7  # cross-bank mapping

        # Novelty: never repeat
        if candidate.question.id in asked_question_ids:
            return 0.0
        novelty = 1.0

        # Sector bonus: if user is in this sector, boost professional questions
        sector_bonus = 1.0
        if self._sector and candidate.bank == f"professional_{self._sector.value}":
            sector_bonus = 1.3
        elif candidate.bank.startswith("professional_") and candidate.bank != f"professional_{self._sector.value}":
            sector_bonus = 0.8

        # Information gain: prefer questions with higher min_words_for_depth
        # (they tend to elicit richer answers)
        info_gain = min(1.0, candidate.question.min_words_for_depth / 80.0)

        # Diversity penalty: if this dimension already has many questions asked,
        # slightly penalize to encourage spreading coverage
        dim_questions_asked = sum(
            1 for qid in asked_question_ids
            if self._get_dimension_for_question_id(qid) == gap_dim
        )
        diversity_penalty = max(0.7, 1.0 - (dim_questions_asked * 0.05))

        final_score = gap_score * relevance * novelty * sector_bonus * info_gain * diversity_penalty
        return final_score

    def _find_by_id(self, question_id: str) -> InterviewQuestion | None:
        """Look up a question by ID across all banks."""
        for q in INTERVIEW_QUESTIONS:
            if q.id == question_id:
                return q
        for q in WORK_SESSION_QUESTIONS:
            if q.id == question_id:
                return q
        if self._sector:
            for q in get_questions(self._sector):
                if q.id == question_id:
                    return q
        return None

    def _get_dimension_for_question_id(self, question_id: str) -> str:
        """Return the canonical dimension for a given question ID."""
        q = self._find_by_id(question_id)
        if not q:
            return "unknown"
        if q.dimension in _PROF_DIM_MAP:
            return _PROF_DIM_MAP[q.dimension]
        return q.dimension
