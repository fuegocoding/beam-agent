"""Gap identifier and termination check for the reactive interview.

Given dimension coverage scores, determines:
1. Which dimensions are under-covered (gaps)
2. Whether the interview has collected enough data to terminate
"""

import logging
from dataclasses import dataclass

from brain_platform.pipeline.interview.coverage_types import DimensionScore

logger = logging.getLogger(__name__)

# Coverage thresholds
CRITICAL_GAP_THRESHOLD = 0.30
MODERATE_GAP_THRESHOLD = 0.60
ADEQUATE_THRESHOLD = 0.60
WELL_COVERED_THRESHOLD = 0.85

# Interview bounds
MIN_QUESTIONS = 8
MAX_QUESTIONS = 19
MIN_CORE_QUESTIONS = 3  # must ask at least this many from the core personality bank


@dataclass
class GapAnalysis:
    """Result of analyzing dimension coverage gaps."""

    critical_gaps: list[str]           # dimensions with score < 0.30
    moderate_gaps: list[str]           # dimensions with 0.30 <= score < 0.60
    adequate_dimensions: list[str]     # dimensions with score >= 0.60
    total_score: float                 # average score across all dimensions
    should_terminate: bool             # True if interview is complete
    termination_reason: str | None     # why we terminated (if should_terminate)


class GapIdentifier:
    """Identify coverage gaps and decide when to stop the interview."""

    def analyze(
        self,
        coverage: dict[str, DimensionScore],
        questions_asked: int,
    ) -> GapAnalysis:
        """Analyze coverage scores and determine if the interview should continue."""
        critical_gaps = []
        moderate_gaps = []
        adequate_dimensions = []
        total_score = 0.0

        for dim, score in coverage.items():
            total_score += score.score
            if score.score < CRITICAL_GAP_THRESHOLD:
                critical_gaps.append(dim)
            elif score.score < MODERATE_GAP_THRESHOLD:
                moderate_gaps.append(dim)
            else:
                adequate_dimensions.append(dim)

        avg_score = total_score / len(coverage) if coverage else 0.0

        # ── Termination rules ──

        # Rule 1: Hard minimum not yet reached
        if questions_asked < MIN_QUESTIONS:
            return GapAnalysis(
                critical_gaps=critical_gaps,
                moderate_gaps=moderate_gaps,
                adequate_dimensions=adequate_dimensions,
                total_score=round(avg_score, 3),
                should_terminate=False,
                termination_reason=None,
            )

        # Rule 2: Hard maximum reached
        if questions_asked >= MAX_QUESTIONS:
            return GapAnalysis(
                critical_gaps=critical_gaps,
                moderate_gaps=moderate_gaps,
                adequate_dimensions=adequate_dimensions,
                total_score=round(avg_score, 3),
                should_terminate=True,
                termination_reason=f"max_questions_reached ({MAX_QUESTIONS})",
            )

        # Rule 3: All dimensions are at least moderate (no critical gaps)
        # AND average score is above threshold
        if not critical_gaps and avg_score >= ADEQUATE_THRESHOLD:
            return GapAnalysis(
                critical_gaps=critical_gaps,
                moderate_gaps=moderate_gaps,
                adequate_dimensions=adequate_dimensions,
                total_score=round(avg_score, 3),
                should_terminate=True,
                termination_reason="all_dimensions_adequate",
            )

        # Rule 4: All dimensions are well-covered (>= 0.85)
        # (can terminate early even if below min questions in theory, but min_questions guards above)
        if all(s.score >= WELL_COVERED_THRESHOLD for s in coverage.values()):
            return GapAnalysis(
                critical_gaps=critical_gaps,
                moderate_gaps=moderate_gaps,
                adequate_dimensions=adequate_dimensions,
                total_score=round(avg_score, 3),
                should_terminate=True,
                termination_reason="all_dimensions_well_covered",
            )

        # Otherwise: keep going
        return GapAnalysis(
            critical_gaps=critical_gaps,
            moderate_gaps=moderate_gaps,
            adequate_dimensions=adequate_dimensions,
            total_score=round(avg_score, 3),
            should_terminate=False,
            termination_reason=None,
        )

    def get_target_dimensions(self, analysis: GapAnalysis) -> list[str]:
        """Return dimensions that should be targeted next, in priority order."""
        # Critical gaps first, then moderate gaps, ordered by lowest score
        all_gaps = analysis.critical_gaps + analysis.moderate_gaps
        # Note: caller should sort by actual score if they have the coverage dict
        return all_gaps
