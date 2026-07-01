"""Coverage types — standalone dataclasses used by gap_identifier and coverage_scorer.

Kept as a leaf module (no DB imports) so the scoring and gap-detection
logic stays portable and unit-testable without SQLAlchemy/PostgreSQL.
"""
from dataclasses import dataclass


@dataclass
class DimensionScore:
    """Coverage score for a single dimension."""

    dimension: str
    score: float  # 0.0-1.0
    node_count: int
    edge_count: int
    avg_summary_len: float
    diversity: float  # unique_names / node_count (1.0 = all unique)
