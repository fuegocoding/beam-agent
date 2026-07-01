"""Coverage types — standalone dataclasses used by gap_identifier and coverage_scorer.

This file exists so `gap_identifier.py` can be lifted as-is in Chunk 1
without pulling in `coverage_scorer.py`'s SQLAlchemy/PostgreSQL dependencies.
The full `coverage_scorer.py` (with the actual scoring logic) is a Chunk 2 lift.
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
