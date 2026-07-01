"""Bi-temporal edge logic — pure functions for time-aware graph operations.

The cloud's beam_mind uses Graphiti against a live Neo4j instance for
bi-temporal edge tracking. Each edge carries two independent time axes:

- **Event time** (``valid_at`` / ``invalid_at``): When the fact became
  true / stopped being true in the real world.
- **Ingestion time** (``created_at`` / ``expired_at``): When the system
  recorded / invalidated the fact.

Old facts are marked ``expired_at``, never deleted — this enables
"what did I used to believe about X?" queries.

The local port keeps the same bi-temporal semantics but operates on
in-memory lists/dicts. The :class:`LocalGraphWriter` (in
``brain_platform.services.local_graph_writer``) calls these helpers
to stamp edges and decide which existing edges to expire.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    """Return the current UTC time (testable via monkeypatch)."""
    return datetime.now(timezone.utc)


@dataclass
class BiTemporalEdge:
    """An edge with bi-temporal validity.

    Mirrors Graphiti's EntityEdge bi-temporal model:
    - ``valid_at`` / ``invalid_at``: event time (real-world)
    - ``created_at`` / ``expired_at``: ingestion time (system)

    An edge is "currently active" when ``invalid_at is None`` and
    ``expired_at is None``. Setting ``expired_at`` (without
    ``invalid_at``) means the system learned the fact is no longer
    true, but the event itself may still be historically valid.
    """

    uuid: str
    source_name: str
    target_name: str
    name: str  # relation type (e.g. "HOLDS", "HAS_TRAIT")
    fact: str
    group_id: str
    created_at: datetime
    valid_at: datetime
    invalid_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    superseded_by: Optional[str] = None

    def is_active(self, at_time: Optional[datetime] = None) -> bool:
        """True if this edge is currently active (not expired, not invalidated)."""
        if self.expired_at is not None:
            return False
        if self.invalid_at is not None:
            when = at_time or _now_utc()
            if when >= self.invalid_at:
                return False
        return True


def _same_edge_key(source: str, target: str, name: str) -> tuple[str, str, str]:
    """Canonical key for edge identity (source, target, relation name).

    Two edges with the same key conflict — adding a new one expires
    the old (same-relation edges are mutually exclusive in time).
    """
    return (source, target, name)


def find_conflicting_edges(
    existing: Iterable[BiTemporalEdge],
    new_source: str,
    new_target: str,
    new_name: str,
) -> List[BiTemporalEdge]:
    """Return currently-active edges that conflict with a proposed new edge.

    A conflict is defined as: same (source, target, name) AND
    still active (no expired_at, and no invalid_at in the past).
    """
    key = _same_edge_key(new_source, new_target, new_name)
    return [
        e for e in existing
        if _same_edge_key(e.source_name, e.target_name, e.name) == key
        and e.is_active()
    ]


def expire_edges(
    edges: List[BiTemporalEdge],
    to_expire: Iterable[BiTemporalEdge],
    superseded_by_uuid: str,
    at_time: Optional[datetime] = None,
) -> int:
    """Mark the given edges as expired (system learned they're superseded).

    Sets ``expired_at`` to ``at_time`` (or now) and ``superseded_by``
    to the new edge's UUID. Does NOT set ``invalid_at`` — the
    original event may still be historically valid; only the system
    record is superseded.

    Returns the number of edges expired.
    """
    when = at_time or _now_utc()
    targets = {e.uuid: e for e in to_expire}
    count = 0
    for edge in edges:
        if edge.uuid in targets and edge.expired_at is None:
            edge.expired_at = when
            edge.superseded_by = superseded_by_uuid
            count += 1
    return count


def currently_active_edges(
    edges: Iterable[BiTemporalEdge],
    at_time: Optional[datetime] = None,
) -> List[BiTemporalEdge]:
    """Return only the currently-active edges at ``at_time`` (or now)."""
    return [e for e in edges if e.is_active(at_time)]


def edges_for_group(
    edges: Iterable[BiTemporalEdge],
    group_id: str,
) -> List[BiTemporalEdge]:
    """Filter edges to a specific group (multi-tenant isolation)."""
    return [e for e in edges if e.group_id == group_id]


__all__ = [
    "BiTemporalEdge",
    "find_conflicting_edges",
    "expire_edges",
    "currently_active_edges",
    "edges_for_group",
]
