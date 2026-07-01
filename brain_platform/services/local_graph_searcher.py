"""Local graph searcher — port of beam_mind's Retriever.

Wraps :meth:`LocalGraphStore.search` with the same return shape as
the cloud's ``Retriever.retrieve()``: a list of fact strings extracted
from matching edges.

The cloud's retriever is part of the agent runtime (called on every
agent turn to inject relevant memories into context). The local port
is the same — the agent runtime's context_builder calls this to
retrieve facts relevant to the user's message.

Public API:

  searcher = LocalGraphSearcher(store)
  facts = searcher.search(query="what does the user believe about X", group_id="user_123")
  # ["THE_USER HOLDS belief X", ...]
"""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


class LocalGraphSearcher:
    """Sync wrapper over :class:`LocalGraphStore` that returns fact strings.

    Mirrors the cloud's ``Retriever.retrieve()`` return shape (list[str]).
    Skips the cloud's Redis cache layer (single-user local app, no
    cross-process cache needed).
    """

    def __init__(self, store: "LocalGraphStore"):  # noqa: F821
        self._store = store

    def search(
        self,
        query: str,
        group_id: str,
        num_results: int = 5,
    ) -> List[str]:
        """Return up to ``num_results`` facts relevant to ``query``.

        Mirrors the cloud's ``Retriever.retrieve()``: pulls the
        ``.fact`` and ``.name`` attributes off each returned edge
        and flattens to a list of strings. Edges without a fact
        fall back to their relation name.
        """
        try:
            results = self._store.search(
                query=query,
                group_id=group_id,
                num_results=num_results,
            )
        except Exception:
            logger.debug("Local graph search failed, returning empty context")
            return []

        facts: List[str] = []
        for edge in results:
            if hasattr(edge, "fact") and edge.fact:
                facts.append(edge.fact)
            elif hasattr(edge, "name") and edge.name:
                facts.append(edge.name)
        return facts


__all__ = ["LocalGraphSearcher"]
