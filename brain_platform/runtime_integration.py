"""Runtime integration — wire brain_platform into the agent's memory retrieval.

The agent runtime uses :class:`brain.brain_retriever.BrainRetriever` to
fetch relevant personality facts for each user message. This module
provides a drop-in replacement that uses
:class:`brain_platform.services.local_graph_searcher.LocalGraphSearcher`
when Neo4j is configured, and falls back to the offline retriever
otherwise.

Usage (in the agent runtime or any context builder):

  from brain_platform.runtime_integration import GraphBackedBrainRetriever

  retriever = GraphBackedBrainRetriever()
  facts = retriever.retrieve(query="what does the user believe about X")

The class auto-detects whether Neo4j is reachable. If yes, it uses
the graphiti-backed search. If no, it falls back to the offline
keyword retriever (same behavior as before Chunk 4).
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def _neo4j_configured() -> bool:
    """True if NEO4J_URI is set in the environment."""
    return bool(os.environ.get("NEO4J_URI"))


class GraphBackedBrainRetriever:
    """Drop-in replacement for BrainRetriever that uses Neo4j when available.

    Falls back to the offline BrainRetriever (keyword search on the
    local personality_graph.json) when Neo4j is not configured or
    not reachable. The fallback is silent — callers get the same
    list[str] return shape either way.

    The class auto-detects Neo4j on first use. Set the
    ``BRAIN_RETRIEVER`` env var to force a specific backend:
      - ``"graphiti"`` — always use LocalGraphSearcher (raise on failure)
      - ``"local"`` — always use the offline BrainRetriever
      - unset — auto-detect (graphiti if configured, local otherwise)
    """

    def __init__(self, *, group_id: str = "default_user"):
        self._group_id = group_id
        self._backend: Optional[str] = None
        self._graphiti_retriever: Any = None
        self._local_retriever: Any = None

    def retrieve(self, query: str, num_results: int = 5) -> List[str]:
        """Return up to ``num_results`` facts relevant to ``query``.

        Mirrors the cloud's ``Retriever.retrieve()`` return shape
        (list[str]) and the offline ``BrainRetriever`` return shape.
        """
        backend = self._select_backend()
        if backend == "graphiti":
            return self._retrieve_graphiti(query, num_results)
        return self._retrieve_local(query, num_results)

    def _select_backend(self) -> str:
        """Decide which backend to use for this call (cached after first pick)."""
        if self._backend is not None:
            return self._backend

        forced = os.environ.get("BRAIN_RETRIEVER", "").lower().strip()
        if forced == "graphiti":
            self._backend = "graphiti"
        elif forced == "local":
            self._backend = "local"
        elif _neo4j_configured():
            # Try to initialize; fall back on any error
            try:
                from brain_platform.services.local_graph_store import LocalGraphStore
                from brain_platform.services.local_graph_searcher import LocalGraphSearcher

                store = LocalGraphStore()
                store.initialize()
                self._graphiti_retriever = LocalGraphSearcher(store)
                self._backend = "graphiti"
                logger.debug("GraphBackedBrainRetriever: using graphiti backend")
            except Exception as e:
                logger.debug("GraphBackedBrainRetriever: graphiti init failed (%s), using local", e)
                self._backend = "local"
        else:
            self._backend = "local"
            logger.debug("GraphBackedBrainRetriever: NEO4J_URI not set, using local backend")

        return self._backend

    def _retrieve_graphiti(self, query: str, num_results: int) -> List[str]:
        try:
            return self._graphiti_retriever.search(
                query=query,
                group_id=self._group_id,
                num_results=num_results,
            )
        except Exception as e:
            logger.debug("Graphiti search failed (%s), falling back to local", e)
            return self._retrieve_local(query, num_results)

    def _retrieve_local(self, query: str, num_results: int) -> List[str]:
        if self._local_retriever is None:
            from brain.brain_retriever import BrainRetriever
            self._local_retriever = BrainRetriever()
        try:
            return self._local_retriever.build_context_for_query(query, num_results=num_results)
        except Exception as e:
            logger.debug("Local BrainRetriever failed: %s", e)
            return []


__all__ = ["GraphBackedBrainRetriever"]
