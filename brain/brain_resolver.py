"""Brain resolver — local-only brain access.

Every brain in the system is now installed as a local personality_graph.json
(this module used to also support a remote proxy mode, but that was removed
in favor of pure offline operation). Brains are downloaded once via
`beam install` and queried locally via `BrainRetriever`.

This module is kept as a thin compatibility shim so existing import sites
(`from brain.brain_resolver import ...`) keep working.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from brain.brain_retriever import BrainRetriever
from brain.paths import (
    get_active_brain_graph_path,
    get_active_brain_name,
    get_brain_graph_path,
)

logger = logging.getLogger(__name__)

BEAM_HOME = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam"))


class BrainInterface(ABC):
    """Abstract interface for brain queries. Local-only now."""

    @abstractmethod
    def search(
        self, query: str, trust_level: str = "owner", brain_power: str = "standard"
    ) -> dict:
        """Search the brain for relevant nodes and return structured results."""
        ...

    @abstractmethod
    def get_soul(self) -> str:
        """Return SOUL.md content."""
        ...

    @abstractmethod
    def get_context(self) -> dict:
        """Return lightweight behavioral context."""
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        """Return brain statistics."""
        ...

    @abstractmethod
    def export_soul(self) -> dict:
        """Export SOUL.md content as a dict."""
        ...


class LocalBrain(BrainInterface):
    """Brain backed by a local personality_graph.json.

    The previous ProxyBrain class (which talked to api.openbeam.me for every
    query) is gone. All brains are downloaded at install time and used
    offline from then on.
    """

    def __init__(self, graph: dict):
        self._graph = graph
        self._retriever = BrainRetriever()

    def search(
        self, query: str, trust_level: str = "owner", brain_power: str = "standard"
    ) -> dict:
        return self._retriever.search(query, self._graph, trust_level, brain_power)

    def get_soul(self) -> str:
        result = self._retriever.export_soul(self._graph)
        return result.get("soul_md", "")

    def get_context(self) -> dict:
        return self._retriever.build_context(self._graph)

    def get_stats(self) -> dict:
        return self._retriever.get_stats(self._graph)

    def export_soul(self) -> dict:
        return self._retriever.export_soul(self._graph)


def _load_local_graph(name: str) -> dict | None:
    """Load a local personality_graph.json if it exists."""
    path = get_brain_graph_path(name)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load local graph for '%s': %s", name, e)
    # Fallback: old path (pre-migration)
    old_path = BEAM_HOME / "brain" / name / "personality_graph.json"
    if old_path.exists():
        try:
            return json.loads(old_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load old-path graph for '%s': %s", name, e)
    return None


def resolve_brain(name: str | None = None) -> BrainInterface:
    """Resolve a brain by name (or the active brain if name is None).

    Returns a `LocalBrain`. The old proxy-resolution branch is gone — the
    marketplace no longer ships proxy-only brains.
    """
    if name is None:
        name = get_active_brain_name()

    graph = _load_local_graph(name)
    if graph is not None:
        return LocalBrain(graph)

    logger.warning("No brain data found for '%s', returning empty local brain", name)
    return LocalBrain({})


def is_proxy_brain(name: str | None = None) -> bool:
    """Always returns False. Proxy brains no longer exist.

    Kept for backwards-compat with any import sites that still call it.
    """
    return False


def get_active_brain_interface() -> BrainInterface:
    """Convenience wrapper for the currently active brain."""
    return resolve_brain()
