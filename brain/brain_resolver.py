"""Brain resolver — unified interface for local and remote (proxy) brains.

When a user installs a marketplace brain, the full personality_graph.json
is NOT downloaded. Instead, a lightweight brain_config.json is stored locally
and all queries go through the Beam API (BrainProxyClient).

This module provides a single abstraction so the rest of the codebase
doesn't need to care whether a brain is local or remote.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from brain.brain_proxy_client import BrainProxyClient
from brain.brain_retriever import BrainRetriever
from brain.paths import (
    get_active_brain_graph_path,
    get_active_brain_path,
    get_brain_graph_path,
    get_brain_path,
)

logger = logging.getLogger(__name__)

BEAM_HOME = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam"))


class BrainInterface(ABC):
    """Abstract interface for brain queries."""

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
    """Brain backed by a local personality_graph.json."""

    def __init__(self, graph: dict):
        self._graph = graph
        self._retriever = BrainRetriever()

    def search(
        self, query: str, trust_level: str = "owner", brain_power: str = "standard"
    ) -> dict:
        return self._retriever.search(self._graph, query, trust_level, brain_power)

    def get_soul(self) -> str:
        result = self._retriever.export_soul(self._graph)
        return result.get("soul_md", "")

    def get_context(self) -> dict:
        return self._retriever.build_context(self._graph)

    def get_stats(self) -> dict:
        return self._retriever.get_stats(self._graph)

    def export_soul(self) -> dict:
        return self._retriever.export_soul(self._graph)


class ProxyBrain(BrainInterface):
    """Brain accessed remotely via BrainProxyClient."""

    def __init__(self, slug: str, token: str, api_url: str | None = None):
        self._client = BrainProxyClient(slug, token, api_url)

    def search(
        self, query: str, trust_level: str = "owner", brain_power: str = "standard"
    ) -> dict:
        try:
            results = self._client.search(query, trust_level, brain_power)
            # BrainProxyClient.search now normalizes both old (list) and new (dict)
            # server response shapes.
            if isinstance(results, dict) and "nodes" in results:
                return results
            return {
                "nodes": results if isinstance(results, list) else [],
                "edges": [],
                "context": "",
                "total_matches": len(results) if isinstance(results, list) else 0,
            }
        except ConnectionError as e:
            logger.warning("Proxy brain offline: %s", e)
            return {
                "nodes": [],
                "edges": [],
                "context": "Brain is offline. Using cached identity only.",
                "total_matches": 0,
                "offline": True,
            }

    def get_soul(self) -> str:
        try:
            return self._client.get_soul()
        except ConnectionError as e:
            logger.warning("Proxy brain offline, cannot fetch SOUL.md: %s", e)
            return ""

    def get_context(self) -> dict:
        try:
            ctx = self._client.get_context()
            if isinstance(ctx, dict) and "context" in ctx:
                return ctx
            return {"context": str(ctx) if ctx else ""}
        except ConnectionError as e:
            logger.warning("Proxy brain offline, cannot fetch context: %s", e)
            return {"context": "Brain is offline. Using cached identity only."}

    def get_stats(self) -> dict:
        # Proxy stats are best-effort; we don't have the full graph locally.
        try:
            ctx = self._client.get_context()
            if isinstance(ctx, dict):
                return {
                    "user_summary": bool(ctx.get("user_summary")),
                    "total_nodes": len(ctx.get("top_traits", [])) + len(ctx.get("top_values", [])),
                    "total_edges": 0,
                    "coverage": {
                        "traits": len(ctx.get("top_traits", [])),
                        "values": len(ctx.get("top_values", [])),
                        "voice_dna": bool(ctx.get("voice_dna", {}).get("humor_style")),
                    },
                }
        except Exception:
            pass
        return {"status": "proxy", "note": "Full stats unavailable for remote brain"}

    def export_soul(self) -> dict:
        soul = self.get_soul()
        return {"soul_md": soul}


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


def _load_proxy_config(name: str) -> dict | None:
    """Load a brain_config.json proxy reference if it exists."""
    brain_path = get_brain_path(name)
    config_path = brain_path / "brain_config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load proxy config for '%s': %s", name, e)
    return None


def resolve_brain(name: str | None = None) -> BrainInterface:
    """Resolve a brain by name (or active brain if name is None).

    Returns a BrainInterface implementation — either LocalBrain or ProxyBrain.
    """
    if name is None:
        from brain.paths import get_active_brain_name
        name = get_active_brain_name()

    # 1. Check for proxy config first
    proxy_config = _load_proxy_config(name)
    if proxy_config and proxy_config.get("type") == "proxy":
        return ProxyBrain(
            slug=proxy_config["slug"],
            token=proxy_config["token"],
            api_url=proxy_config.get("api_url"),
        )

    # 2. Fall back to local graph
    graph = _load_local_graph(name)
    if graph is not None:
        return LocalBrain(graph)

    # 3. Ultimate fallback: empty local brain
    logger.warning("No brain data found for '%s', returning empty local brain", name)
    return LocalBrain({})


def is_proxy_brain(name: str | None = None) -> bool:
    """Check whether the named (or active) brain is a proxy brain."""
    if name is None:
        from brain.paths import get_active_brain_name
        name = get_active_brain_name()
    proxy_config = _load_proxy_config(name)
    return proxy_config is not None and proxy_config.get("type") == "proxy"


def get_active_brain_interface() -> BrainInterface:
    """Convenience wrapper for the currently active brain."""
    return resolve_brain()
