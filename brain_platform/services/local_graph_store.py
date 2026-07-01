"""Local graph store — Neo4j + Graphiti wrapper for single-user local use.

The cloud's beam_mind uses Graphiti against a managed Neo4j cluster
(see ``beam_mind.services.graphiti.GraphitiService``). The local port
uses the same Graphiti client + same EntityNode/EntityEdge schema +
same group_id isolation, but points at a **local** Neo4j instance
(Docker, Desktop, or embedded).

**Neo4j is a required dependency.** Users must have a running Neo4j
instance (bolt://localhost:7687 by default) before the brain pipeline
can persist graphs. The local port does NOT fall back to SQLite or
in-memory storage — the schema, search semantics, and bi-temporal
edge model are all Graphiti-specific and would not be faithfully
reproducible in a different store.

Public API (sync facade over Graphiti's async client):

  store = LocalGraphStore(uri="bolt://localhost:7687", user="neo4j", password="...")
  store.initialize()
  try:
      group_id = store.group_id_for_user("user_123")
      edges = store.search(query="beliefs about honesty", group_id=group_id)
  finally:
      store.close()

The async→sync bridge uses ``asyncio.run()`` internally. This is safe
for a single-threaded local app (no concurrent event loops). If the
caller is already inside an event loop, use :meth:`asyncio_search`
and :meth:`asyncio_close` instead.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# Default local Neo4j connection. Override via env vars (NEO4J_URI,
# NEO4J_USER, NEO4J_PASSWORD) or constructor args. Works with:
# - Local Docker: bolt://localhost:7687 (the default)
# - Neo4j Desktop: bolt://localhost:7687
# - Neo4j Aura (managed cloud): neo4j+s://xxxxx.databases.neo4j.io
DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "neo4j"  # Neo4j's default dev password; users MUST change


def _env(name: str, default: str) -> str:
    """Read an env var, returning ``default`` if unset or empty.

    Checks the process environment first, then falls back to
    ``~/.hermes/.env`` (where the ``beam brain setup-neo4j`` wizard
    writes the config). The .env fallback is important because the
    setup wizard writes to the file but doesn't export to the parent
    shell — without this fallback, users would need to manually
    ``source ~/.hermes/.env`` after running the wizard.
    """
    import os
    val = os.environ.get(name)
    if val:
        return val
    # Fall back to ~/.hermes/.env via beam-agent's existing loader
    try:
        from hermes_cli.config import load_env
        env = load_env()
        val = env.get(name)
        if val:
            return val
    except Exception:
        pass
    return default


def _ensure_openai_compat_env():
    """If Graphiti's embedder needs OPENAI_API_KEY but only OPENROUTER_API_KEY
    is set, derive OPENAI_API_KEY + OPENAI_BASE_URL from OpenRouter.

    Graphiti instantiates its own OpenAI client for embeddings (it does
    not go through beam-agent's ``call_llm``). If the user has only
    configured OpenRouter, the embedder would fail with "api_key client
    option must be set". This helper bridges that gap: any user with
    ``OPENROUTER_API_KEY`` can run brain_platform without separately
    setting ``OPENAI_API_KEY``.

    Idempotent — only sets the vars if they're not already set.

    IMPORTANT: this must run BEFORE ``import graphiti_core`` because
    Graphiti creates its default embedder at import time. The
    module-level call below handles that.
    """
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            os.environ["OPENAI_API_KEY"] = openrouter_key
            # Only set the base URL if not already configured
            os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


# Run the env-derivation at import time so it's in place before
# ``import graphiti_core`` triggers the default embedder's OpenAI
# client construction. This is what makes the OpenRouter-only
# configuration work without users having to set OPENAI_API_KEY.
_ensure_openai_compat_env()


class LocalGraphStore:
    """Sync facade over a local Graphiti + Neo4j instance.

    Mirrors the cloud's ``GraphitiService`` but with sync methods.
    Uses ``asyncio.run()`` internally to bridge to Graphiti's async
    client. Single-threaded use only.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        llm_client: Any = None,
        embedder: Any = None,
    ):
        self._uri = uri or _env("NEO4J_URI", DEFAULT_URI)
        self._user = user or _env("NEO4J_USER", DEFAULT_USER)
        self._password = password or _env("NEO4J_PASSWORD", DEFAULT_PASSWORD)
        self._llm_client = llm_client
        self._embedder = embedder
        self._graphiti: Any = None
        # Persistent event loop — Graphiti's connection pool is bound to
        # the loop that created it. Using asyncio.run() per-call would
        # create a new loop each time, and the Neo4j driver's async
        # resources would be bound to the (now-closed) old loop, causing
        # "got Future attached to a different loop" errors on reuse.
        self._loop: Any = None

    def initialize(self) -> None:
        """Connect to Neo4j and build Graphiti's indices/constraints.

        Mirrors the cloud's ``GraphitiService.initialize()``. Must
        be called before any search/write operations.
        """
        from graphiti_core import Graphiti

        from brain_platform.pipeline.graphiti_prompts import apply_prompt_overrides

        # If the user configured OpenRouter, derive OPENAI_API_KEY from
        # it so Graphiti's embedder (which uses the OpenAI client
        # directly) can authenticate.
        _ensure_openai_compat_env()

        # Apply personality-aware extraction prompts before any
        # add_episode() calls. The cloud does this in initialize()
        # for the same reason.
        apply_prompt_overrides()

        # Create a persistent event loop. All async operations on this
        # store run on this same loop, so connection pools stay valid
        # across multiple sync calls.
        self._loop = asyncio.new_event_loop()
        try:
            self._graphiti = Graphiti(
                uri=self._uri,
                user=self._user,
                password=self._password,
                llm_client=self._llm_client,
                embedder=self._embedder,
            )
            # build_indices_and_constraints is async — run on the persistent loop
            self._loop.run_until_complete(
                self._graphiti.build_indices_and_constraints()
            )
        except Exception:
            self._loop.close()
            self._loop = None
            raise
        logger.info("LocalGraphStore connected to %s (personality prompts active)", self._uri)

    def close(self) -> None:
        """Close the Graphiti client and release the Neo4j connection."""
        if self._graphiti and self._loop:
            try:
                self._loop.run_until_complete(self._graphiti.close())
            except Exception:
                logger.warning("Error closing Graphiti client", exc_info=True)
            self._graphiti = None
        if self._loop:
            self._loop.close()
            self._loop = None

    @property
    def client(self) -> Any:
        """Return the raw Graphiti client (for advanced/async use)."""
        if not self._graphiti:
            raise RuntimeError("LocalGraphStore not initialized — call initialize() first")
        return self._graphiti

    @staticmethod
    def group_id_for_user(user_id: str | uuid.UUID) -> str:
        """Convert a user identifier to a Graphiti group_id.

        Graphiti's group_id must match ``^[a-zA-Z0-9_-]+$`` — UUIDs
        use hyphens which are allowed, but the cloud normalizes to
        underscores for consistency. We follow the cloud's pattern.
        """
        return str(user_id).replace("-", "_")

    def search(
        self,
        query: str,
        group_id: str,
        num_results: int = 5,
    ) -> List[Any]:
        """Search the graph for facts relevant to ``query``.

        Sync wrapper over Graphiti's async ``search()``. Returns a
        list of edge objects with ``.fact`` and ``.name`` attributes
        (same shape as the cloud's ``Retriever.retrieve()``).
        """
        if not self._loop:
            raise RuntimeError("LocalGraphStore not initialized — call initialize() first")
        return self._loop.run_until_complete(
            self._graphiti.search(
                query=query,
                group_ids=[group_id],
                num_results=num_results,
            )
        )

    def health_check(self) -> bool:
        """Verify Neo4j is responsive.

        Same pattern as the cloud's ``GraphitiService.health_check()``:
        attempt a simple retrieval against a sentinel group_id.
        """
        try:
            if not self._loop:
                return False
            self._loop.run_until_complete(
                self._graphiti.retrieve_episodes(
                    reference_time=None,
                    last_n=1,
                    group_ids=["__healthcheck__"],
                )
            )
            return True
        except Exception:
            logger.warning("LocalGraphStore health check failed", exc_info=True)
            return False

    def __enter__(self) -> "LocalGraphStore":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


__all__ = ["LocalGraphStore", "DEFAULT_URI", "DEFAULT_USER", "DEFAULT_PASSWORD"]
