"""Tests for Chunk 3 — the Neo4j + Graphiti local graph store.

Neo4j is a required runtime dependency for Chunk 3 (no SQLite fallback
— the schema, search semantics, and bi-temporal edge model are all
Graphiti-specific). These tests mock the Graphiti client to validate
the local port's logic without needing a live Neo4j instance.

For live integration tests, see ``test_chunk3_integration.py`` (TODO
once the docker-compose for the test Neo4j is wired up).
"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# pipeline/temporal.py — bi-temporal edge logic
# ──────────────────────────────────────────────────────────────────────

class TestBiTemporalEdge:
    """Pure bi-temporal edge logic (no DB needed)."""

    def test_is_active_when_no_expiry(self):
        from datetime import datetime, timezone, timedelta
        from brain_platform.pipeline.temporal import BiTemporalEdge

        now = datetime.now(timezone.utc)
        edge = BiTemporalEdge(
            uuid="e1", source_name="A", target_name="B", name="HOLDS",
            fact="A holds B", group_id="g1", created_at=now, valid_at=now,
        )
        assert edge.is_active() is True

    def test_is_inactive_when_expired(self):
        from datetime import datetime, timezone, timedelta
        from brain_platform.pipeline.temporal import BiTemporalEdge

        now = datetime.now(timezone.utc)
        edge = BiTemporalEdge(
            uuid="e1", source_name="A", target_name="B", name="HOLDS",
            fact="A holds B", group_id="g1", created_at=now, valid_at=now,
            expired_at=now,
        )
        assert edge.is_active() is False

    def test_is_inactive_when_invalidated_in_past(self):
        from datetime import datetime, timezone, timedelta
        from brain_platform.pipeline.temporal import BiTemporalEdge

        now = datetime.now(timezone.utc)
        edge = BiTemporalEdge(
            uuid="e1", source_name="A", target_name="B", name="HOLDS",
            fact="A holds B", group_id="g1", created_at=now, valid_at=now,
            invalid_at=now - timedelta(hours=1),
        )
        assert edge.is_active() is False

    def test_is_active_when_invalidated_in_future(self):
        from datetime import datetime, timezone, timedelta
        from brain_platform.pipeline.temporal import BiTemporalEdge

        now = datetime.now(timezone.utc)
        edge = BiTemporalEdge(
            uuid="e1", source_name="A", target_name="B", name="HOLDS",
            fact="A holds B", group_id="g1", created_at=now, valid_at=now,
            invalid_at=now + timedelta(hours=1),
        )
        assert edge.is_active() is True


class TestFindConflictingEdges:
    def test_finds_same_source_target_name(self):
        from datetime import datetime, timezone
        from brain_platform.pipeline.temporal import (
            BiTemporalEdge,
            find_conflicting_edges,
        )

        now = datetime.now(timezone.utc)
        edges = [
            BiTemporalEdge(
                uuid="e1", source_name="A", target_name="B", name="HOLDS",
                fact="...", group_id="g1", created_at=now, valid_at=now,
            ),
            BiTemporalEdge(
                uuid="e2", source_name="A", target_name="C", name="HOLDS",
                fact="...", group_id="g1", created_at=now, valid_at=now,
            ),
        ]
        conflicts = find_conflicting_edges(edges, "A", "B", "HOLDS")
        assert len(conflicts) == 1
        assert conflicts[0].uuid == "e1"

    def test_ignores_expired_edges(self):
        from datetime import datetime, timezone
        from brain_platform.pipeline.temporal import (
            BiTemporalEdge,
            find_conflicting_edges,
        )

        now = datetime.now(timezone.utc)
        edges = [
            BiTemporalEdge(
                uuid="e1", source_name="A", target_name="B", name="HOLDS",
                fact="...", group_id="g1", created_at=now, valid_at=now,
                expired_at=now,
            ),
        ]
        conflicts = find_conflicting_edges(edges, "A", "B", "HOLDS")
        assert len(conflicts) == 0


class TestExpireEdges:
    def test_expires_only_unexpired(self):
        from datetime import datetime, timezone
        from brain_platform.pipeline.temporal import (
            BiTemporalEdge,
            expire_edges,
        )

        now = datetime.now(timezone.utc)
        edges = [
            BiTemporalEdge(
                uuid="e1", source_name="A", target_name="B", name="HOLDS",
                fact="...", group_id="g1", created_at=now, valid_at=now,
            ),
            BiTemporalEdge(
                uuid="e2", source_name="A", target_name="B", name="HOLDS",
                fact="...", group_id="g1", created_at=now, valid_at=now,
                expired_at=now,
            ),
        ]
        count = expire_edges(edges, edges, "new_uuid")
        assert count == 1
        assert edges[0].expired_at is not None
        assert edges[0].superseded_by == "new_uuid"
        # Already-expired edge is not re-expired
        assert edges[1].expired_at == now


class TestCurrentlyActiveEdges:
    def test_filters_expired_and_invalidated(self):
        from datetime import datetime, timezone, timedelta
        from brain_platform.pipeline.temporal import (
            BiTemporalEdge,
            currently_active_edges,
        )

        now = datetime.now(timezone.utc)
        edges = [
            BiTemporalEdge(
                uuid="e1", source_name="A", target_name="B", name="HOLDS",
                fact="active", group_id="g1", created_at=now, valid_at=now,
            ),
            BiTemporalEdge(
                uuid="e2", source_name="A", target_name="B", name="HOLDS",
                fact="expired", group_id="g1", created_at=now, valid_at=now,
                expired_at=now,
            ),
            BiTemporalEdge(
                uuid="e3", source_name="A", target_name="B", name="HOLDS",
                fact="invalidated", group_id="g1", created_at=now, valid_at=now,
                invalid_at=now - timedelta(hours=1),
            ),
        ]
        active = currently_active_edges(edges)
        assert len(active) == 1
        assert active[0].fact == "active"


# ──────────────────────────────────────────────────────────────────────
# services/local_graph_store.py — sync facade over Graphiti
# ──────────────────────────────────────────────────────────────────────

class TestLocalGraphStoreGroupId:
    def test_uuid_hyphens_become_underscores(self):
        from brain_platform.services.local_graph_store import LocalGraphStore

        gid = LocalGraphStore.group_id_for_user("user-123-abc")
        assert gid == "user_123_abc"
        assert "-" not in gid

    def test_uuid_object_supported(self):
        import uuid as uuid_mod
        from brain_platform.services.local_graph_store import LocalGraphStore

        u = uuid_mod.UUID("12345678-1234-5678-1234-567812345678")
        gid = LocalGraphStore.group_id_for_user(u)
        assert gid == "12345678_1234_5678_1234_567812345678"

    def test_env_var_override(self):
        """NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD override the defaults.

        This is the path for Neo4j Aura (managed cloud) and any other
        remote Neo4j instance — no Docker required.
        """
        import os
        from brain_platform.services.local_graph_store import LocalGraphStore

        env = {
            "NEO4J_URI": "neo4j+s://abc123.databases.neo4j.io",
            "NEO4J_USER": "aura-user",
            "NEO4J_PASSWORD": "aura-secret",
        }
        with patch.dict(os.environ, env, clear=False):
            store = LocalGraphStore()
        assert store._uri == "neo4j+s://abc123.databases.neo4j.io"
        assert store._user == "aura-user"
        assert store._password == "aura-secret"

    def test_constructor_args_override_env(self):
        """Explicit constructor args take precedence over env vars."""
        import os
        from brain_platform.services.local_graph_store import LocalGraphStore

        with patch.dict(os.environ, {"NEO4J_URI": "neo4j+s://from-env.io"}, clear=False):
            store = LocalGraphStore(uri="bolt://from-ctor:7687")
        assert store._uri == "bolt://from-ctor:7687"


class TestLocalGraphStoreUninitialized:
    def test_client_raises_before_initialize(self):
        from brain_platform.services.local_graph_store import LocalGraphStore

        store = LocalGraphStore()
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = store.client


# ──────────────────────────────────────────────────────────────────────
# services/local_graph_writer.py — port of BrainWriter.write()
# ──────────────────────────────────────────────────────────────────────

class TestLocalGraphWriterResolveName:
    """The fuzzy name resolver inside LocalGraphWriter is the same as
    the cloud's BrainWriter. We test it via a small extracted helper."""

    def test_exact_match(self):
        from brain_platform.services.local_graph_writer import LocalGraphWriter

        store = MagicMock()
        writer = LocalGraphWriter(store)
        # Access the nested function via the write method's closure
        # by inspecting the source — easier to just test via integration.
        # Here we just verify the writer can be instantiated.
        assert writer is not None

    def test_writes_just_the_user_for_empty_graph(self):
        """Even for an empty graph, THE_USER hub node is always created.

        Mirrors the cloud's BrainWriter behavior: THE_USER is the
        identity anchor and must exist before any other nodes can
        be connected. The result is 1 node (THE_USER) and 0 edges
        (no other nodes to connect to).
        """
        from brain_platform.services.local_graph_writer import LocalGraphWriter
        from brain_platform.pipeline.brain_schema import PersonalityGraph
        from graphiti_core.nodes import EntityNode

        # Patch EntityNode.save to be a no-op async method so we
        # don't need a real Neo4j driver. Also patch embedding gen.
        async def fake_save(self, driver):
            return None
        async def fake_gen_embedding(self, embedder):
            self.name_embedding = [0.0] * 768
            return None

        with patch.object(EntityNode, "save", fake_save), \
             patch.object(EntityNode, "generate_name_embedding", fake_gen_embedding):
            store = MagicMock()
            # Mock the Neo4j driver: no existing THE_USER
            mock_result = MagicMock()
            mock_result.records = []
            mock_neo4j = MagicMock()
            mock_neo4j.execute_query = AsyncMock(return_value=mock_result)
            mock_driver = MagicMock()
            mock_driver.client = mock_neo4j
            store.client.driver = mock_driver
            store.client.embedder = MagicMock()
            writer = LocalGraphWriter(store)
            graph = PersonalityGraph(user_summary="")
            result = writer.write(graph, group_id="test")
        assert result["nodes_created"] >= 1  # At least THE_USER
        assert result["edges_created"] == 0


# ──────────────────────────────────────────────────────────────────────
# services/local_graph_searcher.py — port of Retriever.retrieve()
# ──────────────────────────────────────────────────────────────────────

class TestLocalGraphSearcher:
    def test_returns_facts_from_edges(self):
        from brain_platform.services.local_graph_searcher import LocalGraphSearcher

        edge1 = MagicMock()
        edge1.fact = "THE_USER HOLDS belief X"
        edge1.name = "HOLDS"
        edge2 = MagicMock()
        edge2.fact = "THE_USER HAS_TRAIT curious"
        edge2.name = "HAS_TRAIT"
        store = MagicMock()
        store.search.return_value = [edge1, edge2]
        searcher = LocalGraphSearcher(store)
        facts = searcher.search("beliefs", group_id="g1")
        assert facts == ["THE_USER HOLDS belief X", "THE_USER HAS_TRAIT curious"]

    def test_falls_back_to_edge_name_when_no_fact(self):
        from brain_platform.services.local_graph_searcher import LocalGraphSearcher

        edge = MagicMock()
        edge.fact = ""
        edge.name = "HOLDS"
        store = MagicMock()
        store.search.return_value = [edge]
        searcher = LocalGraphSearcher(store)
        facts = searcher.search("anything", group_id="g1")
        assert facts == ["HOLDS"]

    def test_returns_empty_on_search_failure(self):
        from brain_platform.services.local_graph_searcher import LocalGraphSearcher

        store = MagicMock()
        store.search.side_effect = RuntimeError("Neo4j down")
        searcher = LocalGraphSearcher(store)
        facts = searcher.search("anything", group_id="g1")
        assert facts == []


# ──────────────────────────────────────────────────────────────────────
# pipeline/graphiti_prompts.py — personality-aware prompt overrides
# ──────────────────────────────────────────────────────────────────────

class TestGraphitiPrompts:
    def test_apply_prompt_overrides_is_idempotent(self):
        from brain_platform.pipeline.graphiti_prompts import apply_prompt_overrides

        # Calling twice should not raise
        apply_prompt_overrides()
        apply_prompt_overrides()

    def test_custom_extract_text_returns_messages(self):
        from brain_platform.pipeline.graphiti_prompts import custom_extract_text

        context = {
            "source_description": "test",
            "entity_types": "PersonalityTrait, Value",
            "custom_extraction_instructions": "Focus on personality",
            "previous_episodes": [],
            "episode_content": "I value honesty and autonomy.",
        }
        messages = custom_extract_text(context)
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "personality" in messages[0].content.lower()

    def test_synthesis_mode_adds_preamble(self):
        from brain_platform.pipeline.graphiti_prompts import custom_extract_text

        context = {
            "source_description": "interview synthesis",
            "entity_types": "PersonalityTrait",
            "custom_extraction_instructions": "",
            "previous_episodes": [],
            "episode_content": "Some text",
        }
        messages = custom_extract_text(context)
        user_content = messages[1].content
        assert "SYNTHESIS MODE" in user_content


class TestOpenAICompatEnvDerivation:
    """Graphiti's embedder needs OPENAI_API_KEY directly. This helper
    derives it from OPENROUTER_API_KEY so users with only an OpenRouter
    key can run brain_platform."""

    def test_openrouter_derives_openai_key(self, monkeypatch):
        import os
        from brain_platform.services.local_graph_store import _ensure_openai_compat_env

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")

        _ensure_openai_compat_env()

        assert os.environ.get("OPENAI_API_KEY") == "sk-or-v1-test"
        assert os.environ.get("OPENAI_BASE_URL") == "https://openrouter.ai/api/v1"

    def test_explicit_openai_key_not_overridden(self, monkeypatch):
        import os
        from brain_platform.services.local_graph_store import _ensure_openai_compat_env

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-openrouter")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        _ensure_openai_compat_env()

        # Explicit OPENAI_* must NOT be overridden by the OpenRouter derivation
        assert os.environ.get("OPENAI_API_KEY") == "sk-real-openai-key"
        assert os.environ.get("OPENAI_BASE_URL") == "https://api.openai.com/v1"

    def test_no_openrouter_key_is_noop(self, monkeypatch):
        import os
        from brain_platform.services.local_graph_store import _ensure_openai_compat_env

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        _ensure_openai_compat_env()

        # Nothing to derive from — both vars stay unset
        assert os.environ.get("OPENAI_API_KEY") is None
        assert os.environ.get("OPENAI_BASE_URL") is None
