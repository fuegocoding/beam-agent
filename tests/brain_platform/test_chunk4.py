"""Tests for Chunk 4 — the brain_platform CLI integration.

These tests validate the CLI surface (argparse setup, env-var
read/write, command dispatch) without requiring a live Neo4j
instance. The Neo4j-dependent code paths (cmd_brain_platform_search,
cmd_brain_platform_ingest, cmd_interview_adaptive) are tested with
mocked LocalGraphStore.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# Env-var helpers
# ──────────────────────────────────────────────────────────────────────

class TestReadWriteEnvValue:
    def test_read_existing_value(self, tmp_path, monkeypatch):
        from brain_platform.cli.integration import _read_env_value, _write_env_value

        env_path = tmp_path / ".env"
        env_path.write_text("NEO4J_URI=bolt://test:7687\nOTHER=foo\n")
        monkeypatch.setattr("brain_platform.cli.integration.HERMES_ENV_PATH", env_path)

        assert _read_env_value("NEO4J_URI") == "bolt://test:7687"
        assert _read_env_value("OTHER") == "foo"
        assert _read_env_value("MISSING") is None

    def test_write_new_value(self, tmp_path, monkeypatch):
        from brain_platform.cli.integration import _write_env_value

        env_path = tmp_path / ".env"
        env_path.write_text("EXISTING=foo\n")
        monkeypatch.setattr("brain_platform.cli.integration.HERMES_ENV_PATH", env_path)

        _write_env_value("NEW", "bar")
        content = env_path.read_text()
        assert "EXISTING=foo" in content
        assert "NEW=bar" in content

    def test_write_updates_existing(self, tmp_path, monkeypatch):
        from brain_platform.cli.integration import _write_env_value

        env_path = tmp_path / ".env"
        env_path.write_text("NEO4J_URI=old\n")
        monkeypatch.setattr("brain_platform.cli.integration.HERMES_ENV_PATH", env_path)

        _write_env_value("NEO4J_URI", "new")
        content = env_path.read_text()
        assert "NEO4J_URI=new" in content
        assert "NEO4J_URI=old" not in content

    def test_write_creates_file_if_missing(self, tmp_path, monkeypatch):
        from brain_platform.cli.integration import _write_env_value

        env_path = tmp_path / ".env"
        monkeypatch.setattr("brain_platform.cli.integration.HERMES_ENV_PATH", env_path)

        _write_env_value("FRESH", "value")
        assert env_path.exists()
        assert "FRESH=value" in env_path.read_text()

    def test_write_also_sets_in_environ(self, tmp_path, monkeypatch):
        from brain_platform.cli.integration import _write_env_value

        env_path = tmp_path / ".env"
        monkeypatch.setattr("brain_platform.cli.integration.HERMES_ENV_PATH", env_path)

        monkeypatch.delenv("MY_VAR", raising=False)
        _write_env_value("MY_VAR", "hello")
        assert os.environ.get("MY_VAR") == "hello"


# ──────────────────────────────────────────────────────────────────────
# cmd_setup_neo4j
# ──────────────────────────────────────────────────────────────────────

class TestCmdSetupNeo4j:
    def test_keeps_existing_when_user_says_no(self, tmp_path, monkeypatch):
        from brain_platform.cli.integration import cmd_setup_neo4j, HERMES_ENV_PATH

        env_path = tmp_path / ".env"
        env_path.write_text("NEO4J_URI=bolt://existing:7687\n")
        monkeypatch.setattr("brain_platform.cli.integration.HERMES_ENV_PATH", env_path)

        # User says "n" to overwrite
        with patch("builtins.input", side_effect=["n"]):
            args = MagicMock()
            result = cmd_setup_neo4j(args)

        assert result == 0
        assert "NEO4J_URI=bolt://existing:7687" in env_path.read_text()

    def test_writes_new_values_on_confirm(self, tmp_path, monkeypatch):
        from brain_platform.cli.integration import cmd_setup_neo4j

        env_path = tmp_path / ".env"
        env_path.write_text("")
        monkeypatch.setattr("brain_platform.cli.integration.HERMES_ENV_PATH", env_path)

        # Inputs: uri, user, password (no existing config, no confirm prompt)
        with patch("builtins.input", side_effect=[
            "bolt://new:7687",  # uri
            "neo4j",             # user
            "secret",            # password
        ]):
            args = MagicMock()
            # Skip the connection test by making LocalGraphStore fail
            with patch("brain_platform.services.local_graph_store.LocalGraphStore") as MockStore:
                MockStore.side_effect = RuntimeError("test skip")
                result = cmd_setup_neo4j(args)

        content = env_path.read_text()
        assert "NEO4J_URI=bolt://new:7687" in content
        assert "NEO4J_USER=neo4j" in content
        assert "NEO4J_PASSWORD=secret" in content


# ──────────────────────────────────────────────────────────────────────
# cmd_brain_platform_search
# ──────────────────────────────────────────────────────────────────────

class TestCmdBrainPlatformSearch:
    def test_returns_facts(self, capsys):
        from brain_platform.cli.integration import cmd_brain_platform_search

        mock_edge = MagicMock()
        mock_edge.fact = "THE_USER HOLDS belief X"
        mock_edge.name = "HOLDS"

        with patch("brain_platform.services.local_graph_store.LocalGraphStore") as MockStore:
            store_instance = MagicMock()
            store_instance.search.return_value = [mock_edge]
            MockStore.return_value = store_instance

            args = MagicMock(query="beliefs", num_results=5, group_id="test_group")
            result = cmd_brain_platform_search(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "THE_USER HOLDS belief X" in captured.out

    def test_handles_connection_error(self, capsys):
        from brain_platform.cli.integration import cmd_brain_platform_search

        with patch("brain_platform.services.local_graph_store.LocalGraphStore") as MockStore:
            MockStore.side_effect = RuntimeError("Neo4j not reachable")
            args = MagicMock(query="beliefs", num_results=5, group_id="test")
            result = cmd_brain_platform_search(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "setup-neo4j" in captured.out

    def test_requires_query(self, capsys):
        from brain_platform.cli.integration import cmd_brain_platform_search

        args = MagicMock(query=None)
        result = cmd_brain_platform_search(args)
        assert result == 1

    def test_no_facts_message(self, capsys):
        from brain_platform.cli.integration import cmd_brain_platform_search

        with patch("brain_platform.services.local_graph_store.LocalGraphStore") as MockStore:
            store_instance = MagicMock()
            store_instance.search.return_value = []
            MockStore.return_value = store_instance

            args = MagicMock(query="nothing", num_results=5, group_id="test")
            result = cmd_brain_platform_search(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No facts found" in captured.out


# ──────────────────────────────────────────────────────────────────────
# cmd_brain_platform_ingest
# ──────────────────────────────────────────────────────────────────────

class TestCmdBrainPlatformIngest:
    def test_ingests_file(self, tmp_path, capsys):
        from brain_platform.cli.integration import cmd_brain_platform_ingest

        test_file = tmp_path / "interview.txt"
        test_file.write_text("Q: Tell me about yourself.\nA: I'm a software engineer.")

        with patch("brain_platform.services.local_graph_store.LocalGraphStore") as MockStore:
            store_instance = MagicMock()
            store_instance.client.llm_client = MagicMock()
            MockStore.return_value = store_instance

            with patch("brain_platform.services.llm_adapter.LLMAdapter"):
                with patch("brain_platform.pipeline.ingestion_orchestrator.IngestionOrchestrator") as MockOrch:
                    orch_instance = MagicMock()
                    orch_instance.ingest_file.return_value = {
                        "documents": 1,
                        "chunks": 1,
                        "nodes_created": 5,
                        "edges_created": 3,
                        "source_type": "txt",
                        "file": str(test_file),
                        "size_bytes": 100,
                    }
                    MockOrch.return_value = orch_instance

                    args = MagicMock(file=str(test_file), group_id="test", type=None)
                    result = cmd_brain_platform_ingest(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Nodes:        5" in captured.out
        assert "Edges:        3" in captured.out

    def test_missing_file(self, capsys):
        from brain_platform.cli.integration import cmd_brain_platform_ingest

        args = MagicMock(file="/nonexistent/path.txt", group_id="test")
        result = cmd_brain_platform_ingest(args)
        assert result == 1

    def test_no_file_arg(self, capsys):
        from brain_platform.cli.integration import cmd_brain_platform_ingest

        args = MagicMock(file=None)
        result = cmd_brain_platform_ingest(args)
        assert result == 1


# ──────────────────────────────────────────────────────────────────────
# Parser registration
# ──────────────────────────────────────────────────────────────────────

class TestRegisterBrainPlatformCommands:
    def test_registers_platform_search(self):
        from brain_platform.cli.integration import register_brain_platform_commands

        # Build a minimal subparsers structure
        top = argparse.ArgumentParser()
        sub = top.add_subparsers(dest="cmd")
        brain = sub.add_parser("brain")
        brain_sub = brain.add_subparsers(dest="brain_action")
        interview = sub.add_parser("interview")

        # Wrap in a parent parser so register can find them
        parent = argparse.ArgumentParser()
        parent_sub = parent.add_subparsers(dest="cmd")
        parent_brain = parent_sub.add_parser("brain")
        parent_brain_sub = parent_brain.add_subparsers(dest="brain_action")
        parent_interview = parent_sub.add_parser("interview")

        register_brain_platform_commands(parent_sub)

        # The brain parser should now have platform-search, platform-ingest, setup-neo4j
        # Test by parsing a command line
        args = parent.parse_args(["brain", "platform-search", "test query"])
        assert args.brain_action == "platform-search"
        assert args.query == "test query"
        assert hasattr(args, "func")

    def test_registers_adaptive_flag_on_interview(self):
        from brain_platform.cli.integration import register_brain_platform_commands

        parent = argparse.ArgumentParser()
        parent_sub = parent.add_subparsers(dest="cmd")
        parent_interview = parent_sub.add_parser("interview")

        register_brain_platform_commands(parent_sub)

        args = parent.parse_args(["interview", "--adaptive"])
        assert args.adaptive is True

        args = parent.parse_args(["interview"])
        assert args.adaptive is False

    def test_registers_age_and_max_questions(self):
        from brain_platform.cli.integration import register_brain_platform_commands

        parent = argparse.ArgumentParser()
        parent_sub = parent.add_subparsers(dest="cmd")
        parent_sub.add_parser("interview")

        register_brain_platform_commands(parent_sub)

        args = parent.parse_args(["interview", "--adaptive", "--age", "25", "--max-questions", "10"])
        assert args.age == 25
        assert args.max_questions == 10


# ──────────────────────────────────────────────────────────────────────
# cmd_interview_adaptive — only test the dispatch path
# ──────────────────────────────────────────────────────────────────────

class TestCmdInterviewAdaptive:
    def test_dispatches_to_brain_platform_orchestrator(self):
        """cmd_interview_adaptive constructs an AdaptiveInterviewOrchestrator
        and calls .start(). We mock the orchestrator to avoid LLM calls.
        """
        from brain_platform.cli.integration import cmd_interview_adaptive

        mock_q = MagicMock()
        mock_q.dimension = "identity"
        mock_q.question = "What's your name?"

        with patch("brain_platform.services.llm_adapter.LLMAdapter"):
            with patch("brain_platform.interview_orchestrator.AdaptiveInterviewOrchestrator") as MockOrch:
                orch_instance = MagicMock()
                orch_instance.start.return_value = mock_q
                orch_instance.questions_asked = [(mock_q, "")]
                MockOrch.return_value = orch_instance

                with patch("builtins.input", side_effect=EOFError):
                    args = MagicMock(age=30, max_questions=19)
                    result = cmd_interview_adaptive(args)

        assert result == 0
        # Verify the orchestrator was constructed with the right age
        MockOrch.assert_called_once()
        call_kwargs = MockOrch.call_args.kwargs
        assert call_kwargs.get("user_age") == 30
        assert call_kwargs.get("max_questions") == 19


# ──────────────────────────────────────────────────────────────────────
# runtime_integration.py — GraphBackedBrainRetriever
# ──────────────────────────────────────────────────────────────────────

class TestGraphBackedBrainRetriever:
    def test_falls_back_to_local_when_neo4j_not_set(self, monkeypatch):
        from brain_platform.runtime_integration import GraphBackedBrainRetriever

        monkeypatch.delenv("NEO4J_URI", raising=False)
        retriever = GraphBackedBrainRetriever()

        with patch("brain.brain_retriever.BrainRetriever") as MockLocal:
            local_instance = MagicMock()
            local_instance.build_context_for_query.return_value = ["local fact"]
            MockLocal.return_value = local_instance

            facts = retriever.retrieve("test query")

        assert facts == ["local fact"]
        assert retriever._backend == "local"

    def test_uses_graphiti_when_neo4j_configured(self, monkeypatch):
        from brain_platform.runtime_integration import GraphBackedBrainRetriever

        monkeypatch.setenv("NEO4J_URI", "bolt://test:7687")

        with patch("brain_platform.services.local_graph_store.LocalGraphStore") as MockStore:
            store_instance = MagicMock()
            MockStore.return_value = store_instance

            retriever = GraphBackedBrainRetriever()
            retriever._graphiti_retriever = MagicMock()
            retriever._graphiti_retriever.search.return_value = ["graphiti fact"]
            retriever._backend = "graphiti"  # force the path

            facts = retriever.retrieve("test query")

        assert facts == ["graphiti fact"]

    def test_force_local_backend_via_env(self, monkeypatch):
        from brain_platform.runtime_integration import GraphBackedBrainRetriever

        monkeypatch.setenv("NEO4J_URI", "bolt://test:7687")
        monkeypatch.setenv("BRAIN_RETRIEVER", "local")

        retriever = GraphBackedBrainRetriever()
        assert retriever._select_backend() == "local"

    def test_force_graphiti_backend_via_env(self, monkeypatch):
        from brain_platform.runtime_integration import GraphBackedBrainRetriever

        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.setenv("BRAIN_RETRIEVER", "graphiti")

        retriever = GraphBackedBrainRetriever()
        # Backend is forced to graphiti even without NEO4J_URI
        assert retriever._select_backend() == "graphiti"

    def test_graphiti_failure_falls_back_to_local(self, monkeypatch):
        from brain_platform.runtime_integration import GraphBackedBrainRetriever

        monkeypatch.setenv("NEO4J_URI", "bolt://test:7687")
        retriever = GraphBackedBrainRetriever()
        retriever._backend = "graphiti"
        retriever._graphiti_retriever = MagicMock()
        retriever._graphiti_retriever.search.side_effect = RuntimeError("Neo4j down")

        with patch("brain.brain_retriever.BrainRetriever") as MockLocal:
            local_instance = MagicMock()
            local_instance.build_context_for_query.return_value = ["fallback fact"]
            MockLocal.return_value = local_instance

            facts = retriever.retrieve("test")

        assert facts == ["fallback fact"]

    def test_returns_empty_list_when_all_backends_fail(self, monkeypatch):
        from brain_platform.runtime_integration import GraphBackedBrainRetriever

        monkeypatch.delenv("NEO4J_URI", raising=False)
        retriever = GraphBackedBrainRetriever()

        with patch("brain.brain_retriever.BrainRetriever") as MockLocal:
            local_instance = MagicMock()
            local_instance.build_context_for_query.side_effect = RuntimeError("everything broken")
            MockLocal.return_value = local_instance

            facts = retriever.retrieve("test")

        assert facts == []


# ──────────────────────────────────────────────────────────────────────
# Chunk 6: cmd_brain_platform_ingest with --type flag and orchestrator
# ──────────────────────────────────────────────────────────────────────

class TestCmdBrainPlatformIngestWithType:
    def test_type_flag_registered(self):
        from brain_platform.cli.integration import register_brain_platform_commands
        import argparse

        parent = argparse.ArgumentParser()
        parent_sub = parent.add_subparsers(dest="cmd")
        parent_brain = parent_sub.add_parser("brain")
        parent_brain.add_subparsers(dest="brain_action")

        register_brain_platform_commands(parent_sub)

        args = parent.parse_args(["brain", "platform-ingest", "file.txt", "--type", "code"])
        assert args.type == "code"

    def test_type_flag_optional(self):
        from brain_platform.cli.integration import register_brain_platform_commands
        import argparse

        parent = argparse.ArgumentParser()
        parent_sub = parent.add_subparsers(dest="cmd")
        parent_brain = parent_sub.add_parser("brain")
        parent_brain.add_subparsers(dest="brain_action")

        register_brain_platform_commands(parent_sub)

        args = parent.parse_args(["brain", "platform-ingest", "file.txt"])
        assert args.type is None  # Auto-detect

    def test_invalid_type_rejected(self, tmp_path, monkeypatch):
        from brain_platform.cli.integration import cmd_brain_platform_ingest

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        args = MagicMock(file=str(test_file), group_id="test", type="bogus_type")
        result = cmd_brain_platform_ingest(args)
        assert result == 1


class TestCmdBrainPlatformIngestUsesOrchestrator:
    def test_routes_through_ingestion_orchestrator(self, tmp_path):
        """The CLI should use the IngestionOrchestrator, not read raw text."""
        from brain_platform.cli.integration import cmd_brain_platform_ingest

        test_file = tmp_path / "essay.md"
        test_file.write_text("# Test\n\nContent here.")

        with patch("brain_platform.pipeline.ingestion_orchestrator.IngestionOrchestrator") as MockOrch:
            orch_instance = MagicMock()
            orch_instance.ingest_file.return_value = {
                "documents": 1,
                "chunks": 2,
                "nodes_created": 10,
                "edges_created": 5,
                "source_type": "obsidian",
                "file": str(test_file),
                "size_bytes": 100,
            }
            MockOrch.return_value = orch_instance

            with patch("brain_platform.services.local_graph_store.LocalGraphStore"):
                with patch("brain_platform.services.llm_adapter.LLMAdapter"):
                    args = MagicMock(file=str(test_file), group_id="test", type=None)
                    result = cmd_brain_platform_ingest(args)

        assert result == 0
        # Verify IngestionOrchestrator.ingest_file was called
        orch_instance.ingest_file.assert_called_once()
        call_kwargs = orch_instance.ingest_file.call_args.kwargs
        assert call_kwargs["file_path"] == str(test_file)
        assert call_kwargs["group_id"] == "test"
        # No explicit source_type → auto-detect (None)
        assert call_kwargs["source_type"] is None


# ──────────────────────────────────────────────────────────────────────
# Chunk 7: interview progress bar + question numbering
# ──────────────────────────────────────────────────────────────────────

class TestInterviewProgressBar:
    """The adaptive interview should show a progress bar and question
    numbering so users know how far along they are."""

    def test_progress_bar_renders_correctly(self, capsys):
        from brain_platform.cli.integration import _progress_bar

        # 0% — empty bar
        result = _progress_bar(0, 10, width=10)
        assert "0%" in result
        assert "░" * 10 in result

        # 50% — half filled
        result = _progress_bar(5, 10, width=10)
        assert "50%" in result
        assert "█" * 5 in result
        assert "░" * 5 in result

        # 100% — full bar
        result = _progress_bar(10, 10, width=10)
        assert "100%" in result
        assert "█" * 10 in result

    def test_progress_bar_handles_zero_total(self):
        from brain_platform.cli.integration import _progress_bar

        # Should not crash on zero total
        result = _progress_bar(5, 0)
        assert result == ""

    def test_interview_intro_shows_question_count(self, capsys):
        """The intro should tell the user the approximate number of questions."""
        # Just verify the intro text is in the right format by checking
        # the source code (cheap test that doesn't require running the full interview)
        import inspect
        from brain_platform.cli.integration import cmd_interview_adaptive
        source = inspect.getsource(cmd_interview_adaptive)

        # The intro should mention "core questions" and a number
        assert "core questions" in source
        assert "max_questions" in source
        assert "Question" in source
        assert "progress" in source.lower() or "%" in source


# ──────────────────────────────────────────────────────────────────────
# Chunk 8: default group_id should match the active brain
# ──────────────────────────────────────────────────────────────────────

class TestDefaultGroupIdFromActiveBrain:
    """platform-search / platform-ingest / platform-deepen should
    default to the active brain's group_id so users don't have to
    pass --group-id every time."""

    def test_get_default_group_id_returns_active_brain(self, monkeypatch):
        from brain_platform.cli.integration import _get_default_group_id

        # Mock the active brain name
        import brain.paths
        monkeypatch.setattr(brain.paths, "get_active_brain_name", lambda: "bill-gates")

        assert _get_default_group_id() == "bill-gates"

    def test_falls_back_to_default_user_on_error(self, monkeypatch):
        """If get_active_brain_name raises (no config, etc.), fall back."""
        from brain_platform.cli.integration import _get_default_group_id

        import brain.paths
        def boom():
            raise RuntimeError("no config")
        monkeypatch.setattr(brain.paths, "get_active_brain_name", boom)

        assert _get_default_group_id() == "default_user"

    def test_search_uses_active_brain_by_default(self, capsys, monkeypatch):
        """beam brain platform-search should use the active brain's group_id."""
        from brain_platform.cli.integration import cmd_brain_platform_search

        import brain.paths
        monkeypatch.setattr(brain.paths, "get_active_brain_name", lambda: "bill-gates")

        # Mock the searcher so we don't need a real Neo4j
        with patch("brain_platform.services.local_graph_store.LocalGraphStore") as MockStore:
            store_instance = MagicMock()
            store_instance.search.return_value = []
            MockStore.return_value = store_instance

            # group_id=None simulates "user didn't pass --group-id"
            args = MagicMock(query="microsoft", num_results=5, group_id=None)
            result = cmd_brain_platform_search(args)

        # The search should have used "bill-gates" as the group_id
        # (from get_active_brain_name), not "default_user"
        store_instance.search.assert_called_once()
        call_kwargs = store_instance.search.call_args.kwargs
        assert call_kwargs["group_id"] == "bill-gates"

    def test_search_honors_explicit_group_id(self, capsys, monkeypatch):
        """If the user explicitly passes --group-id, use that even if
        it's not the active brain."""
        from brain_platform.cli.integration import cmd_brain_platform_search

        import brain.paths
        monkeypatch.setattr(brain.paths, "get_active_brain_name", lambda: "bill-gates")

        with patch("brain_platform.services.local_graph_store.LocalGraphStore") as MockStore:
            store_instance = MagicMock()
            store_instance.search.return_value = []
            MockStore.return_value = store_instance

            args = MagicMock(query="x", num_results=5, group_id="explicit-brain")
            cmd_brain_platform_search(args)

        store_instance.search.assert_called_once()
        call_kwargs = store_instance.search.call_args.kwargs
        assert call_kwargs["group_id"] == "explicit-brain"


# ──────────────────────────────────────────────────────────────────────
# Chunk 9: setup-neo4j detects missing graphiti_core
# ──────────────────────────────────────────────────────────────────────

class TestSetupNeo4jMissingGraphitiCore:
    """If graphiti_core isn't installed, setup-neo4j should fail fast
    with a clear install instruction rather than the cryptic
    'No module named graphiti_core' error."""

    def test_fails_fast_when_graphiti_core_missing(self, capsys, monkeypatch):
        """The wizard should detect the missing dep and return exit code 1."""
        import builtins
        from brain_platform.cli.integration import cmd_setup_neo4j

        # Block the import of graphiti_core
        import importlib
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "graphiti_core" or name.startswith("graphiti_core."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        args = MagicMock()
        result = cmd_setup_neo4j(args)

        # Should fail fast (not save creds, not try to connect)
        assert result == 1
        captured = capsys.readouterr()
        assert "graphiti-core is not installed" in captured.out
        assert "pip install" in captured.out
        assert "brain-platform-graph" in captured.out

    def test_continues_when_graphiti_core_installed(self, capsys, monkeypatch):
        """If graphiti_core IS importable, the wizard proceeds normally
        and asks for connection details (doesn't fail fast)."""
        from brain_platform.cli.integration import cmd_setup_neo4j

        # The wizard should proceed past the import check
        # (it'll then ask for URI input; we simulate "no" to keep
        # existing values)

        # Mock the env-reading functions
        monkeypatch.setattr(
            "brain_platform.cli.integration._read_env_value",
            lambda name: "neo4j+s://existing.io" if "URI" in name else "existing_user"
        )

        with patch("builtins.input", return_value="n"):  # Don't overwrite
            args = MagicMock()
            result = cmd_setup_neo4j(args)

        # Should NOT fail fast — proceeds to the "keep existing" path
        assert result == 0
        captured = capsys.readouterr()
        # The graphiti_core check passed (no error message about it)
        assert "graphiti-core is not installed" not in captured.out
        assert "Keeping existing values" in captured.out
