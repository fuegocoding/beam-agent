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
            MockStore.return_value = store_instance

            with patch("brain_platform.services.local_graph_writer.LocalGraphWriter") as MockWriter:
                writer_instance = MagicMock()
                writer_instance.write_interview_session.return_value = {
                    "nodes_created": 5, "edges_created": 3,
                }
                MockWriter.return_value = writer_instance

                args = MagicMock(file=str(test_file), group_id="test")
                result = cmd_brain_platform_ingest(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Nodes created: 5" in captured.out
        assert "Edges created: 3" in captured.out

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
