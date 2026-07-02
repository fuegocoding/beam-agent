"""Tier 1: Offline smoke test — fast structural verification of the brain_platform pipeline.

This test mocks all external services (LLM, Neo4j) and verifies the
WIRING of every service is correct: the right functions are called in
the right order, the CLI argument parsing works, every parser loads,
the schema validates, and the orchestrator routes by source type.

Designed to run in CI without any API keys or Neo4j instance. Fast
(target: <5 seconds). Complements the unit tests in test_chunk{1..5}.py
by exercising the *integration between modules* rather than the
modules in isolation.

For the live end-to-end test (which actually runs the pipeline
against real LLM + Neo4j), see :mod:`tests.brain_platform.test_integration`.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# 1. Parser smoke tests — every parser can be loaded and parses something
# ──────────────────────────────────────────────────────────────────────

class TestParserSmoke:
    """Every parser in brain_platform.pipeline.parsers should be importable,
    instantiable, and produce a non-empty ParseResult for sample input.

    This is the "structural" test — it doesn't validate parsing quality,
    just that the parsers don't crash and return the expected shape.
    """

    def test_all_parsers_importable(self):
        from brain_platform.pipeline.parsers import PARSER_MAP
        from brain_platform.models.enums import DataSourceType

        # All source types should have a registered parser (except types
        # without local implementations like audio, tweet, ai_memory)
        implemented = {
            DataSourceType.OBSIDIAN,
            DataSourceType.TXT,
            DataSourceType.CODE,
            DataSourceType.PROMPT,
            DataSourceType.INSTRUCTIONS,
            DataSourceType.EMAIL,
            DataSourceType.JOURNAL,
            DataSourceType.REDDIT,
        }
        for source_type in implemented:
            assert source_type in PARSER_MAP, f"Missing parser for {source_type.value}"

    def test_obsidian_parser_handles_zip_vault(self):
        """The Obsidian parser expects a vault ZIP, not raw markdown."""
        import io
        import zipfile
        from brain_platform.pipeline.parsers import get_parser
        from brain_platform.models.enums import DataSourceType

        # Create a minimal Obsidian vault ZIP
        md_content = (
            b"---\n"
            b"title: My Note\n"
            b"date: 2026-01-15\n"
            b"---\n"
            b"\n"
            b"Some [[wikilink]] content. #tag #another-tag\n"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("notes/my-note.md", md_content)

        parser = get_parser(DataSourceType.OBSIDIAN)
        results = parser.parse(buf.getvalue(), "vault.zip")
        assert len(results) >= 1
        assert "wikilink" in results[0].text

    def test_txt_parser_handles_plain_text(self):
        from brain_platform.pipeline.parsers import get_parser
        from brain_platform.models.enums import DataSourceType

        parser = get_parser(DataSourceType.TXT)
        results = parser.parse(b"Line 1\nLine 2\nLine 3\n", "notes.txt")
        assert len(results) >= 1
        assert "Line 1" in results[0].text

    def test_code_parser_handles_python(self):
        from brain_platform.pipeline.parsers import get_parser
        from brain_platform.models.enums import DataSourceType

        parser = get_parser(DataSourceType.CODE)
        py = b"def hello():\n    print('world')\n\nclass Foo:\n    pass\n"
        results = parser.parse(py, "main.py")
        assert len(results) >= 1
        assert "def hello" in results[0].text

    def test_journal_parser_handles_dated_entries(self):
        from brain_platform.pipeline.parsers import get_parser
        from brain_platform.models.enums import DataSourceType

        parser = get_parser(DataSourceType.JOURNAL)
        text = b"2026-01-15\n\nToday I worked on the brain platform port.\n\n2026-01-14\n\nYesterday I started the audit.\n"
        results = parser.parse(text, "journal.md")
        assert len(results) >= 1
        assert "brain platform" in results[0].text

    def test_email_parser_handles_json_bundle(self):
        """The EmailParser expects a JSON bundle of exported emails."""
        import json
        from brain_platform.pipeline.parsers import get_parser
        from brain_platform.models.enums import DataSourceType

        emails = json.dumps([{
            "id": "msg-1",
            "subject": "Test email",
            "from": "alice@example.com",
            "to": "bob@example.com",
            "date": "2026-01-15T10:30:00Z",
            "body": "Hello, this is a test email about sustainability.",
            "thread_id": "thread-1",
        }]).encode("utf-8")

        parser = get_parser(DataSourceType.EMAIL)
        results = parser.parse(emails, "inbox.json")
        assert len(results) >= 1
        assert "Test email" in results[0].text or "sustainability" in results[0].text

    def test_get_parser_raises_for_unknown_type(self):
        from brain_platform.pipeline.parsers import get_parser
        from brain_platform.models.enums import DataSourceType

        with pytest.raises(ValueError, match="No parser"):
            get_parser(DataSourceType.AUDIO)


# ──────────────────────────────────────────────────────────────────────
# 2. Source-type auto-detection smoke test
# ──────────────────────────────────────────────────────────────────────

class TestSourceTypeAutoDetection:
    """The ingestion orchestrator's file-extension → source-type mapping
    should cover the common file types users will feed the brain."""

    @pytest.mark.parametrize("filename,expected", [
        ("essay.md", "obsidian"),
        ("notes.markdown", "obsidian"),
        ("readme.txt", "txt"),
        ("script.py", "code"),
        ("app.js", "code"),
        ("component.tsx", "code"),
        ("lib.rs", "code"),
        ("Main.go", "code"),
        ("inbox.eml", "email"),
        ("paper.pdf", "pdf"),
        ("report.docx", "docx"),
        ("comments.json", "reddit"),
    ])
    def test_extension_maps_to_expected_type(self, filename, expected):
        from brain_platform.pipeline.ingestion_orchestrator import detect_source_type
        from brain_platform.models.enums import DataSourceType

        detected = detect_source_type(filename)
        assert detected == DataSourceType(expected)

    def test_unknown_extension_falls_back_to_txt(self):
        from brain_platform.pipeline.ingestion_orchestrator import detect_source_type
        from brain_platform.models.enums import DataSourceType

        assert detect_source_type("data.xyz") == DataSourceType.TXT
        assert detect_source_type("noextension") == DataSourceType.TXT


# ──────────────────────────────────────────────────────────────────────
# 3. Ingestion orchestrator pipeline smoke test
# ──────────────────────────────────────────────────────────────────────

class TestIngestionPipelineSmoke:
    """The IngestionOrchestrator should correctly chain:
    parse → chunk → extract → write.

    LLM and Neo4j are mocked; we verify the wiring is right.
    """

    def test_pipeline_chains_correctly(self, tmp_path):
        from brain_platform.pipeline.ingestion_orchestrator import IngestionOrchestrator
        from brain_platform.pipeline.parsers.base import ParseResult
        from brain_platform.models.enums import DataSourceType

        test_file = tmp_path / "essay.md"
        test_file.write_text("# Story\n\nI value autonomy.")

        # Mock every component
        mock_parser = MagicMock()
        mock_parser.parse.return_value = [ParseResult(
            text="I value autonomy.",
            title="Story",
            metadata={"source": "obsidian"},
        )]

        mock_chunker = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "I value autonomy."
        mock_chunker.chunk.return_value = [mock_chunk]

        mock_extractor = MagicMock()
        mock_graph = MagicMock()
        mock_extractor.extract.return_value = mock_graph

        mock_writer = MagicMock()
        mock_writer.write.return_value = {"nodes_created": 3, "edges_created": 2}

        mock_store = MagicMock()
        mock_store.client.llm_client = MagicMock()

        with patch("brain_platform.pipeline.ingestion_orchestrator.get_parser", return_value=mock_parser), \
             patch("brain_platform.pipeline.ingestion_orchestrator.SemanticChunker", return_value=mock_chunker), \
             patch("brain_platform.services.local_graph_writer.LocalGraphWriter", return_value=mock_writer), \
             patch("brain_platform.extractor.brain_extractor.BrainExtractor", return_value=mock_extractor):
            orch = IngestionOrchestrator(store=mock_store, llm=MagicMock())
            result = orch.ingest_file(file_path=str(test_file), group_id="test")

        # Verify the pipeline chained in the right order
        mock_parser.parse.assert_called_once()
        mock_chunker.chunk.assert_called_once()
        mock_extractor.extract.assert_called_once()
        mock_writer.write.assert_called_once()

        # Verify the data flowed correctly between stages
        chunk_arg = mock_chunker.chunk.call_args[0][0]
        assert chunk_arg.text == "I value autonomy."

        extract_arg = mock_extractor.extract.call_args.kwargs["interview_text"]
        assert "autonomy" in extract_arg

        write_arg = mock_writer.write.call_args.kwargs
        assert write_arg["graph"] is mock_graph
        assert write_arg["group_id"] == "test"

        # Verify the result aggregates correctly
        assert result["nodes_created"] == 3
        assert result["edges_created"] == 2
        assert result["chunks"] == 1


# ──────────────────────────────────────────────────────────────────────
# 4. Brain file generation smoke test
# ──────────────────────────────────────────────────────────────────────

class TestBrainFileGenerationSmoke:
    """BrainFileGenerator chains: graph read → style analysis → personality
    extraction → assembly. Verify the wiring."""

    def test_generator_assembles_brain_file(self):
        from brain_platform.pipeline.brain_file.generator import BrainFileGenerator
        from brain_platform.pipeline.brain_file.graph_reader import GraphData
        from brain_platform.pipeline.brain_file.schema import GraphNode, GraphEdge

        # Mock the graph reader
        mock_reader = MagicMock()
        mock_reader.read_all.return_value = GraphData(
            nodes=[GraphNode(id="1", type="Value", label="autonomy", summary="Values independence")],
            edges=[GraphEdge(id="e1", source="1", target="2", relation="INFORMS", fact="A informs B")],
            node_summaries=["[Value] autonomy: Values independence"],
            edge_facts=["A informs B"],
        )

        # Mock the personality extractor
        mock_extractor = MagicMock()
        from brain_platform.pipeline.brain_file.schema import PersonalityProfile
        mock_extractor.extract.return_value = PersonalityProfile(
            values=["autonomy"],
            core_beliefs=["sustainability matters"],
        )

        # Patch the dependencies
        store = MagicMock()
        llm = MagicMock()
        with patch("brain_platform.pipeline.brain_file.generator.GraphReader", return_value=mock_reader), \
             patch("brain_platform.pipeline.brain_file.generator.PersonalityExtractor", return_value=mock_extractor):
            generator = BrainFileGenerator(store=store, llm=llm)
            result = generator.generate(
                group_id="test_group",
                raw_texts=["Sample text for style analysis."],
            )

        # Verify the assembled brain file
        assert result.knowledge_graph.nodes[0].label == "autonomy"
        assert result.knowledge_graph.edges[0].relation == "INFORMS"
        assert "autonomy" in result.personality_profile.values
        assert result.metadata.graphiti_group_id == "test_group"

    def test_generator_to_file_writes_valid_json(self, tmp_path):
        from brain_platform.pipeline.brain_file.generator import BrainFileGenerator
        from brain_platform.pipeline.brain_file.graph_reader import GraphData
        from brain_platform.pipeline.brain_file.schema import GraphNode

        mock_reader = MagicMock()
        mock_reader.read_all.return_value = GraphData(
            nodes=[GraphNode(id="1", type="Value", label="test")],
        )
        mock_extractor = MagicMock()
        from brain_platform.pipeline.brain_file.schema import PersonalityProfile
        mock_extractor.extract.return_value = PersonalityProfile()

        output_path = tmp_path / "brain.json"
        with patch("brain_platform.pipeline.brain_file.generator.GraphReader", return_value=mock_reader), \
             patch("brain_platform.pipeline.brain_file.generator.PersonalityExtractor", return_value=mock_extractor):
            generator = BrainFileGenerator(store=MagicMock(), llm=MagicMock())
            result = generator.generate_to_file(
                group_id="test",
                output_path=str(output_path),
            )

        # File exists, valid JSON
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "metadata" in data
        assert "personality_profile" in data
        assert "knowledge_graph" in data
        assert result["path"] == str(output_path)
        assert result["size_bytes"] > 0


# ──────────────────────────────────────────────────────────────────────
# 5. Exporters smoke test — all 3 formats produce valid output
# ──────────────────────────────────────────────────────────────────────

class TestExporterSmoke:
    """All 3 exporters (claude, jsonld, obsidian) should produce valid
    output from a complete BrainFileSchema."""

    def _make_brain_file(self):
        from brain_platform.pipeline.brain_file.schema import (
            BrainFileSchema, BrainFileMetadata, PersonalityProfile,
            WritingStyle, KnowledgeDomain, KnowledgeGraph, GraphNode, GraphEdge,
        )
        from datetime import datetime, timezone
        return BrainFileSchema(
            metadata=BrainFileMetadata(
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                user_id="test",
                source_count=1,
                graphiti_group_id="test",
            ),
            personality_profile=PersonalityProfile(
                values=["autonomy", "honesty"],
                core_beliefs=["sustainability"],
            ),
            writing_style=WritingStyle(
                avg_sentence_length=15.0,
                vocabulary_level="advanced",
                tone="analytical",
            ),
            knowledge_domains=[KnowledgeDomain(topic="Tech", confidence=0.8, source_count=2)],
            knowledge_graph=KnowledgeGraph(
                nodes=[GraphNode(id="1", type="Value", label="autonomy", summary="Independent")],
                edges=[GraphEdge(id="e1", source="1", target="2", relation="INFORMS", fact="A informs B")],
            ),
        )

    def test_jsonld_export(self):
        from brain_platform.pipeline.brain_file.exporters.jsonld import export_jsonld

        result = export_jsonld(self._make_brain_file())
        assert isinstance(result, bytes)
        json.loads(result.decode("utf-8"))  # Validates JSON

    def test_claude_export(self):
        from brain_platform.pipeline.brain_file.exporters.claude import export_claude

        result = export_claude(self._make_brain_file())
        assert "system_prompt" in result
        assert "knowledge_files" in result
        assert "autonomy" in result["system_prompt"]
        assert "sustainability" in result["system_prompt"]

    def test_obsidian_export(self):
        from brain_platform.pipeline.brain_file.exporters.obsidian import export_obsidian
        import io
        import zipfile

        result = export_obsidian(self._make_brain_file())
        assert isinstance(result, bytes)
        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            names = zf.namelist()
        assert "_index.md" in names
        assert "_personality.md" in names
        assert "_writing_style.md" in names


# ──────────────────────────────────────────────────────────────────────
# 6. Deepen smoke test — gap analysis + probe generation
# ──────────────────────────────────────────────────────────────────────

class TestDeepenSmoke:
    """analyze_brain_gaps + generate_probe_questions should correctly
    chain: count nodes per dimension → identify gaps → ask LLM for probes."""

    def test_deepen_identifies_gaps(self):
        from brain_platform.pipeline.interview.deepen import analyze_brain_gaps
        from brain_platform.pipeline.brain_schema import PersonalityGraph, TraitNode, BeliefNode

        # Empty graph → all dimensions are gaps
        graph = PersonalityGraph(user_summary="")
        result = analyze_brain_gaps(graph)
        assert result.completeness_pct == 0
        assert len(result.gaps) > 0

    def test_deepen_probes_fallback_on_llm_failure(self):
        from brain_platform.pipeline.interview.deepen import (
            analyze_brain_gaps, generate_probe_questions,
        )
        from brain_platform.pipeline.brain_schema import PersonalityGraph

        graph = PersonalityGraph(user_summary="")
        result = analyze_brain_gaps(graph)
        llm = MagicMock()
        llm.generate_response.side_effect = RuntimeError("LLM down")
        probes = generate_probe_questions(
            gaps=result.gaps[:3],
            graph=graph,
            covered_questions=[],
            llm_client=llm,
        )
        # Fallback produces per-gap probes
        assert len(probes) == 3
        assert all(p.question for p in probes)


# ──────────────────────────────────────────────────────────────────────
# 7. CLI smoke test — all commands parse args correctly
# ──────────────────────────────────────────────────────────────────────

class TestCLISmoke:
    """All CLI commands should have correct argparse wiring."""

    def test_all_brain_platform_subcommands_registered(self):
        from brain_platform.cli.integration import register_brain_platform_commands
        import argparse

        parent = argparse.ArgumentParser()
        parent_sub = parent.add_subparsers(dest="cmd")
        parent_brain = parent_sub.add_parser("brain")
        parent_brain.add_subparsers(dest="brain_action")
        parent_interview = parent_sub.add_parser("interview")

        register_brain_platform_commands(parent_sub)

        # Every subcommand should parse without error
        for cmd in [
            ["brain", "platform-search", "test query"],
            ["brain", "platform-ingest", "file.txt"],
            ["brain", "platform-generate", "out.json"],
            ["brain", "platform-export", "out.md", "--format", "claude"],
            ["brain", "platform-deepen"],
            ["brain", "setup-neo4j"],
        ]:
            args = parent.parse_args(cmd)
            assert args.brain_action.startswith("platform-") or args.brain_action == "setup-neo4j"

    def test_interview_adaptive_flag(self):
        from brain_platform.cli.integration import register_brain_platform_commands
        import argparse

        parent = argparse.ArgumentParser()
        parent_sub = parent.add_subparsers(dest="cmd")
        parent_sub.add_parser("interview")

        register_brain_platform_commands(parent_sub)

        # With --adaptive
        args = parent.parse_args(["interview", "--adaptive", "--age", "30", "--max-questions", "15"])
        assert args.adaptive is True
        assert args.age == 30
        assert args.max_questions == 15

        # Without --adaptive
        args = parent.parse_args(["interview"])
        assert args.adaptive is False
        assert args.age == 30  # default
        assert args.max_questions == 19  # default


# ──────────────────────────────────────────────────────────────────────
# 8. Retriever fallback smoke test — GraphBackedBrainRetriever auto-detects
# ──────────────────────────────────────────────────────────────────────

class TestRetrieverFallbackSmoke:
    """The agent's retriever should auto-detect Neo4j vs offline and
    fall back gracefully when Neo4j is down."""

    def test_no_neo4j_uri_uses_local(self, monkeypatch):
        from brain_platform.runtime_integration import GraphBackedBrainRetriever

        monkeypatch.delenv("NEO4J_URI", raising=False)
        retriever = GraphBackedBrainRetriever()
        assert retriever._select_backend() == "local"

    def test_force_local_via_env(self, monkeypatch):
        from brain_platform.runtime_integration import GraphBackedBrainRetriever

        monkeypatch.setenv("NEO4J_URI", "bolt://test:7687")
        monkeypatch.setenv("BRAIN_RETRIEVER", "local")
        retriever = GraphBackedBrainRetriever()
        assert retriever._select_backend() == "local"

    def test_force_graphiti_via_env(self, monkeypatch):
        from brain_platform.runtime_integration import GraphBackedBrainRetriever

        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.setenv("BRAIN_RETRIEVER", "graphiti")
        retriever = GraphBackedBrainRetriever()
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


# ──────────────────────────────────────────────────────────────────────
# 9. Style analyzer smoke test
# ──────────────────────────────────────────────────────────────────────

class TestStyleAnalyzerSmoke:
    def test_analyzes_formal_text(self):
        from brain_platform.pipeline.brain_file.style_analyzer import StyleAnalyzer

        sa = StyleAnalyzer()
        formal_text = (
            "Furthermore, the data suggests a correlation. The methodology "
            "was systematic and the analysis was thorough. Consequently, the "
            "evidence supports the hypothesis. Moreover, the empirical "
            "findings align with the framework."
        )
        result = sa.analyze([formal_text])
        assert result.tone in ("analytical", "neutral")
        assert result.vocabulary_level in ("basic", "intermediate", "advanced", "technical")

    def test_analyzes_casual_text(self):
        from brain_platform.pipeline.brain_file.style_analyzer import StyleAnalyzer

        sa = StyleAnalyzer()
        casual_text = "Hey, that's awesome! Yeah I kinda love it tbh. Cool stuff."
        result = sa.analyze([casual_text])
        assert result.avg_sentence_length > 0
