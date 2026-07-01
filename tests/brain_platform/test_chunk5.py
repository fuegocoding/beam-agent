"""Tests for Chunk 5 — brain_file generation + deepen + refiner.

These tests cover the brain_file package (graph_reader, style_analyzer,
style_embedder, personality_extractor, generator, exporters) and the
interview deepening / personality refinement logic. LLM calls are
mocked; Neo4j operations are mocked via LocalGraphStore.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# pipeline/brain_file/style_analyzer.py
# ──────────────────────────────────────────────────────────────────────

class TestStyleAnalyzer:
    def test_empty_input(self):
        from brain_platform.pipeline.brain_file.style_analyzer import StyleAnalyzer
        from brain_platform.pipeline.brain_file.schema import WritingStyle

        sa = StyleAnalyzer()
        result = sa.analyze([])
        assert isinstance(result, WritingStyle)

    def test_basic_analysis(self):
        from brain_platform.pipeline.brain_file.style_analyzer import StyleAnalyzer

        sa = StyleAnalyzer()
        texts = [
            "Furthermore, the data suggests a correlation. The methodology was systematic.",
            "Moreover, the empirical evidence supports this hypothesis.",
        ]
        result = sa.analyze(texts)
        assert result.tone in ("analytical", "optimistic", "critical", "neutral")
        assert result.avg_sentence_length > 0
        assert result.vocabulary_level in ("basic", "intermediate", "advanced", "technical")

    def test_casual_tone(self):
        from brain_platform.pipeline.brain_file.style_analyzer import StyleAnalyzer

        sa = StyleAnalyzer()
        texts = ["Hey, that was awesome! Yeah I kinda love it tbh. Cool stuff."]
        result = sa.analyze(texts)
        # Casual markers should pull formality down
        assert result.tone in ("neutral", "optimistic")


# ──────────────────────────────────────────────────────────────────────
# pipeline/brain_file/personality_extractor.py
# ──────────────────────────────────────────────────────────────────────

class TestPersonalityExtractor:
    def test_empty_input(self):
        from brain_platform.pipeline.brain_file.personality_extractor import PersonalityExtractor

        llm = MagicMock()
        pe = PersonalityExtractor(llm=llm)
        result = pe.extract(node_summaries=[], edge_facts=[], community_summaries=[])
        assert result.communication_style == "analytical"
        assert result.values == []
        assert llm.generate_response.call_count == 0

    def test_extracts_typed_nodes(self):
        from brain_platform.pipeline.brain_file.personality_extractor import PersonalityExtractor
        from brain_platform.pipeline.brain_file.schema import GraphNode

        llm = MagicMock()
        llm.generate_response.return_value = json.dumps({
            "communication_style": "narrative",
            "formality": 0.7,
            "humor_frequency": 0.3,
            "empathy_indicators": 0.8,
        })

        pe = PersonalityExtractor(llm=llm)
        typed_nodes = [
            GraphNode(id="1", type="Value", label="autonomy", attributes={"value_name": "autonomy"}),
            GraphNode(id="2", type="Belief", label="sustainability", attributes={"position": "sustainability matters"}),
            GraphNode(id="3", type="Boundary", label="won't lie", attributes={"description": "won't lie about impact"}),
        ]
        result = pe.extract(
            node_summaries=["some node"],
            edge_facts=["some fact"],
            community_summaries=[],
            typed_nodes=typed_nodes,
        )
        assert "autonomy" in result.values
        assert "sustainability matters" in result.core_beliefs
        assert any("will not: won't lie about impact" in v for v in result.values)
        assert result.communication_style == "narrative"
        assert result.formality == 0.7

    def test_llm_failure_uses_defaults(self):
        from brain_platform.pipeline.brain_file.personality_extractor import PersonalityExtractor
        from brain_platform.pipeline.brain_file.schema import GraphNode

        llm = MagicMock()
        llm.generate_response.side_effect = RuntimeError("LLM down")
        pe = PersonalityExtractor(llm=llm)
        typed_nodes = [
            GraphNode(id="1", type="Value", label="honesty", attributes={"value_name": "honesty"}),
        ]
        result = pe.extract(
            node_summaries=["some node"],
            edge_facts=[],
            community_summaries=[],
            typed_nodes=typed_nodes,
        )
        # LLM failed — values still extracted, scores are defaults
        assert "honesty" in result.values
        assert result.communication_style == "analytical"  # default
        assert result.formality == 0.5  # default

    def test_parses_fenced_json(self):
        from brain_platform.pipeline.brain_file.personality_extractor import PersonalityExtractor

        llm = MagicMock()
        llm.generate_response.return_value = (
            "```json\n"
            '{"communication_style": "directive", "formality": 0.4, '
            '"humor_frequency": 0.1, "empathy_indicators": 0.5}\n'
            "```"
        )
        pe = PersonalityExtractor(llm=llm)
        result = pe.extract(node_summaries=["x"], edge_facts=[], community_summaries=[])
        assert result.communication_style == "directive"
        assert result.formality == 0.4


# ──────────────────────────────────────────────────────────────────────
# pipeline/brain_file/exporters/
# ──────────────────────────────────────────────────────────────────────

class TestExporters:
    def _make_brain_file(self):
        from brain_platform.pipeline.brain_file.schema import (
            BrainFileSchema, BrainFileMetadata, PersonalityProfile,
            WritingStyle, StyleEmbedding, KnowledgeDomain, KnowledgeGraph,
            GraphNode, GraphEdge, SourceManifestEntry,
        )
        from datetime import datetime, timezone

        return BrainFileSchema(
            metadata=BrainFileMetadata(
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                user_id="test_user",
                source_count=2,
                graphiti_group_id="test_user",
            ),
            personality_profile=PersonalityProfile(
                communication_style="analytical",
                formality=0.7,
                humor_frequency=0.2,
                empathy_indicators=0.6,
                values=["autonomy", "honesty"],
                core_beliefs=["sustainability matters"],
            ),
            writing_style=WritingStyle(
                avg_sentence_length=15.0,
                vocabulary_level="advanced",
                common_phrases=["the data suggests"],
                tone="analytical",
            ),
            style_embedding=StyleEmbedding(
                authorship_vector=[0.0] * 768,
                model="AnnaWegmann/Style-Embedding",
                sample_count=5,
            ),
            knowledge_domains=[
                KnowledgeDomain(
                    topic="Sustainable Materials",
                    confidence=0.8,
                    source_count=3,
                    key_entities=["mycelium", "bioplastics"],
                    community_summary="Deep expertise in compostable polymers",
                ),
            ],
            knowledge_graph=KnowledgeGraph(
                nodes=[GraphNode(id="1", type="PersonalityTrait", label="analytical")],
                edges=[GraphEdge(id="e1", source="1", target="2", relation="INFORMS", fact="A informs B")],
            ),
            source_manifest=[
                SourceManifestEntry(
                    source_type="pdf",
                    import_date=datetime.now(timezone.utc),
                    record_count=5,
                ),
            ],
        )

    def test_export_jsonld(self):
        from brain_platform.pipeline.brain_file.exporters.jsonld import export_jsonld

        bf = self._make_brain_file()
        result = export_jsonld(bf)
        assert isinstance(result, bytes)
        # Should be valid JSON
        data = json.loads(result.decode("utf-8"))
        assert "personality_profile" in data or "metadata" in data

    def test_export_claude(self):
        from brain_platform.pipeline.brain_file.exporters.claude import export_claude

        bf = self._make_brain_file()
        result = export_claude(bf)
        assert "system_prompt" in result
        assert "knowledge_files" in result
        assert len(result["system_prompt"]) > 0
        assert "autonomy" in result["system_prompt"]
        # Knowledge file
        kf = result["knowledge_files"][0]
        assert kf["filename"] == "knowledge.md"
        assert "Sustainable Materials" in kf["content"]

    def test_export_obsidian(self):
        from brain_platform.pipeline.brain_file.exporters.obsidian import export_obsidian

        bf = self._make_brain_file()
        result = export_obsidian(bf)
        assert isinstance(result, bytes)
        # Should be a valid zip
        import io
        import zipfile
        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            names = zf.namelist()
        assert "_index.md" in names
        assert "_personality.md" in names
        assert "_writing_style.md" in names


# ──────────────────────────────────────────────────────────────────────
# pipeline/interview/deepen.py
# ──────────────────────────────────────────────────────────────────────

class TestDeepen:
    def test_empty_graph(self):
        from brain_platform.pipeline.interview.deepen import analyze_brain_gaps
        from brain_platform.pipeline.brain_schema import PersonalityGraph

        graph = PersonalityGraph(user_summary="")
        result = analyze_brain_gaps(graph)
        assert result.completeness_pct == 0
        assert len(result.gaps) > 0
        assert "0%" in result.summary or "complete" in result.summary

    def test_well_populated_graph(self):
        from brain_platform.pipeline.interview.deepen import analyze_brain_gaps
        from brain_platform.pipeline.brain_schema import (
            PersonalityGraph, TraitNode, BeliefNode, ValueNode, BoundaryNode,
            LifeEventNode, MemoryNode, PatternNode, SocialNode,
            ExpertiseNode, StyleNode, PersonNode,
        )

        # Create a well-populated graph (5+ in each dimension)
        def mk(cls, n):
            return [cls(name=f"n_{i}", summary="") for i in range(n)]

        graph = PersonalityGraph(
            user_summary="",
            traits=mk(TraitNode, 10),
            beliefs=mk(BeliefNode, 10),
            values=mk(ValueNode, 6),
            boundaries=mk(BoundaryNode, 5),
            life_events=mk(LifeEventNode, 8),
            memories=mk(MemoryNode, 8),
            patterns=mk(PatternNode, 6),
            social=mk(SocialNode, 5),
            expertise=mk(ExpertiseNode, 4),
            style=mk(StyleNode, 6),
            people=mk(PersonNode, 5),
        )
        result = analyze_brain_gaps(graph)
        assert result.completeness_pct >= 90
        assert len(result.gaps) == 0
        assert "well-populated" in result.summary.lower()

    def test_type_error_on_non_graph(self):
        from brain_platform.pipeline.interview.deepen import analyze_brain_gaps

        with pytest.raises(TypeError, match="Expected PersonalityGraph"):
            analyze_brain_gaps({"not": "a graph"})

    def test_gaps_sorted_by_severity(self):
        from brain_platform.pipeline.interview.deepen import analyze_brain_gaps
        from brain_platform.pipeline.brain_schema import (
            PersonalityGraph, TraitNode, BeliefNode,
        )

        graph = PersonalityGraph(
            user_summary="",
            traits=[TraitNode(name="t1", summary="")],  # 1/9 — big gap
            beliefs=[BeliefNode(name=f"b{i}", summary="") for i in range(7)],  # 7/8 — small gap
        )
        result = analyze_brain_gaps(graph)
        # Largest gap (traits, gap=8) should be first
        assert result.gaps[0].dimension == "traits"

    def test_generate_probes_fallback(self):
        """When LLM fails, the fallback generator returns simple per-gap probes."""
        from brain_platform.pipeline.interview.deepen import (
            analyze_brain_gaps, generate_probe_questions, GapAnalysis,
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
        # Fallback produces one probe per gap
        assert len(probes) == 3
        assert all(p.question for p in probes)


# ──────────────────────────────────────────────────────────────────────
# pipeline/brain_file/graph_reader.py
# ──────────────────────────────────────────────────────────────────────

class TestGraphReader:
    def test_handles_unavailable_graphiti(self):
        from brain_platform.pipeline.brain_file.graph_reader import GraphReader, GraphData

        store = MagicMock()
        # Simulate "not initialized"
        type(store).client = property(lambda self: (_ for _ in ()).throw(RuntimeError("not initialized")))
        reader = GraphReader(store)
        data = reader.read_all("test_group")
        assert isinstance(data, GraphData)
        assert len(data.nodes) == 0
        assert len(data.edges) == 0

    def test_reads_nodes_and_edges(self):
        from brain_platform.pipeline.brain_file.graph_reader import GraphReader

        # Mock the Graphiti client
        mock_node = MagicMock()
        mock_node.uuid = "node-uuid-1"
        mock_node.name = "autonomy"
        mock_node.summary = "Values independence"
        mock_node.labels = ["Value", "Entity"]
        mock_node.attributes = {"value_name": "autonomy"}

        mock_edge = MagicMock()
        mock_edge.uuid = "edge-uuid-1"
        mock_edge.source_node_uuid = "a"
        mock_edge.target_node_uuid = "b"
        mock_edge.name = "INFORMS"
        mock_edge.fact = "A informs B"
        mock_edge.valid_at = None
        mock_edge.invalid_at = None

        client = MagicMock()
        client.nodes.entity.get_by_group_ids = AsyncMock(return_value=[mock_node])
        client.nodes.community.get_by_group_ids = AsyncMock(return_value=[])
        client.edges.entity.get_by_group_ids = AsyncMock(return_value=[mock_edge])
        client.edges.community.get_by_group_ids = AsyncMock(return_value=[])

        store = MagicMock()
        store.client = client

        reader = GraphReader(store)
        data = reader.read_all("test_group")

        assert len(data.nodes) == 1
        assert data.nodes[0].label == "autonomy"
        assert data.nodes[0].type == "Value"
        assert len(data.edges) == 1
        assert data.edges[0].relation == "INFORMS"
        assert len(data.node_summaries) == 1
        assert len(data.edge_facts) == 1


# ──────────────────────────────────────────────────────────────────────
# pipeline/personality_refiner.py — HUB_EDGE_MAP + dedup + orphan fix
# ──────────────────────────────────────────────────────────────────────

class TestPersonalityRefinerHUB_EDGE_MAP:
    def test_class_attribute_exists(self):
        """The HUB_EDGE_MAP class attribute must match the cloud."""
        from brain_platform.pipeline.personality_refiner import PersonalityRefiner

        assert hasattr(PersonalityRefiner, "HUB_EDGE_MAP")
        m = PersonalityRefiner.HUB_EDGE_MAP
        # Must have all 10 construct types
        assert "PersonalityTrait" in m
        assert "Belief" in m
        assert "Value" in m
        assert "Boundary" in m
        assert "LifeEvent" in m
        assert "EpisodicMemory" in m
        assert "KnowledgeDomain" in m
        assert "SocialPattern" in m
        assert "StyleProfile" in m
        assert "CognitivePattern" in m
        # Edge types must match the cloud
        assert m["PersonalityTrait"] == "HAS_TRAIT"
        assert m["Belief"] == "HOLDS"
        assert m["Value"] == "DRIVEN_BY"
        assert m["CognitivePattern"] == "HAS_TRAIT"

    def test_used_by_create_hub_edges(self):
        """_create_hub_edges should use HUB_EDGE_MAP (not hardcoded INVOLVES)."""
        from brain_platform.pipeline.personality_refiner import PersonalityRefiner
        import inspect
        src = inspect.getsource(PersonalityRefiner._create_hub_edges)
        assert "HUB_EDGE_MAP" in src
        # Should NOT have the old inline HUB_EDGE_FOR_TYPE dict
        assert "HUB_EDGE_FOR_TYPE" not in src

    def test_used_by_fix_orphan_nodes(self):
        """_fix_orphan_nodes should use HUB_EDGE_MAP for orphan classification."""
        from brain_platform.pipeline.personality_refiner import PersonalityRefiner
        import inspect
        src = inspect.getsource(PersonalityRefiner._fix_orphan_nodes)
        assert "HUB_EDGE_MAP" in src


class TestPersonalityRefinerDedup:
    def test_dedup_person_nodes_finds_duplicates(self):
        """_dedup_person_nodes queries for short-named duplicate nodes."""
        from brain_platform.pipeline.personality_refiner import PersonalityRefiner
        import inspect
        src = inspect.getsource(PersonalityRefiner._dedup_person_nodes)

        # Must have the dedup logic (Cypher + redirect + delete)
        assert "toLower(trim(n.name))" in src, "missing normalize-name step"
        assert "WHERE size(nodes) > 1" in src, "missing duplicate filter"
        assert "RELATES_TO" in src, "missing edge redirect"
        assert "MENTIONS" in src, "missing MENTIONS handling"
        assert "DETACH DELETE" in src, "missing delete step"
        assert "merged_count" in src, "missing return value"
        # Must pick canonical as Entity-only
        assert "Entity" in src, "missing canonical selection logic"


class TestPersonalityRefinerOrphanFix:
    def test_uses_hub_edge_map_for_edge_type(self):
        """_fix_orphan_nodes should look up edge type from HUB_EDGE_MAP,
        not hardcode INVOLVES."""
        from brain_platform.pipeline.personality_refiner import PersonalityRefiner
        import inspect
        src = inspect.getsource(PersonalityRefiner._fix_orphan_nodes)
        # Should call .get() on HUB_EDGE_MAP
        assert "HUB_EDGE_MAP.get" in src or "HUB_EDGE_MAP[" in src


# ──────────────────────────────────────────────────────────────────────
# pipeline/ingestion_orchestrator.py — the import pipeline
# ──────────────────────────────────────────────────────────────────────

class TestDetectSourceType:
    """Auto-detection from file extension."""

    def test_markdown_defaults_to_obsidian(self):
        from brain_platform.pipeline.ingestion_orchestrator import detect_source_type
        from brain_platform.models.enums import DataSourceType

        assert detect_source_type("essay.md") == DataSourceType.OBSIDIAN
        assert detect_source_type("notes.markdown") == DataSourceType.OBSIDIAN

    def test_text_files(self):
        from brain_platform.pipeline.ingestion_orchestrator import detect_source_type
        from brain_platform.models.enums import DataSourceType

        assert detect_source_type("readme.txt") == DataSourceType.TXT
        assert detect_source_type("notes.text") == DataSourceType.TXT

    def test_code_files(self):
        from brain_platform.pipeline.ingestion_orchestrator import detect_source_type
        from brain_platform.models.enums import DataSourceType

        assert detect_source_type("script.py") == DataSourceType.CODE
        assert detect_source_type("app.js") == DataSourceType.CODE
        assert detect_source_type("component.tsx") == DataSourceType.CODE
        assert detect_source_type("lib.rs") == DataSourceType.CODE
        assert detect_source_type("Main.go") == DataSourceType.CODE
        assert detect_source_type("Main.GO") == DataSourceType.CODE  # case insensitive

    def test_email_files(self):
        from brain_platform.pipeline.ingestion_orchestrator import detect_source_type
        from brain_platform.models.enums import DataSourceType

        assert detect_source_type("inbox.eml") == DataSourceType.EMAIL
        assert detect_source_type("all.mbox") == DataSourceType.EMAIL

    def test_pdf_and_docx(self):
        from brain_platform.pipeline.ingestion_orchestrator import detect_source_type
        from brain_platform.models.enums import DataSourceType

        assert detect_source_type("paper.pdf") == DataSourceType.PDF
        assert detect_source_type("report.docx") == DataSourceType.DOCX

    def test_unknown_extension_falls_back_to_txt(self):
        from brain_platform.pipeline.ingestion_orchestrator import detect_source_type
        from brain_platform.models.enums import DataSourceType

        assert detect_source_type("data.xyz") == DataSourceType.TXT
        assert detect_source_type("noextension") == DataSourceType.TXT

    def test_json_defaults_to_reddit(self):
        """Most .json in this context is a Reddit data export."""
        from brain_platform.pipeline.ingestion_orchestrator import detect_source_type
        from brain_platform.models.enums import DataSourceType

        assert detect_source_type("comments.json") == DataSourceType.REDDIT


class TestIngestionOrchestratorFileNotFound:
    def test_missing_file_raises(self):
        from brain_platform.pipeline.ingestion_orchestrator import IngestionOrchestrator

        orch = IngestionOrchestrator(store=MagicMock(), llm=MagicMock())
        with pytest.raises(FileNotFoundError, match="not found"):
            orch.ingest_file(
                file_path="/nonexistent/path.txt",
                group_id="test",
            )


class TestIngestionOrchestratorIngestFile:
    """The full parse → chunk → extract → write pipeline, mocked."""

    def test_runs_full_pipeline(self, tmp_path, capsys):
        """Happy path: parser produces chunks, extractor runs, writer persists."""
        from brain_platform.pipeline.ingestion_orchestrator import IngestionOrchestrator
        from brain_platform.pipeline.parsers.base import ParseResult
        from brain_platform.pipeline.brain_schema import PersonalityGraph
        from brain_platform.models.enums import DataSourceType

        # Create a real file to ingest
        test_file = tmp_path / "essay.md"
        test_file.write_text("# My Story\n\nI value autonomy and honesty.\n")

        # Mock the parser
        mock_parser = MagicMock()
        mock_parser.parse.return_value = [
            ParseResult(
                text="I value autonomy and honesty.",
                title="My Story",
                metadata={"source": "obsidian"},
            ),
        ]

        # Mock the chunker
        mock_chunker = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "I value autonomy and honesty."
        mock_chunker.chunk.return_value = [mock_chunk]

        # Mock the extractor
        mock_extractor = MagicMock()
        mock_graph = MagicMock(spec=PersonalityGraph)
        mock_extractor.extract.return_value = mock_graph

        # Mock the writer
        mock_writer = MagicMock()
        mock_writer.write.return_value = {"nodes_created": 5, "edges_created": 3}

        # Mock the store (used for client.llm_client)
        mock_store = MagicMock()
        mock_store.client.llm_client = MagicMock()

        with patch("brain_platform.pipeline.ingestion_orchestrator.get_parser", return_value=mock_parser), \
             patch("brain_platform.pipeline.ingestion_orchestrator.SemanticChunker", return_value=mock_chunker), \
             patch("brain_platform.services.local_graph_writer.LocalGraphWriter", return_value=mock_writer), \
             patch("brain_platform.extractor.brain_extractor.BrainExtractor", return_value=mock_extractor):
            orch = IngestionOrchestrator(store=mock_store, llm=MagicMock())
            result = orch.ingest_file(
                file_path=str(test_file),
                group_id="test_group",
            )

        assert result["documents"] == 1
        assert result["chunks"] == 1
        assert result["nodes_created"] == 5
        assert result["edges_created"] == 3
        assert result["source_type"] == "obsidian"  # .md → obsidian
        assert mock_extractor.extract.called
        assert mock_writer.write.called

    def test_explicit_type_override(self, tmp_path):
        """--type flag overrides auto-detection."""
        from brain_platform.pipeline.ingestion_orchestrator import IngestionOrchestrator
        from brain_platform.models.enums import DataSourceType

        test_file = tmp_path / "something.md"
        test_file.write_text("content")

        with patch("brain_platform.pipeline.ingestion_orchestrator.get_parser") as mock_get_parser:
            mock_parser = MagicMock()
            mock_parser.parse.return_value = []
            mock_get_parser.return_value = mock_parser

            with patch("brain_platform.pipeline.ingestion_orchestrator.SemanticChunker"):
                orch = IngestionOrchestrator(store=MagicMock(), llm=MagicMock())
                result = orch.ingest_file(
                    file_path=str(test_file),
                    group_id="test",
                    source_type=DataSourceType.CODE,  # Override
                )

        # Should have used CODE parser, not OBSIDIAN
        from brain_platform.models.enums import DataSourceType
        called_with = mock_get_parser.call_args[0][0]
        assert called_with == DataSourceType.CODE
        assert result["source_type"] == "code"

    def test_parser_fallback_to_txt_on_import_error(self, tmp_path):
        """If the chosen parser's dep is missing, fall back to TXT."""
        from brain_platform.pipeline.ingestion_orchestrator import IngestionOrchestrator
        from brain_platform.models.enums import DataSourceType

        test_file = tmp_path / "doc.pdf"
        test_file.write_text("text content")

        call_count = [0]

        def get_parser_side_effect(source_type):
            call_count[0] += 1
            if source_type == DataSourceType.PDF:
                raise ImportError("pymupdf not installed")
            # TXT parser works
            mock_parser = MagicMock()
            mock_parser.parse.return_value = []
            return mock_parser

        with patch("brain_platform.pipeline.ingestion_orchestrator.get_parser", side_effect=get_parser_side_effect), \
             patch("brain_platform.pipeline.ingestion_orchestrator.SemanticChunker"):
            orch = IngestionOrchestrator(store=MagicMock(), llm=MagicMock())
            result = orch.ingest_file(
                file_path=str(test_file),
                group_id="test",
            )

        # PDF parser was tried, then fell back to TXT
        assert call_count[0] == 2
        assert result["source_type"] == "txt"

    def test_handles_no_chunks(self, tmp_path):
        """If parsing produces no chunks, return zeros without crashing."""
        from brain_platform.pipeline.ingestion_orchestrator import IngestionOrchestrator

        test_file = tmp_path / "empty.md"
        test_file.write_text("")

        with patch("brain_platform.pipeline.ingestion_orchestrator.get_parser") as mock_get_parser, \
             patch("brain_platform.pipeline.ingestion_orchestrator.SemanticChunker") as mock_chunker_cls:
            mock_parser = MagicMock()
            mock_parser.parse.return_value = []  # No parse results
            mock_get_parser.return_value = mock_parser

            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = []
            mock_chunker_cls.return_value = mock_chunker

            orch = IngestionOrchestrator(store=MagicMock(), llm=MagicMock())
            result = orch.ingest_file(
                file_path=str(test_file),
                group_id="test",
            )

        assert result["chunks"] == 0
        assert result["nodes_created"] == 0
        assert result["edges_created"] == 0


class TestIngestionOrchestratorIngestText:
    """ingest_text — raw text without a file or parser."""

    def test_basic_text_ingestion(self):
        from brain_platform.pipeline.ingestion_orchestrator import IngestionOrchestrator

        with patch("brain_platform.pipeline.ingestion_orchestrator.SemanticChunker") as mock_chunker_cls, \
             patch("brain_platform.services.local_graph_writer.LocalGraphWriter") as mock_writer_cls, \
             patch("brain_platform.extractor.brain_extractor.BrainExtractor") as mock_extractor_cls:
            mock_chunker = MagicMock()
            mock_chunk = MagicMock()
            mock_chunk.text = "raw text"
            mock_chunker.chunk.return_value = [mock_chunk]
            mock_chunker_cls.return_value = mock_chunker

            mock_writer = MagicMock()
            mock_writer.write.return_value = {"nodes_created": 2, "edges_created": 1}
            mock_writer_cls.return_value = mock_writer

            mock_extractor = MagicMock()
            mock_extractor.extract.return_value = MagicMock()
            mock_extractor_cls.return_value = mock_extractor

            mock_store = MagicMock()
            mock_store.client.llm_client = MagicMock()

            orch = IngestionOrchestrator(store=mock_store, llm=MagicMock())
            result = orch.ingest_text(
                text="raw text content",
                group_id="test",
            )

        assert result["documents"] == 1
        assert result["chunks"] == 1
        assert result["nodes_created"] == 2
