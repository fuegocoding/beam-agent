"""Tests for lifted brain_platform schemas and parsers.

These tests validate that the Tier 1 lift from cloud beam_mind produced
working Python — schemas instantiate, parsers parse, the question bank
loads, and the 4-rule termination logic is computable.

Chunk 1 deliverable: prove the lift is correct, no behavior change yet
(no LLM calls, no Neo4j, no async).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────
# brain_schema.py — 22 Pydantic node-type models
# ──────────────────────────────────────────────────────────────────────

class TestBrainSchema:
    """The 22 node-type Pydantic models instantiate and round-trip."""

    def test_personality_graph_minimal(self):
        from brain_platform.pipeline.brain_schema import PersonalityGraph

        g = PersonalityGraph(user_summary="A short test summary.")
        assert g.user_summary == "A short test summary."
        assert g.traits == []
        assert g.beliefs == []
        assert g.values == []
        # voice_dna defaults to None (Optional) — must be explicitly set.
        # The legacy flat schema in the offline `brain/brain_schema.py`
        # fills in defaults; the cloud's v2 schema treats nested models
        # as Optional to keep the wire format compact.
        assert g.voice_dna is None
        # voice_dna can be set explicitly and then accessed normally
        from brain_platform.pipeline.brain_schema import VoiceDNA
        g.voice_dna = VoiceDNA(characteristic_phrases=["let's think"])
        assert g.voice_dna.characteristic_phrases == ["let's think"]

    def test_personality_graph_full_fixture(self):
        """Build a populated PersonalityGraph from a fixture dict —
        mirrors the structure produced by the cloud's BrainExtractor."""
        from brain_platform.pipeline.brain_schema import (
            PersonalityGraph,
            TraitNode,
            BeliefNode,
            ValueNode,
        )

        g = PersonalityGraph(
            user_summary="Test user.",
            traits=[
                TraitNode(name="analytical", strength=0.8, summary="Thinks in systems."),
                TraitNode(name="creative", strength=0.6, summary="Explores ideas."),
            ],
            beliefs=[
                BeliefNode(name="ai_safety", confidence=0.9, summary="Critical."),
            ],
            values=[
                ValueNode(name="autonomy", importance=0.85, summary="Self-direction."),
            ],
        )
        assert len(g.traits) == 2
        assert g.traits[0].name == "analytical"
        assert g.traits[0].strength == 0.8
        assert len(g.beliefs) == 1
        assert g.beliefs[0].confidence == 0.9

    def test_string_coercion_via_model_validator(self):
        """The brain_schema uses model_validator to coerce string nodes
        into minimal valid dicts (legacy compatibility)."""
        from brain_platform.pipeline.brain_schema import PersonalityGraph

        g = PersonalityGraph.model_validate({
            "traits": ["curious", "kind"],
            "beliefs": ["honesty"],
        })
        assert len(g.traits) == 2
        assert g.traits[0].name == "curious"
        # Default strength of 0.5 applied by the model_validator
        assert g.traits[0].strength == 0.5


# ──────────────────────────────────────────────────────────────────────
# edges.py + edge_types.py + entity_types.py
# ──────────────────────────────────────────────────────────────────────

class TestEdgesAndEntityTypes:
    def test_edge_descriptor(self):
        from brain_platform.pipeline.edges import (
            EdgeDescriptor,
            EDGE_TYPES,
            EDGE_REGISTRY,
            render_edge_types_block,
        )

        # EDGE_TYPES is derived from the registry (frozen set of names)
        assert isinstance(EDGE_TYPES, frozenset)
        assert len(EDGE_TYPES) >= 20
        # The canonical 20+ edge types are exposed
        for name in ("ENFORCES", "SHAPED", "GUIDES", "INFORMS", "EXPERIENCED"):
            assert name in EDGE_TYPES, f"missing edge type {name}"

        # EDGE_REGISTRY holds the descriptors
        assert "ENFORCES" in EDGE_REGISTRY
        descriptor = EDGE_REGISTRY["ENFORCES"]
        assert isinstance(descriptor, EdgeDescriptor)
        # Descriptor has the documentation fields used by the LLM prompt
        assert descriptor.name == "ENFORCES"
        assert descriptor.source_target_hint
        assert descriptor.examples

    def test_edge_types_pydantic_models(self):
        """The 20 typed edge Pydantic models instantiate. They have
        edge-type-specific fields (causal_strength, held, etc.) rather
        than a generic source_name/target_name/fact — the generic edge
        spec is EdgeSpec in brain_schema.py."""
        from brain_platform.pipeline.edge_types import (
            ShapedBy, EnforcedAs, Guides, LearnedFrom,
            ExpressedThrough, TestedBy, Involves,
        )

        # Spot-check a few — ShapedBy and TestedBy have specific fields
        sb = ShapedBy(
            causal_strength=0.8,
            lived_experience="",
            reinforcement_pattern="",
        )
        assert sb.causal_strength == 0.8

        tb = TestedBy(held=True)
        assert tb.held is True

        # Others are pass-through dataclasses
        for cls in [EnforcedAs, Guides, LearnedFrom, ExpressedThrough, Involves]:
            inst = cls()
            assert inst is not None

    def test_entity_types_pydantic_models(self):
        """The 14 typed entity Pydantic models instantiate with their
        canonical field names (trait_name, topic, event, etc. — not the
        generic legacy 'name' field)."""
        from brain_platform.pipeline.entity_types import (
            PersonalityTrait, Belief, EpisodicMemory, StyleProfile,
            Value, Boundary, CognitivePattern, SocialPattern,
            KnowledgeDomain, LifeEvent, ProceduralPattern,
            WorkLoop, PromptingStyle, TechnicalGap,
        )

        # Spot-check each category
        t = PersonalityTrait(
            trait_name="analytical",
            strength=0.8,
            evidence_count=3,
        )
        b = Belief(topic="ai safety", position="critical", confidence=0.9)
        em = EpisodicMemory(event="first day at work", emotional_valence=0.5, salience=0.7)
        s = StyleProfile(formality=0.4, vocabulary_level="advanced")
        v = Value(value_name="autonomy", importance=0.85)
        bd = Boundary(description="won't compromise on honesty", tested=True)

        assert t.trait_name == "analytical"
        assert t.strength == 0.8
        assert b.confidence == 0.9
        assert em.emotional_valence == 0.5
        assert s.formality == 0.4
        assert v.importance == 0.85
        assert bd.tested is True

    def test_entity_type_registry(self):
        """BEAM_MIND_ENTITY_TYPES is a dict mapping canonical name -> Pydantic class."""
        from brain_platform.pipeline.entity_types import BEAM_MIND_ENTITY_TYPES

        # Registry is non-empty and includes the canonical types
        assert len(BEAM_MIND_ENTITY_TYPES) >= 10
        assert "PersonalityTrait" in BEAM_MIND_ENTITY_TYPES
        assert "Belief" in BEAM_MIND_ENTITY_TYPES
        assert "Value" in BEAM_MIND_ENTITY_TYPES

    def test_render_edge_types_block(self):
        """The edge-type rendering produces a prompt-ready block listing
        all edges with examples. Used by the LLM extraction prompts."""
        from brain_platform.pipeline.edges import render_edge_types_block

        block = render_edge_types_block()
        assert "ENFORCES" in block
        assert "SHAPED" in block
        assert isinstance(block, str)
        # Without procedural edges: still works
        no_proc = render_edge_types_block(include_procedural=False)
        assert "ENFORCES" in no_proc


# ──────────────────────────────────────────────────────────────────────
# brain_file/schema.py — v2.2.0 BrainFileSchema
# ──────────────────────────────────────────────────────────────────────

class TestBrainFileSchema:
    def test_minimal_brain_file(self):
        from brain_platform.pipeline.brain_file.schema import BrainFileSchema

        bf = BrainFileSchema(
            metadata={
                "version": "2.2.0",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "user_id": "test-user",
            },
        )
        assert bf.metadata.version == "2.2.0"
        assert bf.metadata.user_id == "test-user"
        assert bf.personality_profile.communication_style == "analytical"  # default
        # All the v2.2.0 blocks exist with defaults
        assert bf.knowledge_graph is not None
        assert bf.voice_dna is not None
        assert bf.behavioral_rules == []
        assert bf.contradiction_patterns == []
        assert bf.emotional_triggers == []
        assert bf.contextual_moods == []
        assert bf.procedural_patterns == []
        assert bf.work_loops == []
        assert bf.prompting_styles == []
        assert bf.technical_gaps == []

    def test_full_brain_file_round_trip(self):
        """A BrainFile with all v2.2.0 blocks serializes and deserializes."""
        from brain_platform.pipeline.brain_file.schema import (
            BrainFileSchema,
            PersonalityProfile,
            VoiceDNAEntry,
            KnowledgeDomain,
            GraphNode,
        )

        bf = BrainFileSchema(
            metadata={
                "version": "2.2.0",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "user_id": "test-user",
            },
            personality_profile=PersonalityProfile(
                communication_style="direct",
                formality=0.7,
                humor_frequency=0.2,
                values=["Honesty (0.99): tell the truth"],
                core_beliefs=["Free will exists"],
            ),
            voice_dna=VoiceDNAEntry(
                characteristic_phrases=["let's think about this"],
                humor_style="dry",
            ),
            knowledge_graph={
                "nodes": [
                    GraphNode(
                        id="n1",
                        type="PersonalityTrait",
                        label="analytical",
                        summary="Thinks in systems.",
                    ).model_dump(),
                ],
                "edges": [],
            },
            knowledge_domains=[
                KnowledgeDomain(
                    topic="Engineering",
                    confidence=0.9,
                    key_entities=["systems", "patterns"],
                    community_summary="Deep focus on systems thinking.",
                ),
            ],
        )
        # Round-trip via JSON
        d = bf.model_dump()
        j = json.dumps(d, default=str)
        restored = BrainFileSchema.model_validate_json(j)
        assert restored.metadata.user_id == "test-user"
        assert restored.personality_profile.formality == 0.7
        assert restored.voice_dna.humor_style == "dry"
        # KnowledgeGraph is a Pydantic model — access via .nodes
        assert len(restored.knowledge_graph.nodes) == 1


# ──────────────────────────────────────────────────────────────────────
# interview/questions.py — question bank
# ──────────────────────────────────────────────────────────────────────

class TestInterviewQuestions:
    def test_question_bank_loads(self):
        from brain_platform.pipeline.interview.questions import (
            INTERVIEW_QUESTIONS,
            TIER_STANDARD,
            TIER_YOUNG,
            TIER_EMERGING,
            age_to_tier,
            get_question_for_tier,
            get_min_words_for_tier,
        )

        # Question bank has 19 questions (cloud's core set)
        assert len(INTERVIEW_QUESTIONS) >= 15
        # All questions have dimension + purpose + follow_up_hint
        # (order is optional — the AGE_QUESTION has order=0, the rest >= 1)
        for q in INTERVIEW_QUESTIONS:
            assert q.id
            assert q.dimension
            assert q.question
            assert q.purpose
            assert q.follow_up_hint
            assert q.order >= 0

    def test_age_tiers(self):
        from brain_platform.pipeline.interview.questions import age_to_tier

        assert age_to_tier(15) == "young"
        assert age_to_tier(20) == "emerging"
        assert age_to_tier(40) == "standard"

    def test_age_adaptive_question_selection(self):
        """A question with a young_question variant returns the
        young-appropriate text for tier=young."""
        from brain_platform.pipeline.interview.questions import (
            INTERVIEW_QUESTIONS,
            TIER_YOUNG,
            TIER_STANDARD,
            get_question_for_tier,
        )

        # Find a question with a young variant
        with_variant = next(
            (q for q in INTERVIEW_QUESTIONS if q.young_question is not None),
            None,
        )
        if with_variant is None:
            pytest.skip("no questions with age variants in this bank")
        # Young tier returns the young variant
        young_text = get_question_for_tier(with_variant, TIER_YOUNG)
        assert young_text == with_variant.young_question
        # Standard tier returns the default
        std_text = get_question_for_tier(with_variant, TIER_STANDARD)
        assert std_text == with_variant.question

    def test_min_words_scales_with_tier(self):
        """Younger users have a lower word-count floor for depth."""
        from brain_platform.pipeline.interview.questions import (
            INTERVIEW_QUESTIONS,
            get_min_words_for_tier,
            TIER_YOUNG,
            TIER_EMERGING,
            TIER_STANDARD,
        )

        q = INTERVIEW_QUESTIONS[0]
        assert q.min_words_for_depth > 0
        young_min = get_min_words_for_tier(q, TIER_YOUNG)
        emerging_min = get_min_words_for_tier(q, TIER_EMERGING)
        standard_min = get_min_words_for_tier(q, TIER_STANDARD)
        assert young_min <= emerging_min <= standard_min


# ──────────────────────────────────────────────────────────────────────
# interview/gap_identifier.py — 4 termination rules
# ──────────────────────────────────────────────────────────────────────

class TestGapIdentifier:
    def test_gap_analysis_minimal(self):
        from brain_platform.pipeline.interview.gap_identifier import (
            GapIdentifier, GapAnalysis,
        )
        from brain_platform.pipeline.interview.coverage_types import DimensionScore

        gi = GapIdentifier()
        # CRITICAL_GAP_THRESHOLD = 0.30 (strict <) — use 0.2 to land in critical
        coverage = {
            "episodic_memory": DimensionScore("episodic_memory", 0.5, 2, 1, 50.0, 1.0),
            "core_beliefs": DimensionScore("core_beliefs", 0.7, 3, 2, 60.0, 1.0),
            "values": DimensionScore("values", 0.2, 1, 0, 30.0, 1.0),  # < 0.30 → critical
        }
        result = gi.analyze(coverage, questions_asked=5)
        assert isinstance(result, GapAnalysis)
        # 1 critical (values=0.2), 1 moderate (episodic_memory), 1 adequate
        assert "values" in result.critical_gaps
        assert "episodic_memory" in result.moderate_gaps
        assert "core_beliefs" in result.adequate_dimensions
        # Below min questions, should not terminate
        assert result.should_terminate is False

    def test_terminates_at_min_questions_when_all_adequate(self):
        from brain_platform.pipeline.interview.gap_identifier import GapIdentifier
        from brain_platform.pipeline.interview.coverage_types import DimensionScore

        gi = GapIdentifier()
        # All dimensions adequate, questions_asked >= MIN_QUESTIONS (8)
        coverage = {
            f"dim_{i}": DimensionScore(f"dim_{i}", 0.8, 3, 2, 50.0, 1.0)
            for i in range(5)
        }
        result = gi.analyze(coverage, questions_asked=10)
        assert result.should_terminate is True
        assert result.termination_reason == "all_dimensions_adequate"

    def test_terminates_at_max_questions(self):
        from brain_platform.pipeline.interview.gap_identifier import GapIdentifier
        from brain_platform.pipeline.interview.coverage_types import DimensionScore

        gi = GapIdentifier()
        coverage = {
            "dim_a": DimensionScore("dim_a", 0.2, 1, 0, 20.0, 1.0),  # critical
        }
        result = gi.analyze(coverage, questions_asked=25)  # > MAX_QUESTIONS=19
        assert result.should_terminate is True
        assert "max_questions_reached" in result.termination_reason

    def test_get_target_dimensions(self):
        """get_target_dimensions returns critical gaps first, then moderate."""
        from brain_platform.pipeline.interview.gap_identifier import GapIdentifier
        from brain_platform.pipeline.interview.coverage_types import DimensionScore

        gi = GapIdentifier()
        coverage = {
            "episodic_memory": DimensionScore("episodic_memory", 0.5, 2, 1, 50.0, 1.0),
            "core_beliefs": DimensionScore("core_beliefs", 0.2, 1, 0, 20.0, 1.0),
            "values": DimensionScore("values", 0.7, 3, 2, 60.0, 1.0),
        }
        analysis = gi.analyze(coverage, questions_asked=5)
        targets = gi.get_target_dimensions(analysis)
        # core_beliefs (critical) should be ahead of episodic_memory (moderate)
        assert targets[0] == "core_beliefs"
        assert "episodic_memory" in targets


# ──────────────────────────────────────────────────────────────────────
# parsers/* — file parsers
# ──────────────────────────────────────────────────────────────────────

class TestParsers:
    def test_parser_map_has_all_stdlib_parsers(self):
        from brain_platform.pipeline.parsers import PARSER_MAP, get_parser
        from brain_platform.models.enums import DataSourceType

        # Pure-stdlib parsers (no third-party deps) are always present
        for st in (
            DataSourceType.OBSIDIAN,
            DataSourceType.TXT,
            DataSourceType.CODE,
            DataSourceType.EMAIL,
            DataSourceType.PROMPT,
            DataSourceType.INSTRUCTIONS,
            DataSourceType.JOURNAL,
            DataSourceType.REDDIT,
        ):
            assert st in PARSER_MAP, f"missing parser for {st.value}"

    def test_get_parser_raises_for_unknown_type(self):
        from brain_platform.pipeline.parsers import get_parser
        from brain_platform.models.enums import DataSourceType

        with pytest.raises(ValueError, match="No parser"):
            get_parser(DataSourceType.AI_MEMORY)

    def test_txt_parser_round_trip(self):
        """TxtParser is the simplest parser — round-trip bytes."""
        from brain_platform.pipeline.parsers import get_parser
        from brain_platform.models.enums import DataSourceType

        text = "Hello, this is a test note with some words to chunk.\n\n" * 5
        parser = get_parser(DataSourceType.TXT)
        results = parser.parse(text.encode("utf-8"), "note.txt")
        assert len(results) == 1
        assert "Hello" in results[0].text
        assert results[0].title == "note"
        assert results[0].metadata.get("source_file") == "note.txt"

    def test_parse_result_dataclass(self):
        """ParseResult exposes text, title, metadata, created_at, links."""
        from brain_platform.pipeline.parsers.base import ParseResult

        r = ParseResult(text="hello", title="note", metadata={"k": "v"})
        assert r.text == "hello"
        assert r.title == "note"
        assert r.metadata == {"k": "v"}
        assert r.links == []


# ──────────────────────────────────────────────────────────────────────
# chunker.py — SemanticChunker
# ──────────────────────────────────────────────────────────────────────

class TestChunker:
    def _make_parse_result(self, text: str):
        from brain_platform.pipeline.parsers.base import ParseResult
        return ParseResult(text=text, title="test", metadata={})

    def test_chunker_short_input(self):
        from brain_platform.pipeline.chunker import SemanticChunker

        text = "This is a short piece of text."
        chunks = SemanticChunker(max_chars=6000).chunk(self._make_parse_result(text))
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].index == 0

    def test_chunker_long_input_splits(self):
        from brain_platform.pipeline.chunker import SemanticChunker

        text = ("First paragraph.\n\n" * 30) + ("Second paragraph.\n\n" * 30)
        chunks = SemanticChunker(max_chars=500).chunk(self._make_parse_result(text))
        # Long input should produce multiple chunks
        assert len(chunks) > 1
        # Each chunk has the expected shape
        for i, c in enumerate(chunks):
            assert c.index == i
            assert c.text
            assert c.metadata  # carries ParseResult metadata through

    def test_chunk_dataclass(self):
        """Chunk holds text, index, metadata."""
        from brain_platform.pipeline.chunker import Chunk

        c = Chunk(text="hello", index=0, metadata={"src": "test"})
        assert c.text == "hello"
        assert c.index == 0
        assert c.metadata == {"src": "test"}


# ──────────────────────────────────────────────────────────────────────
# artifacts/exclusions.py + secret_scanner.py
# ──────────────────────────────────────────────────────────────────────

class TestArtifacts:
    def test_secret_scanner_detects_openai_key(self):
        from brain_platform.pipeline.artifacts import secret_scanner

        text = 'api_key = "sk-abcdefghijklmnopqrstuvwxyz1234567890"'
        # The function returns findings (a list of strings naming the matches).
        findings = secret_scanner.scan(text)
        assert isinstance(findings, list)
        assert len(findings) > 0  # at least one secret was found

    def test_secret_scanner_clean_text(self):
        from brain_platform.pipeline.artifacts import secret_scanner

        # is_clean returns True when no secrets are present
        assert secret_scanner.is_clean("This is just a normal sentence about coding.")
        assert secret_scanner.is_clean("def hello(): return 42")
        # is_clean returns False when a secret is present
        assert not secret_scanner.is_clean(
            'api_key = "sk-abcdefghijklmnopqrstuvwxyz1234567890"'
        )

    def test_secret_scanner_handles_none(self):
        """scan() and is_clean() must accept None without crashing."""
        from brain_platform.pipeline.artifacts import secret_scanner

        assert secret_scanner.scan(None) == []
        assert secret_scanner.is_clean(None) is True

    def test_exclusions_looks_like_code(self):
        from brain_platform.pipeline.artifacts.exclusions import looks_like_code

        # Code-shaped text is detected
        assert looks_like_code("def foo(x):\n    return x * 2")
        # Prose is not
        assert not looks_like_code("This is a normal sentence about coding.")
        # None is not code
        assert not looks_like_code(None)

    def test_exclusions_verify_features_clean(self):
        from brain_platform.pipeline.artifacts.exclusions import verify_features_clean

        # Clean feature dict
        assert verify_features_clean({"name": "x", "count": 5, "flag": True}) == []
        # Raw text leaking into a feature
        violations = verify_features_clean({"name": "x" * 200})
        assert len(violations) > 0


# ──────────────────────────────────────────────────────────────────────
# models/enums.py stub
# ──────────────────────────────────────────────────────────────────────

class TestModelsEnums:
    def test_data_source_type_values(self):
        from brain_platform.models.enums import DataSourceType

        # The 13 source types the cloud supports
        assert DataSourceType.OBSIDIAN.value == "obsidian"
        assert DataSourceType.PDF.value == "pdf"
        assert DataSourceType.DOCX.value == "docx"
        assert DataSourceType.JOURNAL.value == "journal"
        # Used as dict keys
        d = {DataSourceType.PDF: "pdf"}
        assert d[DataSourceType.PDF] == "pdf"

    def test_data_source_type_total(self):
        """The stub has all 13 source types the lifted parsers need."""
        from brain_platform.models.enums import DataSourceType

        expected = {
            "obsidian", "pdf", "docx", "txt", "audio", "tweet",
            "email", "ai_memory", "reddit", "journal", "code",
            "prompt", "instructions",
        }
        actual = {st.value for st in DataSourceType}
        assert expected == actual
