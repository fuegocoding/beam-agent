"""End-to-end test for beam-agent brain flow.

Tests: interview → build brain → SOUL.md generation → brain_search

Uses mocked Rust subprocess calls since the actual binaries need
a full build environment. Tests the Python integration layer.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def beam_home(tmp_path):
    """Create a temporary BEAM_HOME directory."""
    beam_dir = tmp_path / ".beam"
    beam_dir.mkdir()
    (beam_dir / "brain" / "default").mkdir(parents=True)
    return beam_dir


@pytest.fixture
def hermes_home(tmp_path):
    """Create a temporary HERMES_HOME directory."""
    hermes_dir = tmp_path / ".hermes"
    hermes_dir.mkdir()
    return hermes_dir


@pytest.fixture
def sample_graph():
    """Sample personality graph for testing."""
    return {
        "user_id": "test_user",
        "traits": [
            {"name": "analytical", "strength": 0.8, "summary": "Tends to analyze problems systematically"},
            {"name": "creative", "strength": 0.6, "summary": "Enjoys exploring unconventional solutions"},
        ],
        "beliefs": [
            {"name": "ai_safety", "confidence": 0.9, "summary": "Believes AI safety research is critical"},
        ],
        "values": [
            {"name": "autonomy", "importance": 0.85, "summary": "Values independence and self-direction"},
        ],
        "boundaries": [
            {"topic": "politics", "comfort_level": 0.3, "summary": "Prefers not to discuss partisan politics"},
        ],
        "voice_dna": {
            "characteristic_phrases": ["let's think about this", "the key insight is"],
            "phrases_to_avoid": ["just my two cents", "I'm no expert but"],
            "humor_style": "dry wit, occasional self-deprecation",
            "response_length_pattern": "concise but thorough",
            "formality_range": "casual-professional, adapts to context",
        },
        "work_dna": {
            "decomposition_style": "top-down, identify core abstractions first",
            "debugging_approach": "reproduce, isolate, fix, verify",
            "risk_posture": "calculated — willing to try new approaches with rollback plans",
            "delegation_style": "clear boundaries, explicit success criteria",
        },
        "emotional_profile": {
            "triggers": [
                {"stimulus": "unclear requirements", "reaction": "frustration", "intensity": 0.6},
            ],
            "energy_sources": ["solving hard problems", "learning new things"],
            "energy_drains": ["repetitive tasks", "unclear communication"],
        },
    }


class TestBrainSchema:
    """Test the Python brain schema."""

    def test_brain_schema_import(self):
        from brain.brain_schema import PersonalityGraph
        assert PersonalityGraph is not None

    def test_graph_validation(self, sample_graph):
        from brain.brain_schema import PersonalityGraph
        graph = PersonalityGraph(**sample_graph)
        assert len(graph.traits) == 2
        assert graph.traits[0].name == "analytical"
        assert graph.beliefs[0].confidence == 0.9


class TestSubprocessBridge:
    """Test the Rust subprocess bridge."""

    def test_bridge_import(self):
        from brain.subprocess_bridge import call_rust_binary
        assert callable(call_rust_binary)


class TestBrainRetriever:
    """Test the brain retriever (search, context, export, stats)."""

    def test_search_with_mock(self, sample_graph):
        """Test brain_search with mocked Rust binary."""
        mock_result = {
            "nodes": [
                {"name": "analytical", "type": "trait", "relevance": 0.9},
            ],
            "edges": [],
            "context": "The user is highly analytical.",
        }

        with patch("brain.brain_retriever.call_rust_binary", return_value=mock_result):
            from brain.brain_retriever import BrainRetriever
            retriever = BrainRetriever()
            result = retriever.search("analytical thinking", sample_graph)

            assert "nodes" in result
            assert len(result["nodes"]) == 1
            assert result["nodes"][0]["name"] == "analytical"

    def test_stats_with_mock(self, sample_graph):
        """Test brain_stats with mocked Rust binary."""
        mock_result = {
            "total_nodes": 5,
            "total_edges": 3,
            "coverage": {"traits": 2, "beliefs": 1, "values": 1},
        }

        with patch("brain.brain_retriever.call_rust_binary", return_value=mock_result):
            from brain.brain_retriever import BrainRetriever
            retriever = BrainRetriever()
            result = retriever.get_stats(sample_graph)

            assert result["total_nodes"] == 5
            assert result["coverage"]["traits"] == 2


class TestSoulGenerator:
    """Test SOUL.md generation."""

    def test_generate_soul_md(self, sample_graph, tmp_path):
        """Test SOUL.md file generation."""
        mock_result = {
            "soul_md": """# SOUL.md

## Who I Am
An analytical and creative thinker who values autonomy.

## Core Traits
- Analytical (80%): Tends to analyze problems systematically
- Creative (60%): Enjoys exploring unconventional solutions

## Values
- Autonomy (85%): Values independence and self-direction

## Communication Style
- Characteristic phrases: "let's think about this", "the key insight is"
- Humor: dry wit, occasional self-deprecation
""",
        }

        output_path = tmp_path / "SOUL.md"

        with patch("brain.soul_generator.call_rust_binary", return_value=mock_result):
            from brain.soul_generator import generate_soul_md
            content = generate_soul_md(sample_graph, output_path)

            assert output_path.exists()
            assert "analytical" in content.lower() or "Analytical" in content
            assert "autonomy" in content.lower() or "Autonomy" in content


class TestMDMemory:
    """Test MD file memory system."""

    def test_write_and_read_episodic(self, beam_home):
        with patch("brain.md_memory.BEAM_HOME", beam_home):
            from brain.md_memory import MDMemory
            memory = MDMemory(user_id="default")

            path = memory.write_episodic(
                title="First Meeting",
                content="Had a great conversation about AI safety.",
                emotional_tone=0.8,
                tags=["ai", "safety"],
            )

            assert Path(path).exists()
            content = Path(path).read_text()
            assert "First Meeting" in content
            assert "AI safety" in content

    def test_write_and_read_semantic(self, beam_home):
        with patch("brain.md_memory.BEAM_HOME", beam_home):
            from brain.md_memory import MDMemory
            memory = MDMemory(user_id="default")

            path = memory.write_semantic(
                topic="AI Safety Views",
                content="Believes alignment research is the most important area.",
                confidence=0.9,
                source="interview",
            )

            assert Path(path).exists()
            content = Path(path).read_text()
            assert "AI Safety Views" in content

    def test_write_style(self, beam_home):
        with patch("brain.md_memory.BEAM_HOME", beam_home):
            from brain.md_memory import MDMemory
            memory = MDMemory(user_id="default")

            voice_dna = {
                "characteristic_phrases": ["let's think about this"],
                "phrases_to_avoid": ["just my two cents"],
                "humor_style": "dry wit",
                "response_length_pattern": "concise",
                "formality_range": "casual-professional",
                "storytelling_style": "structured with examples",
            }
            work_dna = {
                "decomposition_style": "top-down",
                "debugging_approach": "reproduce, isolate, fix",
                "risk_posture": "calculated",
                "delegation_style": "clear boundaries",
                "documentation_habit": "thorough for public APIs",
            }

            path = memory.write_style(voice_dna, work_dna)
            assert Path(path).exists()
            content = Path(path).read_text()
            assert "dry wit" in content
            assert "top-down" in content


class TestBrainTools:
    """Test brain tools (brain_search, brain_export, brain_status)."""

    def test_brain_search_no_graph(self, beam_home):
        with patch("tools.brain_tools.BEAM_HOME", beam_home):
            from tools.brain_tools import brain_search
            result = json.loads(brain_search("test query"))
            assert "error" in result

    def test_brain_status_no_graph(self, beam_home):
        with patch("tools.brain_tools.BEAM_HOME", beam_home):
            from tools.brain_tools import brain_status
            result = json.loads(brain_status())
            assert result["status"] == "empty"

    def test_brain_search_with_graph(self, beam_home, sample_graph):
        graph_path = beam_home / "brain" / "default" / "personality_graph.json"
        graph_path.write_text(json.dumps(sample_graph))

        mock_result = {
            "nodes": [{"name": "analytical", "type": "trait"}],
            "edges": [],
            "context": "User is analytical.",
        }

        with patch("tools.brain_tools.BEAM_HOME", beam_home), \
             patch("brain.brain_retriever.call_rust_binary", return_value=mock_result):
            from tools.brain_tools import brain_search
            result = json.loads(brain_search("analytical"))
            assert "nodes" in result

    def test_brain_status_with_graph(self, beam_home, sample_graph):
        graph_path = beam_home / "brain" / "default" / "personality_graph.json"
        graph_path.write_text(json.dumps(sample_graph))

        mock_result = {
            "total_nodes": 5,
            "total_edges": 3,
            "coverage": {"traits": 2, "beliefs": 1, "values": 1},
        }

        with patch("tools.brain_tools.BEAM_HOME", beam_home), \
             patch("brain.brain_retriever.call_rust_binary", return_value=mock_result):
            from tools.brain_tools import brain_status
            result = json.loads(brain_status())
            assert result["total_nodes"] == 5


class TestInterviewOrchestrator:
    """Test the interview orchestrator."""

    def test_start_interview(self):
        mock_result = {
            "question_id": "q1",
            "question": "Tell me about yourself — who are you at your core?",
            "domain": "identity",
            "pass": 1,
        }

        with patch("brain.interview_orchestrator.call_rust_binary", return_value=mock_result):
            from brain.interview_orchestrator import InterviewOrchestrator
            orchestrator = InterviewOrchestrator()
            result = orchestrator.start()

            assert result["question_id"] == "q1"
            assert result["domain"] == "identity"

    def test_continue_interview(self):
        mock_start = {
            "question_id": "q1",
            "question": "Tell me about yourself.",
            "domain": "identity",
            "pass": 1,
        }
        mock_continue = {
            "question_id": "q2",
            "question": "What drives you?",
            "domain": "identity",
            "pass": 1,
        }

        with patch("brain.interview_orchestrator.call_rust_binary", return_value=mock_continue):
            from brain.interview_orchestrator import InterviewOrchestrator
            orchestrator = InterviewOrchestrator()
            orchestrator.answers = [{"question_id": "q1", "question": "Tell me", "answer": "I am...", "domain": "identity"}]
            result = orchestrator.answer("q1", "Tell me", "I am...", "identity")

            assert result["question_id"] == "q2"

    def test_get_full_transcript(self):
        from brain.interview_orchestrator import InterviewOrchestrator
        orchestrator = InterviewOrchestrator()
        orchestrator.answers = [
            {"question_id": "q1", "question": "Who are you?", "answer": "I'm analytical.", "domain": "identity"},
            {"question_id": "q2", "question": "What drives you?", "answer": "Curiosity.", "domain": "identity"},
        ]

        transcript = orchestrator.get_full_transcript()
        assert len(transcript["answers"]) == 2
        assert "analytical" in transcript["transcript"]


class TestPluginRegistration:
    """Test that plugins register correctly."""

    def test_brain_tools_plugin_has_register(self):
        from plugins.brain_tools import register
        assert callable(register)

    def test_interview_plugin_has_register(self):
        from plugins.interview import register
        assert callable(register)


class TestEndToEndFlow:
    """Test the complete flow: interview → brain → SOUL.md → search."""

    def test_full_flow(self, beam_home, hermes_home, sample_graph):
        """Simulate the full brain-building flow."""
        mock_interview_start = {
            "question_id": "q1",
            "question": "Tell me about yourself.",
            "domain": "identity",
            "pass": 1,
        }
        mock_interview_complete = {
            "status": "complete",
            "summary": "Interview complete. 30 answers collected across 6 domains.",
        }
        mock_build_result = {"graph": sample_graph}
        mock_soul_result = {"soul_md": "# SOUL.md\n\nAnalytical thinker who values autonomy."}
        mock_search_result = {
            "nodes": [{"name": "analytical", "type": "trait", "relevance": 0.9}],
            "edges": [],
            "context": "User is highly analytical.",
        }

        with patch("brain.interview_orchestrator.call_rust_binary", return_value=mock_interview_start), \
             patch("brain.brain_builder.call_rust_binary", return_value=mock_build_result), \
             patch("brain.soul_generator.call_rust_binary", return_value=mock_soul_result), \
             patch("brain.brain_retriever.call_rust_binary", return_value=mock_search_result), \
             patch("tools.brain_tools.BEAM_HOME", beam_home), \
             patch("brain.soul_generator.BEAM_HOME", beam_home), \
             patch("brain.md_memory.BEAM_HOME", beam_home):

            # Step 1: Start interview
            from brain.interview_orchestrator import InterviewOrchestrator
            orchestrator = InterviewOrchestrator()
            start_result = orchestrator.start()
            assert start_result["question_id"] == "q1"

            # Step 2: Build brain (simulating interview completion)
            from brain.brain_builder import BrainBuilder
            builder = BrainBuilder()
            interview_data = orchestrator.get_full_transcript()
            build_result = builder.extract(interview_data)

            # Save graph
            graph_path = beam_home / "brain" / "default" / "personality_graph.json"
            graph_path.write_text(json.dumps(build_result.get("graph", {})))

            # Step 3: Generate SOUL.md
            from brain.soul_generator import generate_soul_md
            soul_path = hermes_home / "SOUL.md"
            soul_content = generate_soul_md(build_result.get("graph", {}), soul_path)
            assert soul_path.exists()
            assert "SOUL" in soul_content

            # Step 4: Search brain
            from tools.brain_tools import brain_search
            search_result = json.loads(brain_search("analytical thinking"))
            assert "nodes" in search_result
            assert len(search_result["nodes"]) > 0
