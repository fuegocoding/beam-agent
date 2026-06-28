"""End-to-end test for beam-agent brain flow.

Tests: install → load brain → search → export → SOUL.md.

The brain subsystem is fully offline (no LLM, no network). Brains are
shipped as personality_graph.json files and searched in-process.
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
    (beam_dir / "brains" / "default").mkdir(parents=True)
    (beam_dir / "brains" / "default" / "memory" / "episodic").mkdir(parents=True)
    (beam_dir / "brains" / "default" / "memory" / "semantic").mkdir(parents=True)
    (beam_dir / "brains" / "default" / "memory" / "procedural").mkdir(parents=True)
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
            {"name": "politics", "comfort_level": 0.3, "summary": "Prefers not to discuss partisan politics"},
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

    def test_search(self, sample_graph):
        """BrainRetriever.search hits on trait names from the sample graph."""
        from brain.brain_retriever import BrainRetriever
        retriever = BrainRetriever()
        result = retriever.search("analytical thinking", sample_graph)

        assert "nodes" in result
        assert result["total_matches"] >= 1
        names = [n["name"] for n in result["nodes"]]
        assert "analytical" in names

    def test_search_memories(self, sample_graph):
        """BrainRetriever.search now also indexes memories."""
        from brain.brain_retriever import BrainRetriever
        sample_graph["memories"] = [
            {"name": "interview-chunk-1", "summary": "I love hiking in the mountains on weekends."},
        ]
        retriever = BrainRetriever()
        result = retriever.search("hiking", sample_graph)

        types = {n["type"] for n in result["nodes"]}
        assert "memory" in types

    def test_search_raw_transcript(self, sample_graph):
        """BrainRetriever.search surfaces a transcript excerpt for raw queries."""
        from brain.brain_retriever import BrainRetriever
        sample_graph["raw_transcript"] = (
            "Interviewer: Tell me about your childhood.\n"
            "Subject: I grew up in a small coastal town in Maine, near Acadia National Park."
        )
        retriever = BrainRetriever()
        result = retriever.search("Maine", sample_graph)

        types = {n["type"] for n in result["nodes"]}
        assert "transcript_excerpt" in types

    def test_stats(self, sample_graph):
        """BrainRetriever.get_stats includes memories and raw_transcript."""
        from brain.brain_retriever import BrainRetriever
        retriever = BrainRetriever()
        stats = retriever.get_stats(sample_graph)
        assert stats["user_summary"] is False
        assert stats["coverage"]["traits"] == 2
        assert stats["coverage"]["values"] == 1
        assert stats["coverage"]["memories"] == 0
        assert "raw_transcript" in stats["coverage"]


class TestMarketplaceSchemaBrain:
    """Regression tests for marketplace-schema brains.

    api.openbeam.me ships brains in a richer schema
    (``personality_profile`` / ``knowledge_graph`` /
    ``knowledge_domains`` / ``episodic_memories``) than the legacy
    flat schema the in-tree brain_builder produces. Before
    :mod:`brain.schema_adapter` was introduced, the retriever and
    SOUL.md generator only read the legacy flat keys, so a downloaded
    marketplace brain came back as a 582-byte empty persona with zero
    searchable nodes — the agent had no personality to draw on.

    These tests assert the behavior contract, not specific counts:
    any marketplace brain (regardless of which historical figure /
    community author built it) must produce a non-empty search
    result for a relevant query, populate ``coverage`` with the
    marketplace keys, and render a SOUL.md that mentions the
    personality content.
    """

    @pytest.fixture
    def marketplace_graph(self):
        """Minimal marketplace-schema graph (the only fields the
        schema adapter actually needs). Avoids the full 210 KB
        Seneca fixture so the test stays fast and the assertions
        don't depend on which brains happen to be in the catalog
        today (change-detector trap)."""
        return {
            "personality_profile": {
                "values": [
                    "Temperance (0.95): Moderation in all things is the path to clarity.",
                    "Courage (0.90): Face what must be faced, even when afraid.",
                ],
                "core_beliefs": [
                    "Discipline is freedom. (confidence: 0.97) [Letters]",
                    "Time is the only true wealth. (confidence: 0.92)",
                ],
                "cognitive_patterns": [
                    "First principles: strip a problem to its base before solving.",
                ],
                "communication_style": "Direct, plainspoken, occasionally pointed.",
                "formality": "Low — talks like a craftsman, not a lecturer.",
                "humor_frequency": "Rare but dry.",
            },
            "knowledge_graph": {
                "nodes": [
                    {
                        "id": "n1",
                        "type": "PersonalityTrait",
                        "label": "Resilient",
                        "summary": "Endures hardship without losing focus on the goal.",
                        "attributes": {"strength": 0.8},
                    },
                    {
                        "id": "n2",
                        "type": "PersonalityTrait",
                        "label": "Pragmatic",
                        "summary": "Picks the working solution over the elegant one.",
                        "attributes": {"strength": 0.7},
                    },
                    {
                        "id": "n3",
                        "type": "PersonalityTrait",
                        "label": "Stoic",
                        "summary": "Separates what can be controlled from what cannot.",
                        "attributes": {"strength": 0.6},
                    },
                ],
                "edges": [
                    {"id": "e1", "source": "n1", "target": "n2", "relation": "COMPLEMENTS"},
                ],
            },
            "knowledge_domains": [
                {
                    "topic": "Craftsmanship",
                    "community_summary": "Deep focus on doing the work well, with patience and care.",
                    "key_entities": ["practice", "patience", "tooling"],
                    "confidence": 0.9,
                    "source_count": 7,
                },
            ],
            "episodic_memories": [
                {
                    "name": "The forge",
                    "content": "Spent the night at the forge, hammering a blade that refused to straighten.",
                    "emotional_tone": 0.4,
                },
            ],
            "voice_dna": {
                "humor_style": "dry, observational",
                "response_length_pattern": "concise by default",
                "characteristic_phrases": ["the work speaks"],
            },
        }

    def test_search_hits_personality_profile_value(self, marketplace_graph):
        """A query matching a ``personality_profile.values`` entry
        must return a hit — before the schema adapter this returned
        0 even for an obvious match like 'temperance'."""
        from brain.brain_retriever import BrainRetriever
        retriever = BrainRetriever()
        result = retriever.search("temperance", marketplace_graph, "owner", "standard")
        assert result["total_matches"] >= 1
        types = {n.get("type") for n in result["nodes"]}
        assert "value" in types

    def test_search_hits_knowledge_graph_node(self, marketplace_graph):
        """A query matching a ``knowledge_graph.nodes[*].summary`` must
        return a hit. Before the schema adapter the retriever only
        iterated the legacy flat keys and missed this entirely."""
        from brain.brain_retriever import BrainRetriever
        retriever = BrainRetriever()
        result = retriever.search("endures hardship", marketplace_graph, "owner", "standard")
        assert result["total_matches"] >= 1
        # The fixture node has type=PersonalityTrait which the adapter
        # canonicalizes to 'trait' so it stays consistent with the
        # legacy schema's node type vocabulary.
        assert any(
            "Resilient" in n.get("name", "") for n in result["nodes"]
        )

    def test_search_extracts_embedded_score(self, marketplace_graph):
        """Marketplace values like ``'Temperance (0.95): ...'`` embed
        the numeric importance in the name string. The adapter must
        parse it out so the importance field is 0.95, not the 0.5
        default."""
        from brain.brain_retriever import BrainRetriever
        retriever = BrainRetriever()
        result = retriever.search("temperance", marketplace_graph, "owner", "standard")
        matches = [n for n in result["nodes"] if "Temperance" in n.get("name", "")]
        assert matches, "expected a Temperance value match"
        # Score extraction: importance 0.95 should propagate to the
        # node, not stay at the 0.5 default.
        assert any(
            abs(m.get("importance", 0) - 0.95) < 0.01 for m in matches
        ), f"expected importance 0.95, got nodes: {matches}"

    def test_stats_includes_marketplace_keys(self, marketplace_graph):
        """``get_stats`` must surface the marketplace-schema node
        counts via the ``knowledge_graph_nodes``, ``knowledge_domains``,
        and ``episodic_memories`` coverage keys — without these, the
        CLI's `brain status` / `brain info` outputs report 0 for any
        downloaded brain."""
        from brain.brain_retriever import BrainRetriever
        retriever = BrainRetriever()
        stats = retriever.get_stats(marketplace_graph)
        coverage = stats["coverage"]
        # Invariant: marketplace brains carry knowledge_graph + domains
        # + episodic memories; the coverage dict must reflect that
        # (previously these fields were always 0).
        assert coverage["knowledge_graph_nodes"] >= 3
        assert coverage["knowledge_domains"] >= 1
        assert coverage["episodic_memories"] >= 1
        # Legacy keys can legitimately be 0 for marketplace brains —
        # don't assert non-zero (would be a change-detector).
        assert "traits" in coverage
        assert "knowledge_graph_nodes" in coverage

    def test_stats_includes_total_edges(self, marketplace_graph):
        """``get_stats`` must report ``total_edges`` from both
        ``knowledge_graph.edges`` and the legacy top-level
        ``edges`` field. Previously always 0."""
        from brain.brain_retriever import BrainRetriever
        retriever = BrainRetriever()
        stats = retriever.get_stats(marketplace_graph)
        assert stats["total_edges"] >= 1

    def test_soul_md_contains_personality_content(self, marketplace_graph):
        """The template-only SOUL.md generator must produce content
        drawn from the marketplace schema. Before the fix it emitted
        a 582-byte stub with only the voice_dna block populated,
        even when the graph carried 50+ nodes."""
        from brain.soul_generator import _template_soul
        soul = _template_soul(marketplace_graph)
        # Invariant: a marketplace brain with personality_profile +
        # knowledge_graph + knowledge_domains + voice_dna must
        # produce a SOUL.md larger than the 582-byte legacy-only
        # ceiling (which is what `_template_soul` produced when it
        # only knew about the legacy flat schema — see PR
        # "brains arent properly injected into the beam agent").
        LEGACY_ONLY_STUB_CEILING = 600
        assert len(soul) > LEGACY_ONLY_STUB_CEILING, (
            f"SOUL.md too small ({len(soul)} chars); legacy-only stub "
            f"(<= {LEGACY_ONLY_STUB_CEILING} chars) suggests the "
            f"schema adapter isn't wired up."
        )
        # The marketplace value "Temperance" must appear in the
        # rendered SOUL.md so the model can use it.
        assert "Temperance" in soul
        # The knowledge domain "Craftsmanship" must surface — it's
        # exactly the kind of high-signal data the marketplace adds.
        assert "Craftsmanship" in soul
        # The forge episodic memory should also show up in
        # the "Memorable Context" section.
        assert "Memorable Context" in soul
        assert "forge" in soul.lower()

    def test_schema_adapter_legacy_still_works(self, sample_graph):
        """Adding the marketplace schema path must not break the
        legacy flat schema. The retriever's node count, type set, and
        relevance ranking for a query like 'analytical' must remain
        stable."""
        from brain.brain_retriever import BrainRetriever
        retriever = BrainRetriever()
        result = retriever.search("analytical", sample_graph, "owner", "standard")
        # Legacy invariant: the analytical trait is the top hit.
        assert result["nodes"][0]["name"] == "analytical"
        assert result["nodes"][0]["type"] == "trait"
        assert result["nodes"][0]["relevance"] > 0


class TestBrainBuilder:
    """Test offline transcript → graph conversion."""

    def test_extract_returns_minimal_graph(self):
        from brain.brain_builder import BrainBuilder
        builder = BrainBuilder()
        interview = {
            "answers": [
                {"question_id": "i_1", "question": "Who are you?", "answer": "I am a builder.",
                 "domain": "identity"},
            ]
        }
        result = builder.extract(interview)
        assert "graph" in result
        graph = result["graph"]
        # Memories are populated from the transcript.
        assert any(m.get("name", "").startswith("interview-") for m in graph["memories"])
        # Raw transcript is preserved for the retriever to index.
        assert "builder" in graph["raw_transcript"]
        # A user_summary was derived locally.
        assert graph["user_summary"]

    def test_merge_dedupes_by_name(self, sample_graph):
        from brain.brain_builder import BrainBuilder
        builder = BrainBuilder()
        new_graph = {
            "traits": [
                {"name": "analytical", "strength": 0.9, "summary": "duplicate"},
                {"name": "curious", "strength": 0.7, "summary": "new trait"},
            ],
        }
        merged = builder.merge(sample_graph, new_graph)
        # Existing trait was kept (not overwritten by the dup).
        assert merged["traits"][0]["name"] == "analytical"
        # New trait was appended.
        names = {t["name"] for t in merged["traits"]}
        assert "curious" in names

    def test_validate_flags_empty(self):
        from brain.brain_builder import BrainBuilder
        builder = BrainBuilder()
        result = builder.validate({})
        assert result["valid"] is False
        assert "Missing user_summary" in result["issues"]


class TestSoulGenerator:
    """Test offline template-based SOUL.md generation."""

    def test_generate_soul_md(self, sample_graph, tmp_path):
        from brain.soul_generator import generate_soul_md
        output_path = tmp_path / "SOUL.md"

        content = generate_soul_md(sample_graph, output_path)

        assert output_path.exists()
        assert "analytical" in content.lower() or "Analytical" in content
        assert "autonomy" in content.lower() or "Autonomy" in content

    def test_generate_soul_md_handles_empty_graph(self, tmp_path):
        from brain.soul_generator import generate_soul_md
        output_path = tmp_path / "SOUL.md"
        content = generate_soul_md({}, output_path)
        assert output_path.exists()
        assert content.startswith("# Soul")


class TestInterviewOrchestrator:
    """Test the offline scripted interview orchestrator."""

    def test_start_interview(self):
        from brain.interview_orchestrator import InterviewOrchestrator
        orchestrator = InterviewOrchestrator()
        result = orchestrator.start()

        assert result["question_id"] == "identity_1"
        assert result["domain"] == "identity"

    def test_continue_interview(self):
        from brain.interview_orchestrator import InterviewOrchestrator
        orchestrator = InterviewOrchestrator()
        orchestrator.answers = [
            {"question_id": "q1", "question": "Tell me", "answer": "I am...",
             "domain": "identity"}
        ]
        # Force the scripted progression to advance by claiming 3 questions
        # in the identity domain already.
        orchestrator.questions_asked["identity"] = 3
        result = orchestrator.answer("q1", "Tell me", "I am...", "identity")
        assert "question_id" in result
        # The scripted progression moved to the next domain.
        assert result["domain"] != "identity"

    def test_get_full_transcript(self):
        from brain.interview_orchestrator import InterviewOrchestrator
        orchestrator = InterviewOrchestrator()
        orchestrator.answers = [
            {"question_id": "q1", "question": "Who are you?",
             "answer": "I'm analytical.", "domain": "identity"},
            {"question_id": "q2", "question": "What drives you?",
             "answer": "Curiosity.", "domain": "identity"},
        ]

        transcript = orchestrator.get_full_transcript()
        assert len(transcript["answers"]) == 2
        assert "analytical" in transcript["transcript"]


class TestBrainResolver:
    """Test the brain resolver abstraction (local-only)."""

    def test_resolve_returns_local_brain(self, beam_home, sample_graph):
        graph_path = beam_home / "brains" / "default" / "personality_graph.json"
        graph_path.write_text(json.dumps(sample_graph))

        with patch("brain.brain_resolver.BEAM_HOME", beam_home), \
             patch("brain.paths.BEAM_HOME", beam_home):
            from brain.brain_resolver import resolve_brain
            brain = resolve_brain("default")
            assert brain is not None
            result = brain.search("analytical", "owner", "standard")
            assert "nodes" in result
            names = [n.get("name") for n in result["nodes"]]
            assert "analytical" in names

    def test_is_proxy_brain_always_false(self):
        """Proxy brains are gone — is_proxy_brain must always return False."""
        from brain.brain_resolver import is_proxy_brain
        assert is_proxy_brain() is False
        assert is_proxy_brain("nonexistent") is False


class TestBrainProxyClientLocalOnly:
    """Test the legacy BrainProxyClient shim now reads from disk."""

    def test_local_search(self, beam_home, sample_graph):
        graph_path = beam_home / "brains" / "default" / "personality_graph.json"
        graph_path.write_text(json.dumps(sample_graph))

        with patch.dict(os.environ, {"BEAM_HOME": str(beam_home)}), \
             patch("brain.paths.BEAM_HOME", beam_home):
            from brain.paths import set_active_brain
            set_active_brain("default")
            from brain.brain_proxy_client import BrainProxyClient
            client = BrainProxyClient(slug="test")  # slug/token ignored
            results = client.search("analytical")
            names = [r.get("name") for r in results]
            assert "analytical" in names


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
        with patch("tools.brain_tools.BEAM_HOME", beam_home), \
             patch("brain.brain_resolver.BEAM_HOME", beam_home), \
             patch("brain.paths.BEAM_HOME", beam_home):
            from tools.brain_tools import brain_search
            result = json.loads(brain_search("test query"))
            # No graph → either an error or empty result; either is acceptable.
            assert "error" in result or result.get("total_matches", 0) == 0

    def test_brain_status_no_graph(self, beam_home):
        with patch("tools.brain_tools.BEAM_HOME", beam_home), \
             patch("brain.brain_resolver.BEAM_HOME", beam_home), \
             patch("brain.paths.BEAM_HOME", beam_home):
            from tools.brain_tools import brain_status
            result = json.loads(brain_status())
            # No graph → either "empty" status or empty coverage.
            assert result.get("status") == "empty" or result.get("coverage", {}).get("traits", 0) == 0

    def test_brain_search_with_graph(self, beam_home, sample_graph):
        graph_path = beam_home / "brains" / "default" / "personality_graph.json"
        graph_path.write_text(json.dumps(sample_graph))

        with patch("tools.brain_tools.BEAM_HOME", beam_home), \
             patch("brain.brain_resolver.BEAM_HOME", beam_home), \
             patch("brain.paths.BEAM_HOME", beam_home):
            from tools.brain_tools import brain_search
            result = json.loads(brain_search("analytical"))
            assert "nodes" in result
            names = [n.get("name") for n in result["nodes"]]
            assert "analytical" in names

    def test_brain_status_with_graph(self, beam_home, sample_graph):
        graph_path = beam_home / "brains" / "default" / "personality_graph.json"
        graph_path.write_text(json.dumps(sample_graph))

        with patch("tools.brain_tools.BEAM_HOME", beam_home), \
             patch("brain.brain_resolver.BEAM_HOME", beam_home), \
             patch("brain.paths.BEAM_HOME", beam_home):
            from tools.brain_tools import brain_status
            result = json.loads(brain_status())
            assert "coverage" in result
            assert result["coverage"]["traits"] == 2


class TestBrainUpdate:
    """Test the brain update command re-downloads from the marketplace."""

    def test_update_official_brain(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from hermes_cli import brain_cmds

        beam_home = tmp_path / ".beam"
        brain_dir = beam_home / "brains" / "official-writer"
        brain_dir.mkdir(parents=True)
        # v1 on disk
        (brain_dir / "personality_graph.json").write_text(
            json.dumps({"user_summary": "v1", "traits": []})
        )
        # config
        (beam_home / "config.yaml").write_text(
            "active_brain: official-writer\n"
            "brains:\n"
            "  official-writer:\n"
            "    source: marketplace-official\n"
            "    installed_at: '2026-06-01T00:00:00+00:00'\n"
        )

        v2 = {"user_summary": "v2", "traits": [{"name": "fresh", "strength": 0.9}]}
        fake_resp = MagicMock()
        fake_resp.json.return_value = v2
        fake_resp.raise_for_status.return_value = None
        fake_client = MagicMock()
        fake_client.__enter__.return_value.get.return_value = fake_resp

        with patch("brain.paths.BEAM_HOME", beam_home), \
             patch("hermes_cli.install_cmd.httpx.Client", return_value=fake_client):
            brain_cmds.cmd_brain_update("official-writer")

        on_disk = json.loads((brain_dir / "personality_graph.json").read_text())
        assert on_disk["user_summary"] == "v2"
        assert on_disk["traits"][0]["name"] == "fresh"

    def test_update_rejects_local_brain(self, tmp_path):
        from unittest.mock import patch
        from hermes_cli import brain_cmds

        beam_home = tmp_path / ".beam"
        brain_dir = beam_home / "brains" / "default"
        brain_dir.mkdir(parents=True)
        (brain_dir / "personality_graph.json").write_text(
            json.dumps({"user_summary": "local", "traits": []})
        )
        (beam_home / "config.yaml").write_text(
            "active_brain: default\n"
            "brains:\n"
            "  default:\n"
            "    source: local\n"
            "    installed_at: '2026-06-01T00:00:00+00:00'\n"
        )

        with patch("brain.paths.BEAM_HOME", beam_home):
            try:
                brain_cmds.cmd_brain_update("default")
            except SystemExit as e:
                assert e.code == 1
            else:
                raise AssertionError("expected SystemExit for local brain")


class TestEndToEndFlow:
    """Test the complete flow: interview → brain → SOUL.md → search."""

    def test_full_flow(self, beam_home, hermes_home, sample_graph):
        """Simulate the full brain-building flow with the offline pipeline."""
        # Step 1: Start the interview
        from brain.interview_orchestrator import InterviewOrchestrator
        orchestrator = InterviewOrchestrator()
        start_result = orchestrator.start()
        assert start_result["question_id"] == "identity_1"

        # Step 2: Drive a few scripted answers
        for i in range(2):
            r = orchestrator.answer(
                f"identity_{i+1}",
                f"Question {i+1}",
                "I am thoughtful and curious.",
                "identity",
            )
        interview_data = orchestrator.get_full_transcript()
        assert "I am thoughtful" in interview_data["transcript"]

        # Step 3: Build brain from the transcript (offline, no LLM)
        from brain.brain_builder import BrainBuilder
        builder = BrainBuilder()
        build_result = builder.extract(interview_data)
        assert "graph" in build_result
        graph = build_result["graph"]
        assert "raw_transcript" in graph
        assert graph["memories"]

        # Save the graph and run the rest against the on-disk file
        graph_path = beam_home / "brains" / "default" / "personality_graph.json"
        graph_path.write_text(json.dumps(graph))

        # Step 4: Generate SOUL.md (template-based, offline)
        from brain.soul_generator import generate_soul_md
        soul_path = hermes_home / "SOUL.md"
        soul_content = generate_soul_md(graph, soul_path)
        assert soul_path.exists()
        assert soul_content

        # Step 5: Search the saved brain
        with patch("tools.brain_tools.BEAM_HOME", beam_home), \
             patch("brain.brain_resolver.BEAM_HOME", beam_home), \
             patch("brain.paths.BEAM_HOME", beam_home):
            from tools.brain_tools import brain_search
            search_result = json.loads(brain_search("thoughtful"))
            assert "nodes" in search_result
            # The transcript is searchable too — we should find a memory or
            # transcript_excerpt hit for "thoughtful".
            types = {n.get("type") for n in search_result["nodes"]}
            assert types & {"memory", "transcript_excerpt"}


class TestBrainInstallMaterializesSoul:
    """`beam install <slug>` must eagerly regenerate ~/.hermes/SOUL.md
    from the downloaded marketplace brain so the next session has a
    populated identity — without this, the user had to wait for the
    on_session_start hook (which only fires on edge cases) or
    manually call `brain_export`."""

    def test_cmd_install_writes_soul_md(
        self, beam_home, hermes_home, marketplace_graph_factory
    ):
        import argparse
        from unittest.mock import MagicMock, patch
        from hermes_cli import install_cmd

        # Stub the marketplace download so the test doesn't hit the
        # network. We return a marketplace-schema graph that exercises
        # the schema adapter end-to-end.
        fake_resp = MagicMock()
        fake_resp.json.return_value = marketplace_graph_factory()
        fake_resp.raise_for_status.return_value = None
        fake_client = MagicMock()
        fake_client.__enter__.return_value.get.return_value = fake_resp

        args = argparse.Namespace(slug="creative-writer", no_activate=False)

        with patch("hermes_cli.install_cmd.httpx.Client", return_value=fake_client), \
             patch("brain.paths.BEAM_HOME", beam_home), \
             patch("hermes_constants.get_hermes_home", return_value=hermes_home):
            install_cmd.cmd_install(args)

        soul_path = hermes_home / "SOUL.md"
        assert soul_path.exists(), "cmd_install should have materialized SOUL.md"
        soul = soul_path.read_text(encoding="utf-8")
        # Invariant: the materialized SOUL.md must contain the
        # marketplace brain's personality content, not just the
        # legacy-only voice_dna stub.
        assert "Temperance" in soul or "Courage" in soul, (
            "marketplace values missing from materialized SOUL.md — "
            "schema adapter not wired into soul_generator"
        )


@pytest.fixture
def marketplace_graph_factory():
    """Factory for a minimal marketplace-schema graph used in install tests."""
    def _make():
        return {
            "personality_profile": {
                "values": [
                    "Temperance (0.95): Moderation in all things.",
                    "Courage (0.90): Face what must be faced.",
                ],
                "core_beliefs": [
                    "Discipline is freedom. (confidence: 0.97)",
                ],
            },
            "voice_dna": {
                "humor_style": "dry",
                "response_length_pattern": "concise",
            },
        }
    return _make
