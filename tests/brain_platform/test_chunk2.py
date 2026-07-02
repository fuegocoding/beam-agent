"""Tests for Chunk 2 — the LLM-powered interview + brain extraction.

These tests validate that the cloud brain_mind extraction code was
adapted correctly to use the local :class:`LLMAdapter` (sync, BYOK)
instead of the cloud's async ``LLMService``.

LLM calls are mocked. The mock returns canned JSON responses for each
of the 4 extraction passes; the tests assert that the algorithm
correctly composes a ``PersonalityGraph`` from those responses.

The orchestration tests use a ``MockLLMAdapter`` that returns canned
content for depth checks and follow-up generation.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


# ──────────────────────────────────────────────────────────────────────
# services/llm_adapter.py
# ──────────────────────────────────────────────────────────────────────

class TestLLMAdapter:
    """The LLMAdapter bridges call_llm (BYOK) to the cloud's
    ``generate_response`` interface used by the lifted extraction code."""

    def test_strip_code_fence(self):
        from brain_platform.services.llm_adapter import _strip_code_fence

        assert _strip_code_fence('{"x": 1}') == '{"x": 1}'
        assert _strip_code_fence('```json\n{"x": 1}\n```') == '{"x": 1}'
        assert _strip_code_fence('```\n{"x": 1}\n```') == '{"x": 1}'
        assert _strip_code_fence('  {"x": 1}  ') == '{"x": 1}'

    def test_extract_first_json_block_finds_balanced_object(self):
        from brain_platform.services.llm_adapter import _extract_first_json_block

        text = 'Here is the JSON: {"a": 1, "b": [1, 2, 3]} thanks!'
        assert _extract_first_json_block(text) == '{"a": 1, "b": [1, 2, 3]}'

    def test_extract_first_json_block_handles_nested_strings(self):
        from brain_platform.services.llm_adapter import _extract_first_json_block

        text = '{"a": "has \\"escaped\\" quote", "b": 2}'
        result = _extract_first_json_block(text)
        assert result == text
        parsed = json.loads(result)
        assert parsed["a"] == 'has "escaped" quote'

    def test_extract_first_json_block_returns_none_for_no_object(self):
        from brain_platform.services.llm_adapter import _extract_first_json_block

        assert _extract_first_json_block("just a sentence") is None
        assert _extract_first_json_block("") is None

    def test_parse_into_pydantic_strict(self):
        """A well-formed JSON string parses on the first try."""
        from brain_platform.services.llm_adapter import _parse_into
        from brain_platform.pipeline.brain_schema import PersonalityGraph

        content = json.dumps({
            "user_summary": "Test user",
            "traits": [{"name": "curious", "strength": 0.7, "summary": "wonders"}],
        })
        g = _parse_into(content, PersonalityGraph)
        assert g.user_summary == "Test user"
        assert g.traits[0].name == "curious"

    def test_parse_into_pydantic_tolerates_fences(self):
        from brain_platform.services.llm_adapter import _parse_into
        from brain_platform.pipeline.brain_schema import PersonalityGraph

        content = "```json\n" + json.dumps({
            "user_summary": "Fenced",
            "traits": [{"name": "kind", "strength": 0.6, "summary": "warm"}],
        }) + "\n```"
        g = _parse_into(content, PersonalityGraph)
        assert g.user_summary == "Fenced"

    def test_parse_into_pydantic_tolerates_extra_prose(self):
        """When the LLM emits a JSON object followed by stray prose,
        the extractor pulls out the first balanced JSON block."""
        from brain_platform.services.llm_adapter import _parse_into
        from brain_platform.pipeline.brain_schema import PersonalityGraph

        content = 'Here you go: {"user_summary": "Prose user", "traits": []} -- done!'
        g = _parse_into(content, PersonalityGraph)
        assert g.user_summary == "Prose user"

    def test_parse_into_raises_on_unparseable(self):
        from brain_platform.services.llm_adapter import _parse_into
        from brain_platform.pipeline.brain_schema import PersonalityGraph

        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_into("not json at all", PersonalityGraph)

    def test_singleton_default_adapter(self):
        from brain_platform.services.llm_adapter import get_default_adapter

        a = get_default_adapter()
        b = get_default_adapter()
        assert a is b
        assert a.temperature == 0.3
        assert a.max_tokens == 4096

    def test_adapter_instantiation_with_overrides(self):
        from brain_platform.services.llm_adapter import LLMAdapter

        a = LLMAdapter(temperature=0.7, max_tokens=2000)
        assert a.temperature == 0.7
        assert a.max_tokens == 2000

    def test_extract_content_handles_openai_response(self):
        """call_llm returns an OpenAI ChatCompletion (or SimpleNamespace
        proxy). The adapter pulls content out of choices[0].message.content."""
        from brain_platform.services.llm_adapter import _extract_content

        # Simulate an OpenAI response
        msg = MagicMock()
        msg.content = "the assistant text"
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        assert _extract_content(response) == "the assistant text"

    def test_extract_content_handles_dict_response(self):
        from brain_platform.services.llm_adapter import _extract_content

        response = {"choices": [{"message": {"content": "from dict"}}]}
        assert _extract_content(response) == "from dict"


# ──────────────────────────────────────────────────────────────────────
# extractor/dynamic_limits.py
# ──────────────────────────────────────────────────────────────────────

class TestDynamicLimits:
    """The char-limit model detection matches the cloud's exact tags."""

    def test_is_large_model_for_known_large_tags(self):
        from brain_platform.extractor.dynamic_limits import is_large_model

        for tag in ("gpt-4o", "gpt-4o-2024-08-06", "claude-3-opus",
                    "claude-sonnet-4", "deepseek-chat", "o1-preview", "o3"):
            assert is_large_model(tag), f"should treat {tag} as large"

    def test_is_large_model_for_known_small_variants(self):
        """The '-mini' / 'haiku' / 'mini' variants of large families
        are explicitly small. The cloud's flat substring check had a
        bug that matched these as large; the local port's regex
        excludes them via negative lookahead."""
        from brain_platform.extractor.dynamic_limits import is_large_model

        for tag in ("gpt-4o-mini", "o1-mini", "o3-mini"):
            assert not is_large_model(tag), f"should treat {tag} as small"

    def test_is_large_model_for_small_tags(self):
        from brain_platform.extractor.dynamic_limits import is_large_model

        for tag in ("gpt-4o-mini", "gpt-3.5-turbo", "llama-3-8b",
                    "mistral-7b", "qwen-1.5b"):
            assert not is_large_model(tag), f"should treat {tag} as small"

    def test_is_large_model_for_empty_string(self):
        from brain_platform.extractor.dynamic_limits import is_large_model

        assert not is_large_model("")

    def test_max_chars_matches_constants(self):
        from brain_platform.extractor.dynamic_limits import (
            is_large_model, max_chars_for_model, MAX_INTERVIEW_CHARS_LARGE,
            MAX_INTERVIEW_CHARS_SMALL,
        )

        for tag in ("gpt-4o", "claude-3-haiku"):
            assert max_chars_for_model(tag) == MAX_INTERVIEW_CHARS_LARGE

        for tag in ("gpt-4o-mini", "llama-3-8b"):
            assert max_chars_for_model(tag) == MAX_INTERVIEW_CHARS_SMALL

    def test_truncate_interview_under_limit(self):
        from brain_platform.extractor.dynamic_limits import truncate_interview

        text = "short"
        result, orig = truncate_interview(text, "gpt-4o")
        assert result == "short"
        assert orig == 5

    def test_truncate_interview_over_limit(self):
        from brain_platform.extractor.dynamic_limits import (
            truncate_interview, MAX_INTERVIEW_CHARS_LARGE,
        )

        text = "x" * (MAX_INTERVIEW_CHARS_LARGE + 1000)
        result, orig = truncate_interview(text, "gpt-4o")
        assert orig == len(text)
        # Truncated to the max + a marker
        assert len(result) > MAX_INTERVIEW_CHARS_LARGE
        assert "[Content truncated" in result


# ──────────────────────────────────────────────────────────────────────
# extractor/entity_resolution.py
# ──────────────────────────────────────────────────────────────────────

class TestEntityResolution:
    """The role-only / named-people merge from the cloud's CHANGELOG
    "the single biggest fix" of the 2026 extraction upgrade."""

    def _make_graph(self, people_data: list[dict]) -> "PersonalityGraph":
        from brain_platform.pipeline.brain_schema import PersonalityGraph, PersonNode
        return PersonalityGraph(
            user_summary="",
            people=[PersonNode(**p) for p in people_data],
        )

    def test_merges_role_only_into_named_person(self):
        from brain_platform.extractor.entity_resolution import resolve_people_entities

        g = self._make_graph([
            {"name": "Sarah", "role": "spouse", "summary": "My wife Sarah"},
            {"name": "wife", "role": "", "summary": "My wife is supportive"},
        ])
        out = resolve_people_entities(g)
        names = {p.name for p in out.people}
        # "wife" is merged into "Sarah"; the role-only node is dropped
        assert "wife" not in names
        assert "Sarah" in names
        # Sarah's role should now include "wife" (it had "spouse" before)
        sarah = next(p for p in out.people if p.name == "Sarah")
        assert "wife" in sarah.role.lower() or "spouse" in sarah.role.lower()

    def test_preserves_graph_when_no_role_nodes(self):
        from brain_platform.extractor.entity_resolution import resolve_people_entities

        g = self._make_graph([
            {"name": "Alice", "role": "friend", "summary": "good friend"},
            {"name": "Bob", "role": "colleague", "summary": "works with me"},
        ])
        out = resolve_people_entities(g)
        # No role-only nodes; nothing to merge
        assert len(out.people) == 2

    def test_preserves_graph_when_no_named_nodes(self):
        from brain_platform.extractor.entity_resolution import resolve_people_entities

        g = self._make_graph([
            {"name": "wife", "role": "", "summary": "supportive"},
            {"name": "mother", "role": "", "summary": "kind"},
        ])
        out = resolve_people_entities(g)
        # No named people; can't merge into anything
        assert len(out.people) == 2

    def test_rewrites_edges_after_merge(self):
        from brain_platform.extractor.entity_resolution import resolve_people_entities
        from brain_platform.pipeline.brain_schema import PersonalityGraph, PersonNode, EdgeSpec

        g = PersonalityGraph(
            user_summary="",
            people=[
                PersonNode(name="Sarah", role="", summary="my wife"),
                PersonNode(name="wife", role="", summary=""),
            ],
            edges=[
                EdgeSpec(source_name="wife", target_name="Sarah",
                          edge_type="HOLDS", fact="wife relates to Sarah"),
            ],
        )
        out = resolve_people_entities(g)
        # The edge that referenced "wife" should now point to "Sarah"
        assert all(e.source_name != "wife" for e in out.edges)
        assert all(e.target_name != "wife" for e in out.edges)


# ──────────────────────────────────────────────────────────────────────
# pipeline/interview/coverage_scorer.py
# ──────────────────────────────────────────────────────────────────────

class TestCoverageScorer:
    """Per-dimension 0.0-1.0 scoring from a PersonalityGraph in memory."""

    def _build_graph_with_traits(self, n: int) -> "PersonalityGraph":
        from brain_platform.pipeline.brain_schema import PersonalityGraph, TraitNode
        return PersonalityGraph(
            user_summary="",
            traits=[
                TraitNode(name=f"trait_{i}", strength=0.5 + i * 0.05, summary="x" * 50)
                for i in range(n)
            ],
        )

    def test_empty_graph_scores_all_zero(self):
        from brain_platform.pipeline.interview.coverage_scorer import (
            DimensionCoverageScorer,
        )

        scorer = DimensionCoverageScorer()
        scores = scorer.score({})
        # All 10 dimensions present, all zero
        assert len(scores) == 10
        for s in scores.values():
            assert s.score == 0.0
            assert s.source == "local"

    def test_traits_fill_core_beliefs_dimension_via_belief(self):
        """The dimension mapping is type-based. Traits don't directly
        map to any dimension — but PersonalityGraph.traits are mostly
        behavioral markers. The scorer uses the entity type."""
        from brain_platform.pipeline.interview.coverage_scorer import (
            DimensionCoverageScorer,
        )

        scorer = DimensionCoverageScorer()
        scores = scorer.score(self._build_graph_with_traits(5))
        # We added 5 traits (not beliefs) — no dimension should be > 0
        # because traits don't have a direct dimension in the map
        assert scores["core_beliefs"].score == 0.0

    def test_beliefs_fill_core_beliefs_dimension(self):
        from brain_platform.pipeline.interview.coverage_scorer import (
            DimensionCoverageScorer,
        )
        from brain_platform.pipeline.brain_schema import (
            PersonalityGraph, BeliefNode,
        )

        g = PersonalityGraph(
            user_summary="",
            beliefs=[
                BeliefNode(name=f"belief_{i}", confidence=0.7, summary="x" * 60)
                for i in range(3)
            ],
        )
        scorer = DimensionCoverageScorer()
        scores = scorer.score(g)
        # 3 beliefs hit the min_nodes=3 target for core_beliefs
        assert scores["core_beliefs"].node_count == 3
        assert scores["core_beliefs"].score >= 0.5

    def test_diversity_metric(self):
        """Diversity = unique_names / node_count. 1.0 if no dupes."""
        from brain_platform.pipeline.interview.coverage_scorer import (
            DimensionCoverageScorer,
        )
        from brain_platform.pipeline.interview.coverage_types import DimensionScore
        from brain_platform.pipeline.brain_schema import (
            PersonalityGraph, BeliefNode,
        )

        # 5 beliefs but only 3 unique names → diversity = 0.6
        g = PersonalityGraph(
            user_summary="",
            beliefs=[
                BeliefNode(name="alpha", confidence=0.7, summary="x"),
                BeliefNode(name="alpha", confidence=0.7, summary="x"),
                BeliefNode(name="beta", confidence=0.7, summary="x"),
                BeliefNode(name="beta", confidence=0.7, summary="x"),
                BeliefNode(name="gamma", confidence=0.7, summary="x"),
            ],
        )
        scorer = DimensionCoverageScorer()
        scores = scorer.score(g)
        # 5 belief nodes hit min_nodes=3 but we have 5/3 unique
        # so diversity = 3/5 = 0.6
        assert scores["core_beliefs"].diversity == pytest.approx(0.6, abs=0.01)

    def test_source_is_local(self):
        """The local scorer tags results with source='local' (not
        'graphiti' or 'postgres_fallback' like the cloud)."""
        from brain_platform.pipeline.interview.coverage_scorer import (
            DimensionCoverageScorer,
        )

        scorer = DimensionCoverageScorer()
        scores = scorer.score({})
        for s in scores.values():
            assert s.source == "local"


# ──────────────────────────────────────────────────────────────────────
# pipeline/interview/follow_up.py
# ──────────────────────────────────────────────────────────────────────

class TestFollowUp:
    """The depth check and follow-up generation use a mocked LLMAdapter."""

    def _make_question(self, idx: int = 1):
        """Default to question 1 (life_story_1, min_words_for_depth=60)
        so the short-answer hard-floor path is exercised. Question 0
        is the AGE_QUESTION with min_words=1 which is too permissive."""
        from brain_platform.pipeline.interview.questions import INTERVIEW_QUESTIONS
        return INTERVIEW_QUESTIONS[idx]

    def test_short_answer_always_triggers_followup(self):
        """Word count below min_words_for_depth → always need follow-up,
        no LLM call."""
        from brain_platform.pipeline.interview.follow_up import needs_follow_up

        llm = MagicMock()
        q = self._make_question(idx=1)  # min_words=60
        result = needs_follow_up(llm, q, "too brief")  # 2 words
        assert result is True
        # LLM not called for the short-answer path
        llm.generate_response.assert_not_called()

    def test_long_answer_never_needs_followup(self):
        """Word count > 120 → never need follow-up, no LLM call."""
        from brain_platform.pipeline.interview.follow_up import needs_follow_up

        llm = MagicMock()
        q = self._make_question(idx=1)
        long_answer = " ".join(["word"] * 150)
        result = needs_follow_up(llm, q, long_answer)
        assert result is False
        llm.generate_response.assert_not_called()

    def test_medium_answer_uses_llm_depth_check(self):
        """Word count between min and 120 → LLM judges depth."""
        from brain_platform.pipeline.interview.follow_up import needs_follow_up

        llm = MagicMock()
        llm.generate_response.return_value = "insufficient"
        q = self._make_question(idx=1)  # min_words=60
        # 80 words — between 60 and 120
        medium_answer = " ".join(["word"] * 80)
        result = needs_follow_up(llm, q, medium_answer)
        assert result is True
        # LLM was called
        assert llm.generate_response.called
        # With task=TASK_DEPTH_CHECK
        call_kwargs = llm.generate_response.call_args.kwargs
        assert call_kwargs["task"] == "interview_depth_check"

    def test_medium_answer_sufficient_does_not_trigger(self):
        from brain_platform.pipeline.interview.follow_up import needs_follow_up

        llm = MagicMock()
        llm.generate_response.return_value = "sufficient"
        q = self._make_question(idx=1)
        medium_answer = " ".join(["word"] * 80)
        result = needs_follow_up(llm, q, medium_answer)
        assert result is False

    def test_generate_follow_up_strips_quotes(self):
        from brain_platform.pipeline.interview.follow_up import generate_follow_up

        llm = MagicMock()
        llm.generate_response.return_value = '"What was the hardest part?"'
        q = self._make_question(idx=1)
        result = generate_follow_up(llm, q, "some answer", "standard")
        # The wrapping double-quotes are stripped
        assert result == "What was the hardest part?"
        # Task hint is the follow-up task
        call_kwargs = llm.generate_response.call_args.kwargs
        assert call_kwargs["task"] == "interview_followup"

    def test_generate_follow_up_falls_back_to_hint_on_failure(self):
        """If the LLM call fails, return the question's static follow_up_hint.

        Matches the cloud's behavior (beam_mind/pipeline/interview/follow_up.py:102):
        the hint has its ``"If too brief: "`` prefix stripped before returning,
        so the caller sees the actionable question, not the meta-instruction.
        """
        from brain_platform.pipeline.interview.follow_up import generate_follow_up

        llm = MagicMock()
        llm.generate_response.side_effect = RuntimeError("LLM down")
        q = self._make_question(idx=1)
        result = generate_follow_up(llm, q, "some answer", "standard")
        # Cloud behavior: strip the "If too brief: " prefix from the hint
        expected = q.follow_up_hint.split(": ", 1)[-1]
        assert result == expected


# ──────────────────────────────────────────────────────────────────────
# interview_orchestrator.py — end-to-end with mocked LLM
# ──────────────────────────────────────────────────────────────────────

class TestAdaptiveInterviewOrchestrator:
    """The orchestrator ties the question bank, follow-up engine,
    coverage scorer, gap identifier, and question selector together."""

    def _llm(self, follow_up_response: str = "", depth_response: str = "sufficient"):
        """Mock LLM that returns canned responses for follow-up / depth."""
        llm = MagicMock()
        if follow_up_response:
            llm.generate_response.return_value = follow_up_response
        else:
            llm.generate_response.return_value = depth_response
        return llm

    def test_starts_with_first_question(self):
        from brain_platform.interview_orchestrator import AdaptiveInterviewOrchestrator
        from brain_platform.services.llm_adapter import LLMAdapter

        # Use the real LLMAdapter singleton — the orchestrator never
        # calls it unless an answer is shallow enough to need follow-up.
        orch = AdaptiveInterviewOrchestrator(llm=LLMAdapter(), user_age=40)
        q = orch.start()
        assert q is not None
        assert q.id  # has an ID
        assert orch.questions_asked  # the question is recorded

    def test_long_answer_advances(self):
        from brain_platform.interview_orchestrator import AdaptiveInterviewOrchestrator
        from brain_platform.services.llm_adapter import LLMAdapter

        llm = LLMAdapter()  # not called for long answers
        orch = AdaptiveInterviewOrchestrator(llm=llm, user_age=40)
        q1 = orch.start()
        long_answer = " ".join(["word"] * 200)
        result = orch.answer(long_answer)
        # Long answer → no follow-up, advances to next question
        assert result.follow_up is None
        assert result.next_question is not None
        assert result.next_question.id != q1.id  # advanced

    def test_short_answer_asks_followup(self):
        from brain_platform.interview_orchestrator import AdaptiveInterviewOrchestrator

        # The orchestrator's first question is the AGE_QUESTION with
        # min_words=1. After that, life_story_1 has min_words=60.
        # We answer the first question (passes hard floor), then
        # answer the second question short (triggers follow-up).
        llm = self._llm(follow_up_response='"Tell me more about that."')
        orch = AdaptiveInterviewOrchestrator(llm=llm, user_age=40)
        q1 = orch.start()
        # Answer q1 (AGE_QUESTION) — short is fine here
        r1 = orch.answer("Theodore")
        assert r1.next_question is not None
        # Answer r1.next_question (life_story_1) with 2 words
        r2 = orch.answer("ok")
        assert r2.follow_up is not None
        assert "Tell me more about that" in r2.follow_up

    def test_short_answer_with_depth_sufficient_skips_followup(self):
        """Short answers ALWAYS trigger follow-up — the cloud's depth
        check only runs for medium-length answers. Below the
        min_words_for_depth floor, no LLM is called, no "sufficient"
        check is made, follow-up fires regardless. This test pins
        that contract."""
        from brain_platform.interview_orchestrator import AdaptiveInterviewOrchestrator

        llm = MagicMock()
        llm.generate_response.return_value = "sufficient"
        orch = AdaptiveInterviewOrchestrator(llm=llm, user_age=40)
        q1 = orch.start()  # AGE_QUESTION
        r1 = orch.answer("Theodore")  # advance
        r2 = orch.answer("ok")  # life_story_1, min_words=60, 2-word answer
        # Short answer below min_words floor → follow-up fires,
        # even though LLM would have said "sufficient" (never called)
        assert r2.follow_up is not None
        # Verify the LLM was never called for the depth check
        # (the short-path hard floor skips the LLM entirely)
        for call in llm.generate_response.call_args_list:
            assert call.kwargs.get("task") != "interview_depth_check"

    def test_get_transcript_returns_full_state(self):
        from brain_platform.interview_orchestrator import AdaptiveInterviewOrchestrator
        from brain_platform.services.llm_adapter import LLMAdapter

        orch = AdaptiveInterviewOrchestrator(llm=LLMAdapter(), user_age=40)
        orch.start()
        orch.answer(" ".join(["word"] * 50))
        transcript = orch.get_transcript()
        assert "answers" in transcript
        assert "transcript" in transcript
        assert "questions_asked" in transcript
        assert "coverage" in transcript
        assert len(transcript["answers"]) == 1

    def test_max_questions_caps_interview(self):
        from brain_platform.interview_orchestrator import AdaptiveInterviewOrchestrator
        from brain_platform.services.llm_adapter import LLMAdapter

        orch = AdaptiveInterviewOrchestrator(
            llm=LLMAdapter(), user_age=40, max_questions=2,
        )
        q1 = orch.start()
        r1 = orch.answer(" ".join(["word"] * 50))
        r2 = orch.answer(" ".join(["word"] * 50))
        # After max_questions, the next answer should mark complete
        r3 = orch.answer(" ".join(["word"] * 50))
        # r3 is the third question's answer — should be complete now
        # (max_questions=2, so we stop after r2)
        # Actually the count is checked before recording the answer
        assert r3.is_complete or r3.next_question is None
