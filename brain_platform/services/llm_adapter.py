"""LLM adapter — sync wrapper that bridges the cloud brain_mind extraction code
to beam-agent's :func:`agent.auxiliary_client.call_llm`.

The cloud code (e.g. ``BrainExtractor.extract``) calls::

    await llm_client.generate_response(
        messages=[...],
        response_model=PersonalityGraph,
    )

This module exposes :class:`LLMAdapter` with the same call signature
(sync, not async) so the lifted extraction code is mechanically close
to the cloud source. Inside, it routes through beam-agent's BYOK
``call_llm`` (which already handles 6+ providers, credit-exhaustion
fallback, and per-task model overrides via ``auxiliary.*`` config).

The return value of :meth:`LLMAdapter.generate_response` is a parsed
Pydantic instance of ``response_model`` (or the raw dict if
``response_model`` is None). The cloud's Graphiti client returns the
same shape, so the lifted code can treat the return value identically
after dropping the ``await``.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# Auxiliary task names registered by this package. Users can pin a
# different provider/model per task via ``auxiliary.<task>`` in
# ``~/.hermes/config.yaml`` (e.g. ``auxiliary.brain_extract.model: gpt-4o``).
TASK_BRAIN_EXTRACT = "brain_extract"
TASK_INTERVIEW = "interview"
TASK_FOLLOW_UP = "interview_followup"
TASK_DEPTH_CHECK = "interview_depth_check"
TASK_CLONE_SPEC = "brain_clone_spec"
TASK_EMOTIONAL = "brain_emotional"
TASK_REPAIR = "brain_repair"
TASK_INGEST = "brain_ingest"

# Strip markdown code fences that some models (especially older gpt-4o-mini)
# wrap JSON in even when instructed not to. The cloud's Graphiti client
# does this too.
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _JSON_FENCE_RE.sub("", text).strip()
    return text


class LLMAdapter:
    """Sync adapter that mimics the cloud's ``Graphiti-compatible generate_response`` API.

    Usage (mirrors the cloud's BrainExtractor call site)::

        from brain_platform.services.llm_adapter import LLMAdapter
        from brain_platform.pipeline.brain_schema import PersonalityGraph

        llm = LLMAdapter()
        graph = llm.generate_response(
            messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
            response_model=PersonalityGraph,
            task="brain_extract",
        )
        # graph is a PersonalityGraph instance

    All calls are routed through ``agent.auxiliary_client.call_llm``, so
    the user controls provider/model via the existing ``auxiliary.*``
    config (BYOK). The cloud's runtime-injected ``llm_client`` is
    replaced with a process-level singleton.
    """

    def __init__(self, *, temperature: float = 0.3, max_tokens: int = 4096):
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_response(
        self,
        messages: List[dict],
        response_model: Optional[Type[BaseModel]] = None,
        *,
        task: str = TASK_BRAIN_EXTRACT,
        prompt_name: str = "brain_platform.extract",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[BaseModel, dict, str]:
        """Call the LLM, parse the response into ``response_model`` if given.

        Args:
            messages: OpenAI-style chat messages (already in the
                ``[{"role": "system", "content": "..."}, ...]`` shape).
                The cloud's Graphiti Message objects are accepted too —
                they expose ``.role`` and ``.content`` attributes.
            response_model: Pydantic class to parse the response into.
                If None, returns the raw string content.
            task: Auxiliary task name — routes to the user's
                ``auxiliary.<task>.*`` config so they can pin a
                different model per task.
            prompt_name: Debug label, logged on the request.
            temperature: Override the default 0.3 (lower = more deterministic).
            max_tokens: Override the default 4096.

        Returns:
            If ``response_model`` is set: an instance of that class.
            Otherwise: the raw string content.
        """
        from agent.auxiliary_client import call_llm

        # The cloud's Graphiti Message dataclass is a duck-typed equivalent
        # of the OpenAI message dict. Normalize here so callers can pass
        # either shape (the lifted brain_extractor passes Message objects).
        normalized = [
            {"role": m.role if hasattr(m, "role") else m["role"],
             "content": m.content if hasattr(m, "content") else m["content"]}
            for m in messages
        ]

        kwargs = {
            "task": task,
            "messages": normalized,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        logger.debug(
            "LLMAdapter.generate_response task=%s prompt_name=%s messages=%d",
            task, prompt_name, len(normalized),
        )

        response = call_llm(**kwargs)

        # ``call_llm`` returns a SimpleNamespace or a real OpenAI response
        # with .choices[0].message.content — depends on the path. Pull
        # the content string out of the response.
        content = _extract_content(response)

        if response_model is None:
            return content

        return _parse_into(content, response_model, prompt_name=prompt_name)


def _extract_content(response: Any) -> str:
    """Pull the assistant text out of whatever shape ``call_llm`` returns.

    ``call_llm`` returns one of:
    - An OpenAI ``ChatCompletion`` (has ``.choices[0].message.content``)
    - A ``SimpleNamespace`` proxy (has ``.choices[0].message.content``)
    - A dict (legacy path)
    """
    # OpenAI / SimpleNamespace response
    choices = getattr(response, "choices", None)
    if choices:
        msg = choices[0].message
        content = getattr(msg, "content", None)
        if content is not None:
            return content

    # Dict response (legacy)
    if isinstance(response, dict):
        choices = response.get("choices")
        if choices:
            return choices[0].get("message", {}).get("content", "")

    # Fallback: stringify
    return str(response)


def _parse_into(
    content: str,
    response_model: Type[BaseModel],
    *,
    prompt_name: str = "",
) -> BaseModel:
    """Parse an LLM string response into a Pydantic model.

    Strips markdown code fences, then tries strict JSON parse first,
    then a tolerant extractor (greedy brace match) as a fallback.
    Raises ``ValueError`` if the content can't be parsed — the caller
    should treat this as a recoverable LLM error and retry or skip.
    """
    stripped = _strip_code_fence(content)
    try:
        return response_model.model_validate_json(stripped)
    except Exception as first_err:
        # Fallback: find the largest {...} or [...] block in the response
        # and try to parse that. Some models emit a JSON object followed
        # by stray prose, especially with shorter max_tokens settings.
        candidate = _extract_first_json_block(stripped)
        if candidate and candidate != stripped:
            try:
                return response_model.model_validate_json(candidate)
            except Exception:
                pass
        logger.warning(
            "LLMAdapter._parse_into failed to parse response_model=%s prompt_name=%s: %s",
            response_model.__name__, prompt_name, first_err,
        )
        raise ValueError(
            f"Failed to parse LLM response into {response_model.__name__}: {first_err}"
        ) from first_err


def _extract_first_json_block(text: str) -> Optional[str]:
    """Find the first balanced JSON object/array in the text and return it.

    Used as a tolerant fallback when the LLM emits extra prose around
    the JSON. Scans for the first ``{`` or ``[`` and matches balanced
    braces/brackets (handling nested strings and escapes).
    """
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        opener, closer = ch, "}" if ch == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for j in range(i, len(text)):
            c = text[j]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return text[i:j + 1]
    return None


# Process-level singleton — the lifted extraction code constructs an
# LLMAdapter() and uses it across the 3+ LLM calls in a single
# extraction. A module-level singleton saves the per-instance dict
# construction cost (negligible) but mainly signals "this is stateless,
# no per-user state".
_default_adapter: Optional[LLMAdapter] = None


def get_default_adapter() -> LLMAdapter:
    """Return the process-level default LLMAdapter singleton."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = LLMAdapter()
    return _default_adapter


__all__ = [
    "LLMAdapter",
    "get_default_adapter",
    "TASK_BRAIN_EXTRACT",
    "TASK_INTERVIEW",
    "TASK_FOLLOW_UP",
    "TASK_DEPTH_CHECK",
    "TASK_CLONE_SPEC",
    "TASK_EMOTIONAL",
    "TASK_REPAIR",
    "TASK_INGEST",
]
