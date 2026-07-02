"""Dynamic char limits for the brain extraction LLM call.

The cloud's ``BrainExtractor`` (``brain_extractor.py:871-913``) decides
how much of the interview text to send to the LLM based on the model's
context window. Large models (gpt-4o, claude-3, deepseek, o1/o3) get
80,000 chars; small models (gpt-4o-mini, Groq free tier) get 20,000.

The cloud pulls the model name from the injected ``llm_client.config.model``.
The local port uses the user's auxiliary config — read the model name
from the auxiliary client router so the same limit logic applies
without re-implementing the provider resolution chain.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


MAX_INTERVIEW_CHARS_LARGE = 80_000
MAX_INTERVIEW_CHARS_SMALL = 20_000


# Substring matches that indicate a large-context model.
#
# The cloud's original check was a flat substring match on tags like
# "gpt-4o" / "o1" / "o3" / "claude-3". That has a known bug: the
# substring also matches the smaller variants ("gpt-4o" in "gpt-4o-mini",
# "o1" in "o1-mini"), so the smaller models get the 80K-char limit
# when they should get 20K. The CHANGELOG explicitly states
# "gpt-4o-mini, groq: stays at 20,000".
#
# The local port uses word-boundary matching via a negative lookahead
# for the small-variant suffixes, so "gpt-4o" matches "gpt-4o" but
# not "gpt-4o-mini". Same model catalog, fewer surprises.
#
# Note: All Claude 3 models (opus, sonnet, haiku) have 200K context,
# so they're all "large" by char-limit. The cloud's "claude-3" tag
# treated them all as large; the local port does the same.
_LARGE_MODEL_PATTERNS = (
    r"gpt-4o(?!-mini)",       # gpt-4o but not gpt-4o-mini
    r"gpt-4-turbo",
    r"claude-3",              # all Claude 3 variants (opus/sonnet/haiku) have 200K
    r"claude-sonnet-4",
    r"claude-opus-4",
    r"deepseek",             # all deepseek variants (v3, chat, etc.)
    r"o1(?!-mini)",           # o1, o1-preview, not o1-mini
    r"o3(?!-mini)",           # o3, not o3-mini
)

import re as _re_module
_LARGE_MODEL_RE = _re_module.compile(
    "(" + "|".join(_LARGE_MODEL_PATTERNS) + ")"
)


def is_large_model(model_name: str) -> bool:
    """True if the model has ~128K context and can handle 80K chars of input."""
    if not model_name:
        return False
    return bool(_LARGE_MODEL_RE.search(model_name))


def max_chars_for_model(model_name: str) -> int:
    """Return the interview-char limit for a given model name."""
    return MAX_INTERVIEW_CHARS_LARGE if is_large_model(model_name) else MAX_INTERVIEW_CHARS_SMALL


def get_model_name() -> str:
    """Return the active model name for brain-extraction LLM calls.

    Reads from ``agent.auxiliary_client``'s runtime cache so the
    adapter doesn't need a separate provider-resolution chain. Returns
    an empty string if the model can't be determined (the caller
    should fall back to the small-model limit in that case).
    """
    try:
        from agent.auxiliary_client import _resolve_auto, _resolve_api_key_provider
        # Prefer the resolution that picks the user's active provider
        client, model = _resolve_auto()
        if model:
            return model
    except Exception:
        pass
    return ""


def truncate_interview(interview_text: str, model_name: str) -> tuple[str, int]:
    """Truncate the interview to the model's char limit, appending a marker.

    Returns ``(truncated_text, original_len)``. The marker tells the
    LLM that the source was truncated so it doesn't hallucinate
    content for the missing tail.
    """
    max_chars = max_chars_for_model(model_name)
    if len(interview_text) <= max_chars:
        return interview_text, len(interview_text)

    logger.warning(
        "Interview text truncated %d → %d chars (model=%s)",
        len(interview_text), max_chars, model_name or "unknown",
    )
    return (
        interview_text[:max_chars]
        + "\n[Content truncated — full extraction requires a higher-capacity LLM]"
    ), len(interview_text)
