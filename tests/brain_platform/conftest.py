"""Local conftest for brain_platform tests.

Overrides the parent conftest's ``_hermetic_environment`` for integration
tests so credential env vars (NEO4J_*, OPENROUTER_API_KEY, etc.) survive
into the test. The parent's autouse fixture blanks all credential-shaped
env vars to keep unit tests hermetic — but the integration tests need
real credentials to talk to Neo4j + the LLM.

Graphiti's embedder uses the OpenAI Python client directly (not beam-agent's
``call_llm``), so it needs ``OPENAI_API_KEY`` + optionally ``OPENAI_BASE_URL``.
If you're using OpenRouter, set:

    OPENAI_API_KEY=<your-openrouter-key>
    OPENAI_BASE_URL=https://openrouter.ai/api/v1

to route Graphiti's embeddings through OpenRouter.
"""
from __future__ import annotations

import os

import pytest


# Capture credentials at import time (before the parent's autouse fixture
# runs and blanks them). These are read from the parent process's env.
_INTEGRATION_ENV: dict[str, str] = {}
for _name in (
    "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_TLS",
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "DEEPSEEK_API_KEY", "NOUS_API_KEY", "MINIMAX_API_KEY",
):
    _val = os.environ.get(_name)
    if _val:
        _INTEGRATION_ENV[_name] = _val


@pytest.fixture(autouse=True)
def _restore_integration_credentials(monkeypatch):
    """Re-set credential env vars that the parent's hermetic fixture blanked.

    Runs after the parent's ``_hermetic_environment`` (because child
    conftest fixtures resolve after parent conftest fixtures). Uses
    monkeypatch.setenv to restore the values, so they're still scoped
    to the test and cleaned up afterward.
    """
    for name, value in _INTEGRATION_ENV.items():
        monkeypatch.setenv(name, value)

