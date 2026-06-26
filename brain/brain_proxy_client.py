"""Local-only brain access (legacy compatibility shim).

The agent used to proxy every search/soul/context query through the Beam
API when a paid brain was installed. That meant no offline use, no
guarantees about latency, and the brain was useless without a working
connection.

That flow has been removed. The marketplace now ships every brain as a
full personality_graph.json downloaded once at install time. The class
below is kept so any 3rd-party code that imported `BrainProxyClient`
keeps working, but every method now reads from disk. There is no
network I/O — that was the whole point of removing the proxy.

New code should use `brain.brain_resolver.resolve_brain` or
`brain.brain_retriever.BrainRetriever` directly.
"""
import json
import os
from pathlib import Path

from brain.paths import (
    get_active_brain_graph_path,
    get_active_brain_name,
)


class _NetworkCallsRemoved(RuntimeError):
    """The brain subsystem no longer talks to the Beam API at runtime.

    If you see this, you're either:
      - Trying to install a brain that doesn't exist (run
        `beam install <slug>` first), or
      - Re-introducing a network call somewhere in the brain code path
        (don't — brains are local-only).
    """


def _load_local_graph() -> dict:
    """Load the active brain's personality graph from disk."""
    path = get_active_brain_graph_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No brain installed at {path}. Run 'beam install <slug>' first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


class BrainProxyClient:
    """Reads from the locally-downloaded graph. No network involved.

    The class signature is preserved so any lingering imports keep
    working, but the `token` and `api_url` arguments are ignored — the
    active brain is whatever the user has set via `beam brain switch`.
    """

    def __init__(self, slug: str, token: str | None = None, api_url: str | None = None):
        # slug/token/api_url are accepted for backwards-compat but ignored.
        self.slug = slug
        self._token = token
        self._api_url = api_url
        self._graph = _load_local_graph()

    def _enforce_offline(self) -> None:
        """Refuse to make network calls. The brain is local now."""
        raise _NetworkCallsRemoved(
            "BrainProxyClient no longer talks to the Beam API. "
            f"Active brain '{get_active_brain_name()}' is read from disk. "
            "If you need fresh data, re-run `beam install <slug>`."
        )

    def _search_local(
        self,
        query: str,
        trust_level: str = "visitor",
        brain_power: str = "standard",
    ) -> list[dict]:
        from brain.brain_retriever import BrainRetriever
        return (
            BrainRetriever()
            .search(query, self._graph, trust_level, brain_power)
            .get("nodes", [])
        )

    def search(
        self,
        query: str,
        trust_level: str = "visitor",
        brain_power: str = "standard",
    ) -> list[dict]:
        """Search the local brain. No network involved."""
        return self._search_local(query, trust_level, brain_power)

    def get_soul(self) -> str:
        """Return the locally-stored SOUL.md, or build one from the graph."""
        from brain.soul_generator import generate_soul_md
        soul_path = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam")) / "SOUL.md"
        if soul_path.exists():
            return soul_path.read_text(encoding="utf-8")
        return generate_soul_md(self._graph)

    def get_context(self) -> dict:
        """Return behavioral context for the active brain (computed locally)."""
        from brain.brain_retriever import BrainRetriever
        return BrainRetriever().build_context(self._graph)

    def ping(self) -> bool:
        """Returns True iff a local brain is installed and readable."""
        return get_active_brain_graph_path().exists()


__all__ = ["BrainProxyClient"]
