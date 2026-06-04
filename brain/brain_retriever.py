"""Calls beam-brain-runtime for graph queries."""

from brain.subprocess_bridge import call_rust_binary


class BrainRetriever:
    """Queries the personality graph at runtime."""

    def search(self, query: str, graph: dict, trust_level: str = "owner", brain_power: str = "standard") -> dict:
        """Search the brain for relevant nodes/edges."""
        return call_rust_binary("beam-brain-runtime", {
            "command": "search",
            "query": query,
            "graph": graph,
            "trust_level": trust_level,
            "brain_power": brain_power,
        })

    def build_context(self, graph: dict, trust_level: str = "owner", brain_power: str = "standard") -> dict:
        """Build structured context from the graph."""
        return call_rust_binary("beam-brain-runtime", {
            "command": "context",
            "graph": graph,
            "trust_level": trust_level,
            "brain_power": brain_power,
        })

    def export_soul(self, graph: dict) -> dict:
        """Generate SOUL.md from the graph."""
        return call_rust_binary("beam-brain-runtime", {
            "command": "export_soul",
            "graph": graph,
            "trust_level": "owner",
            "brain_power": "full",
        })

    def get_stats(self, graph: dict) -> dict:
        """Get graph statistics."""
        return call_rust_binary("beam-brain-runtime", {
            "command": "stats",
            "graph": graph,
            "trust_level": "owner",
            "brain_power": "full",
        })
