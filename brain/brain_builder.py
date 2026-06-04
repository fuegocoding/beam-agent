"""Calls beam-brain-builder to extract personality graph."""

from brain.subprocess_bridge import call_rust_binary


class BrainBuilder:
    """Builds a PersonalityGraph from interview data."""

    def extract(self, interview_data: dict, existing_graph: dict = None) -> dict:
        """Extract personality graph from interview data."""
        return call_rust_binary("beam-brain-builder", {
            "command": "extract",
            "interview_data": interview_data,
            "existing_graph": existing_graph,
        })

    def merge(self, existing_graph: dict, new_graph: dict) -> dict:
        """Merge new extraction into existing graph."""
        return call_rust_binary("beam-brain-builder", {
            "command": "merge",
            "interview_data": {"answers": [], "transcript": ""},
            "existing_graph": existing_graph,
        })

    def validate(self, graph: dict) -> dict:
        """Validate graph completeness."""
        return call_rust_binary("beam-brain-builder", {
            "command": "validate",
            "interview_data": {"answers": [], "transcript": ""},
            "existing_graph": graph,
        })
