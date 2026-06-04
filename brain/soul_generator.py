"""Generates SOUL.md from the personality graph."""

import os
from pathlib import Path

from brain.subprocess_bridge import call_rust_binary

BEAM_HOME = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam"))


def generate_soul_md(graph: dict, output_path: Path = None) -> str:
    """Generate SOUL.md from a PersonalityGraph."""
    if output_path is None:
        output_path = BEAM_HOME / "SOUL.md"

    result = call_rust_binary("beam-brain-runtime", {
        "command": "export_soul",
        "graph": graph,
        "trust_level": "owner",
        "brain_power": "full",
    })

    soul_content = result.get("soul_md", "# SOUL.md\n\nNo brain data available yet.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(soul_content, encoding="utf-8")
    return soul_content
