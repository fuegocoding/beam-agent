"""Canonical JSON-LD export."""

import json

from brain_platform.pipeline.brain_file.schema import BrainFileSchema


def export_jsonld(brain_file: BrainFileSchema) -> bytes:
    """Serialize brain file to JSON-LD bytes."""
    data = brain_file.to_jsonld()
    return json.dumps(data, indent=2, default=str).encode("utf-8")
