"""interview plugin — adaptive multi-pass personality interview for beam-agent.

Registers start_interview and continue_interview tools that orchestrate
the multi-pass personality interview via the Rust beam-interview binary.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BEAM_HOME = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam"))

_interview_state: dict = {}


def _start_interview_handler(args: dict, **kw: Any) -> str:
    from brain.interview_orchestrator import InterviewOrchestrator

    global _interview_state
    orchestrator = InterviewOrchestrator()
    _interview_state["orchestrator"] = orchestrator

    result = orchestrator.start()
    return json.dumps(result, indent=2)


def _continue_interview_handler(args: dict, **kw: Any) -> str:
    from brain.brain_builder import BrainBuilder
    from brain.interview_orchestrator import InterviewOrchestrator

    global _interview_state
    orchestrator = _interview_state.get("orchestrator")

    if not orchestrator:
        return json.dumps({"error": "No active interview. Call start_interview first."})

    question_id = args.get("question_id", "")
    question = args.get("question", "")
    answer = args.get("answer", "")
    domain = args.get("domain", "")

    result = orchestrator.answer(question_id, question, answer, domain)

    if result.get("status") == "complete":
        from brain.brain_builder import BrainBuilder
        from brain.soul_generator import generate_soul_md

        interview_data = orchestrator.get_full_transcript()
        build_result = builder.extract(interview_data)

        user_id = "default"
        brain_dir = BEAM_HOME / "brain" / user_id
        brain_dir.mkdir(parents=True, exist_ok=True)
        graph_path = brain_dir / "personality_graph.json"
        graph = build_result.get("graph", {})
        graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

        # Auto-generate SOUL.md to Hermes home so the agent loads it
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
        hermes_home.mkdir(parents=True, exist_ok=True)
        try:
            generate_soul_md(graph, hermes_home / "SOUL.md")
        except Exception as exc:
            logger.warning("Failed to generate SOUL.md: %s", exc)

        result["brain_built"] = True
        result["graph_path"] = str(graph_path)
        result["soul_md"] = str(hermes_home / "SOUL.md")

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

START_INTERVIEW_SCHEMA = {
    "name": "start_interview",
    "description": "Start an adaptive interview to build your digital brain. Asks deep questions about personality, beliefs, values, work style, and emotional patterns across 3 passes.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

CONTINUE_INTERVIEW_SCHEMA = {
    "name": "continue_interview",
    "description": "Record your answer to an interview question and get the next question.",
    "parameters": {
        "type": "object",
        "properties": {
            "question_id": {
                "type": "string",
                "description": "The ID of the question being answered",
            },
            "question": {
                "type": "string",
                "description": "The question text",
            },
            "answer": {
                "type": "string",
                "description": "Your answer to the question",
            },
            "domain": {
                "type": "string",
                "enum": ["identity", "relationships", "work", "emotional", "beliefs", "procedural"],
                "description": "The domain of the question",
            },
        },
        "required": ["question_id", "question", "answer", "domain"],
    },
}


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_tool(
        name="start_interview",
        toolset="interview",
        schema=START_INTERVIEW_SCHEMA,
        handler=_start_interview_handler,
        description="Start an adaptive interview to build your digital brain.",
        emoji="🎤",
    )
    ctx.register_tool(
        name="continue_interview",
        toolset="interview",
        schema=CONTINUE_INTERVIEW_SCHEMA,
        handler=_continue_interview_handler,
        description="Record your answer and get the next interview question.",
        emoji="💬",
    )
