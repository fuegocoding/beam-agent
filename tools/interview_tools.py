"""Interview tools for beam-agent.

Provides start_interview and continue_interview tools
that orchestrate the multi-pass personality interview.
"""

import json
import os
from pathlib import Path

BEAM_HOME = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam"))

_interview_state = {}


def start_interview() -> str:
    """Start an adaptive interview to build your digital brain.

    The interview asks deep questions about your personality, beliefs, values,
    work style, and emotional patterns. It has 3 passes:
    - Pass 1: Surface — broad questions across all domains
    - Pass 2: Deep — targeted follow-ups on interesting signals
    - Pass 3: Gaps — fill remaining holes

    Returns:
        JSON string with the first question.
    """
    from brain.interview_orchestrator import InterviewOrchestrator

    global _interview_state
    orchestrator = InterviewOrchestrator()
    _interview_state["orchestrator"] = orchestrator

    result = orchestrator.start()
    return json.dumps(result, indent=2)


def continue_interview(question_id: str, question: str, answer: str, domain: str) -> str:
    """Record your answer to an interview question and get the next question.

    Args:
        question_id: The ID of the question being answered
        question: The question text
        answer: Your answer
        domain: The domain (identity, relationships, work, emotional, beliefs, procedural)

    Returns:
        JSON string with the next question or analysis results.
    """
    from brain.brain_builder import BrainBuilder
    from brain.interview_orchestrator import InterviewOrchestrator

    global _interview_state
    orchestrator = _interview_state.get("orchestrator")

    if not orchestrator:
        return json.dumps({"error": "No active interview. Call start_interview first."})

    result = orchestrator.answer(question_id, question, answer, domain)

    if result.get("status") == "complete":
        builder = BrainBuilder()
        interview_data = orchestrator.get_full_transcript()
        build_result = builder.extract(interview_data)

        user_id = "default"
        brain_dir = BEAM_HOME / "brain" / user_id
        brain_dir.mkdir(parents=True, exist_ok=True)
        graph_path = brain_dir / "personality_graph.json"
        graph_path.write_text(json.dumps(build_result.get("graph", {}), indent=2), encoding="utf-8")

        result["brain_built"] = True
        result["graph_path"] = str(graph_path)

    return json.dumps(result, indent=2)
