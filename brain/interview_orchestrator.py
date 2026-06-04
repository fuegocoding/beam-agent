"""Orchestrates the multi-pass interview by calling beam-interview."""

from brain.subprocess_bridge import call_rust_binary


class InterviewOrchestrator:
    """Manages the multi-pass interview process."""

    def __init__(self):
        self.answers: list[dict] = []
        self.current_pass = 1
        self.current_domain = None
        self.graph_draft = None

    def start(self) -> dict:
        """Start a new interview — get the first question."""
        result = call_rust_binary("beam-interview", {
            "command": "start",
            "pass": 1,
            "existing_graph": None,
            "answers": [],
            "domain": None,
        })
        return result

    def answer(self, question_id: str, question: str, answer: str, domain: str) -> dict:
        """Record an answer and get the next question."""
        self.answers.append({
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "domain": domain,
        })
        result = call_rust_binary("beam-interview", {
            "command": "continue",
            "pass": self.current_pass,
            "existing_graph": self.graph_draft,
            "answers": self.answers,
            "domain": self.current_domain,
        })
        return result

    def analyze(self) -> dict:
        """Analyze current state — get gaps and next steps."""
        result = call_rust_binary("beam-interview", {
            "command": "analyze",
            "pass": self.current_pass,
            "existing_graph": self.graph_draft,
            "answers": self.answers,
            "domain": None,
        })
        if result.get("graph_draft"):
            self.graph_draft = result["graph_draft"]
        return result

    def next_pass(self):
        """Advance to the next pass."""
        self.current_pass += 1

    def get_full_transcript(self) -> dict:
        """Get the full interview data for brain building."""
        return {
            "answers": self.answers,
            "transcript": "\n".join(
                f"Q: {a['question']}\nA: {a['answer']}" for a in self.answers
            ),
        }
