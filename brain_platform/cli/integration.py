"""Chunk 4 — CLI integration for brain_platform.

Adds brain_platform-specific commands to the beam CLI without touching
the existing offline brain/ commands. Wired into hermes_cli/main.py via
:func:`register_brain_platform_commands`.

Commands added:
  - ``beam brain platform-search <query>`` — Neo4j-backed graph search
  - ``beam brain platform-ingest <file>`` — extract + persist to Neo4j
  - ``beam interview --adaptive`` — LLM-powered adaptive interview
  - ``beam brain setup-neo4j`` — interactive wizard for NEO4J_URI etc.

The context_builder integration (using LocalGraphSearcher for agent
memory retrieval) lives in :mod:`brain_platform.runtime_integration`.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

HERMES_ENV_PATH = Path.home() / ".hermes" / ".env"


def _read_env_value(name: str) -> Optional[str]:
    """Read a value from ~/.hermes/.env (returns None if missing)."""
    if not HERMES_ENV_PATH.exists():
        return None
    for line in HERMES_ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            if key.strip() == name:
                return val.strip().strip('"').strip("'")
    return None


def _write_env_value(name: str, value: str) -> None:
    """Write or update a value in ~/.hermes/.env."""
    HERMES_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if HERMES_ENV_PATH.exists():
        lines = HERMES_ENV_PATH.read_text().splitlines()
    else:
        lines = []

    found = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{name}="):
            lines[i] = f"{name}={value}"
            found = True
            break
    if not found:
        lines.append(f"{name}={value}")

    HERMES_ENV_PATH.write_text("\n".join(lines) + "\n")
    # Also set in current process so subsequent commands see it
    os.environ[name] = value


def cmd_setup_neo4j(args: Any) -> int:
    """Interactive wizard to configure Neo4j connection for brain_platform.

    Writes NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD to ~/.hermes/.env.
    If the values are already set, shows them and asks before overwriting.
    """
    print("\n" + "=" * 60)
    print("  Neo4j setup for brain_platform")
    print("=" * 60)
    print()
    print("brain_platform needs a Neo4j instance to store the personality")
    print("knowledge graph. You can use any of:")
    print()
    print("  1. Neo4j Aura (managed cloud, free tier) — RECOMMENDED")
    print("       Sign up: https://neo4j.com/cloud/aura/")
    print("       Copy the connection URI + credentials from the Aura console")
    print()
    print("  2. Neo4j Desktop (local GUI)")
    print("       Download: https://neo4j.com/download/")
    print()
    print("  3. Local Docker")
    print("       docker run -p 7687:7687 -p 7474:7474 \\")
    print("         -e NEO4J_AUTH=neo4j/password neo4j")
    print()

    # Show current values
    current_uri = _read_env_value("NEO4J_URI")
    current_user = _read_env_value("NEO4J_USER")
    if current_uri:
        print(f"Current NEO4J_URI: {current_uri}")
        print(f"Current NEO4J_USER: {current_user or '(not set)'}")
        if input("\nOverwrite? [y/N]: ").strip().lower() != "y":
            print("Keeping existing values.")
            return 0
        print()

    uri = input("Neo4j URI [bolt://localhost:7687]: ").strip()
    if not uri:
        uri = "bolt://localhost:7687"

    user = input("Neo4j user [neo4j]: ").strip()
    if not user:
        user = "neo4j"

    # For Aura, the password is shown once at creation time —
    # make sure the user pastes it correctly.
    if "neo4j+s://" in uri or "neo4j+ssc://" in uri:
        print("\n⚠  You're connecting to Neo4j Aura (TLS).")
        print("   Paste the password you saved when creating the instance.")
    password = input("Neo4j password: ").strip()
    if not password:
        print("No password provided. Aborting.")
        return 1

    # Write to .env
    _write_env_value("NEO4J_URI", uri)
    _write_env_value("NEO4J_USER", user)
    _write_env_value("NEO4J_PASSWORD", password)
    print(f"\n✓ Neo4j config saved to {HERMES_ENV_PATH}")

    # Test connection (best-effort)
    print("\nTesting connection...")
    try:
        from brain_platform.services.local_graph_store import LocalGraphStore

        store = LocalGraphStore(uri=uri, user=user, password=password)
        store.initialize()
        if store.health_check():
            print("✓ Connection successful.")
        else:
            print("⚠  Health check failed — Neo4j may not be reachable yet.")
            print("   The brain_platform will retry on first use.")
        store.close()
    except Exception as e:
        print(f"⚠  Could not connect: {e}")
        print("   Double-check the URI, user, and password.")
        print("   The config is still saved — brain_platform will retry on first use.")

    return 0


def cmd_brain_platform_search(args: Any) -> int:
    """Search the Neo4j-backed personality graph for relevant facts.

    Usage: beam brain platform-search <query> [--num-results N]
    """
    query = getattr(args, "query", None)
    if not query:
        print("Usage: beam brain platform-search <query>")
        return 1

    num_results = getattr(args, "num_results", 5)
    group_id = getattr(args, "group_id", "default_user")

    try:
        from brain_platform.services.local_graph_store import LocalGraphStore
        from brain_platform.services.local_graph_searcher import LocalGraphSearcher

        store = LocalGraphStore()
        store.initialize()
        try:
            searcher = LocalGraphSearcher(store)
            facts = searcher.search(query=query, group_id=group_id, num_results=num_results)
        finally:
            store.close()
    except Exception as e:
        print(f"Error: {e}")
        print("\nIf you haven't set up Neo4j yet, run: beam brain setup-neo4j")
        return 1

    if not facts:
        print(f"No facts found for query: {query!r}")
        return 0

    print(f"\nFound {len(facts)} fact(s) for query: {query!r}\n")
    for i, fact in enumerate(facts, 1):
        print(f"  {i}. {fact}")
    return 0


def cmd_brain_platform_ingest(args: Any) -> int:
    """Extract a PersonalityGraph from a file and persist to Neo4j.

    Usage: beam brain platform-ingest <file> [--group-id <id>]
    """
    file_path = getattr(args, "file", None)
    if not file_path:
        print("Usage: beam brain platform-ingest <file>")
        return 1

    path = Path(file_path)
    if not path.exists():
        print(f"Error: file not found: {file_path}")
        return 1

    group_id = getattr(args, "group_id", "default_user")

    # Read the file
    try:
        text = path.read_text()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 1

    # Extract + persist
    try:
        from brain_platform.services.local_graph_store import LocalGraphStore
        from brain_platform.services.local_graph_writer import LocalGraphWriter

        store = LocalGraphStore()
        store.initialize()
        try:
            writer = LocalGraphWriter(store)
            result = writer.write_interview_session(
                interview_text=text,
                group_id=group_id,
            )
        finally:
            store.close()
    except Exception as e:
        print(f"Error: {e}")
        print("\nIf you haven't set up Neo4j yet, run: beam brain setup-neo4j")
        return 1

    print(f"\n✓ Persisted to Neo4j (group_id={group_id!r}):")
    print(f"  Nodes created: {result['nodes_created']}")
    print(f"  Edges created: {result['edges_created']}")
    return 0


def cmd_interview_adaptive(args: Any) -> int:
    """Run the LLM-powered adaptive interview (brain_platform)."""
    user_age = getattr(args, "age", 30) or 30
    max_questions = getattr(args, "max_questions", 19) or 19

    try:
        from brain_platform.interview_orchestrator import AdaptiveInterviewOrchestrator
        from brain_platform.services.llm_adapter import LLMAdapter
    except ImportError as e:
        print(f"Error: brain_platform not available: {e}")
        return 1

    print("\n" + "=" * 60)
    print("  Adaptive Interview (LLM-powered)")
    print("=" * 60)
    print()
    print("This is the cloud-quality interview path. It uses an LLM to:")
    print("  - Judge the depth of each answer")
    print("  - Generate adaptive follow-ups for shallow answers")
    print("  - Pick the next question based on coverage gaps")
    print()

    llm = LLMAdapter()
    orch = AdaptiveInterviewOrchestrator(
        llm=llm,
        user_age=user_age,
        max_questions=max_questions,
    )

    question = orch.start()
    print(f"Q ({orch.questions_asked[-1][0].dimension}): {question.question}\n")

    while question is not None:
        try:
            answer = input("Your answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInterview interrupted.")
            return 0

        if not answer:
            print("Please provide an answer, or Ctrl+C to exit.")
            continue

        if answer.lower() in ("quit", "exit", "done"):
            print("Ending interview early.")
            return 0

        result = orch.answer(answer)

        # Handle follow-up
        if result.follow_up:
            print(f"\n  Follow-up: {result.follow_up}\n")
            try:
                follow_up_answer = input("Your answer: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nInterview interrupted.")
                return 0
            if follow_up_answer:
                result = orch.answer(follow_up_answer)

        if result.is_complete:
            print("\n✓ Interview complete!")
            transcript = orch.get_transcript()
            print(f"  Questions answered: {transcript['turn_count']}")
            coverage = transcript.get("coverage", {})
            if coverage:
                print(f"  Dimensions covered: {sum(1 for s in coverage.values() if s > 0)}")
            return 0

        question = result.next_question
        if question:
            dim = orch.questions_asked[-1][0].dimension
            print(f"\nQ ({dim}): {question.question}\n")

    return 0


def register_brain_platform_commands(subparsers: Any) -> None:
    """Register brain_platform-specific subcommands on the brain parser.

    Called from hermes_cli/main.py's argparse setup. Adds:
      - brain platform-search
      - brain platform-ingest
      - brain setup-neo4j
      - interview --adaptive

    Accepts either a ``_SubParsersAction`` (the top-level subparsers)
    or any parser with a ``choices`` dict mapping name → subparser.
    """
    # ``subparsers`` is a ``_SubParsersAction`` with a ``.choices``
    # dict mapping command name → subparser.
    choices: dict = getattr(subparsers, "choices", {}) or {}
    brain_parser = choices.get("brain")
    interview_parser = choices.get("interview")

    if brain_parser is not None:
        # Find the brain sub-subparsers action (brain_action dispatch).
        brain_subs_action = _find_subparsers_action(brain_parser, dest="brain_action")
        if brain_subs_action is not None:
            # beam brain platform-search
            p = brain_subs_action.add_parser(
                "platform-search",
                help="Search the Neo4j-backed graph (brain_platform)",
                description="Query the personality knowledge graph stored in Neo4j",
            )
            p.add_argument("query", help="Search query")
            p.add_argument("--num-results", type=int, default=5, help="Max facts to return")
            p.add_argument("--group-id", default="default_user", help="Graphiti group_id")
            p.set_defaults(func=cmd_brain_platform_search)

            # beam brain platform-ingest
            p = brain_subs_action.add_parser(
                "platform-ingest",
                help="Extract + persist a file to Neo4j (brain_platform)",
                description="Run BrainExtractor over a file and persist the graph to Neo4j",
            )
            p.add_argument("file", help="Path to .txt/.md file to ingest")
            p.add_argument("--group-id", default="default_user", help="Graphiti group_id")
            p.set_defaults(func=cmd_brain_platform_ingest)

            # beam brain setup-neo4j
            p = brain_subs_action.add_parser(
                "setup-neo4j",
                help="Configure Neo4j connection (brain_platform)",
                description="Interactive wizard to set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD",
            )
            p.set_defaults(func=cmd_setup_neo4j)

    if interview_parser is not None:
        interview_parser.add_argument(
            "--adaptive",
            action="store_true",
            help="Use the LLM-powered adaptive interview (brain_platform) instead of the default",
        )
        interview_parser.add_argument(
            "--age",
            type=int,
            default=30,
            help="Your age (used for tier-adaptive question selection)",
        )
        interview_parser.add_argument(
            "--max-questions",
            type=int,
            default=19,
            help="Maximum number of questions to ask",
        )


def _find_subparsers_action(parser: Any, dest: str) -> Any:
    """Return the ``_SubParsersAction`` with the given ``dest``, or None.

    Walks ``parser._actions`` looking for a ``_SubParsersAction``
    whose ``dest`` matches. Used to add sub-subparsers to an
    existing parser (e.g. ``brain`` → ``brain platform-search``).
    """
    import argparse

    for action in getattr(parser, "_actions", []):
        if isinstance(action, argparse._SubParsersAction) and action.dest == dest:
            return action
    return None


__all__ = [
    "register_brain_platform_commands",
    "cmd_setup_neo4j",
    "cmd_brain_platform_search",
    "cmd_brain_platform_ingest",
    "cmd_interview_adaptive",
]
