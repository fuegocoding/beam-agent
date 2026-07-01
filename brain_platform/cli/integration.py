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

import json
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

    # Check if graphiti-core is installed. If not, fail fast with a
    # clear install instruction — the user will hit a confusing
    # "No module named 'graphiti_core'" error otherwise.
    try:
        import graphiti_core  # noqa: F401
    except ImportError:
        print("⚠  graphiti-core is not installed in this Python environment.")
        print()
        print("   To use brain_platform with Neo4j, you need to install the")
        print("   graphiti-core extra. Pick ONE of:")
        print()
        print("     pip install 'beam-agent[brain-platform-graph]'")
        print("     pip install 'beam-agent[brain-platform]'   # full meta-extra")
        print("     pip install 'beam-agent[all]'              # everything")
        print()
        print("   Then re-run this command.")
        print()
        return 1

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


def _get_default_group_id() -> str:
    """Return the active brain name as the default group_id.

    The Neo4j group_id is per-user (or per-brain), so searching without
    specifying a group_id should default to whatever brain the user
    currently has active. This way ``beam brain platform-search`` just
    works for whatever the user is currently using.
    """
    try:
        from brain.paths import get_active_brain_name
        return get_active_brain_name()
    except Exception:
        return "default_user"


def cmd_brain_platform_search(args: Any) -> int:
    """Search the Neo4j-backed graph for relevant facts.

    Usage: beam brain platform-search <query> [--num-results N] [--group-id ID]

    Defaults to searching the active brain's graph. Use --group-id to
    search a different Neo4j partition.
    """
    query = getattr(args, "query", None)
    if not query:
        print("Usage: beam brain platform-search <query>")
        return 1

    num_results = getattr(args, "num_results", 5)
    # Default to the active brain's group_id when the user didn't
    # explicitly pass --group-id. The argparse default is None (not
    # "default_user") so we can detect "not specified" here.
    group_id = getattr(args, "group_id", None) or _get_default_group_id()

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
        print(f"No facts found for query: {query!r} (group_id={group_id!r})")
        print()
        print("This usually means the graph has no data for this group yet.")
        print("The marketplace brain is stored as a JSON file at:")
        try:
            from brain.paths import get_active_brain_graph_path
            path = get_active_brain_graph_path()
            if path.exists():
                print(f"  {path}")
        except Exception:
            pass
        print()
        print("To search the active brain via Neo4j, first ingest it:")
        print(f"  beam brain platform-ingest {path if path.exists() else '<brain.json>'}")
        print()
        print("Or run a new interview to build the graph:")
        print("  beam interview --adaptive")
        return 0

    print(f"\nFound {len(facts)} fact(s) for query: {query!r} (group_id={group_id!r})\n")
    for i, fact in enumerate(facts, 1):
        print(f"  {i}. {fact}")
    return 0


def _is_brain_file_schema_json(path: Path) -> bool:
    """True if the file is a BrainFileSchema JSON (the marketplace brain format).

    These files have the shape:
        {
          "metadata": {"schema_version": 2, ...},
          "personality_profile": {...},
          "knowledge_graph": {"nodes": [...], "edges": [...]},
          ...
        }

    Re-extracting them with BrainExtractor would just produce noise
    (the LLM can't extract from a JSON dump). Instead, we ingest them
    directly by writing the existing nodes/edges to Neo4j.
    """
    if path.suffix.lower() != ".json":
        return False
    try:
        data = json.loads(path.read_text())
        return (
            isinstance(data, dict)
            and "metadata" in data
            and "knowledge_graph" in data
            and isinstance(data["knowledge_graph"], dict)
            and "nodes" in data["knowledge_graph"]
        )
    except Exception:
        return False


def _ingest_brain_file_json(path: Path, group_id: str) -> dict:
    """Ingest a pre-built BrainFileSchema JSON directly to Neo4j.

    Reads the JSON, extracts the nodes/edges from knowledge_graph,
    and writes them to Neo4j via LocalGraphWriter. This is the
    direct-write path (no LLM extraction) — the JSON already has
    structured nodes/edges that the marketplace brain shipped.
    """
    from brain_platform.services.local_graph_store import LocalGraphStore
    from brain_platform.services.local_graph_writer import LocalGraphWriter
    from brain_platform.pipeline.brain_file.schema import (
        BrainFileSchema, GraphNode, GraphEdge, GraphCluster,
    )

    data = json.loads(path.read_text())
    brain_file = BrainFileSchema.model_validate(data)
    kg = brain_file.knowledge_graph

    # Convert Pydantic models to the dict shape LocalGraphWriter expects
    nodes = [n.model_dump() if hasattr(n, "model_dump") else n for n in kg.nodes]
    edges = [e.model_dump() if hasattr(e, "model_dump") else e for e in kg.edges]

    store = LocalGraphStore()
    store.initialize()
    try:
        writer = LocalGraphWriter(store)
        # The marketplace brain JSON has typed nodes (PersonalityTrait,
        # Belief, etc.) with real labels, summaries, and attributes.
        # We need to write them directly — building a PersonalityGraph
        # stub from label counts would throw away all the content.
        from graphiti_core.nodes import EntityNode
        from graphiti_core.edges import EntityEdge
        from datetime import datetime, timezone

        client = store.client
        driver = client.driver
        embedder = client.embedder
        loop = store._loop

        nodes_created = 0
        nodes_by_id: dict[str, str] = {}
        now = datetime.now(timezone.utc)

        # Create EntityNodes for each marketplace node
        for n in nodes:
            node_id = n.get("id", "")
            label = n.get("label", "").strip()
            if not label:
                continue
            ntype = n.get("type", "Entity")
            labels_list = [ntype, "Entity"] if ntype != "Entity" else ["Entity"]
            summary = n.get("summary", "")
            attributes = n.get("attributes", {}) or {}

            node = EntityNode(
                name=label,
                group_id=group_id,
                labels=labels_list,
                summary=summary,
                attributes=attributes,
            )
            loop.run_until_complete(node.generate_name_embedding(embedder))
            loop.run_until_complete(node.save(driver))
            nodes_by_id[node_id] = node.uuid
            nodes_created += 1

        # Create EntityEdges
        edges_created = 0
        skipped_edges = 0
        for e in edges:
            src_id = e.get("source", "")
            tgt_id = e.get("target", "")
            src_uuid = nodes_by_id.get(src_id)
            tgt_uuid = nodes_by_id.get(tgt_id)
            if not src_uuid or not tgt_uuid:
                skipped_edges += 1
                continue
            edge = EntityEdge(
                source_node_uuid=src_uuid,
                target_node_uuid=tgt_uuid,
                name=e.get("relation", "INFORMS"),
                group_id=group_id,
                fact=e.get("fact", ""),
                created_at=now,
                valid_at=now,
            )
            loop.run_until_complete(edge.generate_embedding(embedder))
            loop.run_until_complete(edge.save(driver))
            edges_created += 1

        # Also create THE_USER as a hub node so the brain is searchable
        # and connect it to the personality-trait nodes
        from brain_platform.services.local_graph_writer import LocalGraphWriter as _Writer
        # Build a stub personality graph with just the user_summary to
        # trigger THE_USER creation + hub edges
        from brain_platform.pipeline.brain_schema import PersonalityGraph
        stub_graph = PersonalityGraph(user_summary="")
        hub_result = writer.write(graph=stub_graph, group_id=group_id)
    finally:
        store.close()

    return {
        "documents": 1,
        "chunks": 1,
        "nodes_created": nodes_created,
        "edges_created": edges_created,
        "source_type": "brain_file_json",
        "file": str(path),
        "size_bytes": path.stat().st_size,
    }


def cmd_brain_platform_ingest(args: Any) -> int:
    """Ingest a file into the brain (parse → chunk → extract → write).

    Usage: beam brain platform-ingest <file> [--type TYPE] [--group-id ID]

    Auto-detects the source type from the file extension (.md → obsidian,
    .pdf → PDF, .py → code, .eml → email, etc.). Use --type to override.

    Supported types: obsidian, pdf, docx, txt, code, prompt, instructions,
    email, journal, reddit

    If the file is a BrainFileSchema JSON (the marketplace brain format
    with pre-extracted nodes/edges), it skips the LLM extraction and
    writes the existing graph directly to Neo4j.
    """
    file_path = getattr(args, "file", None)
    if not file_path:
        print("Usage: beam brain platform-ingest <file> [--type TYPE]")
        return 1

    path = Path(file_path)
    if not path.exists():
        print(f"Error: file not found: {file_path}")
        return 1

    # Default to the active brain's group_id so the command just works
    # against whatever brain the user is currently using.
    group_id = getattr(args, "group_id", None) or _get_default_group_id()
    explicit_type = getattr(args, "type", None)

    # Special case: if the file is a BrainFileSchema JSON (the marketplace
    # brain format), skip LLM extraction and write the existing nodes/edges
    # directly to Neo4j. Re-extracting a JSON dump produces noise.
    if _is_brain_file_schema_json(path):
        try:
            print(f"Ingesting {path.name} (detected: BrainFileSchema JSON)")
            result = _ingest_brain_file_json(path, group_id)
        except Exception as e:
            print(f"Error: {e}")
            print("\nIf you haven't set up Neo4j yet, run: beam brain setup-neo4j")
            return 1

        print(f"\n✓ Ingested {path.name} → Neo4j (group_id={group_id!r}):")
        print(f"  Source type:  {result['source_type']}")
        print(f"  Documents:    {result['documents']}")
        print(f"  Chunks:       {result['chunks']}")
        print(f"  Nodes:        {result['nodes_created']}")
        print(f"  Edges:        {result['edges_created']}")
        return 0

    source_type = None
    if explicit_type:
        from brain_platform.models.enums import DataSourceType
        try:
            source_type = DataSourceType(explicit_type)
        except ValueError:
            valid = ", ".join(t.value for t in DataSourceType)
            print(f"Error: unknown type {explicit_type!r}. Valid: {valid}")
            return 1

    try:
        from brain_platform.services.local_graph_store import LocalGraphStore
        from brain_platform.services.llm_adapter import LLMAdapter
        from brain_platform.pipeline.ingestion_orchestrator import (
            IngestionOrchestrator, detect_source_type,
        )

        store = LocalGraphStore()
        store.initialize()
        try:
            llm = LLMAdapter()
            orch = IngestionOrchestrator(store=store, llm=llm)
            detected = source_type or detect_source_type(str(path))
            print(f"Ingesting {path.name} (detected: {detected.value})")
            result = orch.ingest_file(
                file_path=str(path),
                group_id=group_id,
                source_type=source_type,
            )
        finally:
            store.close()
    except Exception as e:
        print(f"Error: {e}")
        print("\nIf you haven't set up Neo4j yet, run: beam brain setup-neo4j")
        return 1

    print(f"\n✓ Ingested {path.name} → Neo4j (group_id={group_id!r}):")
    print(f"  Source type:  {result['source_type']}")
    print(f"  Documents:    {result['documents']}")
    print(f"  Chunks:       {result['chunks']}")
    print(f"  Nodes:        {result['nodes_created']}")
    print(f"  Edges:        {result['edges_created']}")
    return 0


def cmd_brain_platform_generate(args: Any) -> int:
    """Generate a brain file (JSON) from the Neo4j graph.

    Usage: beam brain platform-generate <output.json> [--group-id ID] [--raw-texts-file FILE]
    """
    output_path = getattr(args, "output", None)
    if not output_path:
        print("Usage: beam brain platform-generate <output.json>")
        return 1

    # Default to the active brain's group_id so the command just works
    # against whatever brain the user is currently using.
    group_id = getattr(args, "group_id", None) or _get_default_group_id()
    raw_texts_file = getattr(args, "raw_texts_file", None)

    raw_texts = None
    if raw_texts_file:
        path = Path(raw_texts_file)
        if not path.exists():
            print(f"Error: file not found: {raw_texts_file}")
            return 1
        raw_texts = [path.read_text()]

    try:
        from brain_platform.services.local_graph_store import LocalGraphStore
        from brain_platform.services.llm_adapter import LLMAdapter
        from brain_platform.pipeline.brain_file.generator import BrainFileGenerator

        store = LocalGraphStore()
        store.initialize()
        try:
            llm = LLMAdapter()
            generator = BrainFileGenerator(store=store, llm=llm)
            result = generator.generate_to_file(
                group_id=group_id,
                output_path=output_path,
                raw_texts=raw_texts,
            )
        finally:
            store.close()
    except Exception as e:
        print(f"Error: {e}")
        print("\nIf you haven't set up Neo4j yet, run: beam brain setup-neo4j")
        return 1

    print(f"\n✓ Brain file generated: {output_path}")
    print(f"  Size: {result['size_bytes']} bytes")
    print(f"  Hash: {result['content_hash']}")
    print(f"  Nodes: {result['node_count']}")
    print(f"  Edges: {result['edge_count']}")
    print(f"  Domains: {result['domain_count']}")
    return 0


def cmd_brain_platform_export(args: Any) -> int:
    """Generate a brain file and export to claude/jsonld/obsidian format.

    Usage: beam brain platform-export <output> --format <format>
    """
    output_path = getattr(args, "output", None)
    if not output_path:
        print("Usage: beam brain platform-export <output> --format <format>")
        return 1

    fmt = getattr(args, "format", "jsonld")
    # Default to the active brain's group_id so the command just works
    # against whatever brain the user is currently using.
    group_id = getattr(args, "group_id", None) or _get_default_group_id()

    try:
        from brain_platform.services.local_graph_store import LocalGraphStore
        from brain_platform.services.llm_adapter import LLMAdapter
        from brain_platform.pipeline.brain_file.generator import BrainFileGenerator
        from brain_platform.pipeline.brain_file.exporters import (
            export_claude, export_jsonld, export_obsidian,
        )

        store = LocalGraphStore()
        store.initialize()
        try:
            llm = LLMAdapter()
            generator = BrainFileGenerator(store=store, llm=llm)
            brain_file = generator.generate(group_id=group_id)
        finally:
            store.close()

        # Export
        if fmt == "claude":
            exported = export_claude(brain_file)
            import json as _json
            with open(output_path, "w") as f:
                _json.dump(exported, f, indent=2)
            print(f"\n✓ Exported as Claude Projects format: {output_path}")
            print(f"  system_prompt: {len(exported.get('system_prompt', ''))} chars")
            print(f"  knowledge_files: {len(exported.get('knowledge_files', []))} files")
        elif fmt == "jsonld":
            exported = export_jsonld(brain_file)
            with open(output_path, "wb") as f:
                f.write(exported)
            print(f"\n✓ Exported as JSON-LD: {output_path} ({len(exported)} bytes)")
        elif fmt == "obsidian":
            exported = export_obsidian(brain_file)
            with open(output_path, "wb") as f:
                f.write(exported)
            print(f"\n✓ Exported as Obsidian vault: {output_path} ({len(exported)} bytes)")
    except Exception as e:
        print(f"Error: {e}")
        print("\nIf you haven't set up Neo4j yet, run: beam brain setup-neo4j")
        return 1

    return 0


def cmd_brain_platform_deepen(args: Any) -> int:
    """Analyze brain gaps and generate probe questions.

    Usage: beam brain platform-deepen [--group-id ID] [--covered-questions FILE]

    Reads the latest extracted graph from Neo4j, analyzes dimension
    coverage, and asks the LLM to generate targeted probe questions
    for the thin dimensions.
    """
    # Default to the active brain's group_id so the command just works
    # against whatever brain the user is currently using.
    group_id = getattr(args, "group_id", None) or _get_default_group_id()
    covered_questions_file = getattr(args, "covered_questions", None)

    covered_questions = []
    if covered_questions_file:
        path = Path(covered_questions_file)
        if path.exists():
            covered_questions = [
                line.strip() for line in path.read_text().splitlines()
                if line.strip()
            ]

    try:
        from brain_platform.services.local_graph_store import LocalGraphStore
        from brain_platform.services.llm_adapter import LLMAdapter
        from brain_platform.pipeline.brain_file.graph_reader import GraphReader
        from brain_platform.pipeline.interview.deepen import (
            analyze_brain_gaps,
            generate_probe_questions,
        )
        from brain_platform.pipeline.brain_schema import PersonalityGraph

        store = LocalGraphStore()
        store.initialize()
        try:
            client = store.client
            # Read the typed graph nodes and build a minimal PersonalityGraph
            # for the gap analysis (uses node_summaries + typed_nodes).
            reader = GraphReader(store)
            graph_data = reader.read_all(group_id)
        finally:
            store.close()

        # Build a minimal PersonalityGraph from the typed nodes
        typed_nodes = graph_data.nodes
        # We can't reconstruct the full PersonalityGraph from graph
        # reader output (it only gives labels + summaries). Use the
        # node counts per label as a proxy.
        from collections import Counter
        label_counts = Counter(n.type for n in typed_nodes)

        # Build a PersonalityGraph stub for analyze_brain_gaps
        from brain_platform.pipeline.brain_schema import (
            TraitNode, BeliefNode, ValueNode, BoundaryNode,
            LifeEventNode, MemoryNode, PatternNode, SocialNode,
            ExpertiseNode, StyleNode, PersonNode,
        )

        def _mk_stub(cls, count):
            return [cls(name=f"placeholder_{i}", summary="") for i in range(count)]

        graph = PersonalityGraph(
            user_summary="",
            traits=_mk_stub(TraitNode, label_counts.get("PersonalityTrait", 0)),
            beliefs=_mk_stub(BeliefNode, label_counts.get("Belief", 0)),
            values=_mk_stub(ValueNode, label_counts.get("Value", 0)),
            boundaries=_mk_stub(BoundaryNode, label_counts.get("Boundary", 0)),
            life_events=_mk_stub(LifeEventNode, label_counts.get("LifeEvent", 0)),
            memories=_mk_stub(MemoryNode, label_counts.get("EpisodicMemory", 0)),
            patterns=_mk_stub(PatternNode, label_counts.get("CognitivePattern", 0)),
            social=_mk_stub(SocialNode, label_counts.get("SocialPattern", 0)),
            expertise=_mk_stub(ExpertiseNode, label_counts.get("KnowledgeDomain", 0)),
            style=_mk_stub(StyleNode, label_counts.get("StyleProfile", 0)),
            people=_mk_stub(PersonNode, label_counts.get("Person", 0)),
        )

        result = analyze_brain_gaps(graph)

        # Generate probe questions if there are gaps
        if result.gaps:
            llm = LLMAdapter()
            probes = generate_probe_questions(
                gaps=result.gaps,
                graph=graph,
                covered_questions=covered_questions,
                llm_client=llm,
            )
            result.probe_questions = probes
    except Exception as e:
        print(f"Error: {e}")
        print("\nIf you haven't set up Neo4j yet, run: beam brain setup-neo4j")
        return 1

    print(f"\n{result.summary}\n")
    if result.gaps:
        print(f"Gaps ({len(result.gaps)}):")
        for g in result.gaps[:10]:
            print(f"  - {g.dimension}: {g.current_count}/{g.target_count} (gap: {g.gap})")
    if result.probe_questions:
        print(f"\nProbe questions ({len(result.probe_questions)}):")
        for q in result.probe_questions[:5]:
            print(f"  [{q.dimension}] {q.question}")
            print(f"    Why: {q.why}")
    return 0


def _progress_bar(current: int, total: int, width: int = 20) -> str:
    """Render a simple ASCII progress bar.

    Returns a string like ``[██████████░░░░░░░░░░] 50%``. Used by the
    adaptive interview to show users how far along they are.
    """
    if total <= 0:
        return ""
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * current / total)
    return f"[{bar}] {pct}%"


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
    print(f"Up to ~{max_questions} core questions. Each may get a follow-up if your")
    print("answer is brief. Total time: ~15-25 minutes.")
    print()

    llm = LLMAdapter()
    orch = AdaptiveInterviewOrchestrator(
        llm=llm,
        user_age=user_age,
        max_questions=max_questions,
    )

    question = orch.start()
    core_q_num = 1
    print(f"\n{_progress_bar(core_q_num, max_questions)}  Question {core_q_num} of ~{max_questions}")
    print(f"[{orch.questions_asked[-1][0].dimension}] {question.question}\n")

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
            print(f"\n  ↳ Follow-up: {result.follow_up}\n")
            try:
                follow_up_answer = input("Your answer: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nInterview interrupted.")
                return 0
            if follow_up_answer:
                result = orch.answer(follow_up_answer)
            # If the follow-up is still being asked (user gave no answer),
            # the next_question will be the same core question — don't advance counter
            if result.next_question and result.next_question.id == question.id:
                # Still on the same core question
                print(f"\n{_progress_bar(core_q_num, max_questions)}  Question {core_q_num} of ~{max_questions}")
                print(f"[{orch.questions_asked[-1][0].dimension}] {result.next_question.question}\n")
                question = result.next_question
                continue

        if result.is_complete:
            print(f"\n{_progress_bar(core_q_num, max_questions)}  Interview complete!")
            transcript = orch.get_transcript()
            print(f"  Core questions answered: {len(transcript['questions_asked'])}")
            print(f"  Total turns (with follow-ups): {transcript['turn_count']}")
            coverage = transcript.get("coverage", {})
            if coverage:
                print(f"  Dimensions covered: {sum(1 for s in coverage.values() if s > 0)}")
            return 0

        question = result.next_question
        if question:
            core_q_num += 1
            dim = orch.questions_asked[-1][0].dimension
            print(f"\n{_progress_bar(core_q_num, max_questions)}  Question {core_q_num} of ~{max_questions}")
            print(f"[{dim}] {question.question}\n")

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
            p.add_argument("--group-id", default=None, help="Graphiti group_id (default: active brain)")
            p.set_defaults(func=cmd_brain_platform_search)

            # beam brain platform-ingest
            p = brain_subs_action.add_parser(
                "platform-ingest",
                help="Ingest a file (parse → chunk → extract → write to Neo4j)",
                description="Run the full ingestion pipeline: detect file type, parse, "
                            "chunk, extract, and persist to Neo4j. Supports PDF, DOCX, "
                            "Obsidian markdown, code, email, journal, reddit exports, etc.",
            )
            p.add_argument("file", help="Path to file to ingest (.pdf/.docx/.md/.txt/.py/.eml/...)")
            p.add_argument("--type", help="Override detected source type (obsidian/pdf/docx/txt/code/prompt/instructions/email/journal/reddit)")
            p.add_argument("--group-id", default=None, help="Graphiti group_id (default: active brain)")
            p.set_defaults(func=cmd_brain_platform_ingest)

            # beam brain platform-generate — assemble the brain file
            p = brain_subs_action.add_parser(
                "platform-generate",
                help="Generate the brain file (JSON) from Neo4j (brain_platform)",
                description="Assemble a BrainFileSchema from the Neo4j graph and write to disk",
            )
            p.add_argument("output", help="Path to write the brain file (.json)")
            p.add_argument("--group-id", default=None, help="Graphiti group_id (default: active brain)")
            p.add_argument("--raw-texts-file", help="Optional .txt file of raw texts for style analysis")
            p.set_defaults(func=cmd_brain_platform_generate)

            # beam brain platform-export — generate + export to format
            p = brain_subs_action.add_parser(
                "platform-export",
                help="Generate + export the brain file (claude/jsonld/obsidian)",
                description="Generate a brain file and export it in the chosen format",
            )
            p.add_argument("output", help="Path to write the export")
            p.add_argument("--format", choices=["claude", "jsonld", "obsidian"], default="jsonld",
                          help="Export format (default: jsonld)")
            p.add_argument("--group-id", default=None, help="Graphiti group_id (default: active brain)")
            p.set_defaults(func=cmd_brain_platform_export)

            # beam brain platform-deepen — analyze gaps + generate probes
            p = brain_subs_action.add_parser(
                "platform-deepen",
                help="Analyze brain gaps and generate probe questions",
                description="Find thin dimensions and ask the LLM for probe questions to fill them",
            )
            p.add_argument("--group-id", default=None, help="Graphiti group_id (default: active brain)")
            p.add_argument("--covered-questions", help="Optional .txt file of already-asked question IDs")
            p.set_defaults(func=cmd_brain_platform_deepen)

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
