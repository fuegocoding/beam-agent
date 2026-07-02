"""Brain File Generator — orchestrates full brain file creation.

Faithful port of the cloud's
``beam_mind.pipeline.brain_file.generator.BrainFileGenerator``
adapted for the local single-user runtime:

- No SQLAlchemy: passes ``group_id`` instead of loading a ``BrainFile``
  row from PostgreSQL
- No S3: writes to a local file path (or returns the JSON bytes for
  the caller to persist)
- Sync facade over Graphiti's async client (same pattern as
  ``LocalGraphWriter``)

Public API:

  generator = BrainFileGenerator(store, llm)
  brain_file = generator.generate(group_id="user_123", output_path="/path/to/brain.json")
  # brain_file is a BrainFileSchema; output_path is a JSON file
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from brain_platform.pipeline.brain_file.graph_reader import GraphReader
from brain_platform.pipeline.brain_file.personality_extractor import PersonalityExtractor
from brain_platform.pipeline.brain_file.schema import (
    BrainFileMetadata,
    BrainFileSchema,
    KnowledgeDomain,
    KnowledgeGraph,
    SourceManifestEntry,
    StyleEmbedding,
    WritingStyle,
)
from brain_platform.pipeline.brain_file.style_analyzer import StyleAnalyzer
from brain_platform.pipeline.brain_file.style_embedder import StyleEmbedder

logger = logging.getLogger(__name__)


class BrainFileGenerator:
    """Generates a complete BrainFileSchema for one user.

    Mirrors the cloud's BrainFileGenerator but exposes a sync API
    and writes to a local file (no S3).
    """

    def __init__(self, store: Any, llm: Any):
        """Args:
            store: A :class:`LocalGraphStore` (or any object with
                ``.client`` and ``.group_id_for_user``).
            llm: An :class:`LLMAdapter` (or any object exposing
                ``generate_response(messages=..., task=...)``).
        """
        self._store = store
        self._llm = llm
        self._graph_reader = GraphReader(store)
        self._style_analyzer = StyleAnalyzer()
        self._style_embedder = StyleEmbedder()
        self._personality_extractor = PersonalityExtractor(llm)

    def generate(
        self,
        group_id: str,
        raw_texts: Optional[List[str]] = None,
        source_manifest: Optional[List[SourceManifestEntry]] = None,
    ) -> BrainFileSchema:
        """Generate a BrainFileSchema for the given group.

        Args:
            group_id: Graphiti group_id for this user's graph partition.
            raw_texts: Optional list of raw texts for style analysis.
                If not provided, style analysis is skipped (empty WritingStyle).
            source_manifest: Optional list of source manifest entries.
                If not provided, returns an empty manifest.

        Returns:
            BrainFileSchema with all dimensions populated.
        """
        # ── 1. Read knowledge graph ──
        graph_data = self._graph_reader.read_all(group_id)

        # ── 2. Compute style metrics (if texts provided) ──
        # Always provide a WritingStyle — the schema requires it
        # (defaults to empty when no texts are available).
        writing_style = (
            self._style_analyzer.analyze(raw_texts)
            if raw_texts
            else WritingStyle()
        )

        # ── 3. Compute style embedding (optional — may not have torch) ──
        # Always provide a StyleEmbedding — the schema requires it.
        style_embedding = (
            self._style_embedder.compute(raw_texts)
            if raw_texts
            else StyleEmbedding()
        )

        # ── 4. Extract personality profile (typed nodes + LLM scores) ──
        personality = self._personality_extractor.extract(
            node_summaries=graph_data.node_summaries,
            edge_facts=graph_data.edge_facts,
            community_summaries=graph_data.community_summaries,
            typed_nodes=graph_data.nodes,
        )

        # ── 5. Build knowledge domains from communities ──
        knowledge_domains = [
            KnowledgeDomain(
                topic=cluster.name,
                confidence=0.8,
                source_count=len(cluster.member_node_ids),
                key_entities=[
                    n.label
                    for n in graph_data.nodes
                    if n.id in cluster.member_node_ids
                ][:5],
                community_summary=cluster.summary,
            )
            for cluster in graph_data.clusters
        ]

        # ── 6. Assemble brain file ──
        now = datetime.now(timezone.utc)
        manifest = source_manifest or []
        brain_file_schema = BrainFileSchema(
            metadata=BrainFileMetadata(
                created_at=now,
                updated_at=now,
                user_id=group_id,
                source_count=sum(e.record_count for e in manifest),
                graphiti_group_id=group_id,
            ),
            knowledge_domains=knowledge_domains,
            personality_profile=personality,
            writing_style=writing_style,
            style_embedding=style_embedding,
            knowledge_graph=KnowledgeGraph(
                nodes=graph_data.nodes,
                edges=graph_data.edges,
                clusters=graph_data.clusters,
            ),
            source_manifest=manifest,
        )

        logger.info(
            "Brain file generated: %d nodes, %d edges, %d domains",
            len(graph_data.nodes),
            len(graph_data.edges),
            len(knowledge_domains),
        )

        return brain_file_schema

    def generate_to_file(
        self,
        group_id: str,
        output_path: str,
        raw_texts: Optional[List[str]] = None,
        source_manifest: Optional[List[SourceManifestEntry]] = None,
    ) -> dict:
        """Generate a brain file and write it to disk as JSON.

        Returns:
            {"path": str, "size_bytes": int, "content_hash": str,
             "node_count": int, "edge_count": int, "domain_count": int}
        """
        brain_file = self.generate(
            group_id=group_id,
            raw_texts=raw_texts,
            source_manifest=source_manifest,
        )
        jsonld = brain_file.to_jsonld()
        content = json.dumps(jsonld, indent=2, default=str).encode("utf-8")
        content_hash = hashlib.sha256(content).hexdigest()[:16]

        with open(output_path, "wb") as f:
            f.write(content)

        return {
            "path": output_path,
            "size_bytes": len(content),
            "content_hash": content_hash,
            "node_count": len(brain_file.knowledge_graph.nodes),
            "edge_count": len(brain_file.knowledge_graph.edges),
            "domain_count": len(brain_file.knowledge_domains),
        }


__all__ = ["BrainFileGenerator"]
