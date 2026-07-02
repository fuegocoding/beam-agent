"""Ingestion orchestrator — the missing glue for the import pipeline.

Faithful port of the cloud's
``beam_mind.pipeline.orchestrator.IngestionOrchestrator`` adapted
for the local single-user runtime:

- No SQLAlchemy: takes a local file path, no DB session
- No S3: reads the file directly
- No Celery: synchronous, single-threaded
- No Redis: no progress tracking (returns the final result)

Pipeline: **parse → chunk → extract → write**

1. **Parse** — use the right parser based on file extension or
   explicit ``--type`` override. All 9 parsers in
   ``brain_platform.pipeline.parsers`` are available (obsidian, txt,
   reddit, code, prompt, instructions, email, journal, docx, pdf).
2. **Chunk** — use the ``SemanticChunker`` to split long documents
   into token-sized chunks (important for large PDFs / vaults).
3. **Extract** — run ``BrainExtractor`` per chunk. Each chunk produces
   a partial ``PersonalityGraph``; we merge them.
4. **Write** — persist the merged graph to Neo4j via
   ``LocalGraphWriter.write_interview_session``.

Public API:

  orch = IngestionOrchestrator(store, llm)
  result = orch.ingest_file(
      file_path="/path/to/essay.pdf",
      source_type=DataSourceType.PDF,  # optional — auto-detect if None
      group_id="user_123",
  )
  # {"documents": int, "chunks": int, "nodes_created": int, "edges_created": int,
  #  "source_type": str, "file": str, "size_bytes": int}
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from brain_platform.models.enums import DataSourceType
from brain_platform.pipeline.chunker import SemanticChunker
from brain_platform.pipeline.parsers import get_parser
from brain_platform.pipeline.parsers.base import ParseResult

logger = logging.getLogger(__name__)


# Map file extensions to DataSourceType for auto-detection
EXTENSION_MAP: dict[str, DataSourceType] = {
    ".md": DataSourceType.OBSIDIAN,  # Markdown defaults to obsidian (most common)
    ".markdown": DataSourceType.OBSIDIAN,
    ".txt": DataSourceType.TXT,
    ".text": DataSourceType.TXT,
    ".py": DataSourceType.CODE,
    ".js": DataSourceType.CODE,
    ".ts": DataSourceType.CODE,
    ".tsx": DataSourceType.CODE,
    ".jsx": DataSourceType.CODE,
    ".rs": DataSourceType.CODE,
    ".go": DataSourceType.CODE,
    ".java": DataSourceType.CODE,
    ".rb": DataSourceType.CODE,
    ".c": DataSourceType.CODE,
    ".cpp": DataSourceType.CODE,
    ".h": DataSourceType.CODE,
    ".hpp": DataSourceType.CODE,
    ".cs": DataSourceType.CODE,
    ".swift": DataSourceType.CODE,
    ".kt": DataSourceType.CODE,
    ".scala": DataSourceType.CODE,
    ".sh": DataSourceType.CODE,
    ".bash": DataSourceType.CODE,
    ".zsh": DataSourceType.CODE,
    ".eml": DataSourceType.EMAIL,
    ".mbox": DataSourceType.EMAIL,
    ".pdf": DataSourceType.PDF,
    ".docx": DataSourceType.DOCX,
    ".json": DataSourceType.REDDIT,  # Most .json in this context is reddit exports
}


def detect_source_type(file_path: str) -> DataSourceType:
    """Auto-detect the source type from a file path.

    Uses the file extension to map to a DataSourceType. Falls back
    to TXT for unknown extensions. The caller can override with
    an explicit ``source_type`` argument.
    """
    suffix = Path(file_path).suffix.lower()
    return EXTENSION_MAP.get(suffix, DataSourceType.TXT)


class IngestionOrchestrator:
    """Orchestrates the full ingestion pipeline: parse → chunk → extract → write.

    Mirrors the cloud's IngestionOrchestrator but exposes a sync API
    and operates on local files (no S3, no Celery, no PostgreSQL).
    """

    def __init__(self, store: Any, llm: Any):
        """Args:
            store: A :class:`LocalGraphStore` (for writing the graph).
            llm: An :class:`LLMAdapter` (for the BrainExtractor).
        """
        self._store = store
        self._llm = llm
        self._chunker = SemanticChunker()

    def ingest_file(
        self,
        file_path: str,
        group_id: str,
        source_type: Optional[DataSourceType] = None,
    ) -> dict:
        """Ingest a single file into the brain.

        Args:
            file_path: Path to the file (PDF, DOCX, MD, TXT, etc.).
            group_id: Graphiti group_id for the target graph partition.
            source_type: Explicit source type override. If None, auto-detect
                from the file extension.

        Returns:
            Summary dict with documents/chunks/nodes/edges counts and
            the detected source type.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # ── 1. Detect source type ──
        if source_type is None:
            source_type = detect_source_type(str(path))
        logger.info("Ingesting %s as %s", path.name, source_type.value)

        # ── 2. Read file bytes ──
        file_data = path.read_bytes()
        size_bytes = len(file_data)
        logger.info("Read %d bytes from %s", size_bytes, path.name)

        # ── 3. Parse ──
        try:
            parser = get_parser(source_type)
            parse_results: List[ParseResult] = parser.parse(file_data, path.name)
        except (ValueError, ImportError) as e:
            logger.warning("Parser failed for %s: %s — falling back to TXT", source_type.value, e)
            parser = get_parser(DataSourceType.TXT)
            parse_results = parser.parse(file_data, path.name)
            source_type = DataSourceType.TXT
        logger.info("Parsed %d documents from %s", len(parse_results), source_type.value)

        # ── 4. Chunk ──
        all_chunks = []
        for pr in parse_results:
            chunks = self._chunker.chunk(pr)
            all_chunks.extend(chunks)
        logger.info("Created %d chunks", len(all_chunks))

        # ── 5. Extract + write per chunk ──
        from brain_platform.services.local_graph_writer import LocalGraphWriter
        from brain_platform.extractor.brain_extractor import BrainExtractor

        writer = LocalGraphWriter(self._store)
        total_nodes = 0
        total_edges = 0

        if not all_chunks:
            logger.warning("No chunks produced from %s", path.name)
            return {
                "documents": len(parse_results),
                "chunks": 0,
                "nodes_created": 0,
                "edges_created": 0,
                "source_type": source_type.value,
                "file": str(path),
                "size_bytes": size_bytes,
            }

        # Build the LLM client that BrainExtractor expects
        # (Graphiti-compatible; brain_extractor calls generate_response on it)
        graphiti_client = self._store.client
        llm_client = getattr(graphiti_client, "llm_client", self._llm)

        extractor = BrainExtractor()
        for i, chunk in enumerate(all_chunks):
            logger.info("Processing chunk %d/%d (%d chars)", i + 1, len(all_chunks), len(chunk.text))
            try:
                graph = extractor.extract(
                    interview_text=chunk.text,
                    llm_client=llm_client,
                )
                result = writer.write(graph=graph, group_id=group_id)
                total_nodes += result.get("nodes_created", 0)
                total_edges += result.get("edges_created", 0)
            except Exception:
                logger.exception("Failed processing chunk %d", i)
                continue

        logger.info(
            "Ingestion complete: %d docs, %d chunks, %d nodes, %d edges",
            len(parse_results), len(all_chunks), total_nodes, total_edges,
        )

        return {
            "documents": len(parse_results),
            "chunks": len(all_chunks),
            "nodes_created": total_nodes,
            "edges_created": total_edges,
            "source_type": source_type.value,
            "file": str(path),
            "size_bytes": size_bytes,
        }

    def ingest_text(
        self,
        text: str,
        group_id: str,
        source_type: DataSourceType = DataSourceType.TXT,
        source_description: str = "",
    ) -> dict:
        """Ingest raw text (no file, no parser) into the brain.

        Useful for piping content from stdin, scraping web pages, or
        programmatic ingestion. Wraps the text in a minimal ParseResult
        and runs the same chunk → extract → write pipeline.
        """
        from brain_platform.pipeline.parsers.base import ParseResult

        parse_result = ParseResult(
            text=text,
            metadata={
                "source": source_description or "raw_text",
                "type": source_type.value,
            },
        )
        chunks = self._chunker.chunk(parse_result)

        from brain_platform.services.local_graph_writer import LocalGraphWriter
        from brain_platform.extractor.brain_extractor import BrainExtractor

        writer = LocalGraphWriter(self._store)
        graphiti_client = self._store.client
        llm_client = getattr(graphiti_client, "llm_client", self._llm)

        extractor = BrainExtractor()
        total_nodes = 0
        total_edges = 0
        for i, chunk in enumerate(chunks):
            try:
                graph = extractor.extract(
                    interview_text=chunk.text,
                    llm_client=llm_client,
                )
                result = writer.write(graph=graph, group_id=group_id)
                total_nodes += result.get("nodes_created", 0)
                total_edges += result.get("edges_created", 0)
            except Exception:
                logger.exception("Failed processing chunk %d", i)
                continue

        return {
            "documents": 1,
            "chunks": len(chunks),
            "nodes_created": total_nodes,
            "edges_created": total_edges,
            "source_type": source_type.value,
        }


__all__ = [
    "IngestionOrchestrator",
    "detect_source_type",
    "EXTENSION_MAP",
]
