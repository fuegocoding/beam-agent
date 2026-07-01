"""Reads all nodes, edges, and communities from Graphiti for a user.

Faithful port of the cloud's ``beam_mind.pipeline.brain_file.graph_reader.GraphReader``
with the same sync facade pattern used by the rest of brain_platform:
``asyncio.run()`` bridges to Graphiti's async client.

Public API:

  reader = GraphReader(store)  # store is a LocalGraphStore
  data = reader.read_all("user_123")
  # data.nodes, data.edges, data.clusters, etc.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from brain_platform.pipeline.brain_file.schema import GraphCluster, GraphEdge, GraphNode

logger = logging.getLogger(__name__)


def _run(coro: Any, loop: Any) -> Any:
    """Run an async coroutine on the store's persistent event loop.

    Uses ``loop.run_until_complete()`` rather than ``asyncio.run()``
    so the coroutine runs on the same loop that created the Neo4j
    connection pool. This avoids the "Future attached to a different
    loop" error that happens when ``asyncio.run()`` creates a new
    loop each call.

    Falls back to ``asyncio.run()`` when ``loop`` is None or not a
    real event loop (e.g. a MagicMock in tests).
    """
    if loop is None or not isinstance(loop, asyncio.AbstractEventLoop):
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


@dataclass
class GraphData:
    """All graph data for a user, ready for brain file assembly.

    Mirrors the cloud's GraphData dataclass — same fields, same shape.
    """
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    clusters: list[GraphCluster] = field(default_factory=list)
    node_summaries: list[str] = field(default_factory=list)
    edge_facts: list[str] = field(default_factory=list)
    community_summaries: list[str] = field(default_factory=list)


class GraphReader:
    """Reads nodes, edges, and communities from a LocalGraphStore for one user.

    Mirrors the cloud's GraphReader but with a sync API.
    """

    def __init__(self, store: Any):
        """Args:
            store: A :class:`LocalGraphStore` (or any object exposing
                ``.client`` and ``.group_id_for_user``).
        """
        self._store = store

    def read_all(self, group_id: str) -> GraphData:
        """Read all nodes/edges/clusters for ``group_id``.

        Returns an empty GraphData if Graphiti is not available or
        the read fails (matches the cloud's fallback behavior).
        """
        try:
            client = self._store.client
        except RuntimeError:
            logger.warning("Graphiti not available, returning empty graph data")
            return GraphData()

        # Use the store's persistent event loop so the Neo4j connection
        # pool stays valid across multiple sync calls.
        loop = getattr(self._store, "_loop", None)

        data = GraphData()

        try:
            # ── Entity nodes ──
            entity_nodes = _run(
                client.nodes.entity.get_by_group_ids([group_id]),
                loop,
            )
            for node in entity_nodes:
                # Read the node's actual type from its labels (set by
                # entity_types) instead of hardcoding "Entity".
                node_labels = (
                    node.labels
                    if hasattr(node, "labels") and node.labels
                    else []
                )
                node_type = node_labels[0] if node_labels else "Entity"

                data.nodes.append(
                    GraphNode(
                        id=str(node.uuid),
                        type=node_type,
                        label=node.name or "",
                        attributes=(
                            node.attributes
                            if hasattr(node, "attributes") and node.attributes
                            else {}
                        ),
                        summary=node.summary or "",
                        labels=node_labels,
                    )
                )
                if node.summary:
                    data.node_summaries.append(
                        f"[{node_type}] {node.name}: {node.summary}"
                    )

            # ── Community nodes ──
            community_nodes = _run(
                client.nodes.community.get_by_group_ids([group_id]),
                loop,
            )
            for comm in community_nodes:
                member_ids = []
                try:
                    comm_edges = _run(
                        client.edges.community.get_by_group_ids([group_id]),
                        loop,
                    )
                    member_ids = [
                        str(e.target_node_uuid)
                        for e in comm_edges
                        if str(e.source_node_uuid) == str(comm.uuid)
                    ]
                except Exception:
                    pass

                data.clusters.append(
                    GraphCluster(
                        id=str(comm.uuid),
                        name=comm.name or "",
                        member_node_ids=member_ids,
                        summary=comm.summary or "",
                    )
                )
                if comm.summary:
                    data.community_summaries.append(
                        f"{comm.name}: {comm.summary}"
                    )

            # ── Entity edges ──
            entity_edges = _run(
                client.edges.entity.get_by_group_ids([group_id]),
                loop,
            )
            for edge in entity_edges:
                data.edges.append(
                    GraphEdge(
                        id=str(edge.uuid),
                        source=str(edge.source_node_uuid),
                        target=str(edge.target_node_uuid),
                        relation=edge.name or "",
                        fact=edge.fact or "",
                        valid_at=(
                            edge.valid_at
                            if hasattr(edge, "valid_at")
                            else None
                        ),
                        invalid_at=(
                            edge.invalid_at
                            if hasattr(edge, "invalid_at")
                            else None
                        ),
                    )
                )
                if edge.fact:
                    data.edge_facts.append(edge.fact)

            logger.info(
                "Read graph for group %s: %d nodes, %d edges, %d communities",
                group_id,
                len(data.nodes),
                len(data.edges),
                len(data.clusters),
            )
        except Exception:
            logger.exception("Failed to read graph for group %s", group_id)

        return data


__all__ = ["GraphData", "GraphReader"]
