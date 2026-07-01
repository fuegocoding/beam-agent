"""Local graph writer — port of beam_mind's BrainWriter.write().

Writes a ``PersonalityGraph`` to the local Neo4j + Graphiti store.
Faithful port of the cloud's ``beam_mind.pipeline.brain_writer.BrainWriter.write()``
(232 lines) with two local adaptations:

1. **Sync facade** — the cloud uses ``await node.save(driver)``; the
   local port wraps async Graphiti calls in ``asyncio.run()`` so the
   public API is sync (single-user local app, no event loop).

2. **No fuzzy node lookup** — the cloud runs Cypher to load existing
   nodes (for dedup during re-extraction). The local port skips the
   pre-load and relies on EntityNode.save() to upsert by name
   (Graphiti's name_embedding-based dedup handles this).

Everything else — THE_USER hub creation, typed entity node labels,
hub edge types (HAS_TRAIT, HOLDS, etc.), cross-link edges with fuzzy
name resolution, the ``NODE_TYPE_TO_LABEL`` / ``HUB_EDGE_FOR_TYPE``
maps — is copied verbatim from the cloud.

Public API:

  writer = LocalGraphWriter(store)
  result = writer.write(graph, group_id="user_123")
  # {"nodes_created": int, "edges_created": int}
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from brain_platform.pipeline.brain_schema import (
    HUB_EDGE_FOR_TYPE,
    NODE_TYPE_TO_LABEL,
    PersonalityGraph,
)

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously via asyncio.run()."""
    return asyncio.run(coro)


class LocalGraphWriter:
    """Writes a structured PersonalityGraph to a local Neo4j + Graphiti store.

    Mirrors the cloud's BrainWriter.write() but exposes a sync API.
    """

    def __init__(self, store: Any):
        """Args:
            store: A :class:`LocalGraphStore` (or any object with a
                ``.client`` property returning a Graphiti client and a
                ``.group_id_for_user`` method).
        """
        self._store = store

    def write_interview_session(
        self,
        interview_text: str,
        group_id: str,
    ) -> dict:
        """Extract a PersonalityGraph from interview text and persist it.

        Mirrors the cloud's ``GraphWriter.write_interview_session()``:
        runs :class:`BrainExtractor` over the full interview, then
        writes the resulting graph to Neo4j.

        Args:
            interview_text: All interview Q&A concatenated.
            group_id: Graphiti group_id for user partition.

        Returns:
            ``{"nodes_created": int, "edges_created": int}``
        """
        from brain_platform.extractor.brain_extractor import BrainExtractor

        if not interview_text.strip():
            return {"nodes_created": 0, "edges_created": 0}

        # Pass the Graphiti client's LLM client (same shape as cloud's
        # ``client.llm_client``) so the extractor can route through
        # Graphiti's LLM config.
        client = self._store.client
        llm_client = getattr(client, "llm_client", None)

        extractor = BrainExtractor()
        graph = extractor.extract(
            interview_text=interview_text,
            llm_client=llm_client,
        )

        return self.write(graph=graph, group_id=group_id)

    def write(
        self,
        graph: PersonalityGraph,
        group_id: str,
    ) -> dict:
        """Write all nodes and edges from ``graph`` to Neo4j.

        Args:
            graph: The extracted PersonalityGraph.
            group_id: Graphiti group_id for user partition.

        Returns:
            ``{"nodes_created": int, "edges_created": int}``
        """
        from graphiti_core.edges import EntityEdge
        from graphiti_core.nodes import EntityNode

        client = self._store.client
        now = _now_utc()

        name_to_uuid: dict[str, str] = {}
        name_to_label: dict[str, str] = {}
        nodes_created = 0
        edges_created = 0

        # ── 1. Find or create THE_USER node ──
        driver = client.driver
        neo4j_client = driver.client if hasattr(driver, "client") else driver

        existing_user = _run(
            neo4j_client.execute_query(
                "MATCH (n) WHERE n.group_id = $gid AND n.name = 'THE_USER' RETURN n.uuid AS uuid",
                parameters_={"gid": group_id},
            )
        )
        if existing_user.records:
            user_uuid = existing_user.records[0]["uuid"]
            name_to_uuid["THE_USER"] = user_uuid
            name_to_label["THE_USER"] = "Entity"
            logger.info("Reusing existing THE_USER node: %s", user_uuid[:12])
        else:
            user_node = EntityNode(
                name="THE_USER",
                group_id=group_id,
                labels=["Entity"],
                summary=graph.user_summary,
            )
            _run(user_node.generate_name_embedding(client.embedder))
            _run(user_node.save(driver))
            name_to_uuid["THE_USER"] = user_node.uuid
            name_to_label["THE_USER"] = "Entity"
            nodes_created += 1

        # ── 1b. Load existing nodes for dedup during deepening ──
        existing_nodes_result = _run(
            neo4j_client.execute_query(
                """
                MATCH (n) WHERE n.group_id = $gid
                AND NOT 'Episodic' IN labels(n) AND NOT 'Saga' IN labels(n)
                RETURN n.name AS name, n.uuid AS uuid, labels(n) AS labels
                """,
                parameters_={"gid": group_id},
            )
        )
        for record in existing_nodes_result.records:
            ename = record["name"]
            if ename and ename not in name_to_uuid:
                name_to_uuid[ename] = record["uuid"]
                labels = record["labels"]
                name_to_label[ename] = next(
                    (l for l in labels if l != "Entity" and l != "Node"), "Entity"
                )

        # ── 2. Create all typed entity nodes ──
        node_lists = {
            "traits": graph.traits,
            "beliefs": graph.beliefs,
            "values": graph.values,
            "boundaries": graph.boundaries,
            "life_events": graph.life_events,
            "memories": graph.memories,
            "patterns": graph.patterns,
            "social": graph.social,
            "expertise": graph.expertise,
            "style": graph.style,
            "people": graph.people,
            "places": graph.places,
        }

        for list_key, items in node_lists.items():
            label = NODE_TYPE_TO_LABEL[list_key]
            for item in items:
                name = item.name.strip()
                if not name or name in name_to_uuid:
                    continue  # Skip duplicates

                # Build attributes from the item's fields (excluding name/summary)
                attrs = {}
                for field_name, field_value in item.model_dump().items():
                    if field_name not in ("name", "summary"):
                        attrs[field_name] = field_value

                labels = [label, "Entity"] if label != "Entity" else ["Entity"]
                summary = getattr(item, "summary", "") or getattr(item, "significance", "") or ""

                node = EntityNode(
                    name=name,
                    group_id=group_id,
                    labels=labels,
                    summary=summary,
                    attributes=attrs,
                )
                _run(node.generate_name_embedding(client.embedder))
                _run(node.save(driver))
                name_to_uuid[name] = node.uuid
                name_to_label[name] = label
                nodes_created += 1

        logger.info("Created %d nodes (including THE_USER)", nodes_created)

        # ── 3. Create THE_USER hub edges (skip if already connected) ──
        existing_hub_targets = set()
        if existing_user.records:
            hub_check = _run(
                neo4j_client.execute_query(
                    """
                    MATCH (u {name: 'THE_USER', group_id: $gid})-[r:RELATES_TO]->(target)
                    RETURN target.uuid AS uuid
                    """,
                    parameters_={"gid": group_id},
                )
            )
            existing_hub_targets = {r["uuid"] for r in hub_check.records}

        for name, node_uuid in name_to_uuid.items():
            if name == "THE_USER":
                continue
            if node_uuid in existing_hub_targets:
                continue
            label = name_to_label.get(name, "Entity")
            hub_type = HUB_EDGE_FOR_TYPE.get(label)
            if not hub_type:
                hub_type = "INVOLVES"

            edge = EntityEdge(
                source_node_uuid=name_to_uuid["THE_USER"],
                target_node_uuid=node_uuid,
                name=hub_type,
                group_id=group_id,
                fact=f"THE_USER {hub_type.lower().replace('_', ' ')} {name}",
                created_at=now,
                valid_at=now,
            )
            _run(edge.generate_embedding(client.embedder))
            _run(edge.save(driver))
            edges_created += 1

        logger.info("Created %d hub edges (THE_USER → all nodes)", edges_created)

        # ── 4. Create cross-link edges with fuzzy name resolution ──
        # Build a lowercase lookup for fuzzy name matching (LLM
        # sometimes uses slightly different names in edges vs nodes).
        lower_to_name = {name.lower(): name for name in name_to_uuid}

        def resolve_name(edge_name: str) -> str | None:
            """Resolve an edge's entity name to a node name, with fuzzy fallback."""
            if edge_name in name_to_uuid:
                return edge_name
            matched = lower_to_name.get(edge_name.lower())
            if matched:
                return matched
            edge_lower = edge_name.lower()
            for node_lower, node_name in lower_to_name.items():
                if edge_lower in node_lower or node_lower in edge_lower:
                    return node_name
            return None

        cross_edges = 0
        skipped_edges = 0
        for edge_spec in graph.edges:
            src_name = resolve_name(edge_spec.source_name)
            tgt_name = resolve_name(edge_spec.target_name)
            src_uuid = name_to_uuid.get(src_name) if src_name else None
            tgt_uuid = name_to_uuid.get(tgt_name) if tgt_name else None

            if not src_uuid or not tgt_uuid:
                skipped_edges += 1
                continue
            if src_uuid == tgt_uuid:
                logger.debug("Edge skip (self-loop): %s", edge_spec.source_name)
                continue

            edge = EntityEdge(
                source_node_uuid=src_uuid,
                target_node_uuid=tgt_uuid,
                name=edge_spec.edge_type,
                group_id=group_id,
                fact=edge_spec.fact,
                created_at=now,
                valid_at=now,
            )
            _run(edge.generate_embedding(client.embedder))
            _run(edge.save(driver))
            edges_created += 1
            cross_edges += 1

        logger.info(
            "Created %d cross-link edges (%d skipped — name mismatch)",
            cross_edges, skipped_edges,
        )
        logger.info(
            "Total: %d nodes, %d edges written to Neo4j",
            nodes_created, edges_created,
        )

        return {"nodes_created": nodes_created, "edges_created": edges_created}


__all__ = ["LocalGraphWriter"]
