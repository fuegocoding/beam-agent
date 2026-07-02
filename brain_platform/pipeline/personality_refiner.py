"""Post-extraction personality refinement pass.

Faithful port of the cloud's
``beam_mind.pipeline.personality_refiner.PersonalityRefiner``.
After per-episode extraction, this module makes a single LLM call
to identify personality constructs (traits, values, boundaries) that
were missed, then creates them as typed nodes with edges in the
knowledge graph.

Sync facade over Graphiti's async client (same pattern as the rest
of brain_platform). The semantic embedding dedup uses numpy when
available; falls back to exact-name match when not.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ── Structured output models for the refinement LLM call ──


class RefinedTrait(BaseModel):
    trait_name: str = Field(description="Concise trait name, e.g. 'conflict-avoidant'")
    strength: float = Field(description="0-1 how strongly expressed", ge=0, le=1)
    evidence: str = Field(description="Quote or paraphrase from the interview supporting this")
    summary: str = Field(description="2-3 sentence description of how this trait manifests")


class RefinedValue(BaseModel):
    value_name: str = Field(description="Concise value name, e.g. 'autonomy'")
    importance: float = Field(description="0-1 how central to identity", ge=0, le=1)
    evidence: str = Field(description="Specific example of this value in action")
    summary: str = Field(description="2-3 sentence description")


class RefinedBoundary(BaseModel):
    description: str = Field(description="What they won't do, e.g. 'won't lie about product impact'")
    tested: bool = Field(description="Has this boundary been tested in a real situation?")
    cost_paid: str = Field(description="What it cost them to maintain this boundary", default="")
    summary: str = Field(description="2-3 sentence description")


class RefinedEdge(BaseModel):
    source_name: str = Field(description="Name of source entity (must exist in graph)")
    target_name: str = Field(description="Name of target entity (must exist in graph or be newly created)")
    edge_type: str = Field(description="One of: SHAPED_BY, ENFORCED_AS, EVOLVED_INTO, GUIDES, LEARNED_FROM, EXPRESSED_THROUGH, TESTED_BY, INVOLVES, INFORMED_BY")
    fact: str = Field(description="Description of the relationship")


class PersonalityRefinement(BaseModel):
    traits: List[RefinedTrait] = Field(default_factory=list)
    values: List[RefinedValue] = Field(default_factory=list)
    boundaries: List[RefinedBoundary] = Field(default_factory=list)
    missing_edges: List[RefinedEdge] = Field(default_factory=list)


REFINEMENT_PROMPT = """\
You are a personality analyst building a digital replica's knowledge graph.

Below is a complete guided interview with a person, followed by the entities
ALREADY extracted into their knowledge graph. Your job is to identify what
is MISSING — personality constructs that the extraction pipeline overlooked.

<INTERVIEW>
{interview_text}
</INTERVIEW>

<ALREADY EXTRACTED ENTITIES>
{existing_entities}
</ALREADY EXTRACTED ENTITIES>

For each category, identify items that are clearly present in the interview
but NOT yet represented in the graph. Use the ADD/SKIP decision framework:
- ADD: This personality facet is clearly evidenced in the interview but has
  no matching entity in the graph. Create it.
- SKIP: An entity with the same or very similar meaning already exists.
  Do not duplicate it.

CATEGORIES TO CHECK:

1. PERSONALITY TRAITS — behavioral patterns, both stated and implied.
   Look for: "I'm [adjective]", patterns in HOW they tell stories, recurring
   behavioral descriptions. Examples: analytical, conflict-avoidant, stubborn,
   empathetic, risk-averse, perfectionist, anxious.

2. VALUES — what drives their choices and priorities.
   Look for: what they sacrifice for, what makes them feel fulfilled, what they
   refuse to compromise on. Examples: autonomy, honesty, family, creativity,
   rest, transparency.

3. BOUNDARIES — non-negotiables, lines they won't cross.
   Look for: "I won't", "I refuse", "that's my line", stories where they paid
   a cost to maintain a principle. Examples: won't lie about product impact,
   won't take VC at cost of control.

4. MISSING CONNECTIONS — edges between EXISTING entities that should exist
   but don't. Look for causal chains:
   - Which life events SHAPED which beliefs/values?
   - Which values are ENFORCED AS which boundaries?
   - Which beliefs EVOLVED INTO other beliefs?
   - Which expertise INFORMED which beliefs?
   Only reference entities that already exist in the graph or that you are
   creating in this pass.

NAMING RULES:
- Traits: use the adjective form: "conflict-avoidant" not "avoids conflict"
- Values: use the noun: "autonomy" not "value of autonomy"
- Boundaries: describe the line: "won't lie about product impact"
- Keep names under 50 characters
"""


class PersonalityRefiner:
    """Runs a post-extraction refinement pass to catch missing personality constructs.

    Mirrors the cloud's PersonalityRefiner but with a sync API.
    """

    # ── Hub edge type mapping (mirror of cloud's class attribute) ──
    HUB_EDGE_MAP = {
        "PersonalityTrait": "HAS_TRAIT",
        "Belief": "HOLDS",
        "Value": "DRIVEN_BY",
        "Boundary": "DRIVEN_BY",
        "LifeEvent": "EXPERIENCED",
        "EpisodicMemory": "EXPERIENCED",
        "KnowledgeDomain": "EXPERT_IN",
        "SocialPattern": "HANDLES_CONFLICT_BY",
        "StyleProfile": "COMMUNICATES_VIA",
        "CognitivePattern": "HAS_TRAIT",
    }

    def refine(
        self,
        interview_text: str,
        group_id: str,
        graphiti_client: Any,
    ) -> dict:
        """Run the refinement pass.

        Args:
            interview_text: Full concatenated interview Q&A text.
            group_id: Graphiti group_id for this user's graph partition.
            graphiti_client: The Graphiti instance (has .driver, .embedder, .llm_client).

        Returns:
            ``{"nodes_created": int, "edges_created": int}``
        """
        from graphiti_core.edges import EntityEdge
        from graphiti_core.nodes import EntityNode

        driver = graphiti_client.driver
        embedder = graphiti_client.embedder
        llm_client = graphiti_client.llm_client

        # ── 1. Query existing graph state ──
        existing = _run(self._get_existing_entities(driver, group_id))
        existing_text = self._format_existing_entities(existing)

        logger.info(
            "Refinement: %d existing entities in graph for group %s",
            sum(len(v) for v in existing.values()),
            group_id,
        )

        # ── 2. LLM call for targeted extraction ──
        prompt = REFINEMENT_PROMPT.format(
            interview_text=interview_text[:12000],
            existing_entities=existing_text,
        )

        try:
            result = llm_client.generate_response(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a personality analyst. Return structured JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_model=PersonalityRefinement,
                task="brain_personality_refine",
            )
            refinement = PersonalityRefinement(**result)
        except Exception:
            logger.exception("Personality refinement LLM call failed")
            return {"nodes_created": 0, "edges_created": 0}

        logger.info(
            "Refinement LLM returned: %d traits, %d values, %d boundaries, %d edges",
            len(refinement.traits),
            len(refinement.values),
            len(refinement.boundaries),
            len(refinement.missing_edges),
        )

        # ── 3. Dedup + create missing nodes ──
        existing_embeddings = _run(self._load_existing_embeddings(driver, group_id))
        nodes_created = 0
        new_node_names: Dict[str, str] = {}

        for trait in refinement.traits:
            if _run(self._is_duplicate(
                trait.trait_name, "PersonalityTrait", existing_embeddings, embedder
            )):
                logger.debug("SKIP trait (duplicate): %s", trait.trait_name)
                continue
            node = EntityNode(
                name=trait.trait_name,
                group_id=group_id,
                labels=["PersonalityTrait", "Entity"],
                summary=trait.summary,
                attributes={
                    "trait_name": trait.trait_name,
                    "strength": trait.strength,
                    "evidence_count": 1,
                },
            )
            _run(node.generate_name_embedding(embedder))
            _run(node.save(driver))
            new_node_names[trait.trait_name] = node.uuid
            nodes_created += 1
            logger.info("  + PersonalityTrait: %s (%.1f)", trait.trait_name, trait.strength)

        for value in refinement.values:
            if _run(self._is_duplicate(
                value.value_name, "Value", existing_embeddings, embedder
            )):
                logger.debug("SKIP value (duplicate): %s", value.value_name)
                continue
            node = EntityNode(
                name=value.value_name,
                group_id=group_id,
                labels=["Value", "Entity"],
                summary=value.summary,
                attributes={
                    "value_name": value.value_name,
                    "importance": value.importance,
                    "evidence": value.evidence,
                },
            )
            _run(node.generate_name_embedding(embedder))
            _run(node.save(driver))
            new_node_names[value.value_name] = node.uuid
            nodes_created += 1
            logger.info("  + Value: %s (%.1f)", value.value_name, value.importance)

        for boundary in refinement.boundaries:
            if _run(self._is_duplicate(
                boundary.description, "Boundary", existing_embeddings, embedder
            )):
                logger.debug("SKIP boundary (duplicate): %s", boundary.description)
                continue
            node = EntityNode(
                name=boundary.description,
                group_id=group_id,
                labels=["Boundary", "Entity"],
                summary=boundary.summary,
                attributes={
                    "description": boundary.description,
                    "tested": boundary.tested,
                    "cost_paid": boundary.cost_paid,
                },
            )
            _run(node.generate_name_embedding(embedder))
            _run(node.save(driver))
            new_node_names[boundary.description] = node.uuid
            nodes_created += 1
            logger.info("  + Boundary: %s (tested=%s)", boundary.description, boundary.tested)

        # ── 4. Create missing edges ──
        name_to_uuid = {}
        for nodes_list in existing.values():
            for n in nodes_list:
                name_to_uuid[n["name"].lower()] = n["uuid"]
        for name, uid in new_node_names.items():
            name_to_uuid[name.lower()] = uid

        edges_created = 0
        for edge_spec in refinement.missing_edges:
            src_uuid = name_to_uuid.get(edge_spec.source_name.lower())
            tgt_uuid = name_to_uuid.get(edge_spec.target_name.lower())
            if not src_uuid or not tgt_uuid:
                logger.debug(
                    "SKIP edge (node not found): %s --%s--> %s",
                    edge_spec.source_name, edge_spec.edge_type, edge_spec.target_name,
                )
                continue
            edge = EntityEdge(
                source_node_uuid=src_uuid,
                target_node_uuid=tgt_uuid,
                name=edge_spec.edge_type,
                group_id=group_id,
                fact=edge_spec.fact,
                created_at=datetime.now(timezone.utc),
                valid_at=datetime.now(timezone.utc),
            )
            _run(edge.generate_embedding(embedder))
            _run(edge.save(driver))
            edges_created += 1
            logger.info(
                "  + Edge: %s --%s--> %s",
                edge_spec.source_name, edge_spec.edge_type, edge_spec.target_name,
            )

        # ── 5. Create THE_USER hub edges ──
        hub_edges = _run(self._create_hub_edges(driver, embedder, group_id))
        edges_created += hub_edges
        logger.info("  Hub edges created: %d", hub_edges)

        # ── 6. Deduplicate misclassified person nodes ──
        deduped = _run(self._dedup_person_nodes(driver, group_id))
        logger.info("  Person nodes deduplicated: %d", deduped)

        # ── 7. Fix remaining orphan nodes ──
        orphans_fixed = _run(self._fix_orphan_nodes(driver, embedder, group_id))
        edges_created += orphans_fixed
        logger.info("  Orphan edges created: %d", orphans_fixed)

        return {
            "nodes_created": nodes_created,
            "edges_created": edges_created,
        }

    # ── Internal helpers (sync wrappers around the cloud's async logic) ──

    async def _get_existing_entities(
        self, driver: Any, group_id: str
    ) -> Dict[str, list]:
        """Query the graph for all existing nodes per type."""
        neo4j_client = driver.client
        result = await neo4j_client.execute_query(
            """
            MATCH (n) WHERE n.group_id = $gid
            AND NOT 'Episodic' IN labels(n) AND NOT 'Saga' IN labels(n)
            RETURN n.name AS name, n.uuid AS uuid, labels(n) AS labels,
                   n.summary AS summary, n.name_embedding AS embedding
            """,
            parameters_={"gid": group_id},
        )
        grouped: Dict[str, list] = {}
        for record in result.records:
            labels = record["labels"]
            primary = next(
                (l for l in labels if l not in ("Entity", "Node")),
                "Entity",
            )
            grouped.setdefault(primary, []).append({
                "name": record["name"] or "",
                "uuid": record["uuid"],
                "labels": labels,
                "summary": record["summary"] or "",
                "embedding": record["embedding"],
            })
        return grouped

    def _format_existing_entities(self, existing: Dict[str, list]) -> str:
        """Format existing entities for the LLM prompt."""
        lines = []
        for node_type, nodes in existing.items():
            lines.append(f"\n=== {node_type} ({len(nodes)}) ===")
            for n in nodes[:30]:
                summary = (n.get("summary") or "")[:100]
                lines.append(f"  - {n['name']}" + (f": {summary}" if summary else ""))
        return "\n".join(lines)

    async def _load_existing_embeddings(
        self, driver: Any, group_id: str
    ) -> Dict[str, list]:
        """Load name→embedding map for semantic dedup."""
        neo4j_client = driver.client
        result = await neo4j_client.execute_query(
            """
            MATCH (n) WHERE n.group_id = $gid
            AND n.name_embedding IS NOT NULL
            RETURN n.name AS name, n.name_embedding AS embedding
            """,
            parameters_={"gid": group_id},
        )
        return {record["name"]: record["embedding"] for record in result.records}

    async def _is_duplicate(
        self,
        new_name: str,
        node_type: str,
        existing_embeddings: Dict[str, list],
        embedder: Any,
        threshold: float = 0.92,
    ) -> bool:
        """Check if a new entity is a semantic duplicate of an existing one.

        Uses cosine similarity on name embeddings. Falls back to
        case-insensitive exact match if numpy isn't available or
        embeddings can't be computed.
        """
        new_name_lower = new_name.lower().strip()

        # Exact match check (fast path)
        for existing_name in existing_embeddings:
            if existing_name.lower().strip() == new_name_lower:
                return True

        # Semantic match via cosine similarity
        try:
            new_emb = await embedder.create(input_data=[new_name])
            if not new_emb or not new_emb[0]:
                return False
            new_vec = new_emb[0]

            try:
                import numpy as np
            except ImportError:
                return False

            for existing_name, existing_vec in existing_embeddings.items():
                if existing_vec is None:
                    continue
                a = np.array(new_vec)
                b = np.array(existing_vec)
                denom = (np.linalg.norm(a) * np.linalg.norm(b))
                if denom == 0:
                    continue
                sim = float(np.dot(a, b) / denom)
                if sim >= threshold:
                    logger.debug(
                        "DEDUP match: %r ≈ %r (sim=%.3f)",
                        new_name, existing_name, sim,
                    )
                    return True
        except Exception:
            logger.debug("Dedup embedding check failed, falling back to exact match")

        return False

    async def _create_hub_edges(
        self, driver: Any, embedder: Any, group_id: str
    ) -> int:
        """Create THE_USER → all-nodes hub edges for new nodes.

        Mirrors the cloud's _create_hub_edges: queries nodes with
        construct-type labels that aren't already connected to
        THE_USER, then creates the appropriate hub edge based on
        :attr:`HUB_EDGE_MAP`.
        """
        from graphiti_core.edges import EntityEdge

        neo4j_client = driver.client

        # Find THE_USER
        user_result = await neo4j_client.execute_query(
            "MATCH (n) WHERE n.group_id = $gid AND n.name = 'THE_USER' "
            "RETURN n.uuid AS uuid",
            parameters_={"gid": group_id},
        )
        if not user_result.records:
            logger.warning("THE_USER node not found in graph")
            return 0
        user_uuid = user_result.records[0]["uuid"]

        # Find construct nodes NOT already connected to THE_USER
        construct_types = list(self.HUB_EDGE_MAP.keys())
        result = await neo4j_client.execute_query(
            """
            MATCH (n)
            WHERE n.group_id = $gid
            AND any(label IN labels(n) WHERE label IN $types)
            AND n.uuid <> $user_uuid
            AND NOT EXISTS {
                MATCH (u {uuid: $user_uuid})-[:RELATES_TO|MENTIONS]-(n)
            }
            RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels
            """,
            parameters={
                "gid": group_id,
                "types": construct_types,
                "user_uuid": user_uuid,
            },
        )

        edges_created = 0
        for record in result.records:
            node_labels = record["labels"]
            # Pick the most specific type for edge mapping
            node_type = next(
                (lbl for lbl in node_labels if lbl in self.HUB_EDGE_MAP),
                None,
            )
            if not node_type:
                continue

            edge_type = self.HUB_EDGE_MAP[node_type]
            edge = EntityEdge(
                source_node_uuid=user_uuid,
                target_node_uuid=record["uuid"],
                name=edge_type,
                group_id=group_id,
                fact=f"THE_USER {edge_type.lower().replace('_', ' ')} {record['name'][:80]}",
                created_at=datetime.now(timezone.utc),
                valid_at=datetime.now(timezone.utc),
            )
            await edge.generate_embedding(embedder)
            await edge.save(driver)
            edges_created += 1
        return edges_created

    async def _dedup_person_nodes(self, driver: Any, group_id: str) -> int:
        """Merge duplicate person nodes that were misclassified as personality constructs.

        Mirrors the cloud's _dedup_person_nodes: finds short-named
        nodes (likely people) that appear multiple times in the
        graph, picks a canonical one (preferring the Entity-only
        typed node), redirects all RELATES_TO and MENTIONS edges
        from the duplicates to the canonical node, then deletes
        the duplicates.
        """
        neo4j_client = driver.client

        # Find short-named nodes (likely people) that appear multiple times
        result = await neo4j_client.execute_query(
            """
            MATCH (n)
            WHERE n.group_id = $gid
            AND NOT 'Episodic' IN labels(n)
            AND NOT 'Saga' IN labels(n)
            AND NOT 'CommunityNode' IN labels(n)
            AND size(n.name) < 30
            WITH toLower(trim(n.name)) AS normalized, collect(n) AS nodes
            WHERE size(nodes) > 1
            RETURN normalized, [n IN nodes | {uuid: n.uuid, name: n.name, labels: labels(n)}] AS dupes
            """,
            parameters_={"gid": group_id},
        )

        merged_count = 0
        for record in result.records:
            dupes = record["dupes"]
            if len(dupes) <= 1:
                continue

            # Prefer the Entity-only typed node as canonical
            canonical = next(
                (d for d in dupes if set(d["labels"]) == {"Entity"}),
                dupes[0],  # fallback to first
            )
            canonical_uuid = canonical["uuid"]

            for dupe in dupes:
                if dupe["uuid"] == canonical_uuid:
                    continue

                dupe_uuid = dupe["uuid"]
                logger.info(
                    "  Merging '%s' (%s) into canonical '%s'",
                    dupe["name"],
                    [l for l in dupe["labels"] if l != "Entity"],
                    canonical["name"],
                )

                # Redirect outgoing RELATES_TO edges
                await neo4j_client.execute_query(
                    """
                    MATCH (dup {uuid: $dup_uuid})-[r:RELATES_TO]->(target)
                    WHERE NOT EXISTS {
                        MATCH (canon {uuid: $canon_uuid})-[:RELATES_TO]->(target)
                    }
                    CREATE (canon {uuid: $canon_uuid})-[r2:RELATES_TO]->(target)
                    SET r2 = properties(r)
                    DELETE r
                    """,
                    parameters={"dup_uuid": dupe_uuid, "canon_uuid": canonical_uuid},
                )

                # Redirect incoming RELATES_TO edges
                await neo4j_client.execute_query(
                    """
                    MATCH (source)-[r:RELATES_TO]->(dup {uuid: $dup_uuid})
                    WHERE NOT EXISTS {
                        MATCH (source)-[:RELATES_TO]->(canon {uuid: $canon_uuid})
                    }
                    CREATE (source)-[r2:RELATES_TO]->(canon {uuid: $canon_uuid})
                    SET r2 = properties(r)
                    DELETE r
                    """,
                    parameters={"dup_uuid": dupe_uuid, "canon_uuid": canonical_uuid},
                )

                # Redirect MENTIONS edges
                await neo4j_client.execute_query(
                    """
                    MATCH (dup {uuid: $dup_uuid})-[r:MENTIONS]-(other)
                    DELETE r
                    """,
                    parameters={"dup_uuid": dupe_uuid},
                )

                # Delete the duplicate node
                await neo4j_client.execute_query(
                    "MATCH (n {uuid: $uuid}) DETACH DELETE n",
                    parameters={"uuid": dupe_uuid},
                )
                merged_count += 1

        return merged_count

    async def _fix_orphan_nodes(
        self, driver: Any, embedder: Any, group_id: str
    ) -> int:
        """Connect any remaining orphan nodes to THE_USER.

        Mirrors the cloud's _fix_orphan_nodes: uses
        :attr:`HUB_EDGE_MAP` to pick the right edge type for each
        orphan based on its label.
        """
        from graphiti_core.edges import EntityEdge

        neo4j_client = driver.client

        # Find THE_USER
        user_result = await neo4j_client.execute_query(
            "MATCH (n) WHERE n.group_id = $gid AND n.name = 'THE_USER' RETURN n.uuid AS uuid",
            parameters={"gid": group_id},
        )
        if not user_result.records:
            return 0
        user_uuid = user_result.records[0]["uuid"]

        # Find orphan nodes (no RELATES_TO or MENTIONS connections)
        result = await neo4j_client.execute_query(
            """
            MATCH (n)
            WHERE n.group_id = $gid
            AND NOT 'Episodic' IN labels(n)
            AND NOT 'Saga' IN labels(n)
            AND NOT 'CommunityNode' IN labels(n)
            AND n.uuid <> $user_uuid
            AND NOT (n)-[:RELATES_TO|MENTIONS]-()
            RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels
            """,
            parameters={"gid": group_id, "user_uuid": user_uuid},
        )

        edges_created = 0
        for record in result.records:
            node_labels = record["labels"]
            node_type = next(
                (lbl for lbl in node_labels if lbl in self.HUB_EDGE_MAP),
                None,
            )
            edge_type = self.HUB_EDGE_MAP.get(node_type, "INVOLVES")

            edge = EntityEdge(
                source_node_uuid=user_uuid,
                target_node_uuid=record["uuid"],
                name=edge_type,
                group_id=group_id,
                fact=f"THE_USER {edge_type.lower().replace('_', ' ')} {record['name'][:80]}",
                created_at=datetime.now(timezone.utc),
                valid_at=datetime.now(timezone.utc),
            )
            await edge.generate_embedding(embedder)
            await edge.save(driver)
            edges_created += 1
            logger.info("  Fixed orphan: THE_USER --%s--> %s", edge_type, record["name"][:50])

        return edges_created


__all__ = [
    "PersonalityRefiner",
    "RefinedTrait",
    "RefinedValue",
    "RefinedBoundary",
    "RefinedEdge",
    "PersonalityRefinement",
    "REFINEMENT_PROMPT",
]
