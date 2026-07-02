"""Coverage scoring for the adaptive interview.

Lifted from the cloud's ``pipeline/interview/coverage_scorer.py``.
The cloud version had two scoring paths:

  1. ``_score_from_graphiti`` — queries the live Neo4j knowledge graph
  2. ``_score_from_postgres`` — word-count heuristic when Graphiti is down

The local port uses a third path: score from a ``PersonalityGraph`` in
memory. Single-user, no DB, no Graphiti dependency. The scoring
formula and dimension maps (lifted as-is) are preserved.

Public API:

  scorer = DimensionCoverageScorer()
  scores = scorer.score(personality_graph)  # dict[str, DimensionScore]

  # Then feed into gap_identifier
  analysis = GapIdentifier().analyze(scores, questions_asked=N)
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Dimension → node-label maps (lifted from cloud lines 29-53) ──────

_DIMENSION_NODE_MAP: dict[str, list[str]] = {
    "episodic_memory": ["LifeEvent", "EpisodicMemory", "Memory"],
    "core_beliefs": ["Belief"],
    "decision_making": ["Pattern", "CognitivePattern"],
    "values": ["Value"],
    "boundaries": ["Boundary"],
    "social_orientation": ["SocialPattern", "Social", "Person"],
    "emotional_dynamics": ["EmotionalTrigger", "EmotionalProfile", "ContextualMood"],
    "knowledge_domains": ["Expertise", "KnowledgeDomain"],
    "communication_style": ["StyleProfile", "Style"],
    "procedural_memory": ["ProceduralPattern", "WorkLoop", "PromptingStyle", "TechnicalGap"],
}

# Per-dimension target thresholds (lifted from cloud lines 56-67).
_DIMENSION_TARGETS: dict[str, dict[str, int | float]] = {
    "episodic_memory": {"min_nodes": 3, "target_edges_per_node": 1.5},
    "core_beliefs": {"min_nodes": 3, "target_edges_per_node": 1.0},
    "decision_making": {"min_nodes": 2, "target_edges_per_node": 1.0},
    "values": {"min_nodes": 2, "target_edges_per_node": 1.0},
    "boundaries": {"min_nodes": 1, "target_edges_per_node": 1.0},
    "social_orientation": {"min_nodes": 2, "target_edges_per_node": 1.0},
    "emotional_dynamics": {"min_nodes": 3, "target_edges_per_node": 1.0},
    "knowledge_domains": {"min_nodes": 2, "target_edges_per_node": 1.0},
    "communication_style": {"min_nodes": 2, "target_edges_per_node": 1.0},
    "procedural_memory": {"min_nodes": 2, "target_edges_per_node": 1.0},
}

# Map from the schema adapter's canonical node types to dimensions
# (used when scoring from the in-memory graph — the cloud's
# _DIMENSION_NODE_MAP uses Graphiti node labels, but the local graph
# uses the schema adapter's canonical types like "trait" not
# "PersonalityTrait").
_ADAPTER_TYPE_TO_DIMENSION: dict[str, str] = {
    "life_event": "episodic_memory",
    "memory": "episodic_memory",
    "belief": "core_beliefs",
    "pattern": "decision_making",
    "value": "values",
    "boundary": "boundaries",
    "social": "social_orientation",
    "person": "social_orientation",
    "emotional_trigger": "emotional_dynamics",
    "emotional_profile": "emotional_dynamics",
    "contextual_mood": "emotional_dynamics",
    "expertise": "knowledge_domains",
    "style": "communication_style",
    "procedural_pattern": "procedural_memory",
    "work_loop": "procedural_memory",
    "prompting_style": "procedural_memory",
    "technical_gap": "procedural_memory",
    "behavioral_rule": "procedural_memory",  # close enough — captures procedure
    "contradiction": "core_beliefs",  # close enough — captures stance
}


@dataclass
class DimensionScore:
    """Coverage score for a single dimension.

    Mirrors the cloud's dataclass (``brain_extractor.py:71-80``) but
    ``source`` is ``"local"`` instead of ``"graphiti"`` or
    ``"postgres_fallback"``.
    """

    dimension: str
    score: float  # 0.0-1.0
    node_count: int
    edge_count: int
    avg_summary_len: float
    diversity: float  # unique_names / node_count (1.0 = all unique)
    source: str  # "local"


class DimensionCoverageScorer:
    """Score how well each personality dimension is covered in a graph.

    Replaces the cloud's Graphiti+PostgreSQL dual-path scorer. The
    local port scores from a ``PersonalityGraph`` in memory — same
    data shape, no DB round-trip.
    """

    def score(self, graph) -> dict[str, DimensionScore]:
        """Return coverage score for every dimension from the given graph.

        Args:
            graph: A ``brain_platform.pipeline.brain_schema.PersonalityGraph``
                (or any object with the same shape). Reads:
                - ``graph.traits``, ``graph.beliefs``, etc. (legacy flat nodes)
                - ``graph.edges`` (with source_name/target_name for diversity)
                - ``graph.knowledge_graph.nodes`` (marketplace nodes)

        Returns:
            Dict mapping dimension name to ``DimensionScore``.
        """
        # Collect all nodes from the graph — both legacy flat and marketplace
        nodes = self._collect_all_nodes(graph)
        edges = self._collect_all_edges(graph)

        # Bucket nodes by canonical type → dimension
        nodes_by_dimension: dict[str, list[dict]] = {dim: [] for dim in _DIMENSION_NODE_MAP}
        for node in nodes:
            ntype = node.get("type", "")
            dim = _ADAPTER_TYPE_TO_DIMENSION.get(ntype)
            if dim:
                nodes_by_dimension[dim].append(node)

        # Bucket edges by source/dimension
        edges_by_dimension: dict[str, int] = {dim: 0 for dim in _DIMENSION_NODE_MAP}
        if edges:
            for edge in edges:
                source_dim = self._dim_for_node_name(graph, edges, edge.get("source_name", ""))
                if source_dim:
                    edges_by_dimension[source_dim] += 1

        result: dict[str, DimensionScore] = {}
        for dimension, dim_nodes in nodes_by_dimension.items():
            targets = _DIMENSION_TARGETS.get(
                dimension, {"min_nodes": 2, "target_edges_per_node": 1.0}
            )
            min_nodes = int(targets["min_nodes"])
            target_e = float(targets["target_edges_per_node"])

            node_count = len(dim_nodes)
            edge_count = edges_by_dimension.get(dimension, 0)

            # Coverage formula (from cloud's _score_from_graphiti):
            # - 0.0 if no nodes
            # - 1.0 if we have min_nodes + target_e edges
            # - linear in between
            if node_count == 0:
                score = 0.0
            elif node_count >= min_nodes and edge_count >= node_count * target_e:
                score = 1.0
            else:
                node_progress = min(1.0, node_count / max(min_nodes, 1))
                edge_progress = min(1.0, edge_count / max(node_count * target_e, 1.0))
                score = (node_progress + edge_progress) / 2

            # Avg summary length — how "thick" each node is
            summaries = [n.get("summary", "") for n in dim_nodes if n.get("summary")]
            avg_summary_len = (
                sum(len(s) for s in summaries) / len(summaries) if summaries else 0.0
            )

            # Diversity — how many unique names (lower = dedup missed)
            names = [n.get("name", "") for n in dim_nodes]
            diversity = len(set(names)) / len(names) if names else 0.0

            result[dimension] = DimensionScore(
                dimension=dimension,
                score=score,
                node_count=node_count,
                edge_count=edge_count,
                avg_summary_len=avg_summary_len,
                diversity=diversity,
                source="local",
            )

        # Fill in any missing dimensions with zeros (so the gap_identifier
        # sees all 10 dimensions even if the graph is empty for some).
        for dim in _DIMENSION_NODE_MAP:
            if dim not in result:
                result[dim] = DimensionScore(
                    dimension=dim, score=0.0, node_count=0, edge_count=0,
                    avg_summary_len=0.0, diversity=0.0, source="local",
                )

        return result

    def _collect_all_nodes(self, graph) -> list[dict]:
        """Pull all nodes from the graph — both flat and marketplace shapes.

        The cloud's ``PersonalityGraph`` is a Pydantic model (not a
        dict), so we read attributes directly. Dict-shaped graphs
        (e.g. the orchestrator's transient in-memory graph) are read
        via ``graph.get(...)``. The marketplace ``knowledge_graph.nodes``
        are a list of Pydantic models, so they're converted to dicts.
        """
        nodes: list[dict] = []
        for attr in _LEGACY_NODE_FIELDS:
            items = _get_field(graph, attr)
            for n in items or []:
                nodes.append(_node_to_dict(n))
        # Marketplace knowledge_graph.nodes
        kg = _get_field(graph, "knowledge_graph")
        if kg and hasattr(kg, "nodes"):
            for n in kg.nodes or []:
                if isinstance(n, dict):
                    nodes.append(n)
                else:
                    nodes.append(_node_to_dict(n))
        return nodes

    def _collect_all_edges(self, graph) -> list[dict]:
        """Pull all edges (legacy + marketplace) from the graph."""
        edges: list[dict] = []
        for edge in _get_field(graph, "edges") or []:
            edges.append(_edge_to_dict(edge))
        kg = _get_field(graph, "knowledge_graph")
        if kg and hasattr(kg, "edges"):
            for edge in kg.edges or []:
                edges.append(_edge_to_dict(edge))
        return edges

    def _dim_for_node_name(
        self, graph, edges: list[dict], name: str
    ) -> str | None:
        """Return the dimension of the node whose name matches ``name``."""
        if not name:
            return None
        # Look up the node's type via the graph
        for node in self._collect_all_nodes(graph):
            if node.get("name") == name:
                return _ADAPTER_TYPE_TO_DIMENSION.get(node.get("type", ""))
        return None


def _node_to_dict(node) -> dict:
    """Convert a Pydantic node model (or dict) to a dict view.

    Pydantic v2 uses ``model_dump()``; v1 used ``dict()``. Both
    callable forms are checked. For the cloud's marketplace
    ``GraphNode`` and ``PersonalityGraph`` we use the Pydantic path;
    legacy dicts are returned as-is.

    Also adds a ``"type"`` field if the model class name is in
    ``_PYDANTIC_TYPE_TO_CANONICAL`` — the dimension map is keyed
    on the canonical strings ("belief", "trait") not the Pydantic
    class names ("BeliefNode", "TraitNode").
    """
    if isinstance(node, dict):
        return node
    d: dict = {}
    if hasattr(node, "model_dump"):
        d = node.model_dump()
    elif hasattr(node, "dict"):
        d = node.dict()
    # If "type" is missing, fill it from the class name so the
    # dimension map can route the node to the right bucket.
    if "type" not in d or not d["type"]:
        cls_name = type(node).__name__
        d["type"] = _PYDANTIC_TYPE_TO_CANONICAL.get(cls_name, cls_name.lower())
    return d


# Map the cloud's Pydantic model field names to the canonical
# node type strings used by the dimension map. The schema adapter's
# "canonical" types (e.g. "belief", "trait") differ from the
# Pydantic model class names (BeliefNode, TraitNode). This map
# closes the gap so a graph built by the cloud's extractor scores
# correctly against the dimension map.
_PYDANTIC_TYPE_TO_CANONICAL = {
    "TraitNode": "trait",
    "BeliefNode": "belief",
    "ValueNode": "value",
    "BoundaryNode": "boundary",
    "LifeEventNode": "life_event",
    "MemoryNode": "memory",
    "PatternNode": "pattern",
    "SocialNode": "social",
    "ExpertiseNode": "expertise",
    "StyleNode": "style",
    "PersonNode": "person",
    "PlaceNode": "place",
    "ProceduralPatternNode": "procedural_pattern",
    "WorkLoopNode": "work_loop",
    "PromptingStyleNode": "prompting_style",
    "TechnicalGapNode": "technical_gap",
    "BehavioralRule": "behavioral_rule",
    "ContradictionPattern": "contradiction",
    "EmotionalTrigger": "emotional_trigger",
    "ContextualMood": "contextual_mood",
}


# All node fields we walk. Same list in both the cloud's legacy
# schema and the marketplace v2.2.0 schema.
_LEGACY_NODE_FIELDS = (
    "traits", "beliefs", "values", "boundaries", "life_events",
    "memories", "patterns", "social", "expertise", "style",
    "people", "places", "procedural_patterns", "work_loops",
    "prompting_styles", "technical_gaps", "behavioral_rules",
    "contradiction_patterns", "emotional_triggers",
    "contextual_moods",
)


def _get_field(graph, name: str):
    """Read a field from either a dict-shaped or Pydantic-shaped graph.

    Dicts use ``graph.get(name, default)``; Pydantic models use
    ``getattr(graph, name, default)``. The function auto-detects
    which is which, with a callable check to avoid returning
    bound methods when a field is missing on a dict (e.g.
    ``dict.get`` on a dict returns the bound method, not a value).
    """
    if isinstance(graph, dict):
        val = graph.get(name, [])
    else:
        val = getattr(graph, name, [])
    if callable(val) and not isinstance(val, type):
        return []
    return val


def _edge_to_dict(edge) -> dict:
    if hasattr(edge, "model_dump"):
        return edge.model_dump()
    if hasattr(edge, "dict"):
        return edge.dict()
    return dict(edge) if isinstance(edge, dict) else {}


__all__ = [
    "DimensionScore",
    "DimensionCoverageScorer",
    "_DIMENSION_NODE_MAP",
    "_DIMENSION_TARGETS",
    "_ADAPTER_TYPE_TO_DIMENSION",
]
