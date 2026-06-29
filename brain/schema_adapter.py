"""Schema adapter — normalize any brain graph into a uniform node list.

The marketplace at api.openbeam.me ships brains in a richer schema
(``personality_profile`` / ``knowledge_graph`` / ``knowledge_domains`` /
``episodic_memories``) than the legacy flat schema (``traits`` /
``beliefs`` / ``values`` / ``boundaries`` / ``memories``) that the
in-tree brain builder produces.

Both schemas describe the same underlying structure: typed nodes
describing a person's traits, beliefs, values, memories, and
relationships. This module flattens either representation into a
single ``nodes`` list whose entries match the shape that
:class:`brain.brain_retriever.BrainRetriever` already understands
(``{type, name, summary, relevance, ...}``).

Adding a new schema only requires extending :func:`iter_nodes`; the
rest of the brain pipeline keeps working unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# Confidence / strength / importance defaults used when the source
# schema omits the field.  Marketplace entries vary in whether they
# include a numeric score (some ``knowledge_graph.nodes`` do, some
# don't); treating missing scores as 0.5 keeps the retriever's
# relevance ranking stable.
_DEFAULT_STRENGTH = 0.5
_DEFAULT_CONFIDENCE = 0.5
_DEFAULT_IMPORTANCE = 0.5
_DEFAULT_DEPTH = 0.5


def _coerce_text(value: Any) -> str:
    """Best-effort string coercion for summary fields."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(_coerce_text(v) for v in value if v)
    if isinstance(value, dict):
        # Prefer explicit summary/description fields, then the rest.
        for key in ("summary", "description", "text", "content", "label"):
            if key in value and value[key]:
                return _coerce_text(value[key])
        return ", ".join(f"{k}: {_coerce_text(v)}" for k, v in value.items() if v)
    return str(value)


# Patterns the marketplace uses to embed numeric scores in plain
# strings. Recognized forms:
#   "Foo (0.99): bar"                 -> 0.99 from the (0.99) suffix
#   "Foo bar (confidence: 0.99) [ref]" -> 0.99 from (confidence: 0.99)
#   "Foo bar [De Brevitate Vitae]"     -> 0.0, citation kept separately
import re as _re

_LEADING_NAME_SCORE_RE = _re.compile(r"^(?P<name>.+?)\s*\((?P<score>0?\.\d+|1\.0+|\d+(?:\.\d+)?)\)\s*[:\-]\s*(?P<rest>.+)$", _re.DOTALL)
_TRAILING_SCORED_RE = _re.compile(
    r"\((?P<attr>confidence|strength|importance|depth)\s*[:=]\s*"
    r"(?P<score>0?\.\d+|1\.0+|\d+(?:\.\d+)?)\)",
    _re.IGNORECASE,
)
_BRACKET_CITATION_RE = _re.compile(r"\s*\[[^\]]*\]\s*$")


def _split_embedded_score(text: str) -> tuple[str, str, float | None]:
    """Pull a numeric score out of a marketplace-encoded string.

    Returns ``(name, summary, score)`` where ``name`` is the clean
    trait/belief name, ``summary`` is the remaining prose (with
    trailing bracketed citations stripped), and ``score`` is the
    embedded float in ``[0.0, 1.0]`` (or None if no score was
    embedded).

    Recognized forms (see ``_LEADING_NAME_SCORE_RE`` and
    ``_TRAILING_SCORED_RE`` for the exact regexes):
      - ``"Foo (0.99): bar"``                 → name="Foo", score=0.99
      - ``"bar (confidence: 0.99) [ref]"``   → summary="bar", score=0.99
      - ``"Foo"``                            → returns input unchanged
    """
    if not isinstance(text, str):
        return "", _coerce_text(text), None

    score: float | None = None

    leading = _LEADING_NAME_SCORE_RE.match(text)
    if leading:
        try:
            score = float(leading.group("score"))
        except (TypeError, ValueError):
            score = None
        if score is not None and 0.0 <= score <= 1.0:
            return leading.group("name").strip(), leading.group("rest").strip(), score

    trailing_match = _TRAILING_SCORED_RE.search(text)
    cleaned = text
    if trailing_match:
        try:
            score = float(trailing_match.group("score"))
            if not (0.0 <= score <= 1.0):
                score = None
        except (TypeError, ValueError):
            score = None
        if score is not None:
            cleaned = (text[: trailing_match.start()] + text[trailing_match.end():]).strip()

    cleaned = _BRACKET_CITATION_RE.sub("", cleaned).strip()
    return "", cleaned, score


def _coerce_score(value: Any) -> float | None:
    """Coerce a numeric score field, returning None on failure."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= f <= 1.0):
        return None
    return f


def _make_node(
    node_type: str,
    name: str,
    summary: str = "",
    *,
    strength: Optional[float] = None,
    confidence: Optional[float] = None,
    importance: Optional[float] = None,
    depth: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a node dict in the canonical shape.

    The retriever only requires ``type``, ``name``, and ``summary``;
    the numeric fields are optional and are read where present (see
    :func:`brain.brain_retriever._format_context`).
    """
    node: Dict[str, Any] = {
        "type": node_type,
        "name": name or node_type,
        "summary": summary or "",
    }
    if strength is not None:
        node["strength"] = float(strength)
    if confidence is not None:
        node["confidence"] = float(confidence)
    if importance is not None:
        node["importance"] = float(importance)
    if depth is not None:
        node["depth"] = float(depth)
    if extra:
        node.update(extra)
    return node


# ---------------------------------------------------------------------------
# Legacy schema (in-tree brain_builder / offline interview)
# ---------------------------------------------------------------------------


def _iter_legacy_nodes(graph: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield nodes from the legacy flat schema (built by the in-tree
    brain builder and offline interview)."""
    for item in graph.get("traits", []) or []:
        if isinstance(item, str):
            yield _make_node("trait", item, f"Core: {item}", strength=_DEFAULT_STRENGTH)
        elif isinstance(item, dict):
            yield _make_node(
                "trait",
                item.get("name", "trait"),
                item.get("summary", ""),
                strength=item.get("strength", _DEFAULT_STRENGTH),
            )

    for item in graph.get("beliefs", []) or []:
        if isinstance(item, str):
            yield _make_node("belief", item, item, confidence=_DEFAULT_CONFIDENCE)
        elif isinstance(item, dict):
            yield _make_node(
                "belief",
                item.get("name", "belief"),
                item.get("summary", ""),
                confidence=item.get("confidence", _DEFAULT_CONFIDENCE),
            )

    for item in graph.get("values", []) or []:
        if isinstance(item, str):
            yield _make_node("value", item, f"Value: {item}", importance=_DEFAULT_IMPORTANCE)
        elif isinstance(item, dict):
            yield _make_node(
                "value",
                item.get("name", "value"),
                item.get("summary", ""),
                importance=item.get("importance", _DEFAULT_IMPORTANCE),
            )

    for item in graph.get("boundaries", []) or []:
        if isinstance(item, dict):
            yield _make_node(
                "boundary",
                item.get("name") or item.get("topic", "boundary"),
                item.get("summary", ""),
                extra={"comfort_level": item.get("comfort_level", _DEFAULT_STRENGTH)},
            )
        elif isinstance(item, str):
            yield _make_node("boundary", item, item)

    for item in graph.get("life_events", []) or []:
        if isinstance(item, dict):
            yield _make_node(
                "life_event",
                item.get("name") or item.get("event", "event"),
                _coerce_text(item.get("summary") or item.get("impact", "")),
                extra={"year": item.get("year", ""), "impact": item.get("impact", "")},
            )
        elif isinstance(item, str):
            yield _make_node("life_event", item, item)

    for item in graph.get("people", []) or []:
        if isinstance(item, dict):
            yield _make_node(
                "person",
                item.get("name", "person"),
                _coerce_text(item.get("summary") or item.get("significance", "")),
                extra={
                    "relationship": item.get("relationship") or item.get("role", ""),
                    "significance": item.get("significance", ""),
                },
            )
        elif isinstance(item, str):
            yield _make_node("person", item, item)

    for item in graph.get("memories", []) or []:
        if isinstance(item, dict):
            yield _make_node(
                "memory",
                item.get("name", "memory"),
                item.get("summary", ""),
                extra={"emotional_tone": item.get("emotional_tone", 0.0)},
            )
        elif isinstance(item, str):
            yield _make_node("memory", item, item)

    for item in graph.get("patterns", []) or []:
        if isinstance(item, dict):
            yield _make_node("pattern", item.get("name", "pattern"), item.get("summary", ""))
        elif isinstance(item, str):
            yield _make_node("pattern", item, item)

    for item in graph.get("expertise", []) or []:
        if isinstance(item, dict):
            yield _make_node(
                "expertise",
                item.get("name", "expertise"),
                item.get("summary", ""),
                depth=item.get("depth", _DEFAULT_DEPTH),
            )
        elif isinstance(item, str):
            yield _make_node("expertise", item, item, depth=_DEFAULT_DEPTH)

    for item in graph.get("procedural_patterns", []) or []:
        if isinstance(item, dict):
            yield _make_node(
                "procedural_pattern",
                item.get("name", "procedural"),
                item.get("summary") or _coerce_text(item),
            )
        elif isinstance(item, str):
            yield _make_node("procedural_pattern", item, item)

    for item in graph.get("work_loops", []) or []:
        if isinstance(item, dict):
            yield _make_node("work_loop", item.get("name", "work_loop"), _coerce_text(item))
        elif isinstance(item, str):
            yield _make_node("work_loop", item, item)


# ---------------------------------------------------------------------------
# Marketplace schema (api.openbeam.me)
# ---------------------------------------------------------------------------


def _iter_marketplace_personality_profile(graph: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield nodes from the marketplace ``personality_profile`` block.

    The marketplace personality_profile has top-level fields like
    ``values``, ``core_beliefs``, ``cognitive_patterns``,
    ``communication_style``, ``empathy_indicators``, ``formality``,
    ``humor_frequency`` — each a list of strings or string values.

    Marketplace strings embed their numeric score in two ways:
    either as a ``"Name (0.99): text"`` prefix or a trailing
    ``"text (confidence: 0.99) [ref]"``. :func:`_split_embedded_score`
    extracts both forms so the retriever can rank by the real
    confidence/importance rather than a placeholder 0.5.
    """
    pp = graph.get("personality_profile")
    if not isinstance(pp, dict):
        return

    # String-list fields
    list_fields = (
        ("values", "value", "importance"),
        ("core_beliefs", "belief", "confidence"),
        ("cognitive_patterns", "pattern", "confidence"),
        ("empathy_indicators", "trait", "strength"),
    )
    for field, node_type, score_attr in list_fields:
        items = pp.get(field)
        if not items:
            continue
        if isinstance(items, str):
            items = [items]
        for item in items:
            if isinstance(item, str):
                name, summary, score = _split_embedded_score(item)
                yield _make_node(
                    node_type,
                    name or item,
                    summary or item,
                    **{score_attr: score if score is not None else _DEFAULT_STRENGTH},
                )
            elif isinstance(item, dict):
                explicit = _coerce_score(
                    item.get(score_attr) or item.get("score")
                )
                yield _make_node(
                    node_type,
                    item.get("name", node_type),
                    item.get("summary") or _coerce_text(item),
                    **{score_attr: explicit if explicit is not None else _DEFAULT_STRENGTH},
                )

    # Scalar style fields
    for field, node_type, label in (
        ("communication_style", "trait", "Communication style"),
        ("formality", "trait", "Formality"),
        ("humor_frequency", "trait", "Humor frequency"),
    ):
        val = pp.get(field)
        if not val:
            continue
        yield _make_node(node_type, label, _coerce_text(val), strength=_DEFAULT_STRENGTH)


def _iter_marketplace_knowledge_graph(graph: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield nodes from the marketplace ``knowledge_graph.nodes`` block.

    Each node has ``id``, ``type`` (e.g. Person, Trait, Belief), ``label``,
    ``summary``, and ``attributes``. We use ``type`` to pick a sensible
    canonical node type when possible, otherwise fall back to a generic
    ``"knowledge_node"`` type that the retriever's keyword scoring
    still indexes.

    Numeric scores embedded in ``attributes`` (strength / confidence /
    importance / depth) are promoted to first-class fields so the
    retriever and SOUL.md template can rank by them.
    """
    kg = graph.get("knowledge_graph")
    if not isinstance(kg, dict):
        return
    nodes = kg.get("nodes")
    if not isinstance(nodes, list):
        return

    type_map = {
        "Person": "person",
        "Entity": "person",
        "PersonalityTrait": "trait",
        "Trait": "trait",
        "Belief": "belief",
        "Value": "value",
        "Memory": "memory",
        "EpisodicMemory": "memory",
        "Event": "life_event",
        "LifeEvent": "life_event",
        "Pattern": "pattern",
        "CognitivePattern": "pattern",
        "SocialPattern": "social",
        "StyleProfile": "style",
        "Boundary": "boundary",
        "Expertise": "expertise",
        "KnowledgeDomain": "expertise",
        "Place": "place",
        "Concept": "concept",
    }

    for node in nodes:
        if not isinstance(node, dict):
            continue
        kg_type = node.get("type", "Concept")
        canonical = type_map.get(kg_type, "knowledge_node")
        label = node.get("label") or node.get("id") or kg_type
        # Clean the label too — marketplace sometimes encodes scores
        # in the label itself ("Trait (0.99)").
        clean_label, _, _ = _split_embedded_score(label)
        label = clean_label or label
        summary = node.get("summary") or _coerce_text(node.get("attributes", {}))
        attrs = node.get("attributes") or {}

        kwargs: Dict[str, Any] = {}
        if isinstance(attrs, dict):
            for key in ("confidence", "strength", "importance", "depth"):
                if key in attrs and attrs[key] is not None:
                    coerced = _coerce_score(attrs[key])
                    if coerced is not None:
                        kwargs[key] = coerced

        yield _make_node(canonical, label, summary, **kwargs)


def _iter_marketplace_knowledge_domains(graph: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield nodes from the marketplace ``knowledge_domains`` block.

    Each domain has ``topic``, ``community_summary``, ``key_entities``,
    ``confidence``, ``source_count``. The community_summary is
    high-signal — it's exactly what the retriever should index.
    """
    domains = graph.get("knowledge_domains")
    if not isinstance(domains, list):
        return
    for domain in domains:
        if not isinstance(domain, dict):
            continue
        topic = domain.get("topic", "domain")
        summary = domain.get("community_summary", "")
        if not summary:
            # Fall back to key entities if no community summary
            entities = domain.get("key_entities", [])
            if entities:
                summary = f"Key entities: {_coerce_text(entities)}"
        if not summary:
            continue
        yield _make_node(
            "expertise",
            topic,
            summary,
            depth=domain.get("confidence", _DEFAULT_DEPTH),
            extra={"source_count": domain.get("source_count", 0)},
        )


def _iter_marketplace_episodic_memories(graph: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield nodes from the marketplace ``episodic_memories`` block."""
    memories = graph.get("episodic_memories")
    if not isinstance(memories, list):
        return
    for mem in memories:
        if not isinstance(mem, dict):
            continue
        name = mem.get("name") or mem.get("title") or "memory"
        summary = mem.get("summary") or mem.get("content") or _coerce_text(mem)
        yield _make_node(
            "memory",
            name,
            summary,
            extra={"emotional_tone": mem.get("emotional_tone", 0.0)},
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# All adapter functions in iteration order. Order matters: legacy
# nodes first so the in-tree tests that use the legacy schema keep
# producing the same result for the same input. Marketplace adapters
# only add to the output — they never replace legacy nodes.
_NODE_ITERATORS = (
    _iter_legacy_nodes,
    _iter_marketplace_personality_profile,
    _iter_marketplace_knowledge_graph,
    _iter_marketplace_knowledge_domains,
    _iter_marketplace_episodic_memories,
)


def iter_nodes(graph: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield canonical nodes from any supported brain schema.

    See module docstring for which fields each schema maps to. The
    yielded dicts match the shape produced by
    :func:`brain.brain_retriever._search_nodes` so the rest of the
    pipeline (search, context, stats, SOUL.md) keeps working.
    """
    if not isinstance(graph, dict):
        return
    for fn in _NODE_ITERATORS:
        try:
            yield from fn(graph)
        except Exception as exc:  # never let one bad adapter kill the whole graph
            logger.debug("Schema adapter %s failed: %s", fn.__name__, exc)


def collect_nodes(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Materialize :func:`iter_nodes` into a list."""
    return list(iter_nodes(graph))


def iter_edges(graph: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield edges from any supported schema.

    The legacy schema has top-level ``edges`` (a list of dicts); the
    marketplace schema nests them under ``knowledge_graph.edges``.
    Yields copies so callers can mutate freely.
    """
    if not isinstance(graph, dict):
        return
    legacy = graph.get("edges")
    if isinstance(legacy, list):
        for edge in legacy:
            if isinstance(edge, dict):
                yield dict(edge)
    kg = graph.get("knowledge_graph")
    if isinstance(kg, dict):
        kg_edges = kg.get("edges")
        if isinstance(kg_edges, list):
            for edge in kg_edges:
                if isinstance(edge, dict):
                    yield dict(edge)


def has_marketplace_payload(graph: Dict[str, Any]) -> bool:
    """True if *graph* carries marketplace-schema data the legacy
    schema didn't have.

    Used by the brain-tools plugin's session-start hook to detect a
    brain that hasn't been materialized into SOUL.md yet (the legacy
    schema would have produced a 582-byte empty SOUL.md, so an empty
    SOUL.md + ``has_marketplace_payload(graph)`` is the recovery
    signal).
    """
    if not isinstance(graph, dict):
        return False
    if isinstance(graph.get("personality_profile"), dict):
        return True
    kg = graph.get("knowledge_graph")
    if isinstance(kg, dict) and isinstance(kg.get("nodes"), list) and kg.get("nodes"):
        return True
    if isinstance(graph.get("knowledge_domains"), list) and graph.get("knowledge_domains"):
        return True
    if isinstance(graph.get("episodic_memories"), list) and graph.get("episodic_memories"):
        return True
    return False


def coverage(graph: Dict[str, Any]) -> Dict[str, int]:
    """Return per-type node counts derived via :func:`iter_nodes`.

    Cheap to call (single pass over the graph); used by stats and
    by the on_session_start hook to decide whether the brain has
    enough data to be worth surfacing.
    """
    counts: Dict[str, int] = {}
    for node in iter_nodes(graph):
        ntype = node.get("type", "unknown")
        counts[ntype] = counts.get(ntype, 0) + 1
    return counts


__all__ = [
    "iter_nodes",
    "collect_nodes",
    "iter_edges",
    "has_marketplace_payload",
    "coverage",
]
