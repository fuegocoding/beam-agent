"""Entity resolution — merge role-only "wife" / "mother" nodes into named people.

Lifted from the cloud's ``brain_extractor.py:_resolve_people_entities``
(``lines 749-849``). Runs between Pass 1 (entity extraction) and
Pass 2 (edge extraction) so the edge pass sees a deduplicated people
list.

The cloud's CHANGELOG-brain-extraction-upgrade.md §3 calls this
"the single biggest fix" — without it, the LLM creates separate
``wife`` and ``Sarah`` nodes, then can't draw edges between the
right pairs.

Algorithm (unchanged from the cloud):

1. Walk the ``people`` list, separate role-only nodes (name ∈
   ROLE_LABELS) from named nodes (proper-noun heuristic).
2. For each role node, score every named node by co-occurrence in
   summaries/roles/surrounding text. Higher score = more likely
   this role refers to this name.
3. If the best score is >= 2, merge: copy the role into the named
   node's role field, prepend the role's summary if useful, drop
   the role node.
4. Rewrite any edges that referenced the old name.

This is pure data — no LLM calls — and runs in O(P^2 + E) where P
is the people count and E is the edge count.
"""
from __future__ import annotations

import logging

from brain_platform.pipeline.brain_schema import PersonalityGraph, ROLE_LABELS

logger = logging.getLogger(__name__)


def resolve_people_entities(graph: PersonalityGraph) -> PersonalityGraph:
    """Merge role-only nodes into named people based on co-occurrence scoring.

    The cloud's original name was ``_resolve_people_entities`` (private
    leading underscore). The local port renames it to public so
    :class:`BrainExtractor` can call it without underscore-mangling
    in tests.
    """
    role_nodes: dict[str, int] = {}
    named_nodes: list[int] = []

    for i, person in enumerate(graph.people):
        name_lower = person.name.lower().strip()
        if name_lower in ROLE_LABELS:
            role_nodes[name_lower] = i
        elif person.name[0:1].isupper() and not any(c.isdigit() for c in person.name):
            named_nodes.append(i)

    if not role_nodes or not named_nodes:
        return graph

    # Build text corpus for matching
    all_text = " ".join(
        f"{p.name} {p.role} {p.summary}" for p in graph.people
    ).lower()

    # Also include all node summaries for context
    for attr in ("traits", "beliefs", "values", "boundaries", "life_events",
                 "memories", "patterns", "social", "expertise", "style", "places"):
        for node in getattr(graph, attr, []):
            summary = getattr(node, "summary", "") or getattr(node, "significance", "") or ""
            all_text += f" {node.name} {summary}".lower()

    merge_map: dict[str, str] = {}
    indices_to_remove: set[int] = set()

    for role_label, role_idx in role_nodes.items():
        role_person = graph.people[role_idx]
        best_match_idx = None
        best_score = 0

        for named_idx in named_nodes:
            named_person = graph.people[named_idx]
            name_lower = named_person.name.lower()
            score = 0

            # Check if name appears near role in any text
            for p in graph.people:
                text = f"{p.name} {p.role} {p.summary}".lower()
                if name_lower in text and role_label in text:
                    score += 3

            # Check all node summaries for co-occurrence
            for attr in ("life_events", "memories", "social"):
                for node in getattr(graph, attr, []):
                    text = f"{node.name} {node.summary}".lower()
                    if name_lower in text and role_label in text:
                        score += 2

            # Check if role is in the named person's role field
            if role_label in named_person.role.lower():
                score += 5

            # Check if name appears in role node's summary
            if name_lower in role_person.summary.lower():
                score += 3

            if score > best_score:
                best_score = score
                best_match_idx = named_idx

        if best_match_idx is not None and best_score >= 2:
            named_person = graph.people[best_match_idx]
            # Absorb role into named person
            if role_label not in named_person.role.lower():
                named_person.role = (
                    f"{named_person.role}, {role_label}" if named_person.role else role_label
                )
            if role_person.summary and role_person.summary not in named_person.summary:
                named_person.summary = f"{named_person.summary} {role_person.summary}".strip()

            merge_map[role_person.name] = named_person.name
            indices_to_remove.add(role_idx)
            logger.info(
                "Entity resolution: merged '%s' into '%s' (score=%d)",
                role_person.name, named_person.name, best_score,
            )

    if not indices_to_remove:
        return graph

    # Remove merged nodes
    graph.people = [p for i, p in enumerate(graph.people) if i not in indices_to_remove]

    # Fix any edges that referenced the old name
    for edge in graph.edges:
        if edge.source_name in merge_map:
            edge.source_name = merge_map[edge.source_name]
        if edge.target_name in merge_map:
            edge.target_name = merge_map[edge.target_name]

    return graph


# Public alias matching the cloud's private name — kept for source
# compatibility with the cloud's ``brain_extractor.py`` which calls
# ``_resolve_people_entities`` directly.
_resolve_people_entities = resolve_people_entities


__all__ = ["resolve_people_entities", "_resolve_people_entities"]
