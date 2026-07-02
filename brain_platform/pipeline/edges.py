"""Single-source registry of personality-graph edge types (plan/10).

Three places used to need updating per new edge type:
  - `EDGE_TYPES` set in brain_schema.py
  - validation allowlist in brain_extractor.py (twice — Pass 2 and Pass 2.5)
  - prompt body listing edge types (Pass 2 and Pass 2.5)

That split caused a real Phase A bug: I added 5 new edges to the set and
allowlist but forgot the prompt body, so the LLM never generated them.

This module is the single source. `EDGE_TYPES` derives from it. The prompt
text is rendered from the same registry. Validation reads it directly.

Adding a new edge type is one change here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeDescriptor:
    """One row in the edge-type registry.

    Direction reads left-to-right: source VERB target.
    """
    name: str
    source_target_hint: str  # e.g., "Event/Memory → Belief/Value/Trait"
    examples: tuple[str, ...] = ()
    is_hub: bool = False     # THE_USER → personality construct
    layer: str = "personality"  # "personality" | "procedural"


# Canonical registry. Adding a new edge type = adding one entry here.
EDGE_REGISTRY: dict[str, EdgeDescriptor] = {
    # ── Personality cross-link edges ───────────────────────────
    "SHAPED": EdgeDescriptor(
        name="SHAPED",
        source_target_hint="Event/Memory → Belief/Value/Trait",
        examples=('"quitting the agency job" SHAPED "belief in sustainability"',),
    ),
    "EVOLVED_INTO": EdgeDescriptor(
        name="EVOLVED_INTO",
        source_target_hint="OldBelief → NewBelief",
        examples=('"hustle culture is the way" EVOLVED_INTO "rest is productive"',),
    ),
    "ENFORCES": EdgeDescriptor(
        name="ENFORCES",
        source_target_hint="Value → Boundary",
        examples=('"honesty" ENFORCES "won\'t lie about product impact"',),
    ),
    "GUIDES": EdgeDescriptor(
        name="GUIDES",
        source_target_hint="Value/Belief → Pattern",
        examples=('"autonomy" GUIDES "two-column pros/cons analysis"',),
    ),
    "TAUGHT": EdgeDescriptor(
        name="TAUGHT",
        source_target_hint="Event → Pattern",
        examples=('"first product line failure" TAUGHT "test worst conditions first"',),
    ),
    "EXPRESSES": EdgeDescriptor(
        name="EXPRESSES",
        source_target_hint="Trait → Style/Social",
        examples=('"more direct in writing" EXPRESSES "emails 3 words or 3 paragraphs"',),
    ),
    "TESTED": EdgeDescriptor(
        name="TESTED",
        source_target_hint="Event → Boundary",
        examples=('"losing $200K contract" TESTED "won\'t lie about product impact"',),
    ),
    "INFORMS": EdgeDescriptor(
        name="INFORMS",
        source_target_hint="Expertise → Belief/Value",
        examples=('"bioplastics expertise" INFORMS "biodegradable ≠ compostable"',),
    ),
    "INVOLVES": EdgeDescriptor(
        name="INVOLVES",
        source_target_hint="Event/Memory → Person/Place",
        examples=(
            '"quitting the agency job" INVOLVES "Priya"',
            '"first year after quitting" INVOLVES "Portland"',
        ),
    ),
    # ── Hub edges (THE_USER → personality construct) ──────────
    "HAS_TRAIT": EdgeDescriptor(
        name="HAS_TRAIT", source_target_hint="THE_USER → Trait/Pattern", is_hub=True,
    ),
    "HOLDS": EdgeDescriptor(
        name="HOLDS", source_target_hint="THE_USER → Belief", is_hub=True,
    ),
    "DRIVEN_BY": EdgeDescriptor(
        name="DRIVEN_BY", source_target_hint="THE_USER → Value/Boundary", is_hub=True,
    ),
    "EXPERIENCED": EdgeDescriptor(
        name="EXPERIENCED", source_target_hint="THE_USER → LifeEvent/Memory", is_hub=True,
    ),
    "EXPERT_IN": EdgeDescriptor(
        name="EXPERT_IN", source_target_hint="THE_USER → Expertise", is_hub=True,
    ),
    "HANDLES_CONFLICT_BY": EdgeDescriptor(
        name="HANDLES_CONFLICT_BY", source_target_hint="THE_USER → SocialPattern", is_hub=True,
    ),
    "COMMUNICATES_VIA": EdgeDescriptor(
        name="COMMUNICATES_VIA", source_target_hint="THE_USER → StyleProfile", is_hub=True,
    ),
    # ── Procedural memory edges (plan/10 §6.2) ─────────────────
    "OPERATES_IN": EdgeDescriptor(
        name="OPERATES_IN",
        source_target_hint="ProceduralPattern/WorkLoop/PromptingStyle → Expertise",
        examples=('"small-PR refactoring" OPERATES_IN "frontend systems"',),
        layer="procedural",
    ),
    "EXPLAINED_BY": EdgeDescriptor(
        name="EXPLAINED_BY",
        source_target_hint="ProceduralPattern/WorkLoop → Belief/Value/Trait",
        examples=('"small-PR refactoring" EXPLAINED_BY "values reversibility"',),
        layer="procedural",
    ),
    "ASPIRES_TO": EdgeDescriptor(
        name="ASPIRES_TO",
        source_target_hint="Trait/Value → ProceduralPattern (interview-source only)",
        examples=('"values rigor" ASPIRES_TO "hypothesis-first debugging"',),
        layer="procedural",
    ),
    "CONTRADICTS": EdgeDescriptor(
        name="CONTRADICTS",
        source_target_hint=(
            "artifact-source procedural pattern → interview-source procedural pattern"
        ),
        examples=(),  # only emitted when artifact + interview disagree; no interview-only example
        layer="procedural",
    ),
    "WORKS_BY": EdgeDescriptor(
        name="WORKS_BY",
        source_target_hint="THE_USER → ProceduralPattern/WorkLoop/PromptingStyle/TechnicalGap",
        is_hub=True,
        layer="procedural",
    ),
    "SUPERSEDES": EdgeDescriptor(
        name="SUPERSEDES",
        source_target_hint="new ProceduralPattern/WorkLoop → old ProceduralPattern/WorkLoop",
        examples=('"hypothesis-first debugging (2026-05)" SUPERSEDES "print-statement debugging (2026-01)"',),
        layer="procedural",
    ),
}


# Derived. Public API — replaces the old hardcoded set in brain_schema.
EDGE_TYPES: frozenset[str] = frozenset(EDGE_REGISTRY.keys())

# Edge types the LLM may emit in Pass 2 (everything except hub edges, which
# the writer auto-generates from THE_USER).
NON_HUB_EDGE_TYPES: frozenset[str] = frozenset(
    name for name, d in EDGE_REGISTRY.items() if not d.is_hub
)


def render_edge_types_block(*, include_procedural: bool = True) -> str:
    """Render the EDGE TYPES section of the Pass 2 / Pass 2.5 prompts.

    Hub edges are excluded (the writer auto-generates them from THE_USER).
    Procedural edges are included by default but can be skipped for legacy
    extraction paths.
    """
    parts: list[str] = []
    cross_links = [
        d for d in EDGE_REGISTRY.values()
        if not d.is_hub and (include_procedural or d.layer != "procedural")
    ]

    # Group personality first, procedural second, in registry order.
    personality = [d for d in cross_links if d.layer == "personality"]
    procedural = [d for d in cross_links if d.layer == "procedural"]

    for d in personality:
        parts.append(f"  {d.name}: {d.source_target_hint}")
        for ex in d.examples:
            parts.append(f"    {ex}")
        parts.append("")  # blank line between entries

    if procedural:
        parts.append(
            "  ── Procedural memory edges (plan/10 §6.2) — only used if the entity list"
        )
        parts.append(
            "     contains ProceduralPattern / WorkLoop / PromptingStyle / TechnicalGap nodes:"
        )
        parts.append("")
        for d in procedural:
            parts.append(f"  {d.name}: {d.source_target_hint}")
            for ex in d.examples:
                parts.append(f"    {ex}")
            parts.append("")

    return "\n".join(parts).rstrip()


def render_compact_edge_list(*, include_procedural: bool = True) -> str:
    """Compact one-line-per-edge listing for the repair prompt."""
    lines = []
    for d in EDGE_REGISTRY.values():
        if d.is_hub:
            continue
        if not include_procedural and d.layer == "procedural":
            continue
        lines.append(f"  {d.name}: {d.source_target_hint}")
    return "\n".join(lines)
