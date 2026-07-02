"""Extraction prompts for code and prompt artifacts.

These wrap uploaded code/prompts into a format the BrainExtractor can process.
The key difference from interview extraction: these are OBSERVED patterns
(OBSERVED_AS edges), not self-reported (ASPIRES_TO edges).

The prompts reformat the raw artifact + its metadata into an "interview-like"
text that the existing 3-pass extractor can process, with instructions to
focus on procedural patterns.
"""


def build_code_extraction_text(
    code: str,
    language: str,
    language_family: str,
    metadata: dict,
) -> str:
    """Wrap a code file into extraction-ready text for the BrainExtractor.

    The extractor sees this as a special "interview" where the person's code
    speaks for them. We frame it to focus on procedural patterns.
    """
    # Build a language-specific lens
    lang_lens = _get_language_lens(language)

    # Build structural summary from metadata
    structural = []
    if metadata.get("total_lines"):
        structural.append(f"Total lines: {metadata['total_lines']}")
    if metadata.get("code_lines"):
        structural.append(f"Code lines: {metadata['code_lines']}")
    if metadata.get("comment_lines"):
        structural.append(f"Comment lines: {metadata['comment_lines']}")
    if metadata.get("comment_ratio"):
        structural.append(f"Comment ratio: {metadata['comment_ratio']}")
    for feature in [
        "has_tests", "has_docstrings", "has_type_annotations",
        "has_error_handling", "has_async", "has_classes",
    ]:
        if metadata.get(feature):
            structural.append(f"{feature.replace('has_', '').replace('_', ' ').title()}: Yes")

    structural_block = "\n".join(f"  - {s}" for s in structural) if structural else "  (none detected)"

    return f"""[ARTIFACT ANALYSIS — Code File]
This is a code file the person uploaded as representative of how they work.
Extract procedural patterns from it — HOW they think, structure, and solve
problems. These are OBSERVED patterns (not self-reported).

IMPORTANT: Set source="artifact" on all extracted ProceduralPattern,
WorkLoop, and PromptingStyle nodes. These get OBSERVED_AS edges (not
ASPIRES_TO) to enable aspirational/observed gap detection.

Language: {language} ({language_family})
File: {metadata.get('source_file', 'unknown')}

Structural Profile:
{structural_block}

{lang_lens}

WHAT TO EXTRACT FROM THIS CODE:

1. ARCHITECTURE PATTERNS — How do they structure their code?
   - Module/function/class organization
   - Separation of concerns approach
   - Abstraction level (when they extract helpers vs inline)
   - Composition vs inheritance preferences
   - File size preferences (one big module vs many small ones)

2. NAMING CONVENTIONS — What does their naming reveal?
   - Descriptive vs terse names
   - Consistent style or mixed
   - Domain-specific vocabulary in names

3. ERROR HANDLING PHILOSOPHY — How do they handle failure?
   - Defensive (check everything) vs optimistic (happy path first)
   - Error types they define vs generic exceptions
   - Recovery patterns vs fail-fast

4. DOCUMENTATION STYLE — How do they communicate through code?
   - Inline comments: frequent, sparse, or absent
   - Docstrings: formal, informal, or none
   - Self-documenting code vs explained code

5. TESTING APPROACH (if present) — What do they validate?
   - Test granularity (unit vs integration)
   - Edge cases they think about
   - Test naming style

6. PATTERNS & ANTI-PATTERNS — What do they consistently do or avoid?
   - Code they're proud of reveals positive patterns
   - Consistent choices across the file reveal habits

DO NOT extract: specific library names as skills, framework versions,
language features as expertise ratings. Extract HOW they use the language,
not THAT they use it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE CODE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{code}
"""


def build_prompt_extraction_text(
    prompt_text: str,
    metadata: dict,
) -> str:
    """Wrap a user-authored prompt into extraction-ready text.

    Extracts prompting style patterns as OBSERVED behavior.
    """
    # Build feature summary
    features = []
    for feature in [
        "has_examples", "has_constraints", "has_chain_of_thought",
        "has_persona", "has_output_format", "has_context_setting",
    ]:
        if metadata.get(feature):
            features.append(feature.replace("has_", "").replace("_", " ").title())

    features_block = ", ".join(features) if features else "none detected"
    length = metadata.get("length_class", "unknown")

    return f"""[ARTIFACT ANALYSIS — User-Authored Prompt]
This is a prompt the person wrote for directing an AI or structuring a task.
Extract their prompting and delegation style. These are OBSERVED patterns.

IMPORTANT: Set source="artifact" on all extracted PromptingStyle and
ProceduralPattern nodes. These get OBSERVED_AS edges.

Prompt length: {metadata.get('word_count', '?')} words ({length})
Detected features: {features_block}

WHAT TO EXTRACT FROM THIS PROMPT:

1. STRUCTURE — How do they organize instructions?
   - Spec-first (define what you want, then constraints)
   - Example-first (show what good looks like, then rules)
   - Chain-of-thought (walk through reasoning steps)
   - Persona-driven (set a role, then task)
   - Hybrid approaches

2. CONSTRAINT PHRASING — How do they say "don't"?
   - Positive framing ("always X") vs negative ("never Y")
   - Explicit rules vs implicit through examples
   - How many constraints vs how much freedom

3. VERBOSITY & DETAIL — How much do they specify?
   - Terse (trust the AI to figure it out)
   - Moderate (key points, room for interpretation)
   - Detailed (comprehensive spec, little ambiguity)
   - Comprehensive (exhaustive, covers edge cases)

4. CONTEXT-SETTING — How do they frame the task?
   - Background given vs assumed
   - Audience specified vs implicit
   - Success criteria defined vs open-ended

5. OUTPUT SPECIFICATION — How do they define "done"?
   - Format specified (JSON, markdown, etc.)
   - Length/scope bounded
   - Quality criteria stated

DO NOT extract: which AI model they target, specific tool names,
the content of the task itself. Extract HOW they instruct, not WHAT
they instruct about.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE PROMPT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{prompt_text}
"""


def _get_language_lens(language: str) -> str:
    """Return language-specific extraction guidance."""
    lenses = {
        "python": """PYTHON-SPECIFIC PATTERNS TO LOOK FOR:
  - Dataclasses/attrs vs raw dicts — structured thinker vs scripting fast
  - Type hints: strict (mypy-passing) vs loose (occasional) vs absent
  - Async patterns: asyncio, threading, or sync-only
  - Decorator usage: custom decorators suggest metaprogramming comfort
  - List comprehensions vs explicit loops — Pythonic fluency
  - Import organization: stdlib / third-party / local separation""",

        "typescript": """TYPESCRIPT-SPECIFIC PATTERNS TO LOOK FOR:
  - Type safety level: strict mode, no-any, or loose with escape hatches
  - Generics usage: basic vs advanced (conditional types, mapped types)
  - Interface vs type — architectural preference
  - Component patterns (if React): functional vs class, hooks style
  - Enum usage vs union types — design philosophy
  - Null handling: strict null checks, optional chaining depth""",

        "javascript": """JAVASCRIPT-SPECIFIC PATTERNS TO LOOK FOR:
  - No type system is itself a signal — speed over safety preference
  - Module patterns: ESM vs CommonJS, barrel exports
  - Callback vs Promise vs async/await — modernity of approach
  - Prototype manipulation vs class syntax
  - Error handling: try/catch discipline or happy-path coding
  - Variable declarations: const-first, let, or mixed""",

        "java": """JAVA-SPECIFIC PATTERNS TO LOOK FOR:
  - OOP depth: deep hierarchies vs flat composition
  - Design pattern usage: which patterns they reach for
  - Annotation style: heavy Spring-style vs minimal
  - Stream API adoption vs imperative loops
  - Exception handling: checked vs unchecked philosophy
  - Generics usage sophistication""",

        "r": """R-SPECIFIC PATTERNS TO LOOK FOR:
  - Tidyverse vs base R — community alignment and style preference
  - Vectorized operations vs explicit loops — R fluency level
  - Pipe operator usage (%>% or |>) — modern R style
  - Statistical thinking patterns in how they structure analysis
  - Data wrangling approach: dplyr chains vs manual indexing
  - Visualization philosophy: ggplot layers vs base plot""",

        "go": """GO-SPECIFIC PATTERNS TO LOOK FOR:
  - Error handling discipline: if err != nil patterns, wrapping depth
  - Interface design: small interfaces, implicit satisfaction
  - Goroutine/channel patterns — concurrency mental model
  - Package organization: flat vs nested
  - Naming: Go conventions adherence (short names, exported caps)
  - Struct composition vs embedding""",

        "rust": """RUST-SPECIFIC PATTERNS TO LOOK FOR:
  - Ownership patterns: borrowing discipline, lifetime annotations
  - unsafe usage: frequency and justification style
  - Trait design: trait objects vs generics vs enum dispatch
  - Error handling: Result/Option patterns, custom error types
  - Pattern matching exhaustiveness
  - Memory layout awareness in struct design""",

        "sql": """SQL-SPECIFIC PATTERNS TO LOOK FOR:
  - Query complexity: simple selects vs multi-CTE pipelines
  - Naming conventions: tables, columns, aliases
  - JOIN style: explicit vs implicit, LEFT vs INNER defaults
  - Subquery vs CTE preference — readability priority
  - Window function usage — analytical sophistication
  - Schema design philosophy visible in queries""",

        "shell": """SHELL-SPECIFIC PATTERNS TO LOOK FOR:
  - Error handling: set -euo pipefail discipline
  - Variable quoting discipline
  - Function usage vs linear scripts
  - Portability: bash-specific vs POSIX-compatible
  - Pipeline composition style
  - Defensive vs optimistic scripting""",
    }

    return lenses.get(language, f"""GENERAL PATTERNS TO LOOK FOR ({language}):
  - Code organization and structure
  - Naming conventions and consistency
  - Error handling approach
  - Documentation and commenting style
  - Abstraction and composition patterns""")
