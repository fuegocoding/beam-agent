"""Hard exclusions for the artifact pipeline (plan/10 §8.3, §11 invariant).

Two enforcement points:

1. `verify_features_clean(features)` — sanity check that Stage 1 features
   contain only scalars / booleans / short labels, NEVER raw text payloads.
   Run after every Stage 1 extraction; blocks the boundary into Stage 2.

2. `looks_like_code(text)` — heuristic detector for code-bearing strings.
   Run on every Stage-2-generated node's text fields before they leave the
   ingester. Nodes containing code-shaped text are dropped, not sanitized.

These are belt-and-braces. The Stage 2 prompt is also explicitly told not to
emit code; this catches the case where the model regresses anyway.
"""

import re

# Allowed types for ArtifactFeatures.features values. Lists/dicts are allowed
# but only when they bottom out in these primitives.
_ALLOWED_PRIMITIVES = (str, int, float, bool, type(None))

# A "short label" is up to this many characters. Anything longer in a feature
# value is suspicious — likely raw text leaking through.
MAX_FEATURE_STRING_LEN = 80


# Heuristic patterns that indicate code-shaped content. Tuned for low false-
# positive rate on prose; tolerates the occasional natural-language mention
# of "function" or "import" without flagging.
_CODE_PATTERNS: list[re.Pattern[str]] = [
    # Python/JS function or class definitions (with parens, colons, or extends)
    re.compile(r"\b(?:def|function)\s+\w+\s*\("),
    re.compile(r"\bclass\s+\w+\s*(?:\(|:|\s+extends\s+\w+|\s+implements\s+\w+)"),
    # Python imports / from-imports as full statements
    re.compile(r"^(?:from\s+\w+(?:\.\w+)*\s+import|import\s+\w+(?:\.\w+)*)", re.MULTILINE),
    # JS/TS module imports
    re.compile(r"^import\s+[\w{},*\s]+\s+from\s+['\"][^'\"]+['\"]", re.MULTILINE),
    # Arrow functions
    re.compile(r"\bconst\s+\w+\s*=\s*\([^)]*\)\s*=>"),
    # Triple-backtick code fences
    re.compile(r"```[a-zA-Z]*\n"),
    # Diff hunks
    re.compile(r"^@@\s*-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s*@@", re.MULTILINE),
    re.compile(r"^(?:\+\+\+|---)\s+[a-zA-Z]/", re.MULTILINE),
    # Common multi-line code shapes
    re.compile(r"\breturn\s+(?:\w+\.\w+\(|\{|\[)"),
    re.compile(r"\b(?:if|for|while)\s*\([^)]*\)\s*\{"),
    # SQL — split into focused patterns to keep false-positive rate low
    re.compile(r"\bSELECT\b\s+[\w*,\s.]+\s+FROM\s+\w+", re.IGNORECASE),
    re.compile(r"\b(?:CREATE|ALTER|DROP)\s+(?:TABLE|INDEX|VIEW|DATABASE)\s+\w+", re.IGNORECASE),
    re.compile(r"\bINSERT\s+INTO\s+\w+", re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\s+\w+", re.IGNORECASE),
    re.compile(r"\bUPDATE\s+\w+\s+SET\b", re.IGNORECASE),
]


def looks_like_code(text: str | None) -> bool:
    """Heuristic: does `text` look like code or a diff?

    True positives: function defs, import statements, code fences, diff hunks.
    Tolerated false negatives: very small snippets, prose that mentions code
    in passing. The goal is to catch leaks, not to be a parser.
    """
    if not text:
        return False
    for p in _CODE_PATTERNS:
        if p.search(text):
            return True
    return False


def verify_features_clean(features: dict) -> list[str]:
    """Validate that a Stage 1 feature dict contains no raw text payloads.

    Returns a list of violation messages. Empty list = clean.

    Rules:
    - Every leaf must be a scalar/bool/None.
    - String values must be ≤ MAX_FEATURE_STRING_LEN chars.
    - String values must NOT match a code pattern (heuristic).
    - dict / list values are recursed into.
    """
    violations: list[str] = []
    _walk(features, "features", violations)
    return violations


def _walk(value: object, path: str, violations: list[str]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _walk(v, f"{path}.{k}", violations)
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _walk(v, f"{path}[{i}]", violations)
        return
    if isinstance(value, str):
        if len(value) > MAX_FEATURE_STRING_LEN:
            violations.append(
                f"{path}: string of {len(value)} chars exceeds limit of "
                f"{MAX_FEATURE_STRING_LEN} (Stage 1 features must be short labels)"
            )
        if looks_like_code(value):
            violations.append(f"{path}: value looks like code or a diff")
        return
    if not isinstance(value, _ALLOWED_PRIMITIVES):
        violations.append(
            f"{path}: type {type(value).__name__} not allowed in features "
            f"(only str/int/float/bool/None/list/dict)"
        )


def assert_node_clean(node) -> list[str]:
    """Assert a Stage-2 generated node has no code-shaped text in its fields.

    Pass any pydantic model with text fields. Returns violation list.
    """
    violations: list[str] = []
    if hasattr(node, "model_dump"):
        data = node.model_dump()
    else:
        data = dict(node) if isinstance(node, dict) else {}
    for field, value in data.items():
        if isinstance(value, str) and looks_like_code(value):
            violations.append(f"{type(node).__name__}.{field}: contains code-shaped text")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, str) and looks_like_code(v):
                    violations.append(
                        f"{type(node).__name__}.{field}[{i}]: contains code-shaped text"
                    )
    return violations
