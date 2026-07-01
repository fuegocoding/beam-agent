"""Parser for code files — extracts text with language metadata.

Detects language from file extension. JS and TS are distinct languages:
TS reveals type safety habits, interface design, generics usage that JS doesn't.
The absence of types in JS is itself a signal about engineering philosophy.

The parser preserves full source for the extraction prompt but tags it with
structural metadata (language, LOC, has_tests, has_docstrings, etc.) so the
LLM extraction can apply language-specific pattern recognition.

Code is NEVER stored verbatim in the graph — only extracted patterns.
"""

from brain_platform.pipeline.parsers.base import BaseParser, ParseResult

# Extension → (language_name, language_family)
EXTENSION_MAP: dict[str, tuple[str, str]] = {
    ".py": ("python", "scripting"),
    ".pyw": ("python", "scripting"),
    ".ts": ("typescript", "web"),
    ".tsx": ("typescript", "web"),
    ".js": ("javascript", "web"),
    ".jsx": ("javascript", "web"),
    ".mjs": ("javascript", "web"),
    ".cjs": ("javascript", "web"),
    ".java": ("java", "enterprise"),
    ".kt": ("kotlin", "enterprise"),
    ".kts": ("kotlin", "enterprise"),
    ".r": ("r", "data-science"),
    ".rmd": ("r", "data-science"),
    ".go": ("go", "systems"),
    ".rs": ("rust", "systems"),
    ".rb": ("ruby", "scripting"),
    ".swift": ("swift", "mobile"),
    ".cs": ("csharp", "enterprise"),
    ".cpp": ("cpp", "systems"),
    ".cc": ("cpp", "systems"),
    ".cxx": ("cpp", "systems"),
    ".c": ("c", "systems"),
    ".h": ("c", "systems"),
    ".hpp": ("cpp", "systems"),
    ".sql": ("sql", "data"),
    ".sh": ("shell", "devops"),
    ".bash": ("shell", "devops"),
    ".zsh": ("shell", "devops"),
    ".php": ("php", "web"),
    ".scala": ("scala", "enterprise"),
    ".ex": ("elixir", "functional"),
    ".exs": ("elixir", "functional"),
    ".erl": ("erlang", "functional"),
    ".hs": ("haskell", "functional"),
    ".ml": ("ocaml", "functional"),
    ".lua": ("lua", "scripting"),
    ".dart": ("dart", "mobile"),
    ".vue": ("vue", "web"),
    ".svelte": ("svelte", "web"),
}

# Comment syntax per language for structural analysis
COMMENT_SINGLE: dict[str, str] = {
    "python": "#",
    "ruby": "#",
    "r": "#",
    "shell": "#",
    "javascript": "//",
    "typescript": "//",
    "java": "//",
    "kotlin": "//",
    "go": "//",
    "rust": "//",
    "swift": "//",
    "csharp": "//",
    "cpp": "//",
    "c": "//",
    "scala": "//",
    "dart": "//",
    "php": "//",
    "sql": "--",
    "lua": "--",
    "haskell": "--",
    "elixir": "#",
    "erlang": "%",
}


def _detect_language(filename: str) -> tuple[str, str]:
    """Detect language and family from filename extension."""
    lower = filename.lower()
    for ext, (lang, family) in EXTENSION_MAP.items():
        if lower.endswith(ext):
            return lang, family
    return "unknown", "unknown"


def _count_structural_features(text: str, language: str) -> dict:
    """Extract lightweight structural metadata from code text."""
    lines = text.splitlines()
    total_lines = len(lines)
    blank_lines = sum(1 for l in lines if not l.strip())
    comment_prefix = COMMENT_SINGLE.get(language)

    comment_lines = 0
    if comment_prefix:
        comment_lines = sum(
            1 for l in lines if l.strip().startswith(comment_prefix)
        )

    code_lines = total_lines - blank_lines - comment_lines

    # Detect common patterns
    text_lower = text.lower()
    has_tests = any(
        marker in text_lower
        for marker in [
            "def test_", "it(", "describe(", "test(", "@test",
            "assert", "expect(", "#[test]", "func test",
        ]
    )
    has_docstrings = '"""' in text or "'''" in text or "/**" in text
    has_type_annotations = any(
        marker in text
        for marker in [
            "-> ", ": str", ": int", ": float", ": bool", ": list",
            ": Dict", ": List", ": Optional", ": Any",
            "interface ", "type ", ": string", ": number", ": boolean",
        ]
    )
    has_error_handling = any(
        marker in text
        for marker in [
            "try:", "except ", "try {", "catch ", "catch(",
            "rescue", "Result<", "Result::", "if err !=",
        ]
    )
    has_async = any(
        marker in text
        for marker in ["async ", "await ", "asyncio", "Promise", "goroutine"]
    )
    has_classes = any(
        marker in text
        for marker in [
            "class ", "struct ", "impl ", "interface ",
            "abstract class", "data class",
        ]
    )

    return {
        "total_lines": total_lines,
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
        "comment_ratio": round(comment_lines / max(code_lines, 1), 2),
        "has_tests": has_tests,
        "has_docstrings": has_docstrings,
        "has_type_annotations": has_type_annotations,
        "has_error_handling": has_error_handling,
        "has_async": has_async,
        "has_classes": has_classes,
    }


class CodeParser(BaseParser):
    """Parse a code file into a ParseResult with language metadata."""

    def parse(self, data: bytes, filename: str) -> list[ParseResult]:
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            return []

        language, family = _detect_language(filename)
        features = _count_structural_features(text, language)

        title = filename.rsplit("/", 1)[-1] if "/" in filename else filename

        return [
            ParseResult(
                text=text,
                title=title,
                metadata={
                    "source_file": filename,
                    "source_type": "code",
                    "language": language,
                    "language_family": family,
                    **features,
                },
            )
        ]
