"""Parsers — each parser imports a third-party lib (docx, pymupdf, etc.).

Optional deps are wrapped in try/except so the package imports cleanly
when a lib is missing. Callers use :func:`get_parser` which surfaces
an ImportError with a helpful message if the requested parser's dep
isn't installed.
"""
from brain_platform.models.enums import DataSourceType
from brain_platform.pipeline.parsers.base import BaseParser, ParseResult


def _try_import(module: str, attr: str):
    """Best-effort import. Returns the attribute or None if the dep is missing."""
    try:
        from brain_platform.pipeline.parsers import module as _mod  # type: ignore
        return getattr(_mod, attr)
    except ImportError:
        return None


# Pure-stdlib parsers — always importable
from brain_platform.pipeline.parsers.code import CodeParser
from brain_platform.pipeline.parsers.email import EmailParser
from brain_platform.pipeline.parsers.instructions import InstructionsParser
from brain_platform.pipeline.parsers.obsidian import ObsidianParser
from brain_platform.pipeline.parsers.prompt import PromptParser
from brain_platform.pipeline.parsers.reddit import RedditParser
from brain_platform.pipeline.parsers.txt import TxtParser
from brain_platform.pipeline.parsers.journal import JournalParser

# Optional third-party-dep parsers
_docx_cls = _try_import("docx", "DocxParser")
_pdf_cls = _try_import("pdf", "PDFParser")

PARSER_MAP: dict[DataSourceType, type[BaseParser]] = {
    DataSourceType.OBSIDIAN: ObsidianParser,
    DataSourceType.TXT: TxtParser,
    DataSourceType.REDDIT: RedditParser,
    DataSourceType.CODE: CodeParser,
    DataSourceType.PROMPT: PromptParser,
    DataSourceType.INSTRUCTIONS: InstructionsParser,
    DataSourceType.EMAIL: EmailParser,
    DataSourceType.JOURNAL: JournalParser,
}
if _docx_cls is not None:
    PARSER_MAP[DataSourceType.DOCX] = _docx_cls
if _pdf_cls is not None:
    PARSER_MAP[DataSourceType.PDF] = _pdf_cls


def get_parser(source_type: DataSourceType) -> BaseParser:
    """Return a parser instance for the given source type.

    Raises:
        ValueError: if no parser is registered for the source type.
        ImportError: if the parser's third-party dep (docx, pymupdf) is missing.
    """
    parser_cls = PARSER_MAP.get(source_type)
    if not parser_cls:
        raise ValueError(
            f"No parser for source type: {source_type.value}. "
            f"Available: {[t.value for t in PARSER_MAP.keys()]}"
        )
    return parser_cls()


__all__ = ["get_parser", "ParseResult", "PARSER_MAP", "BaseParser"]
