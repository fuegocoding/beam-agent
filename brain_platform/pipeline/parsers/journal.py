from brain_platform.pipeline.parsers.base import BaseParser, ParseResult


class JournalParser(BaseParser):
    """Parses a weekly journal entry (plain text)."""

    def parse(self, data: bytes, filename: str) -> list[ParseResult]:
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            return []

        return [
            ParseResult(
                text=text,
                title="Weekly Journal",
                metadata={"category": "journal"},
            )
        ]
