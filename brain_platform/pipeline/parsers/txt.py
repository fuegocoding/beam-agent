from brain_platform.pipeline.parsers.base import BaseParser, ParseResult


class TxtParser(BaseParser):
    def parse(self, data: bytes, filename: str) -> list[ParseResult]:
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            return []

        title = filename.rsplit(".", 1)[0] if "." in filename else filename

        return [
            ParseResult(
                text=text,
                title=title,
                metadata={"source_file": filename},
            )
        ]
