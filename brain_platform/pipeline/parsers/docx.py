import io

from docx import Document

from brain_platform.pipeline.parsers.base import BaseParser, ParseResult


class DocxParser(BaseParser):
    def parse(self, data: bytes, filename: str) -> list[ParseResult]:
        doc = Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        if not paragraphs:
            return []

        full_text = "\n\n".join(paragraphs)
        title = filename.rsplit(".", 1)[0] if "." in filename else filename

        return [
            ParseResult(
                text=full_text,
                title=title,
                metadata={"paragraph_count": len(paragraphs), "source_file": filename},
            )
        ]
