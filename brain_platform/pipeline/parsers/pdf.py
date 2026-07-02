import pymupdf

from brain_platform.pipeline.parsers.base import BaseParser, ParseResult


class PDFParser(BaseParser):
    def parse(self, data: bytes, filename: str) -> list[ParseResult]:
        doc = pymupdf.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text)
        doc.close()

        if not pages:
            return []

        full_text = "\n\n".join(pages)
        title = filename.rsplit(".", 1)[0] if "." in filename else filename

        return [
            ParseResult(
                text=full_text,
                title=title,
                metadata={"page_count": len(pages), "source_file": filename},
            )
        ]
