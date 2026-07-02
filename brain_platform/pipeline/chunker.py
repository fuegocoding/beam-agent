import re
from dataclasses import dataclass, field

from brain_platform.pipeline.parsers.base import ParseResult

SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
    def __init__(self, max_chars: int = 6000, overlap_sentences: int = 2):
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences

    def chunk(self, parse_result: ParseResult) -> list[Chunk]:
        text = parse_result.text.strip()
        if not text:
            return []

        # If text fits in one chunk, return as-is
        if len(text) <= self.max_chars:
            return [
                Chunk(
                    text=text,
                    index=0,
                    metadata={
                        **parse_result.metadata,
                        "title": parse_result.title,
                    },
                )
            ]

        # Split into paragraphs
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_parts: list[str] = []
        current_len = 0
        overlap_text = ""

        for para in paragraphs:
            para_len = len(para)

            # If a single paragraph exceeds max, split it by sentences
            if para_len > self.max_chars:
                if current_parts:
                    chunks.append(self._make_chunk(current_parts, len(chunks), parse_result, overlap_text))
                    overlap_text = self._get_overlap("\n\n".join(current_parts))
                    current_parts = []
                    current_len = 0

                sentence_chunks = self._split_long_paragraph(para)
                for sc in sentence_chunks:
                    chunks.append(self._make_chunk([sc], len(chunks), parse_result, overlap_text))
                    overlap_text = self._get_overlap(sc)
                continue

            # Would adding this paragraph exceed max?
            if current_len + para_len + 2 > self.max_chars and current_parts:
                chunks.append(self._make_chunk(current_parts, len(chunks), parse_result, overlap_text))
                overlap_text = self._get_overlap("\n\n".join(current_parts))
                current_parts = []
                current_len = 0

            current_parts.append(para)
            current_len += para_len + 2

        if current_parts:
            chunks.append(self._make_chunk(current_parts, len(chunks), parse_result, overlap_text))

        return chunks

    def _make_chunk(
        self, parts: list[str], index: int, parse_result: ParseResult, overlap: str
    ) -> Chunk:
        body = "\n\n".join(parts)
        if overlap and index > 0:
            body = overlap + "\n\n" + body

        return Chunk(
            text=body,
            index=index,
            metadata={
                **parse_result.metadata,
                "title": parse_result.title,
            },
        )

    def _get_overlap(self, text: str) -> str:
        """Extract last N sentences for overlap context."""
        sentences = SENTENCE_END_RE.split(text)
        if len(sentences) <= self.overlap_sentences:
            return ""
        return " ".join(sentences[-self.overlap_sentences :])

    def _split_long_paragraph(self, para: str) -> list[str]:
        """Split a very long paragraph by sentences."""
        sentences = SENTENCE_END_RE.split(para)
        parts = []
        current = []
        current_len = 0
        for sent in sentences:
            if current_len + len(sent) > self.max_chars and current:
                parts.append(" ".join(current))
                current = []
                current_len = 0
            current.append(sent)
            current_len += len(sent) + 1
        if current:
            parts.append(" ".join(current))
        return parts
