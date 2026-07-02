import io
import re
import zipfile
from datetime import datetime

from brain_platform.pipeline.parsers.base import BaseParser, ParseResult

BACKLINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata, body)."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw = match.group(1)
    metadata = {}
    for line in raw.split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip().strip('"').strip("'")
    body = text[match.end() :]
    return metadata, body


class ObsidianParser(BaseParser):
    def parse(self, data: bytes, filename: str) -> list[ParseResult]:
        results = []
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            md_files = [n for n in zf.namelist() if n.endswith(".md") and not n.startswith("__MACOSX")]
            for name in sorted(md_files):
                raw = zf.read(name).decode("utf-8", errors="replace")
                metadata, body = _parse_frontmatter(raw)

                # Extract backlinks
                links = BACKLINK_RE.findall(body)

                # Derive title from filename or frontmatter
                title = metadata.get("title") or name.rsplit("/", 1)[-1].removesuffix(".md")

                # Try to parse date from frontmatter
                created_at = None
                for date_key in ("date", "created", "created_at"):
                    if date_key in metadata:
                        try:
                            created_at = datetime.fromisoformat(metadata[date_key])
                        except (ValueError, TypeError):
                            pass
                        break

                metadata["source_file"] = name
                metadata["backlinks"] = links

                if body.strip():
                    results.append(
                        ParseResult(
                            text=body.strip(),
                            title=title,
                            metadata=metadata,
                            created_at=created_at,
                            links=links,
                        )
                    )

        return results
