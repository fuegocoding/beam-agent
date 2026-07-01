from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ParseResult:
    text: str
    title: str | None = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime | None = None
    links: list[str] = field(default_factory=list)


class BaseParser(ABC):
    @abstractmethod
    def parse(self, data: bytes, filename: str) -> list[ParseResult]:
        """Parse raw file bytes into one or more text documents."""
