"""Email parser — converts a JSON bundle of emails into ParseResult documents."""

import json
import logging
from datetime import datetime

from brain_platform.pipeline.parsers.base import BaseParser, ParseResult

logger = logging.getLogger(__name__)


class EmailParser(BaseParser):
    """Parse a JSON file containing exported emails.

    Expected JSON format (list of email objects):
    [
      {
        "id": "...",
        "subject": "...",
        "from": "...",
        "to": "...",
        "date": "2024-01-15T10:30:00Z",
        "body": "plain text body...",
        "thread_id": "..."
      }
    ]
    """

    def parse(self, data: bytes, filename: str) -> list[ParseResult]:
        try:
            emails = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.error("Failed to decode email JSON: %s", e)
            return []

        if not isinstance(emails, list):
            emails = [emails]

        results: list[ParseResult] = []
        for email in emails:
            subject = email.get("subject", "(no subject)")
            body = email.get("body", "")
            sender = email.get("from", "")
            date_str = email.get("date", "")
            thread_id = email.get("thread_id", "")

            # Build a clean document with metadata
            text = f"From: {sender}\nSubject: {subject}\nDate: {date_str}\n\n{body}"

            created_at = None
            if date_str:
                try:
                    created_at = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

            results.append(
                ParseResult(
                    text=text,
                    title=subject,
                    metadata={
                        "source": "email",
                        "sender": sender,
                        "thread_id": thread_id,
                        "email_id": email.get("id", ""),
                    },
                    created_at=created_at,
                )
            )

        logger.info("EmailParser: parsed %d emails", len(results))
        return results
