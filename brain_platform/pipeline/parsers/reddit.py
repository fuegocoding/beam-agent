"""Parser for Reddit scrape data (JSON format from RedditScraper)."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

from brain_platform.pipeline.parsers.base import BaseParser, ParseResult

# Comments shorter than this are grouped by subreddit into batches
SHORT_COMMENT_THRESHOLD = 300
# Target batch size when grouping short comments
BATCH_TARGET_CHARS = 5000
# Skip posts/comments with text shorter than this
MIN_TEXT_LENGTH = 20
# Texts to treat as deleted/removed
SKIP_TEXTS = {"[deleted]", "[removed]", ""}


class RedditParser(BaseParser):
    def parse(self, data: bytes, filename: str) -> list[ParseResult]:
        raw = json.loads(data.decode("utf-8"))
        results: list[ParseResult] = []

        results.extend(self._parse_posts(raw.get("posts", [])))
        results.extend(self._parse_comments(raw.get("comments", [])))

        return results

    def _parse_posts(self, posts: list[dict]) -> list[ParseResult]:
        results: list[ParseResult] = []
        for post in posts:
            selftext = (post.get("selftext") or "").strip()
            title = (post.get("title") or "").strip()

            # Use selftext for self-posts; for link posts use title only
            text = selftext if selftext not in SKIP_TEXTS else title
            if len(text) < MIN_TEXT_LENGTH:
                continue

            # Prepend title for context if selftext exists
            if selftext and selftext not in SKIP_TEXTS:
                text = f"{title}\n\n{selftext}"

            subreddit = post.get("subreddit", "unknown")
            created_utc = post.get("created_utc", 0)

            results.append(
                ParseResult(
                    text=text,
                    title=f"[r/{subreddit}] {title}",
                    metadata={
                        "source": "reddit",
                        "type": "post",
                        "subreddit": subreddit,
                        "score": post.get("score", 0),
                        "permalink": post.get("permalink", ""),
                        "num_comments": post.get("num_comments", 0),
                    },
                    created_at=datetime.fromtimestamp(created_utc, tz=timezone.utc)
                    if created_utc
                    else None,
                )
            )

        return results

    def _parse_comments(self, comments: list[dict]) -> list[ParseResult]:
        results: list[ParseResult] = []
        # Bucket for short comments grouped by subreddit
        short_buckets: dict[str, list[dict]] = defaultdict(list)

        for comment in comments:
            body = (comment.get("body") or "").strip()
            if body in SKIP_TEXTS or len(body) < MIN_TEXT_LENGTH:
                continue

            subreddit = comment.get("subreddit", "unknown")

            if len(body) >= SHORT_COMMENT_THRESHOLD:
                # Long comment gets its own ParseResult
                link_title = comment.get("link_title", "")
                created_utc = comment.get("created_utc", 0)
                results.append(
                    ParseResult(
                        text=f"Re: {link_title}\n\n{body}" if link_title else body,
                        title=f"Comment in r/{subreddit}",
                        metadata={
                            "source": "reddit",
                            "type": "comment",
                            "subreddit": subreddit,
                            "score": comment.get("score", 0),
                            "permalink": comment.get("permalink", ""),
                        },
                        created_at=datetime.fromtimestamp(created_utc, tz=timezone.utc)
                        if created_utc
                        else None,
                    )
                )
            else:
                short_buckets[subreddit].append(comment)

        # Batch short comments by subreddit
        for subreddit, bucket in short_buckets.items():
            batch_texts: list[str] = []
            batch_chars = 0

            for comment in bucket:
                body = (comment.get("body") or "").strip()
                link_title = comment.get("link_title", "")
                entry = f"Re: {link_title}\n{body}" if link_title else body

                batch_texts.append(entry)
                batch_chars += len(entry)

                if batch_chars >= BATCH_TARGET_CHARS:
                    results.append(self._make_batch_result(subreddit, batch_texts))
                    batch_texts = []
                    batch_chars = 0

            if batch_texts:
                results.append(self._make_batch_result(subreddit, batch_texts))

        return results

    @staticmethod
    def _make_batch_result(subreddit: str, texts: list[str]) -> ParseResult:
        return ParseResult(
            text="\n\n---\n\n".join(texts),
            title=f"Comments in r/{subreddit} ({len(texts)} comments)",
            metadata={
                "source": "reddit",
                "type": "comment_batch",
                "subreddit": subreddit,
                "comment_count": len(texts),
            },
        )
