"""Secret scanner — runs before Stage 1 (plan/10 §8.3).

Policy: hard-fail. Any artifact matching a secret pattern is dropped wholesale.
We do NOT attempt to sanitize: a partial scrub creates a false sense of safety
and risks letting context-bearing residue through.

The patterns are deliberately conservative — false positives drop a few
artifacts (cheap, since the pipeline only needs N samples to fire), false
negatives leak secrets into S3 → unacceptable.
"""

import re

# Each pattern is (name, compiled_regex). Order matters for ranking but every
# match is fatal regardless of which fires first.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # AWS access key id (AKIA / ASIA prefixes)
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    # AWS secret access key — 40 base64-ish chars in env-style assignment
    (
        "aws_secret_access_key",
        re.compile(
            r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{35,}"
        ),
    ),
    # GitHub personal access tokens (classic + fine-grained + OAuth)
    (
        "github_token",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{36,}\b"),
    ),
    # OpenAI API keys
    ("openai_api_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    # Anthropic API keys
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    # Slack tokens
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    # Stripe live keys
    ("stripe_live_key", re.compile(r"\bsk_live_[A-Za-z0-9]{24,}\b")),
    # Google API keys (AIza prefix)
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    # Generic private key headers (RSA, EC, OPENSSH, PGP)
    (
        "private_key_header",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY( BLOCK)?-----"
        ),
    ),
    # JWTs (header.payload.signature, base64url-ish)
    (
        "jwt_token",
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
        ),
    ),
    # Generic high-entropy assignments to typical secret names
    (
        "generic_secret_assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|secret|token|api[_-]?key)\s*[:=]\s*['\"][^'\"\s]{16,}['\"]"
        ),
    ),
]


def scan(text: str | None) -> list[str]:
    """Return the names of secret patterns matched in `text`.

    Empty list = clean. Non-empty = artifact must be dropped.
    Never returns the secret itself.
    """
    if not text:
        return []
    matched: list[str] = []
    for name, pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            matched.append(name)
    return matched


def is_clean(*texts: str | None) -> bool:
    """Convenience: scan multiple text blobs; return True iff all are clean."""
    for t in texts:
        if scan(t):
            return False
    return True


def patterns_in_use() -> list[str]:
    """For diagnostics / tests — names of all active patterns."""
    return [name for name, _ in _SECRET_PATTERNS]
