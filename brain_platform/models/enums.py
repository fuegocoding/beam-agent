"""Models enums stub — minimal set of enums used by Tier 1 lifted files.

Only includes the enums that are imported by the files we lift in Chunk 1.
Chunk 2 (adapt phase) may add more; full enums live in the cloud's
``models/enums.py`` and will be lifted then as needed.
"""
import enum


class DataSourceType(str, enum.Enum):
    """Source of a brain-ingestion input."""

    OBSIDIAN = "obsidian"
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    AUDIO = "audio"
    TWEET = "tweet"
    EMAIL = "email"
    AI_MEMORY = "ai_memory"
    REDDIT = "reddit"
    JOURNAL = "journal"
    CODE = "code"
    PROMPT = "prompt"
    INSTRUCTIONS = "instructions"
