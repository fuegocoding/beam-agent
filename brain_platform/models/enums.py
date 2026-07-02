"""Models enums stub — minimal set of enums used by Tier 1 and Tier 2 lifted files.

Lifted from the cloud's ``models/enums.py`` (private). The cloud has
~20 enums; this stub only includes the ones imported by the files we
lift in Chunks 1-2. Other enums land as needed.
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


class InterviewMode(str, enum.Enum):
    """Routing mode for the guided interview."""

    LINEAR = "linear"       # Original fixed 14-question sequence
    REACTIVE = "reactive"   # Dimension-gap-driven dynamic routing
    EXPRESS = "express"     # 10-question fixed subset


class ProfessionalCategory(str, enum.Enum):
    """Professional sector for the work-session follow-up question bank."""

    SOFTWARE_ENGINEER = "software_engineer"
    PRODUCT_MANAGER = "product_manager"
    EDUCATOR = "educator"
    FINANCIAL_PROFESSIONAL = "financial_professional"
    DATA_SCIENTIST = "data_scientist"
    UX_DESIGNER = "ux_designer"
