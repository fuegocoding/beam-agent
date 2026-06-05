"""Centralized path and config management for Beam brains.

This module provides a single source of truth for all brain-related paths,
replacing the scattered hardcoded paths across the codebase.
"""
import os
from pathlib import Path
from typing import Any

import yaml

# ── Constants ─────────────────────────────────────────────────────────

BEAM_HOME = Path(os.environ.get("BEAM_HOME", Path.home() / ".beam"))
CONFIG_FILE = "config.yaml"
BRAINS_DIR = "brains"
PROJECTS_DIR = "brain-projects"
ACTIVE_BRAIN_FILE = "active_brain"


# ── Path Helpers ──────────────────────────────────────────────────────


def get_beam_home() -> Path:
    """Get the Beam home directory."""
    return BEAM_HOME


def get_config_path() -> Path:
    """Get the path to config.yaml."""
    return BEAM_HOME / CONFIG_FILE


def get_brains_dir() -> Path:
    """Get the directory containing installed brains."""
    return BEAM_HOME / BRAINS_DIR


def get_projects_dir() -> Path:
    """Get the directory containing brain builder projects."""
    return BEAM_HOME / PROJECTS_DIR


def get_brain_path(name: str) -> Path:
    """Get the path to a specific brain's directory."""
    return BEAM_HOME / BRAINS_DIR / name


def get_brain_graph_path(name: str) -> Path:
    """Get the path to a brain's personality_graph.json."""
    return BEAM_HOME / BRAINS_DIR / name / "personality_graph.json"


def get_brain_soul_path(name: str) -> Path:
    """Get the path to a brain's soul.md."""
    return BEAM_HOME / BRAINS_DIR / name / "soul.md"


def get_brain_memory_path(name: str) -> Path:
    """Get the path to a brain's memory directory."""
    return BEAM_HOME / BRAINS_DIR / name / "memory"


def get_project_path(name: str) -> Path:
    """Get the path to a brain builder project."""
    return BEAM_HOME / PROJECTS_DIR / name


def get_project_graph_path(name: str) -> Path:
    """Get the path to a project's personality_graph.json."""
    return BEAM_HOME / PROJECTS_DIR / name / "personality_graph.json"


def get_project_transcript_path(name: str) -> Path:
    """Get the path to a project's interview transcript."""
    return BEAM_HOME / PROJECTS_DIR / name / "interview_transcript.json"


# ── Config Management ─────────────────────────────────────────────────


def _default_config() -> dict:
    """Return a default config structure."""
    return {
        "active_brain": "default",
        "brains": {
            "default": {
                "source": "local",
                "installed_at": None,
            }
        },
    }


def load_config() -> dict:
    """Load config.yaml, creating it with defaults if it doesn't exist."""
    config_path = get_config_path()
    if not config_path.exists():
        config = _default_config()
        save_config(config)
        return config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not config or not isinstance(config, dict):
            config = _default_config()
            save_config(config)
        return config
    except Exception:
        config = _default_config()
        save_config(config)
        return config


def save_config(config: dict) -> None:
    """Save config.yaml."""
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_active_brain_name() -> str:
    """Get the name of the currently active brain."""
    config = load_config()
    return config.get("active_brain", "default")


def get_active_brain_path() -> Path:
    """Get the path to the active brain's directory."""
    name = get_active_brain_name()
    return get_brain_path(name)


def get_active_brain_graph_path() -> Path:
    """Get the path to the active brain's personality_graph.json."""
    name = get_active_brain_name()
    return get_brain_graph_path(name)


def get_active_brain_soul_path() -> Path:
    """Get the path to the active brain's soul.md."""
    name = get_active_brain_name()
    return get_brain_soul_path(name)


def get_active_brain_memory_path() -> Path:
    """Get the path to the active brain's memory directory."""
    name = get_active_brain_name()
    return get_brain_memory_path(name)


def set_active_brain(name: str) -> None:
    """Set the active brain in config."""
    config = load_config()
    config["active_brain"] = name
    save_config(config)


def register_brain(
    name: str,
    source: str = "local",
    slug: str | None = None,
    token: str | None = None,
) -> None:
    """Register a brain in config."""
    from datetime import datetime, timezone

    config = load_config()
    if "brains" not in config:
        config["brains"] = {}

    config["brains"][name] = {
        "source": source,
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    if slug:
        config["brains"][name]["slug"] = slug
    if token:
        config["brains"][name]["token"] = token

    save_config(config)


def unregister_brain(name: str) -> None:
    """Remove a brain from config."""
    config = load_config()
    if "brains" in config and name in config["brains"]:
        del config["brains"][name]
        save_config(config)


def get_brain_info(name: str) -> dict | None:
    """Get config info for a specific brain."""
    config = load_config()
    return config.get("brains", {}).get(name)


def list_brains() -> list[dict]:
    """List all registered brains with their info."""
    config = load_config()
    active = config.get("active_brain", "default")
    brains = config.get("brains", {})

    result = []
    for name, info in brains.items():
        if not isinstance(info, dict):
            info = {}
        result.append({
            "name": name,
            "active": name == active,
            "source": info.get("source", "local"),
            "slug": info.get("slug"),
            "installed_at": info.get("installed_at"),
            "has_token": "token" in info and info["token"] is not None,
        })

    return result


# ── Initialization ────────────────────────────────────────────────────


def ensure_beam_dirs() -> None:
    """Create the Beam directory structure if it doesn't exist."""
    BEAM_HOME.mkdir(parents=True, exist_ok=True)
    (BEAM_HOME / BRAINS_DIR).mkdir(exist_ok=True)
    (BEAM_HOME / PROJECTS_DIR).mkdir(exist_ok=True)

    # Ensure default brain exists
    default_brain = get_brain_path("default")
    default_brain.mkdir(parents=True, exist_ok=True)
    (default_brain / "memory").mkdir(exist_ok=True)
    for subdir in ["episodic", "semantic", "procedural"]:
        (default_brain / "memory" / subdir).mkdir(exist_ok=True)


def migrate_legacy_brain() -> None:
    """Migrate from old ~/.beam/brain/default/ to new ~/.beam/brains/default/.

    This is a one-way migration. If the old path exists and the new one doesn't,
    it moves the data over.
    """
    old_path = BEAM_HOME / "brain" / "default"
    new_path = get_brain_path("default")

    if old_path.exists() and not new_path.exists():
        import shutil
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(new_path))

        # Also migrate memory if it exists at old location
        old_memory = BEAM_HOME / "memory" / "default"
        new_memory = get_brain_memory_path("default")
        if old_memory.exists() and not new_memory.exists():
            shutil.move(str(old_memory), str(new_memory))

        # Clean up old empty directories
        try:
            old_path.parent.rmdir()
        except OSError:
            pass
        try:
            old_memory.parent.rmdir()
        except OSError:
            pass
