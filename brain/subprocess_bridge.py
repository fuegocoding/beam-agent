"""Generic bridge to call Rust binaries via subprocess."""

import json
import subprocess
from pathlib import Path
from typing import Any

BRAIN_RUST_DIR = Path(__file__).parent.parent / "brain-rust"


def call_rust_binary(binary_name: str, input_data: dict, timeout: int = 30) -> dict:
    """Call a Rust binary with JSON input, return JSON output."""
    binary_path = BRAIN_RUST_DIR / "target" / "release" / binary_name
    if not binary_path.exists():
        binary_path = BRAIN_RUST_DIR / "target" / "debug" / binary_name

    if not binary_path.exists():
        raise FileNotFoundError(
            f"Rust binary not found: {binary_path}. Run 'cargo build' in brain-rust/"
        )

    if not binary_path.suffix:
        binary_path = binary_path.with_suffix(".exe")

    result = subprocess.run(
        [str(binary_path)],
        input=json.dumps(input_data),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Rust binary {binary_name} failed: {result.stderr}")

    return json.loads(result.stdout)
