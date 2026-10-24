"""Paths used by the demo app.

Raw takes are personal data and live outside the repository: set ``LSP_RAW_DIR`` to their root
folder (one sub-folder per signer). Trained runs are read from ``LSP_RUNS_DIR`` (default
``outputs/lsp``, the output folder of ``scripts/train.py``).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DEMO_DIR.parent

RAW_DIR = Path(os.environ["LSP_RAW_DIR"]) if os.environ.get("LSP_RAW_DIR") else None
RUNS_DIR = Path(os.environ.get("LSP_RUNS_DIR", REPO_ROOT / "outputs" / "lsp"))
VOCAB_PATH = DEMO_DIR / "vocab.yaml"
SCRIPT_RECORD = REPO_ROOT / "scripts" / "record_keypoints.py"
SCRIPT_LIVE = DEMO_DIR / "live_classifier.py"

# Status and stop files of running sessions (IPC with the recorder or live classifier subprocess).
SESSIONS_DIR = Path(tempfile.gettempdir()) / "lsp_demo_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def python_executable() -> str:
    """Interpreter used to launch subprocesses: the one running the app."""
    return sys.executable


def require_raw_dir() -> Path | None:
    """The raw-data root, or ``None`` (pages then show how to set ``LSP_RAW_DIR``)."""
    if RAW_DIR is None:
        return None
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR
