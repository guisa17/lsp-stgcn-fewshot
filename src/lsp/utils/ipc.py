"""Minimal file-based IPC between a script with its own OpenCV window and the demo app.

A status file (JSON, written atomically) reports progress, and a "stop" file asks the script to
finish gracefully as soon as it exists.
"""

from __future__ import annotations

import json
import os
import time


def write_status(status_file: str | None, **fields) -> None:
    if not status_file:
        return
    fields["ts"] = time.time()
    tmp = status_file + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(fields, f, ensure_ascii=False)
        os.replace(tmp, status_file)
    except OSError:
        pass  # the app may not be listening; this must never crash the script


def should_stop(stop_file: str | None) -> bool:
    return bool(stop_file) and os.path.exists(stop_file)
