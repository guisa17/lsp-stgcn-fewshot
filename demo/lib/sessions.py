"""Subprocess with its own OpenCV window plus file-based IPC (status / stop).

Used to launch the recorder (``scripts/record_keypoints.py``) and the live classifier
(``demo/live_classifier.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid

from . import paths


class ProcSession:
    def __init__(self, proc: subprocess.Popen, status_file: str, stop_file: str):
        self.proc = proc
        self.status_file = status_file
        self.stop_file = stop_file
        self.started_at = time.time()

    @classmethod
    def launch(cls, args: list[str]) -> ProcSession:
        """``args``: script path and arguments (without the interpreter)."""
        run_id = uuid.uuid4().hex[:8]
        status_file = str(paths.SESSIONS_DIR / f"{run_id}.status.json")
        stop_file = str(paths.SESSIONS_DIR / f"{run_id}.stop")
        for f in (status_file, stop_file):
            if os.path.exists(f):
                os.remove(f)

        cmd = [
            paths.python_executable(),
            *args,
            "--status-file",
            status_file,
            "--stop-file",
            stop_file,
        ]
        kwargs = {}
        if sys.platform == "win32":
            # Own console window, so the user sees the script output and can press Ctrl+C.
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        proc = subprocess.Popen(cmd, cwd=str(paths.REPO_ROOT), **kwargs)
        return cls(proc, status_file, stop_file)

    def poll(self) -> int | None:
        return self.proc.poll()

    def read_status(self) -> dict:
        try:
            with open(self.status_file, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def request_stop(self) -> None:
        try:
            open(self.stop_file, "w").close()
        except OSError:
            pass

    def cleanup(self) -> None:
        for path in (self.status_file, self.stop_file):
            try:
                os.remove(path)
            except OSError:
                pass


def start_recording(
    *,
    raw_dir: str,
    signer_id: str,
    classes: list[str],
    n_samples: int,
    camera: int,
    model_complexity: int,
) -> ProcSession:
    return ProcSession.launch(
        [
            str(paths.SCRIPT_RECORD),
            "--out",
            raw_dir,
            "--signer-id",
            signer_id,
            "--classes",
            ",".join(classes),
            "--n-samples",
            str(n_samples),
            "--camera",
            str(camera),
            "--model-complexity",
            str(model_complexity),
        ]
    )


def start_live_classifier(
    *, run: str, camera: int, model_complexity: int, interval: float, top_k: int = 3
) -> ProcSession:
    return ProcSession.launch(
        [
            str(paths.SCRIPT_LIVE),
            "--runs-dir",
            str(paths.RUNS_DIR),
            "--run",
            run,
            "--camera",
            str(camera),
            "--model-complexity",
            str(model_complexity),
            "--interval",
            str(interval),
            "--top-k",
            str(top_k),
        ]
    )
