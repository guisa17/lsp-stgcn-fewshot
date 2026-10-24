"""Prepare -> train -> evaluate on synthetic raw takes, through the real command-line scripts."""

import json
import os
import subprocess
import sys

from .conftest import REPO_ROOT


def _run(*args):
    out = subprocess.run([sys.executable, *args], cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def test_prepare_train_evaluate(tmp_path, raw_dataset):
    bundle, split = tmp_path / "bundle.npz", tmp_path / "split.json"
    _run(
        "scripts/prepare_lsp.py",
        "--config",
        "configs/lsp/prepare.yaml",
        "--raw",
        raw_dataset[0],
        "--raw",
        raw_dataset[1],
        f"out={bundle}",
        f"split_out={split}",
    )
    assert json.loads(split.read_text())["num_samples"] == 64

    overrides = [
        f"data.bundle={bundle}",
        f"data.split={split}",
        f"out_root={tmp_path / 'runs'}",
        "train.epochs=2",
        "eval.latency_runs=2",
    ]
    _run("scripts/train.py", "--config", "configs/lsp/hagcn.yaml", *overrides)
    run_dir = tmp_path / "runs" / "lsp_hagcn"
    for name in ("metrics.json", "history.json", "model.pt", "config.yaml", "run_info.json"):
        assert (run_dir / name).is_file(), name
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["classes"] == ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]

    _run(
        "scripts/evaluate.py",
        "--config",
        "configs/lsp/hagcn.yaml",
        "--checkpoint",
        str(run_dir / "model.pt"),
        "--reference",
        str(run_dir / "metrics.json"),
        *overrides,
    )


def test_recorder_refuses_output_inside_repo():
    out = subprocess.run(
        [
            sys.executable,
            "scripts/record_keypoints.py",
            "--out",
            os.path.join(REPO_ROOT, "data", "raw"),
            "--signer-id",
            "s",
            "--class",
            "A",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0 and "inside the repository" in out.stderr
