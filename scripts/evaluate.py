"""Evaluate a trained classifier on the test split, or summarize several runs into one table.

Usage:
    # Recompute test metrics of a checkpoint (optionally compare with a reference metrics.json)
    python scripts/evaluate.py --config configs/lsp/hagcn.yaml --checkpoint path/to/model.pt \
        --reference results/lsp/hagcn/metrics.json

    # Table of every <runs_dir>/*/metrics.json -> <runs_dir>/comparison.csv
    python scripts/evaluate.py --summarize outputs/lsp
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

from lsp.data.dataset import SignDataset, load_bundle
from lsp.data.splits import load_split
from lsp.evaluation.metrics import classification_metrics, confusion, plot_confusion
from lsp.training.checkpoint import load_classifier
from lsp.training.trainer import evaluate
from lsp.utils.config import load_with_overrides

SUMMARY_COLUMNS = [
    "exp_name",
    "model",
    "streams",
    "num_params",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "best_val_acc",
    "latency_ms",
    "fps",
]


def evaluate_checkpoint(cfg: dict, checkpoint: str, confusion_path: str | None = None) -> dict:
    """Test-split metrics of ``checkpoint`` under the data and model settings of ``cfg``.

    If ``confusion_path`` is given, the test confusion matrix is also saved as an image.
    """
    data_cfg = cfg["data"]
    streams = tuple(data_cfg["streams"])
    X, y, classes, _ = load_bundle(data_cfg["bundle"])
    test_idx = load_split(data_cfg["split"])["test"]
    model = load_classifier(checkpoint, cfg["model"], len(classes), streams)
    loader = DataLoader(SignDataset(X, y, test_idx, streams), batch_size=cfg["train"]["batch_size"])
    ev = evaluate(model, loader, torch.device("cpu"))
    if confusion_path:
        cm = confusion(ev["y_true"], ev["y_pred"], len(classes))
        plot_confusion(cm, classes, confusion_path, title=f"Confusion matrix: {cfg['exp_name']}")
    return {"n_test": len(test_idx), **classification_metrics(ev["y_true"], ev["y_pred"])}


def compare(metrics: dict, reference_path: str, tol: float) -> bool:
    with open(reference_path, encoding="utf-8") as f:
        ref = json.load(f)["test_metrics"]
    ok = True
    for key in ("accuracy", "precision", "recall", "f1"):
        diff = abs(metrics[key] - ref[key])
        status = "OK" if diff <= tol else "MISMATCH"
        ok &= diff <= tol
        print(f"  {key:9s} ref={ref[key]:.6f} new={metrics[key]:.6f} |diff|={diff:.2e} {status}")
    return ok


def summarize(runs_dir: str) -> None:
    rows = []
    for mpath in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.json"))):
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        tm, lat = m.get("test_metrics", {}), m.get("latency", {})
        rows.append(
            {
                "exp_name": m.get("exp_name", os.path.basename(os.path.dirname(mpath))),
                "model": m.get("model", ""),
                "streams": "+".join(m.get("streams", [])),
                "num_params": m.get("num_params", ""),
                "test_accuracy": round(tm.get("accuracy", 0), 4),
                "test_precision": round(tm.get("precision", 0), 4),
                "test_recall": round(tm.get("recall", 0), 4),
                "test_f1": round(tm.get("f1", 0), 4),
                "best_val_acc": round(m.get("best_val_acc", 0), 4),
                "latency_ms": round(lat.get("latency_ms", 0), 2),
                "fps": round(lat.get("fps", 0), 1),
            }
        )
    if not rows:
        raise SystemExit(f"No metrics.json found under {runs_dir}/*/")
    out_csv = os.path.join(runs_dir, "comparison.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print("| " + " | ".join(SUMMARY_COLUMNS) + " |")
    print("|" + "|".join(["---"] * len(SUMMARY_COLUMNS)) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in SUMMARY_COLUMNS) + " |")
    print(f"\nTable saved to {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--reference", help="metrics.json to compare against")
    ap.add_argument("--tol", type=float, default=1e-6, help="Tolerance for --reference")
    ap.add_argument("--out", help="Write the recomputed metrics to this JSON file")
    ap.add_argument("--confusion", help="Save the test confusion matrix to this image file")
    ap.add_argument("--summarize", metavar="RUNS_DIR")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return
    if not (args.config and args.checkpoint):
        ap.error("--config and --checkpoint are required unless --summarize is given")

    cfg = load_with_overrides(args.config, args.overrides)
    metrics = evaluate_checkpoint(cfg, args.checkpoint, args.confusion)
    print(
        f"{cfg['exp_name']}: n_test={metrics['n_test']} acc={metrics['accuracy']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    if args.reference and not compare(metrics, args.reference, args.tol):
        sys.exit(1)


if __name__ == "__main__":
    main()
