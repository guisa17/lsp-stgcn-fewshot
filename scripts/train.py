"""Train a classifier on the LSP bundle from a YAML config (one config per Table 1 row).

Writes ``metrics.json``, ``history.json``, ``confusion.png``, ``model.pt``, the resolved
``config.yaml`` and ``run_info.json`` (git commit, versions) to ``<out_root>/<exp_name>/``.

Usage:
    python scripts/train.py --config configs/lsp/hagcn.yaml
    python scripts/train.py --config configs/lsp/hagcn.yaml train.device=cuda data.bundle=x.npz
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from lsp.data.augment import Augmentor
from lsp.data.dataset import SignDataset, load_bundle
from lsp.data.splits import load_split
from lsp.evaluation.metrics import (
    classification_metrics,
    confusion,
    measure_latency,
    plot_confusion,
)
from lsp.models.build import build_model
from lsp.models.stgcn import count_parameters
from lsp.training.seed import set_seed
from lsp.training.trainer import evaluate, train_model
from lsp.utils.config import load_with_overrides
from lsp.utils.run_info import write_run_info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()
    cfg = load_with_overrides(args.config, args.overrides)

    seed = cfg["seed"]
    set_seed(seed, deterministic=cfg["deterministic"])
    out_dir = os.path.join(cfg["out_root"], cfg["exp_name"])
    write_run_info(out_dir, cfg)

    data_cfg = cfg["data"]
    streams = tuple(data_cfg["streams"])
    X, y, classes, _signers = load_bundle(data_cfg["bundle"])
    split = load_split(data_cfg["split"])

    aug_cfg = dict(cfg["augment"])
    augmentor = Augmentor(seed=seed, **aug_cfg) if aug_cfg.pop("enabled") else None

    train_ds = SignDataset(X, y, split["train"], streams, augmentor=augmentor)
    val_ds = SignDataset(X, y, split["val"], streams)
    test_ds = SignDataset(X, y, split["test"], streams)

    model = build_model(cfg["model"], num_classes=len(classes), streams=streams)
    n_params = count_parameters(model)
    print(
        f"== {cfg['exp_name']} == model={cfg['model']['type']} streams={streams} "
        f"feat_dim={train_ds.feature_dim} params={n_params:,}"
    )

    tr = cfg["train"]
    res = train_model(
        model,
        train_ds,
        val_ds,
        device=tr["device"],
        epochs=tr["epochs"],
        batch_size=tr["batch_size"],
        lr=tr["lr"],
        weight_decay=tr["weight_decay"],
        patience=tr["patience"],
        lr_factor=tr["lr_factor"],
        lr_patience=tr["lr_patience"],
        min_lr=tr["min_lr"],
        num_workers=tr["num_workers"],
    )

    test_loader = DataLoader(test_ds, batch_size=tr["batch_size"], shuffle=False)
    ev = evaluate(model, test_loader, torch.device(tr["device"]))
    metrics = classification_metrics(ev["y_true"], ev["y_pred"])
    lat = measure_latency(
        model,
        test_ds[0][0],
        device="cpu",
        n_warmup=cfg["eval"]["latency_warmup"],
        n_runs=cfg["eval"]["latency_runs"],
    )

    cm = confusion(ev["y_true"], ev["y_pred"], len(classes))
    plot_confusion(
        cm,
        classes,
        os.path.join(out_dir, "confusion.png"),
        title=f"Confusion matrix: {cfg['exp_name']}",
    )

    summary = {
        "exp_name": cfg["exp_name"],
        "model": cfg["model"]["type"],
        "streams": list(streams),
        "num_params": int(n_params),
        "best_val_acc": res["best_val_acc"],
        "best_epoch": res["best_epoch"],
        "test_metrics": metrics,
        "latency": lat,
        "num_classes": len(classes),
        "classes": classes,
        "split_sizes": {k: len(v) for k, v in split.items()},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(res["history"], f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    print(
        f"\n{cfg['exp_name']}: test_acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} | "
        f"latency={lat['latency_ms']:.1f} ms ({lat['fps']:.1f} FPS) | "
        f"val_acc={res['best_val_acc']:.4f}"
    )
    print(f"  saved to {out_dir}/")


if __name__ == "__main__":
    main()
