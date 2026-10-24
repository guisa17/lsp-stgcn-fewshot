"""Few-shot experiments: LSA64 meta-training (Table 2) or LSA64 -> LSP transfer (Table 3).

The config's ``mode`` selects the experiment:

- ``meta_train``: for each seed, hold out ``n_test_signers`` LSA64 signers, (optionally) pre-train
  with cross-entropy, meta-train the encoder episodically, then measure accuracy-vs-K on the
  held-out signers for ProtoNet and the fine-tune-K baseline. Saves one JSON and one encoder per
  seed plus a pooled summary. ``--eval-only`` skips training and re-evaluates the saved encoders.
- ``transfer``: load frozen LSA64 encoders and calibrate them on each LSP signer. Encoders trained
  on the same LSA64 signer split are refused unless ``allow_duplicate_splits`` is true.

Usage:
    python scripts/fewshot_calibrate.py --config configs/lsa64/hagcn_fewshot.yaml
    python scripts/fewshot_calibrate.py --config configs/lsa64/hagcn_fewshot.yaml --eval-only
    python scripts/fewshot_calibrate.py --config configs/transfer/hagcn_lsa64_to_lsp.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from lsp.data.dataset import load_bundle
from lsp.data.splits import pick_signers, random_subject_split, save_split
from lsp.data.streams import stack_streams_batch
from lsp.evaluation.aggregate import aggregate_runs, check_distinct_splits
from lsp.evaluation.fewshot import evaluate_vs_k
from lsp.models.build import build_model
from lsp.training.checkpoint import load_encoder, save_encoder
from lsp.training.episodic import meta_train, pretrain_classifier
from lsp.training.seed import set_seed
from lsp.utils.config import load_with_overrides
from lsp.utils.run_info import write_run_info


def _dump(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _print_curve(title: str, summary: dict, ks) -> None:
    print(f"\n{title}\nK   ProtoNet   fine-tune-K")
    for k in ks:
        p, f = summary["protonet"].get(str(k)), summary["finetune_k"].get(str(k))
        print(f"{k:<3} {'-' if p is None else f'{p:.3f}':<10} {'-' if f is None else f'{f:.3f}'}")


def _calibrate(model, X, y, signers, test_signers, ks, cfg_cal: dict, device: str, seed: int):
    return evaluate_vs_k(
        model,
        X,
        y,
        signers,
        test_signers,
        ks,
        device,
        n_trials=cfg_cal["n_trials"],
        seed=seed,
        finetune_epochs=cfg_cal["finetune_epochs"],
        finetune_lr=cfg_cal["finetune_lr"],
    )


def run_meta_seed(cfg: dict, X, y, classes, signers, seed: int, eval_only: bool) -> dict:
    """Split -> pretrain -> meta-train -> accuracy-vs-K for one seed (one held-out signer pair)."""
    set_seed(seed, deterministic=cfg["deterministic"])
    device, ks = cfg["device"], cfg["calibration"]["ks"]
    model_type = cfg["model"]["type"]
    encoder_path = os.path.join(cfg["encoder_dir"], f"{model_type}_seed{seed}_encoder.pt")
    out_dir = os.path.join(cfg["out_root"], cfg["exp_name"])

    split, test_signers = random_subject_split(signers, cfg["split"]["n_test_signers"], seed=seed)
    save_split(
        split,
        os.path.join(out_dir, f"subject_split_seed{seed}.json"),
        seed=seed,
        protocol="subject_disjoint",
        meta={"test_signers": test_signers},
    )
    print(f"[seed {seed}] held-out test signers: {test_signers}")

    val_cfg = cfg["validation"]
    val_signers = None
    if val_cfg["source"] == "heldout":
        val_signers = pick_signers(signers, val_cfg["n_val_signers"], seed, exclude=test_signers)
        print(f"[seed {seed}] validation signers (from the training pool): {val_signers}")
    elif val_cfg["source"] != "test_signers":
        raise ValueError(
            f"validation.source must be 'test_signers' or 'heldout', got {val_cfg['source']!r}"
        )

    history, best_val = None, None
    if eval_only:
        model, _ = load_encoder(encoder_path, device=device)
        model.to(device)
    else:
        model = build_model(cfg["model"], num_classes=len(classes), streams=cfg["data"]["streams"])
        train_mask = ~np.isin(signers, test_signers)
        if val_signers is not None:
            train_mask &= ~np.isin(signers, val_signers)
        if cfg["pretrain"]["epochs"] > 0:
            p = cfg["pretrain"]
            pretrain_classifier(
                model,
                X[train_mask],
                y[train_mask],
                device=device,
                epochs=p["epochs"],
                batch_size=p["batch_size"],
                lr=p["lr"],
            )

        m = cfg["meta"]
        meta_kwargs = dict(
            device=device,
            n_way=m["n_way"],
            k_shot=m["k_shot"],
            q_query=m["q_query"],
            n_episodes_train=m["n_episodes_train"],
            n_episodes_val=m["n_episodes_val"],
            epochs=m["epochs"],
            lr=m["lr"],
            subject_disjoint=m["subject_disjoint"],
            select_best=val_cfg["select_best"],
            seed=seed,
        )
        if val_signers is None:
            # As in the original runs: the episodic validation uses the held-out test signers.
            result = meta_train(model, X, y, signers, val_signers=test_signers, **meta_kwargs)
        else:
            pool = ~np.isin(signers, test_signers)
            result = meta_train(
                model, X[pool], y[pool], signers[pool], val_signers=val_signers, **meta_kwargs
            )
        model, history, best_val = result["model"], result["history"], result["best_val_acc"]
        save_encoder(
            encoder_path,
            model,
            model_cfg=cfg["model"],
            streams=cfg["data"]["streams"],
            classes=classes,
            test_signers=test_signers,
            seed=seed,
            val_signers=val_signers,
        )
        print(f"[seed {seed}] encoder -> {encoder_path}")

    summary, raw = _calibrate(
        model, X, y, signers, test_signers, ks, cfg["calibration"], device, seed
    )
    _print_curve(f"[seed {seed}] accuracy vs K", summary, ks)
    return {
        "seed": seed,
        "test_signers": test_signers,
        "val_signers": val_signers,
        "meta_train_history": history,
        "best_val_acc_episodic": best_val,
        "accuracy_vs_k": summary,
        "accuracy_vs_k_raw": raw,
    }


def run_meta_train(cfg: dict, eval_only: bool) -> None:
    X_coords, y, classes, signers = load_bundle(cfg["data"]["bundle"])
    X = stack_streams_batch(X_coords, cfg["data"]["streams"])
    print(f"bundle: X={X.shape} classes={len(classes)} signers={len(set(signers.tolist()))}")
    out_dir = os.path.join(cfg["out_root"], cfg["exp_name"])
    write_run_info(out_dir, cfg)

    runs = []
    for seed in cfg["seeds"]:
        res = run_meta_seed(cfg, X, y, classes, signers, seed, eval_only)
        _dump(os.path.join(out_dir, f"seed{seed}.json"), {"classes": classes, **res})
        runs.append(res)

    ks = cfg["calibration"]["ks"]
    agg = aggregate_runs(runs, ks)
    _dump(
        os.path.join(out_dir, "summary.json"),
        {
            "seeds": cfg["seeds"],
            "test_signers_per_seed": {r["seed"]: r["test_signers"] for r in runs},
            "aggregate": agg,
        },
    )
    print("\n=== pooled over seeds ===")
    for k in ks:
        p, f = agg["protonet"][str(k)], agg["finetune_k"][str(k)]
        print(
            f"K={k}: ProtoNet {p['mean']:.3f} +/- {p['std']:.3f} | "
            f"fine-tune-K {f['mean']:.3f} +/- {f['std']:.3f}"
        )
    print(f"results -> {out_dir}/")


def run_transfer(cfg: dict) -> None:
    X_coords, y, classes, signers = load_bundle(cfg["data"]["bundle"])
    print(
        f"LSP bundle: X={X_coords.shape} classes={len(classes)} "
        f"signers={sorted(set(signers.tolist()))}"
    )
    out_dir = os.path.join(cfg["out_root"], cfg["exp_name"])
    write_run_info(out_dir, cfg)
    set_seed(cfg["calibration"]["seed"], deterministic=cfg["deterministic"])

    encoders = [(path, *load_encoder(path, device=cfg["device"])) for path in cfg["encoders"]]
    labels = [os.path.basename(p) for p, _, _ in encoders]
    if not cfg["allow_duplicate_splits"]:
        check_distinct_splits([ck["test_signers"] for _, _, ck in encoders], labels)

    per_signer = {s["id"]: [] for s in cfg["signers"]}
    for (_path, model, ckpt), label in zip(encoders, labels, strict=False):
        X = stack_streams_batch(X_coords, ckpt["streams"])
        model.to(cfg["device"])
        print(
            f"\n--- encoder {label} (LSA64 seed {ckpt.get('seed')}, "
            f"held-out {ckpt.get('test_signers')}) ---"
        )
        for s in cfg["signers"]:
            summary, raw = _calibrate(
                model,
                X,
                y,
                signers,
                [s["id"]],
                s["ks"],
                cfg["calibration"],
                cfg["device"],
                cfg["calibration"]["seed"],
            )
            _print_curve(f"  signer {s['id']}", summary, s["ks"])
            per_signer[s["id"]].append(
                {
                    "encoder": label,
                    "encoder_seed": ckpt.get("seed"),
                    "encoder_test_signers": ckpt.get("test_signers"),
                    "accuracy_vs_k": summary,
                    "accuracy_vs_k_raw": raw,
                }
            )

    results = {
        s["id"]: {
            "ks": s["ks"],
            "per_encoder": per_signer[s["id"]],
            "aggregate": aggregate_runs(per_signer[s["id"]], s["ks"]),
        }
        for s in cfg["signers"]
    }
    _dump(
        os.path.join(out_dir, "results.json"),
        {"encoders": labels, "classes": classes, "signers": results},
    )
    print("\n=== pooled over encoders ===")
    for sid, r in results.items():
        for k in r["ks"]:
            p, f = r["aggregate"]["protonet"][str(k)], r["aggregate"]["finetune_k"][str(k)]
            print(
                f"{sid} K={k}: ProtoNet {p['mean']:.3f} +/- {p['std']:.3f} | "
                f"fine-tune-K {f['mean']:.3f} +/- {f['std']:.3f}"
            )
    print(f"results -> {out_dir}/results.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="meta_train mode: evaluate saved encoders instead of training",
    )
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()
    cfg = load_with_overrides(args.config, args.overrides)

    if cfg["mode"] == "meta_train":
        run_meta_train(cfg, eval_only=args.eval_only)
    elif cfg["mode"] == "transfer":
        run_transfer(cfg)
    else:
        raise SystemExit(f"Unknown mode {cfg['mode']!r} (expected 'meta_train' or 'transfer')")


if __name__ == "__main__":
    main()
