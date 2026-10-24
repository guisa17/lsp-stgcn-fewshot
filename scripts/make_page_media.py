"""Skeleton-only media and failure analysis for the project page (signer_00 data only).

1. Re-runs ProtoNet calibration of the frozen LSA64 encoders on signer_00 with the same draws as
   Table 3, and checks that the pooled accuracy equals the released Table 3 values.
2. Summarizes errors: per-class accuracy and the most frequent confusions (JSON + bar chart).
3. Renders a teaser and success / failure examples as GIFs (body and hands only; no face).

Usage:
    python scripts/make_page_media.py --config configs/page/media.yaml \
        bundle=/path/outside/repo/lsp_signer00_v1.npz
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

from lsp.data.dataset import load_bundle
from lsp.data.streams import stack_streams_batch
from lsp.evaluation.fewshot import protonet_predictions
from lsp.training.checkpoint import load_encoder
from lsp.utils.config import load_with_overrides
from lsp.viz.render import Caption, render_frames, save_gif, view_limits

CORRECTED_TABLE3 = "results/transfer/corrected_3_splits.json"


def run_calibration(cfg: dict, X_coords, y, signers) -> dict[str, list[dict]]:
    """``{encoder_path: trials}`` with the ProtoNet predictions of every trial."""
    cal = cfg["calibration"]
    out = {}
    for path in cal["encoders"]:
        model, ckpt = load_encoder(path, device=cal["device"])
        X = stack_streams_batch(X_coords, ckpt["streams"])
        out[path] = protonet_predictions(
            model, X, y, signers, cfg["signer"], cal["ks"], cal["device"],
            n_trials=cal["n_trials"], seed=cal["seed"],
        )  # fmt: skip
    return out


def check_against_table3(trials_by_encoder: dict, y: np.ndarray, signer: str) -> None:
    """Pooled ProtoNet accuracy per K must equal the released Table 3 (corrected) values."""
    with open(CORRECTED_TABLE3, encoding="utf-8") as f:
        ref = json.load(f)["signers"][signer]["aggregate"]["protonet"]
    for k, r in ref.items():
        accs = [
            float(np.mean(np.array(t["pred"]) == y[t["eval_idx"]]))
            for trials in trials_by_encoder.values()
            for t in trials
            if t["k"] == int(k)
        ]
        diff = abs(float(np.mean(accs)) - r["mean"])
        print(f"  K={k}: pooled ProtoNet {np.mean(accs):.4f} vs Table 3 {r['mean']:.4f}")
        if diff > 1e-9:
            raise SystemExit(f"Mismatch with Table 3 at K={k} (diff {diff:.2e})")


def failure_analysis(trials_by_encoder: dict, y, classes, ks, top_n) -> dict:
    out = {}
    for k in ks:
        correct, total, confusions = Counter(), Counter(), Counter()
        for trials in trials_by_encoder.values():
            for t in (t for t in trials if t["k"] == k):
                for true, pred in zip(y[t["eval_idx"]], t["pred"], strict=True):
                    total[classes[true]] += 1
                    correct[classes[true]] += int(true == pred)
                    if true != pred:
                        confusions[(classes[true], classes[pred])] += 1
        n_errors = sum(confusions.values())
        out[str(k)] = {
            "per_class_accuracy": {c: correct[c] / total[c] for c in classes},
            "n_predictions": sum(total.values()),
            "n_errors": n_errors,
            "top_confusions": [
                {"true": a, "predicted": b, "count": n, "share_of_errors": n / n_errors}
                for (a, b), n in confusions.most_common(top_n)
            ],
        }
    return out


def plot_per_class(analysis: dict, classes, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted(analysis, key=int)
    order = sorted(classes, key=lambda c: analysis[ks[-1]]["per_class_accuracy"][c])
    fig, ax = plt.subplots(figsize=(7, 7.5))
    pos = np.arange(len(order))
    width = 0.8 / len(ks)
    colors = ["#c98a2b", "#2f7f86", "#5b6b7c"]
    for i, k in enumerate(ks):
        vals = [100 * analysis[k]["per_class_accuracy"][c] for c in order]
        ax.barh(pos + i * width, vals, height=width, color=colors[i % 3], label=f"K = {k}")
    ax.set_yticks(pos + width * (len(ks) - 1) / 2)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("accuracy (%), frozen LSA64 encoders, ProtoNet calibration")
    ax.axvline(100 / len(classes), color="#999999", linewidth=0.8, linestyle=":")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pick_examples(trial: dict, y, conf_order: str, correct: bool, n: int, exclude=()) -> list:
    """(index, pred) pairs of distinct true classes, by decreasing confidence."""
    rows = sorted(
        zip(trial["eval_idx"], trial["pred"], trial["confidence"], strict=True),
        key=lambda r: -r[2] if conf_order == "high" else r[2],
    )
    chosen, seen = [], set(exclude)
    for idx, pred, conf in rows:
        if (pred == y[idx]) == correct and y[idx] not in seen:
            chosen.append((int(idx), int(pred), float(conf)))
            seen.add(y[idx])
        if len(chosen) == n:
            break
    return chosen


def save_og_image(frames: list[np.ndarray], og: dict) -> None:
    """Link-preview image: a row of teaser frames centred on a white canvas."""
    from PIL import Image

    width, height = og["size"]
    picks = [frames[i] for i in og["frames"]]
    canvas = Image.new("RGB", (width, height), "white")
    tile = picks[0].shape[0]
    x = (width - tile * len(picks)) // 2
    for frame in picks:
        canvas.paste(Image.fromarray(frame), (x, (height - tile) // 2))
        x += tile
    canvas.save(og["path"], optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()
    cfg = load_with_overrides(args.config, args.overrides)

    X_coords, y, classes, signers = load_bundle(cfg["bundle"])
    if set(signers.tolist()) != {cfg["signer"]}:
        raise SystemExit(
            f"The bundle must contain only {cfg['signer']}, got {set(signers.tolist())}"
        )
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["analysis_out"]), exist_ok=True)

    print("calibrating frozen encoders (same draws as Table 3)...")
    trials = run_calibration(cfg, X_coords, y, signers)
    check_against_table3(trials, y, cfg["signer"])

    a_cfg = cfg["analysis"]
    analysis = failure_analysis(trials, y, classes, a_cfg["ks"], a_cfg["top_confusions"])
    plot_per_class(analysis, classes, os.path.join(cfg["out_dir"], "per_class_accuracy.png"))

    ex = cfg["examples"]
    k_trials = [t for t in trials[ex["encoder"]] if t["k"] == ex["k"]]
    trial = k_trials[ex["trial"]]
    class_idx = {c: i for i, c in enumerate(classes)}
    teaser_ids = {class_idx[c] for c in ex["teaser_classes"]}
    by_class = {y[i]: (i, p, c) for i, p, c in pick_examples(trial, y, "high", True, len(classes))}
    teaser = [by_class[c] for c in (class_idx[n] for n in ex["teaser_classes"]) if c in by_class]
    successes = pick_examples(trial, y, "high", True, ex["n_success"], exclude=teaser_ids)
    failures = pick_examples(trial, y, "high", False, ex["n_failure"])

    r = cfg["render"]
    selected = teaser + successes + failures
    limits = view_limits([X_coords[i] for i, _, _ in selected], r["margin"], r["head_radius"])
    note = f"frozen encoder, K = {ex['k']}"

    def clip(idx, pred):
        caption = Caption(classes[y[idx]], classes[pred], note)
        return render_frames(X_coords[idx], caption, limits=limits, size_px=r["size_px"],
                             head_radius=r["head_radius"])  # fmt: skip

    teaser_frames = [f for idx, pred, _ in teaser for f in clip(idx, pred)]
    save_gif(teaser_frames, os.path.join(cfg["out_dir"], "teaser.gif"), r["fps"])
    save_og_image(teaser_frames, cfg["og_image"])
    for name, items in (("success", successes), ("failure", failures)):
        for n, (idx, pred, _) in enumerate(items, start=1):
            save_gif(clip(idx, pred), os.path.join(cfg["out_dir"], f"{name}_{n}.gif"), r["fps"])

    def describe(items):
        return [{"index": i, "true": classes[y[i]], "predicted": classes[p], "confidence": c}
                for i, p, c in items]  # fmt: skip

    with open(cfg["analysis_out"], "w", encoding="utf-8") as f:
        json.dump({
            "signer": cfg["signer"],
            "setting": "frozen HA-GCN encoders meta-trained on LSA64, ProtoNet calibration, "
                       "same calibration draws as Table 3",
            "analysis": analysis,
            "examples": {"encoder": os.path.basename(ex["encoder"]), "k": ex["k"],
                         "trial": ex["trial"], "teaser": describe(teaser),
                         "success": describe(successes), "failure": describe(failures)},
        }, f, indent=2, ensure_ascii=False)  # fmt: skip

    for name in sorted(os.listdir(cfg["out_dir"])):
        size = os.path.getsize(os.path.join(cfg["out_dir"], name))
        print(f"  {name:28s} {size / 1e6:.2f} MB")
    print(f"analysis -> {cfg['analysis_out']}")


if __name__ == "__main__":
    main()
