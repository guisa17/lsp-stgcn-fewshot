"""Re-pool saved transfer results without some encoders (no retraining, no recalibration).

The original report pooled four HA-GCN encoders, two of which were trained on the same LSA64 signer
split. This script recomputes the pooled mean and std per signer, method and K from the per-trial
values already stored in the results file, keeping only the encoders that are not excluded.

Usage:
    python scripts/reaggregate_transfer.py --results results/transfer/original_4_encoders.json \
        --exclude lsa64_eval_gpu_hagcn_encoder.pt --out results/transfer/corrected_3_splits.json
"""

from __future__ import annotations

import argparse
import json
import os

from lsp.evaluation.aggregate import aggregate_runs


def per_signer_runs(data: dict) -> dict[str, tuple[list[int], list[dict]]]:
    """``{signer: (ks, runs)}`` from the original result format or the format of this repository."""
    if "signers" in data:  # written by scripts/fewshot_calibrate.py
        return {
            sid: (r["ks"], [dict(run, label=run["encoder"]) for run in r["per_encoder"]])
            for sid, r in data["signers"].items()
        }
    out = {}
    for group in ("rich", "sparse"):  # original result format
        runs = data[f"{group}_results_per_encoder"]
        names = data["encoders_used"]
        labeled = [dict(run, label=str(n)) for run, n in zip(runs, names, strict=True)]
        ks = [int(k) for k in runs[0]["accuracy_vs_k"]["protonet"]]
        out[data[f"{group}_signer"]] = (ks, labeled)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--exclude", nargs="+", required=True, help="Encoder file names to drop")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)

    output = {"source": os.path.basename(args.results), "excluded": args.exclude, "signers": {}}
    for sid, (ks, runs) in per_signer_runs(data).items():
        kept = [r for r in runs if r["label"] not in args.exclude]
        if len(kept) == len(runs):
            raise SystemExit(
                f"None of {args.exclude} matches the encoders {[r['label'] for r in runs]}"
            )
        output["signers"][sid] = {
            "ks": ks,
            "encoders": [r["label"] for r in kept],
            "aggregate": aggregate_runs(kept, ks),
            "aggregate_all_encoders": aggregate_runs(runs, ks),
        }
        for k in ks:
            new = output["signers"][sid]["aggregate"]
            old = output["signers"][sid]["aggregate_all_encoders"]
            for method in ("protonet", "finetune_k"):
                n, o = new[method][str(k)], old[method][str(k)]
                print(
                    f"{sid} K={k} {method:10s} all={o['mean']:.4f}+/-{o['std']:.4f} "
                    f"(n={o['n_pooled']}) -> kept={n['mean']:.4f}+/-{n['std']:.4f} "
                    f"(n={n['n_pooled']})"
                )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
