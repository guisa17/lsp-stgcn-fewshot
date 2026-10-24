"""Build the LSA64 bundle (73-node canonical layout) from the Kaggle MediaPipe export.

The export directory must contain ``train_landmarks.pkl``, ``test_landmarks.pkl`` and their label
CSVs (``LSA64_videos_{train,test}.csv``).

Usage:
    # quick check on 2 signers x 5 classes first
    python scripts/prepare_lsa64.py --config configs/lsa64/prepare.yaml --lsa64-dir /path/to/lsa64 \
        --smoke out=data/external/lsa64_smoke.npz
    python scripts/prepare_lsa64.py --config configs/lsa64/prepare.yaml --lsa64-dir /path/to/lsa64
"""

from __future__ import annotations

import argparse

import numpy as np

from lsp.data import canonical as C
from lsp.data.lsa64 import build_bundle, save_bundle
from lsp.utils.config import load_with_overrides


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--lsa64-dir", required=True)
    ap.add_argument("--smoke", action="store_true", help="Process only the smoke-test subset")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()
    cfg = load_with_overrides(args.config, args.overrides)

    splits, signers, classes = cfg["splits"], None, None
    if args.smoke:
        splits, signers, classes = (
            cfg["smoke"]["splits"],
            cfg["smoke"]["signers"],
            cfg["smoke"]["classes"],
        )
        print(f"[smoke] splits={splits} signers={signers} classes={classes}")

    bundle = build_bundle(
        args.lsa64_dir,
        splits=splits,
        T=cfg["T"],
        smooth=cfg["smooth_window"],
        signers=signers,
        classes=classes,
    )
    save_bundle(bundle, cfg["out"])

    X = bundle["X"]
    assert X.shape[1:] == (cfg["T"], C.N, 3), f"unexpected shape {X.shape}"
    assert not np.isnan(X).any(), "the bundle contains NaN"
    print(f"  bundle -> {cfg['out']} X={X.shape}, no NaN")
    print("  signers:", ", ".join(sorted(set(bundle["signers"]))))


if __name__ == "__main__":
    main()
