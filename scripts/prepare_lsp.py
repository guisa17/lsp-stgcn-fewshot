"""Build the LSP bundle and its intra-subject split from raw recorded takes.

Each raw take is a ``(30, 285)`` ``.npy`` file named ``<CLASS>_<timestamp>.npy`` inside one folder
per signer (the folder name is the signer id). The bundle stores, per sample, the normalized
canonical coordinates ``(T, 73, 3)``, the class index and the signer id.

Usage:
    python scripts/prepare_lsp.py --config configs/lsp/prepare.yaml \
        --raw /path/to/raw/signer_00 --raw /path/to/raw/signer_01
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

from lsp.data import canonical as C
from lsp.data.normalize import normalize_sequence
from lsp.data.splits import make_intra_subject_split, save_split
from lsp.utils.config import load_with_overrides


def label_from_filename(path: str) -> str:
    """``AGUA_1783680638791.npy`` -> ``AGUA`` (strips the trailing ``_<digits>``)."""
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(.*)_\d+$", name)
    return (m.group(1) if m else name).strip().upper()


def collect_files(raw_dirs: list[str]) -> list[tuple[str, str]]:
    """(signer_id, path) for every ``.npy`` directly inside each raw folder."""
    files = []
    for raw_dir in raw_dirs:
        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"Raw folder not found: {raw_dir}")
        signer_id = os.path.basename(os.path.normpath(raw_dir))
        found = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))
        if not found:
            print(f"  ! no .npy files in {raw_dir}, skipped")
        files.extend((signer_id, f) for f in found)
    if not files:
        raise SystemExit(f"No .npy files found in: {', '.join(raw_dirs)}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--raw",
        required=True,
        action="append",
        help="Raw folder of one signer; repeat the flag for several signers",
    )
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()
    cfg = load_with_overrides(args.config, args.overrides)

    files = collect_files(args.raw)
    labels_str = [label_from_filename(f) for _, f in files]
    classes = sorted(set(labels_str))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X, y, signers = [], [], []
    for (signer_id, f), lab in zip(files, labels_str, strict=False):
        arr = np.load(f).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != C.RAW_FEATURES:
            print(
                f"  ! skipped {os.path.basename(f)}: shape {arr.shape}, "
                f"expected (*, {C.RAW_FEATURES})"
            )
            continue
        coords = normalize_sequence(C.raw_to_canonical(arr), smooth_w=cfg["smooth_window"])
        X.append(coords.astype(np.float32))
        y.append(class_to_idx[lab])
        signers.append(signer_id)

    X = np.stack(X, axis=0)
    y = np.asarray(y, dtype=np.int64)
    os.makedirs(os.path.dirname(cfg["out"]) or ".", exist_ok=True)
    np.savez_compressed(cfg["out"], X=X, y=y, classes=np.array(classes), signers=np.array(signers))

    split_cfg = cfg["split"]
    split = make_intra_subject_split(
        y, seed=split_cfg["seed"], val_size=split_cfg["val_size"], test_size=split_cfg["test_size"]
    )
    signer_counts = {s: signers.count(s) for s in sorted(set(signers))}
    save_split(
        split,
        cfg["split_out"],
        seed=split_cfg["seed"],
        protocol=split_cfg["protocol"],
        meta={
            "bundle": os.path.basename(cfg["out"]),
            "num_samples": len(y),
            "num_classes": len(classes),
            "signers": signer_counts,
        },
    )

    print(f"samples={len(y)} classes={len(classes)} signers={len(signer_counts)} X={X.shape}")
    print(f"  bundle -> {cfg['out']}")
    print(
        f"  split  -> {cfg['split_out']} "
        f"(train={len(split['train'])} val={len(split['val'])} test={len(split['test'])})"
    )


if __name__ == "__main__":
    main()
