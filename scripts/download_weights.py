"""Download the released LSA64 encoders into ``checkpoints/lsa64/``.

Only encoders meta-trained on the public LSA64 dataset are planned for release. Models trained on
LSP recordings are not released (see the README, "Data").

Usage:
    pip install -e ".[weights]"
    python scripts/download_weights.py --repo-id <hf-user>/<repo>
"""

from __future__ import annotations

import argparse
import os

# TODO(weights): set once the encoders are published on the Hugging Face Hub.
DEFAULT_REPO_ID = None
ENCODER_FILES = [
    "hagcn_seed42_encoder.pt",
    "hagcn_seed7_encoder.pt",
    "hagcn_seed123_encoder.pt",
    "stgcn_seed42_encoder.pt",
    "stgcn_seed7_encoder.pt",
    "stgcn_seed123_encoder.pt",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--out-dir", default="checkpoints/lsa64")
    ap.add_argument("--files", nargs="+", default=ENCODER_FILES)
    args = ap.parse_args()

    if not args.repo_id:
        raise SystemExit("The weights are not published yet. Pass --repo-id once they are.")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit('huggingface_hub is missing: pip install -e ".[weights]"') from e

    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.files:
        path = hf_hub_download(repo_id=args.repo_id, filename=name, local_dir=args.out_dir)
        print(f"  {name} -> {path}")


if __name__ == "__main__":
    main()
