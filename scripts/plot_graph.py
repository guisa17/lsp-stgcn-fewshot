"""Draw the 73-node skeleton graph.

Usage:
    python scripts/plot_graph.py --out docs/figures/skeleton_graph.png
    python scripts/plot_graph.py --out fig.png --bundle data/external/lsa64_v1.npz
"""

from __future__ import annotations

import argparse

from lsp.viz.plot_graph import plot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", help="Use the mean frame of the bundle's first sequence instead of "
                                     "the schematic layout")  # fmt: skip
    args = ap.parse_args()
    plot(args.out, args.bundle)


if __name__ == "__main__":
    main()
