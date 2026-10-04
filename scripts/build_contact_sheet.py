# scripts/build_contact_sheet.py
"""Tile the descriptive figures onto one sheet, so the set can be judged as a set.

    uv run --with matplotlib python -m scripts.build_contact_sheet

WHY. Figure choice is a decision about the set, not about one plot at a time: two
figures answering the same question are only visible as a duplicate when they sit side
by side. This renders every PNG in the descriptive figures directory into a single
labelled sheet for exactly that comparison.

It reads the PNG previews that the figure scripts already wrote and adds nothing of its
own -- no data, no statistic, no ordering claim. Regenerating it after a figure is
removed simply produces a smaller sheet, which is the intended behaviour: the sheet
always shows the set as it currently stands.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIR = _REPO / "results/analysis/heldout_full_matrix/figures"
STEM = "CONTACT_SHEET"
COLUMNS = 3


def collect(directory: Path) -> list[Path]:
    """Every figure preview in the directory, in filename order, sheet itself excluded."""
    return sorted(p for p in directory.glob("*.png") if p.stem != STEM)


def build(images: Sequence[Path], out_dir: Path, columns: int = COLUMNS) -> Path:
    if not images:
        raise SystemExit(f"no PNG figures found in {out_dir}")
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.6 * columns, 3.9 * rows))
    flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, path in zip(flat, images):
        ax.imshow(mpimg.imread(path))
        ax.set_title(path.stem, fontsize=9)
        ax.axis("off")
    for ax in flat[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    out = out_dir / f"{STEM}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--columns", type=int, default=COLUMNS)
    args = ap.parse_args(argv)

    images = collect(args.dir)
    out = build(images, args.dir, args.columns)
    print(f"contact sheet: {len(images)} figures -> {out.relative_to(_REPO)}")
    for p in images:
        print(f"  {p.stem}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
