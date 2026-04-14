"""Open an image in an interactive matplotlib window for pixel inspection.

Hovering over the image shows ``x``, ``y`` and the pixel value in the toolbar
status bar. Use the toolbar's zoom/pan to get down to single pixels. Scroll to
zoom via the mouse wheel.

Usage::

    python -m target_hold.resources.view_image                     # shows rgb_raw.png
    python -m target_hold.resources.view_image --image seg_raw.png
    python -m target_hold.resources.view_image --image /abs/path.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Interactive Qt backend (devcontainer already has X11 forwarding + DISPLAY set).
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_IMAGE = Path(__file__).resolve().parent / "demo_outputs" / "rgb_raw.png"


def _format_cursor(img: np.ndarray):
    """Return a function that matplotlib calls to format the coord status string."""
    if img.ndim == 2:
        def fmt(x, y):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
                return f"x={xi}  y={yi}  v={int(img[yi, xi])}"
            return f"x={x:.1f}  y={y:.1f}"
    else:
        def fmt(x, y):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
                rgb = tuple(int(v) for v in img[yi, xi, :3])
                return f"x={xi}  y={yi}  RGB={rgb}"
            return f"x={x:.1f}  y={y:.1f}"
    return fmt


def _install_scroll_zoom(ax):
    """Scroll to zoom centered on cursor (re-enables after matplotlib zoom resets)."""
    def on_scroll(event):
        if event.inaxes is not ax:
            return
        base = 1.2
        scale = 1 / base if event.button == "up" else base
        xdata, ydata = event.xdata, event.ydata
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        new_w = (x1 - x0) * scale
        new_h = (y1 - y0) * scale
        ax.set_xlim(xdata - (xdata - x0) * scale, xdata + (x1 - xdata) * scale)
        ax.set_ylim(ydata - (ydata - y0) * scale, ydata + (y1 - ydata) * scale)
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("scroll_event", on_scroll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        help=f"Image path (default: {DEFAULT_IMAGE})")
    args = parser.parse_args()

    path = args.image if args.image.is_absolute() else Path.cwd() / args.image
    if not path.exists():
        # Try relative to this file's directory too, for convenience.
        alt = Path(__file__).resolve().parent / "demo_outputs" / args.image.name
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    img = np.array(Image.open(path))
    print(f"[view] {path}  shape={img.shape}  dtype={img.dtype}")

    fig, ax = plt.subplots(figsize=(10, 6))
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    else:
        ax.imshow(img, interpolation="nearest")

    ax.format_coord = _format_cursor(img)
    ax.set_title(f"{path.name}  |  hover for pixel values  |  scroll to zoom  |  home to reset")
    fig.tight_layout()

    _install_scroll_zoom(ax)

    plt.show()


if __name__ == "__main__":
    main()
