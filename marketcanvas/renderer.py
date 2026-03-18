"""
PIL-based visual renderer for MarketCanvas-Env.

Produces an RGB pixel array (numpy) and can save a .png file.
Each element type is rendered as follows:

  shape           → filled rectangle (with optional border & rounded corners)
  text            → filled rectangle + centred text overlay
  image           → filled rectangle + diagonal cross + alt-text label
                    (simulates an image placeholder / lorem picsum box)

Fonts fall back gracefully: tries to load a system TrueType font; if not
available, PIL's built-in bitmap font is used instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from .canvas import Canvas
from .elements import CanvasElement


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

_FONT_CACHE: dict[int, "ImageFont.FreeTypeFont | ImageFont.ImageFont"] = {}

def _get_font(size: int):
    """Return a PIL font of the requested size, with graceful fallback."""
    if not _PIL_AVAILABLE:
        return None
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    # Candidate TrueType font paths (Windows / Linux / macOS)
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    font = None
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                pass

    if font is None:
        font = ImageFont.load_default()

    _FONT_CACHE[size] = font
    return font


# ---------------------------------------------------------------------------
# Hex → PIL colour
# ---------------------------------------------------------------------------

def _hex_to_rgba(hex_color: str, opacity: float = 1.0) -> Tuple[int, int, int, int]:
    c = hex_color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    a = int(opacity * 255)
    return (r, g, b, a)


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

class CanvasRenderer:
    """Renders a Canvas to a PIL Image (RGBA).

    Usage
    -----
    renderer = CanvasRenderer()
    image    = renderer.render(canvas)
    image.save("output.png")
    pixels   = renderer.to_numpy(canvas)   # uint8 RGB array (H, W, 3)
    """

    CANVAS_BG_COLOR = (245, 245, 245, 255)   # light grey stage

    def __init__(self) -> None:
        if not _PIL_AVAILABLE:
            raise ImportError(
                "Pillow is required for visual rendering. "
                "Install it with: pip install Pillow"
            )

    def render(self, canvas: Canvas) -> "Image.Image":
        """Return a PIL RGBA Image of the current canvas state."""
        img = Image.new("RGBA", (canvas.width, canvas.height), self.CANVAS_BG_COLOR)
        draw = ImageDraw.Draw(img, "RGBA")

        # Draw elements in z-index order (lowest first)
        for el in canvas.elements_sorted_by_z():
            self._draw_element(draw, img, el)

        return img.convert("RGB")

    def to_numpy(self, canvas: Canvas):
        """Return a uint8 numpy array of shape (H, W, 3)."""
        if not _NUMPY_AVAILABLE:
            raise ImportError("numpy is required for pixel-array output.")
        img = self.render(canvas)
        return np.array(img, dtype="uint8")

    def save(self, canvas: Canvas, path: str) -> str:
        """Render and save to *path*.  Returns the resolved absolute path."""
        img = self.render(canvas)
        resolved = str(Path(path).resolve())
        img.save(resolved, format="PNG")
        return resolved

    # ------------------------------------------------------------------ #
    # Per-element drawing
    # ------------------------------------------------------------------ #

    def _draw_element(
        self,
        draw: "ImageDraw.ImageDraw",
        img: "Image.Image",
        el: CanvasElement,
    ) -> None:
        x0, y0 = int(el.x), int(el.y)
        x1, y1 = int(el.x + el.width), int(el.y + el.height)
        fill_rgba = _hex_to_rgba(el.color, el.opacity)

        if el.type == "shape" or el.role in ("background", "divider"):
            self._draw_rect(draw, x0, y0, x1, y1, fill_rgba, el)

        elif el.type == "text" or el.role in (
            "headline", "subheadline", "body", "cta", "generic"
        ):
            self._draw_text_element(draw, x0, y0, x1, y1, fill_rgba, el)

        elif el.type == "image" or el.role == "image_placeholder":
            self._draw_image_placeholder(draw, x0, y0, x1, y1, fill_rgba, el)

        else:
            # Fallback: draw as shape
            self._draw_rect(draw, x0, y0, x1, y1, fill_rgba, el)

    def _draw_rect(self, draw, x0, y0, x1, y1, fill, el: CanvasElement) -> None:
        radius = el.corner_radius
        if radius > 0:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=fill)
        if el.border_color and el.border_width > 0:
            border_rgba = _hex_to_rgba(el.border_color)
            draw.rectangle(
                [x0, y0, x1, y1],
                outline=border_rgba,
                width=el.border_width,
            )

    def _draw_text_element(
        self, draw, x0, y0, x1, y1, fill, el: CanvasElement
    ) -> None:
        # Background box
        radius = el.corner_radius
        if radius > 0:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=fill)
        if el.border_color and el.border_width > 0:
            draw.rectangle(
                [x0, y0, x1, y1],
                outline=_hex_to_rgba(el.border_color),
                width=el.border_width,
            )

        if not el.content:
            return

        font = _get_font(el.font_size)
        text_rgba = _hex_to_rgba(el.text_color)

        # Centre text within the bounding box
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        try:
            bbox = draw.textbbox((0, 0), el.content, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx, ty = cx - tw // 2, cy - th // 2
        except AttributeError:
            # Older PIL fallback
            tw, th = draw.textsize(el.content, font=font)  # type: ignore[attr-defined]
            tx, ty = cx - tw // 2, cy - th // 2

        draw.text((tx, ty), el.content, fill=text_rgba, font=font)

    def _draw_image_placeholder(
        self, draw, x0, y0, x1, y1, fill, el: CanvasElement
    ) -> None:
        # Filled rect
        draw.rectangle([x0, y0, x1, y1], fill=fill)
        if el.border_color and el.border_width > 0:
            draw.rectangle(
                [x0, y0, x1, y1],
                outline=_hex_to_rgba(el.border_color),
                width=el.border_width,
            )
        # Diagonal cross to signal "image box"
        cross_color = (0, 0, 0, 60)
        draw.line([x0, y0, x1, y1], fill=cross_color, width=2)
        draw.line([x1, y0, x0, y1], fill=cross_color, width=2)

        if not el.content:
            return

        font = _get_font(max(10, el.font_size - 4))
        text_rgba = _hex_to_rgba(el.text_color)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        try:
            bbox = draw.textbbox((0, 0), el.content, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(el.content, font=font)  # type: ignore[attr-defined]
        draw.text((cx - tw // 2, cy - th // 2), el.content, fill=text_rgba, font=font)
