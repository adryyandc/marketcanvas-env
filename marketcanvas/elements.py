"""
Canvas element definitions for MarketCanvas-Env.

Three element types are supported:
  - text:   A text label with a background box
  - shape:  A solid-colored rectangle (used for backgrounds, buttons, dividers)
  - image:  A colored bounding-box placeholder representing an image asset

Every element also carries a *role* that encodes its semantic purpose
(headline, cta, body, background, image_placeholder, generic).  The reward
function uses roles rather than raw types so the agent can express intent.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Literal, Tuple

# Supported element types
ElementType = Literal["text", "shape", "image"]

# Semantic roles the agent can assign to an element
ElementRole = Literal[
    "headline",
    "subheadline",
    "body",
    "cta",           # Call-to-action button
    "background",
    "image_placeholder",
    "divider",
    "generic",
]


def _new_id() -> str:
    return str(uuid.uuid4())[:8]


@dataclass
class CanvasElement:
    """A single element on the design canvas."""

    id: str = field(default_factory=_new_id)
    type: ElementType = "shape"
    role: ElementRole = "generic"

    # Position and size (canvas pixels)
    x: float = 0.0
    y: float = 0.0
    width: float = 100.0
    height: float = 50.0

    # Layering
    z_index: int = 0

    # Appearance
    color: str = "#FFFFFF"       # fill / background hex colour
    text_color: str = "#000000"  # foreground text hex colour
    content: str = ""            # visible text or alt-text for images
    font_size: int = 16
    border_color: str = ""       # optional border; empty = no border
    border_width: int = 0
    opacity: float = 1.0         # 0.0 (transparent) – 1.0 (opaque)
    corner_radius: int = 0       # rounded corners (px)

    # ------------------------------------------------------------------ #
    # Derived geometry helpers
    # ------------------------------------------------------------------ #

    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2) bounding box."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def area(self) -> float:
        return self.width * self.height

    def intersection_area(self, other: "CanvasElement") -> float:
        """Pixel area of axis-aligned intersection with *other*."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        if x2 > x1 and y2 > y1:
            return (x2 - x1) * (y2 - y1)
        return 0.0

    def is_within(self, canvas_width: float, canvas_height: float) -> bool:
        """True if the element lies fully within the canvas."""
        return (
            self.x >= 0
            and self.y >= 0
            and self.x + self.width <= canvas_width
            and self.y + self.height <= canvas_height
        )

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        d = asdict(self)
        cx, cy = self.center()
        d["center_x"] = round(cx, 2)
        d["center_y"] = round(cy, 2)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "CanvasElement":
        # Strip derived keys that are not constructor args
        data = {k: v for k, v in data.items() if k not in ("center_x", "center_y")}
        return cls(**data)
