"""
Core 2D canvas engine for MarketCanvas-Env.

The Canvas is a deterministic, in-memory scene graph.  All mutations go
through explicit methods so the environment can track every state transition.

Default resolution: 800 × 600 px (landscape email-banner dimensions).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from .elements import CanvasElement, ElementType, ElementRole


class CanvasError(Exception):
    """Raised for invalid canvas operations."""


class Canvas:
    """Lightweight 2D scene graph.

    Attributes
    ----------
    width, height : int
        Canvas resolution in pixels.
    elements : dict[str, CanvasElement]
        All elements keyed by their ``id``.
    _history : list[dict]
        Append-only log of every action applied (useful for replay / debugging).
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        self.width = width
        self.height = height
        self.elements: Dict[str, CanvasElement] = {}
        self._history: List[dict] = []

    # ------------------------------------------------------------------ #
    # Mutation API
    # ------------------------------------------------------------------ #

    def add_element(
        self,
        *,
        type: ElementType = "shape",
        role: ElementRole = "generic",
        x: float = 0.0,
        y: float = 0.0,
        width: float = 100.0,
        height: float = 50.0,
        z_index: int = 0,
        color: str = "#FFFFFF",
        text_color: str = "#000000",
        content: str = "",
        font_size: int = 16,
        border_color: str = "",
        border_width: int = 0,
        opacity: float = 1.0,
        corner_radius: int = 0,
        element_id: Optional[str] = None,
    ) -> CanvasElement:
        """Create and register a new element; return it."""
        el = CanvasElement(
            type=type,
            role=role,
            x=float(x),
            y=float(y),
            width=float(width),
            height=float(height),
            z_index=int(z_index),
            color=color,
            text_color=text_color,
            content=content,
            font_size=int(font_size),
            border_color=border_color,
            border_width=int(border_width),
            opacity=float(opacity),
            corner_radius=int(corner_radius),
        )
        if element_id:
            el.id = element_id
        self.elements[el.id] = el
        self._log("add_element", el.to_dict())
        return el

    def remove_element(self, element_id: str) -> None:
        """Remove an element by id.  Raises CanvasError if not found."""
        if element_id not in self.elements:
            raise CanvasError(f"Element '{element_id}' not found.")
        del self.elements[element_id]
        self._log("remove_element", {"id": element_id})

    def move_element(self, element_id: str, new_x: float, new_y: float) -> CanvasElement:
        """Translate element to an absolute position."""
        el = self._get(element_id)
        el.x = float(new_x)
        el.y = float(new_y)
        self._log("move_element", {"id": element_id, "x": el.x, "y": el.y})
        return el

    def resize_element(
        self, element_id: str, new_width: float, new_height: float
    ) -> CanvasElement:
        el = self._get(element_id)
        if new_width <= 0 or new_height <= 0:
            raise CanvasError("Width and height must be positive.")
        el.width = float(new_width)
        el.height = float(new_height)
        self._log("resize_element", {"id": element_id, "w": el.width, "h": el.height})
        return el

    def change_color(self, element_id: str, hex_color: str) -> CanvasElement:
        """Set the fill / background colour of an element."""
        _validate_hex(hex_color)
        el = self._get(element_id)
        el.color = hex_color
        self._log("change_color", {"id": element_id, "color": hex_color})
        return el

    def change_text_color(self, element_id: str, hex_color: str) -> CanvasElement:
        _validate_hex(hex_color)
        el = self._get(element_id)
        el.text_color = hex_color
        self._log("change_text_color", {"id": element_id, "text_color": hex_color})
        return el

    def change_content(self, element_id: str, content: str) -> CanvasElement:
        el = self._get(element_id)
        el.content = content
        self._log("change_content", {"id": element_id, "content": content})
        return el

    def set_z_index(self, element_id: str, z_index: int) -> CanvasElement:
        el = self._get(element_id)
        el.z_index = int(z_index)
        self._log("set_z_index", {"id": element_id, "z_index": z_index})
        return el

    def set_font_size(self, element_id: str, font_size: int) -> CanvasElement:
        el = self._get(element_id)
        el.font_size = max(6, int(font_size))
        self._log("set_font_size", {"id": element_id, "font_size": el.font_size})
        return el

    def set_opacity(self, element_id: str, opacity: float) -> CanvasElement:
        el = self._get(element_id)
        el.opacity = max(0.0, min(1.0, float(opacity)))
        self._log("set_opacity", {"id": element_id, "opacity": el.opacity})
        return el

    def change_role(self, element_id: str, role: ElementRole) -> CanvasElement:
        el = self._get(element_id)
        el.role = role
        self._log("change_role", {"id": element_id, "role": role})
        return el

    def clear(self) -> None:
        """Remove all elements."""
        self.elements.clear()
        self._log("clear", {})

    # ------------------------------------------------------------------ #
    # Query / observation API
    # ------------------------------------------------------------------ #

    def get_element(self, element_id: str) -> CanvasElement:
        return self._get(element_id)

    def elements_by_role(self, role: ElementRole) -> List[CanvasElement]:
        return [e for e in self.elements.values() if e.role == role]

    def elements_sorted_by_z(self) -> List[CanvasElement]:
        return sorted(self.elements.values(), key=lambda e: (e.z_index, e.id))

    def to_semantic_state(self) -> dict:
        """Return the full canvas as a JSON-serialisable DOM dict.

        The structure is analogous to a browser Accessibility Tree:
        each element node contains all properties plus derived spatial
        relationships (overlaps_with, contained_within_canvas).
        """
        canvas_node: dict = {
            "canvas": {
                "width": self.width,
                "height": self.height,
                "element_count": len(self.elements),
            },
            "elements": [],
        }

        ordered = self.elements_sorted_by_z()
        id_to_idx = {e.id: i for i, e in enumerate(ordered)}

        for el in ordered:
            node = el.to_dict()
            node["within_bounds"] = el.is_within(self.width, self.height)

            # Annotate which other elements this one overlaps
            overlapping_ids = []
            for other in ordered:
                if other.id != el.id and el.intersection_area(other) > 0:
                    overlapping_ids.append(other.id)
            node["overlaps_with"] = overlapping_ids

            canvas_node["elements"].append(node)

        return canvas_node

    def to_json(self) -> str:
        return json.dumps(self.to_semantic_state(), indent=2)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get(self, element_id: str) -> CanvasElement:
        if element_id not in self.elements:
            raise CanvasError(f"Element '{element_id}' not found.")
        return self.elements[element_id]

    def _log(self, action: str, params: dict) -> None:
        self._history.append({"action": action, "params": params})


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _validate_hex(hex_color: str) -> None:
    """Raise ValueError if *hex_color* is not a valid 6-digit hex string."""
    c = hex_color.lstrip("#")
    if len(c) not in (3, 6) or not all(ch in "0123456789abcdefABCDEF" for ch in c):
        raise ValueError(
            f"Invalid hex colour '{hex_color}'. Expected format: #RRGGBB or #RGB."
        )
