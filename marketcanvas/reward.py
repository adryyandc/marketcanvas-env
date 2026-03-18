"""
Heuristic reward function for MarketCanvas-Env.

The reward is a scalar in [-1.0, 1.0] computed at the end of an episode.
It aggregates five independent sub-scores:

  1. element_presence   (0.0 – 0.35)  Required roles present on canvas
  2. wcag_contrast      (0.0 – 0.25)  Text/background contrast passes WCAG AA
  3. layout_alignment   (0.0 – 0.20)  Elements horizontally centred on canvas
  4. overlap_penalty    (-0.30 – 0.0) Penalise illegible element overlaps
  5. boundary_score     (0.0 – 0.10)  All elements inside canvas bounds
  6. content_quality    (0.0 – 0.10)  Non-empty content on headline / CTA

Total possible: -0.30 to 1.00  →  linearly mapped to [-1.0, 1.0].

Potential reward-hacking loopholes (documented deliberately):
  - An agent could satisfy "role presence" by placing all elements at (0,0)
    in a tiny 1×1 stack — overlap penalty partially mitigates this.
  - All-white-text on all-white-background passes "presence" but fails WCAG.
  - Centering is measured per-element; stacking centred elements on top of
    each other still scores full alignment credit.
  See WRITEUP.md for a full discussion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .canvas import Canvas
from .elements import CanvasElement


# ---------------------------------------------------------------------------
# WCAG colour utilities
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    c = hex_color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _relative_luminance(hex_color: str) -> float:
    """WCAG 2.1 relative luminance (0 = black, 1 = white)."""
    r, g, b = (_linearise(ch / 255) for ch in _hex_to_rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _linearise(c: float) -> float:
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def wcag_contrast_ratio(fg: str, bg: str) -> float:
    """Return WCAG contrast ratio between two hex colours."""
    l1 = _relative_luminance(fg)
    l2 = _relative_luminance(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def passes_wcag_aa(fg: str, bg: str, large_text: bool = False) -> bool:
    """True if contrast passes WCAG 2.1 AA (4.5:1 normal, 3:1 large text)."""
    threshold = 3.0 if large_text else 4.5
    return wcag_contrast_ratio(fg, bg) >= threshold


# ---------------------------------------------------------------------------
# Target specification
# ---------------------------------------------------------------------------

@dataclass
class TargetSpec:
    """Structured requirements that the reward function evaluates against.

    Parameters
    ----------
    description : str
        Human-readable prompt, e.g. "Summer Sale email banner".
    required_roles : list[str]
        Element roles the canvas must contain, e.g. ["headline", "cta"].
    canvas_width, canvas_height : int
        Expected canvas dimensions (for alignment checks).
    min_elements : int
        Minimum total number of distinct elements expected.
    center_tolerance : float
        How close (px) an element's centre must be to the canvas horizontal
        midpoint to be counted as "centred".
    """

    description: str = ""
    required_roles: List[str] = field(default_factory=list)
    canvas_width: int = 800
    canvas_height: int = 600
    min_elements: int = 2
    center_tolerance: float = 50.0   # pixels

    @classmethod
    def from_prompt(cls, prompt: str, canvas_width: int = 800, canvas_height: int = 600) -> "TargetSpec":
        """Heuristic keyword-based extraction from a free-text prompt.

        Recognised keywords → roles:
          headline / title / header     → "headline"
          cta / button / call-to-action → "cta"
          image / photo / banner image  → "image_placeholder"
          subheadline / subtitle        → "subheadline"
          body / text / copy            → "body"
        """
        p = prompt.lower()
        roles: list[str] = []

        mapping = {
            "headline": ["headline", "title", "header"],
            "cta": ["cta", "button", "call-to-action", "call to action", "shop now"],
            "image_placeholder": ["image", "photo", "banner image", "graphic"],
            "subheadline": ["subheadline", "subtitle", "sub-headline"],
            "body": ["body", "copy", "description"],
        }
        for role, keywords in mapping.items():
            if any(kw in p for kw in keywords):
                roles.append(role)

        if not roles:
            roles = ["headline", "cta"]   # sensible fallback

        return cls(
            description=prompt,
            required_roles=roles,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    element_presence: float = 0.0
    wcag_contrast: float = 0.0
    layout_alignment: float = 0.0
    overlap_penalty: float = 0.0
    boundary_score: float = 0.0
    content_quality: float = 0.0

    @property
    def raw_total(self) -> float:
        return (
            self.element_presence
            + self.wcag_contrast
            + self.layout_alignment
            + self.overlap_penalty
            + self.boundary_score
            + self.content_quality
        )

    @property
    def normalised(self) -> float:
        """Map raw score from [-0.30, 1.00] to [-1.0, 1.0]."""
        raw = self.raw_total
        raw_min, raw_max = -0.30, 1.00
        normalised = 2 * (raw - raw_min) / (raw_max - raw_min) - 1.0
        return round(max(-1.0, min(1.0, normalised)), 4)

    def to_dict(self) -> dict:
        return {
            "element_presence": round(self.element_presence, 4),
            "wcag_contrast": round(self.wcag_contrast, 4),
            "layout_alignment": round(self.layout_alignment, 4),
            "overlap_penalty": round(self.overlap_penalty, 4),
            "boundary_score": round(self.boundary_score, 4),
            "content_quality": round(self.content_quality, 4),
            "raw_total": round(self.raw_total, 4),
            "reward": self.normalised,
        }


class RewardFunction:
    """Stateless heuristic reward calculator."""

    # Sub-score weights (must sum to ≤ 1.0 for the positive components)
    W_PRESENCE   = 0.35
    W_WCAG       = 0.25
    W_ALIGN      = 0.20
    W_OVERLAP    = 0.30   # max penalty magnitude
    W_BOUNDARY   = 0.10
    W_CONTENT    = 0.10

    def __call__(self, canvas: Canvas, spec: TargetSpec) -> RewardBreakdown:
        bd = RewardBreakdown()
        elements = list(canvas.elements.values())

        if not elements:
            return bd   # all zeros → reward = -1.0

        bd.element_presence = self._score_presence(elements, spec)
        bd.wcag_contrast     = self._score_wcag(canvas, elements, spec)
        bd.layout_alignment  = self._score_alignment(elements, spec)
        bd.overlap_penalty   = self._score_overlap(elements, canvas)
        bd.boundary_score    = self._score_boundary(elements, canvas)
        bd.content_quality   = self._score_content(elements)

        return bd

    # ------------------------------------------------------------------ #
    # Sub-scores
    # ------------------------------------------------------------------ #

    def _score_presence(self, elements: list[CanvasElement], spec: TargetSpec) -> float:
        """Fraction of required roles present, scaled by W_PRESENCE."""
        if not spec.required_roles:
            return self.W_PRESENCE   # no requirements = full score

        present_roles = {e.role for e in elements}
        satisfied = sum(1 for r in spec.required_roles if r in present_roles)
        return self.W_PRESENCE * (satisfied / len(spec.required_roles))

    def _score_wcag(
        self, canvas: Canvas, elements: list[CanvasElement], spec: TargetSpec
    ) -> float:
        """Average WCAG-pass fraction across text-bearing elements."""
        text_elements = [e for e in elements if e.type == "text" or e.role in ("headline", "subheadline", "body", "cta")]
        if not text_elements:
            return 0.0

        passes = 0
        for el in text_elements:
            # Find the topmost element directly beneath this one (by z-index)
            bg_color = self._background_color_under(el, elements, canvas)
            large = el.font_size >= 18 or el.role == "headline"
            if passes_wcag_aa(el.text_color, bg_color, large_text=large):
                passes += 1

        return self.W_WCAG * (passes / len(text_elements))

    def _score_alignment(self, elements: list[CanvasElement], spec: TargetSpec) -> float:
        """Fraction of elements whose horizontal centre is close to the canvas midpoint."""
        canvas_cx = spec.canvas_width / 2
        non_bg = [e for e in elements if e.role != "background"]
        if not non_bg:
            return 0.0

        centred = sum(
            1 for e in non_bg
            if abs(e.center()[0] - canvas_cx) <= spec.center_tolerance
        )
        return self.W_ALIGN * (centred / len(non_bg))

    def _score_overlap(self, elements: list[CanvasElement], canvas: Canvas) -> float:
        """Penalise overlapping non-background elements.

        Penalty is proportional to the total overlapping area divided by the
        canvas area.  Background elements are excluded because deliberate
        layering (bg → image → text) is fine.
        """
        foreground = [e for e in elements if e.role != "background"]
        canvas_area = canvas.width * canvas.height
        total_overlap = 0.0

        for i, a in enumerate(foreground):
            for b in foreground[i + 1:]:
                ia = a.intersection_area(b)
                # Only penalise if overlap is significant (> 10% of smaller element)
                if ia > 0.1 * min(a.area(), b.area()):
                    total_overlap += ia

        if total_overlap == 0:
            return 0.0

        # Normalise: overlap / canvas_area, capped at 1.0
        overlap_fraction = min(1.0, total_overlap / canvas_area)
        return -self.W_OVERLAP * overlap_fraction

    def _score_boundary(self, elements: list[CanvasElement], canvas: Canvas) -> float:
        """Fraction of elements fully within canvas bounds."""
        if not elements:
            return 0.0
        within = sum(1 for e in elements if e.is_within(canvas.width, canvas.height))
        return self.W_BOUNDARY * (within / len(elements))

    def _score_content(self, elements: list[CanvasElement]) -> float:
        """Reward non-empty content strings on semantically important elements."""
        important = [
            e for e in elements
            if e.role in ("headline", "subheadline", "cta", "body")
        ]
        if not important:
            return 0.0
        non_empty = sum(1 for e in important if e.content.strip())
        return self.W_CONTENT * (non_empty / len(important))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _background_color_under(
        self,
        el: CanvasElement,
        elements: list[CanvasElement],
        canvas: Canvas,
    ) -> str:
        """
        Return the effective background hex colour under *el*.

        Strategy: find the element with the highest z-index that is strictly
        below *el* and spatially overlaps it.  If none, fall back to the
        element's own background colour (text on transparent = its own box).
        """
        candidates = [
            e for e in elements
            if e.id != el.id
            and e.z_index < el.z_index
            and el.intersection_area(e) > 0
        ]
        if candidates:
            beneath = max(candidates, key=lambda e: e.z_index)
            return beneath.color
        # No element below — assume white canvas background
        return "#FFFFFF"
