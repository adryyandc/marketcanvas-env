"""
demo.py — MarketCanvas-Env demonstration script
================================================
Initialises the environment with a mock target prompt, executes a short
sequence of programmatic actions derived from the parsed spec, then prints
the final canvas state and reward breakdown.

Optionally saves the rendered canvas as a PNG if Pillow is installed.

Usage
-----
    python demo.py                                  # default prompt, PNG save
    python demo.py --prompt "A product launch banner with a headline and CTA"
    python demo.py --no-render                      # skip PNG rendering
    python demo.py --output my_banner.png
    python demo.py --skip-episode                   # skip the random-action episode
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Ensure Unicode box-drawing characters render correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from marketcanvas import MarketCanvasEnv, TargetSpec

# ---------------------------------------------------------------------------
# Default mock prompt (overridable via --prompt)
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = (
    "Create an email banner with headline 'Summer Sale', "
    "sub-headline 'Up to 50% off everything', "
    "body copy 'Limited time offer. Shop today.', "
    "a 'Shop Now' call-to-action button, and a supporting image."
)

# Canvas dimensions
W, H = 800, 600

# Role → minimal element template
_ROLE_TEMPLATES: dict[str, dict] = {
    "background": {
        "type": "shape", "role": "background",
        "x": 0, "y": 0, "width": W, "height": H,
        "color": "#1A237E", "text_color": "#FFFFFF", "z_index": 0,
    },
    "headline": {
        "type": "text", "role": "headline",
        "x": 40, "y": 80, "width": 400, "height": 90,
        "color": "#1A237E", "text_color": "#FFD700",
        "content": "Headline Text", "font_size": 48, "z_index": 2,
    },
    "subheadline": {
        "type": "text", "role": "subheadline",
        "x": 40, "y": 185, "width": 400, "height": 60,
        "color": "#1A237E", "text_color": "#FFFFFF",
        "content": "Supporting sub-headline", "font_size": 22, "z_index": 2,
    },
    "body": {
        "type": "text", "role": "body",
        "x": 40, "y": 260, "width": 380, "height": 80,
        "color": "#1A237E", "text_color": "#C5CAE9",
        "content": "Body copy goes here.", "font_size": 16, "z_index": 2,
    },
    "cta": {
        "type": "shape", "role": "cta",
        "x": 40, "y": 370, "width": 200, "height": 56,
        "color": "#FFD700", "text_color": "#1A237E",
        "content": "Action", "font_size": 20, "z_index": 3,
        "corner_radius": 28,
    },
    "image_placeholder": {
        "type": "image", "role": "image_placeholder",
        "x": 460, "y": 80, "width": 300, "height": 420,
        "color": "#3949AB", "text_color": "#FFFFFF",
        "content": "Image", "font_size": 14, "z_index": 1,
        "border_color": "#5C6BC0", "border_width": 2,
    },
}

_FALLBACK_ROLE = {
    "type": "shape", "role": "generic",
    "x": 40, "y": 40, "width": 200, "height": 60,
    "color": "#607D8B", "text_color": "#FFFFFF",
    "content": "Element", "font_size": 16, "z_index": 2,
}

# Role → keywords used to locate nearby quoted text in the prompt
_ROLE_CONTEXT_KEYWORDS: dict[str, list[str]] = {
    "headline":          ["headline", "title", "header"],
    "subheadline":       ["sub-headline", "subheadline", "subtitle"],
    "body":              ["body", "copy", "description"],
    "cta":               ["button", "cta", "call-to-action", "call to action"],
    "image_placeholder": ["image", "photo", "graphic"],
}


def _extract_role_content(prompt: str, role: str) -> str | None:
    """Return quoted text from *prompt* closest to a role keyword, or None.

    Scans the full prompt for all quoted strings first (avoiding mid-quote
    window artifacts), then returns the nearest match to any role keyword.
    Only text-bearing roles are considered; image/background placeholders
    have no meaningful inline text to extract.
    """
    _TEXT_ROLES = {"headline", "subheadline", "body", "cta"}
    if role not in _TEXT_ROLES:
        return None

    keywords = _ROLE_CONTEXT_KEYWORDS.get(role, [])
    p_lower = prompt.lower()

    # Collect all properly-quoted strings with their start positions
    all_quotes = [
        (m.start(), m.group(1) or m.group(2))
        for m in re.finditer(r"'([^']{2,80})'|\"([^\"]{2,80})\"", prompt)
    ]
    if not all_quotes:
        return None

    for kw in keywords:
        idx = p_lower.find(kw)
        if idx == -1:
            continue
        nearest_pos, nearest_text = min(all_quotes, key=lambda q: abs(q[0] - idx))
        if abs(nearest_pos - idx) <= 40:
            return nearest_text
    return None


# ---------------------------------------------------------------------------
# Generic demo action sequence (derived from spec)
# ---------------------------------------------------------------------------

def apply_demo_steps(env: MarketCanvasEnv, spec: TargetSpec) -> None:
    """Add one element per required role extracted from *spec*.

    Content text is pulled from quoted strings in the target prompt when
    available, falling back to the generic placeholder in *_ROLE_TEMPLATES*.
    """
    # Always ensure a background exists first
    roles = list(spec.required_roles)
    if "background" not in roles:
        roles.insert(0, "background")
    else:
        roles = ["background"] + [r for r in roles if r != "background"]

    steps = []
    for idx, role in enumerate(roles):
        template = dict(_ROLE_TEMPLATES.get(role, _FALLBACK_ROLE))
        template["role"] = role
        template["id"] = f"{role}_{idx}"
        extracted = _extract_role_content(spec.description, role)
        if extracted:
            template["content"] = extracted
        steps.append({"action_type": "add_element", "params": template})

    print(f"\n{'─'*60}")
    print("  Applying design actions…")
    print(f"{'─'*60}")

    for i, action in enumerate(steps):
        result = env.apply_action(action["action_type"], action["params"])
        status = "OK " if result["success"] else "ERR"
        el_id  = action["params"].get("id", "?")
        role   = action["params"].get("role", "")
        print(f"  [{status}] step {i+1:2d}  add '{el_id}' ({role})")
        if not result["success"]:
            print(f"          Error: {result['error']}")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_state(env: MarketCanvasEnv) -> None:
    state = env.get_semantic_state()
    print(f"\n{'═'*60}")
    print("  CANVAS SEMANTIC STATE")
    print(f"{'═'*60}")
    print(f"  Canvas : {state['canvas']['width']} × {state['canvas']['height']} px")
    print(f"  Elements: {state['canvas']['element_count']}")
    print()
    for el in state["elements"]:
        within   = "✓" if el["within_bounds"] else "✗"
        overlaps = ", ".join(el["overlaps_with"]) if el["overlaps_with"] else "none"
        print(
            f"  [{within}] id={el['id']:16s} role={el['role']:18s} "
            f"type={el['type']:6s} z={el['z_index']}"
        )
        print(
            f"       pos=({el['x']:.0f},{el['y']:.0f})  "
            f"size=({el['width']:.0f}×{el['height']:.0f})  "
            f"color={el['color']}  text={el['text_color']}"
        )
        if el.get("content"):
            preview = el["content"][:40].replace("\n", " ↵ ")
            print(f"       content='{preview}'")
        print(f"       overlaps_with=[{overlaps}]")
    print()


def print_reward(breakdown: dict) -> None:
    print(f"{'═'*60}")
    print("  REWARD BREAKDOWN")
    print(f"{'═'*60}")
    labels = {
        "element_presence": "Element presence",
        "wcag_contrast":    "WCAG contrast",
        "layout_alignment": "Layout alignment",
        "overlap_penalty":  "Overlap penalty",
        "boundary_score":   "Boundary compliance",
        "content_quality":  "Content quality",
    }
    for key, label in labels.items():
        val = breakdown[key]
        bar_len = int(abs(val) * 20)
        bar = ("+" if val >= 0 else "-") * bar_len
        print(f"  {label:22s} {val:+.4f}  [{bar:<20s}]")
    print(f"  {'─'*50}")
    print(f"  {'Raw total':22s} {breakdown['raw_total']:+.4f}")
    print(f"  {'Normalised reward':22s} {breakdown['reward']:+.4f}  (range: -1.0 → 1.0)")
    print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    prompt = args.prompt or DEFAULT_PROMPT

    print("\n" + "═" * 60)
    print("  MarketCanvas-Env  —  Demo")
    print("═" * 60)
    print(f"\n  Target prompt:\n  \"{prompt}\"\n")

    spec = TargetSpec.from_prompt(prompt, W, H)
    print(f"  Parsed required roles: {spec.required_roles}")

    env = MarketCanvasEnv(
        target_spec=spec,
        canvas_width=W,
        canvas_height=H,
        max_steps=50,
    )
    env.reset()

    # --- Apply design actions -------------------------------------------
    apply_demo_steps(env, spec)

    # --- Print canvas state ---------------------------------------------
    print_state(env)

    # --- Compute and print reward ---------------------------------------
    breakdown = env.compute_reward()
    print_reward(breakdown)

    # --- Save PNG -------------------------------------------------------
    if not args.no_render:
        output = args.output or "demo_banner.png"
        try:
            saved = env.save_png(output)
            print(f"  Rendered canvas saved to: {saved}")
        except ImportError:
            print("  [SKIP] PNG rendering requires Pillow: pip install Pillow")
        except Exception as exc:
            print(f"  [WARN] Could not save PNG: {exc}")

    print()

    # --- Demonstrate a full Gymnasium episode (random actions) ----------
    if not args.skip_episode:
        print("─" * 60)
        print("  Running a short random-action Gymnasium episode…")
        print("─" * 60)

        ep_env = MarketCanvasEnv(
            target_prompt="Create a banner with a headline and a CTA button.",
            max_steps=5,
        )
        _, info = ep_env.reset()
        print(f"  Reset OK — canvas is empty (elements: {info['element_count']})")

        random_actions = [
            {"action_type": "add_element", "params": {
                "type": "shape", "role": "background",
                "x": 0, "y": 0, "width": 800, "height": 600,
                "color": "#222222", "z_index": 0,
            }},
            {"action_type": "add_element", "params": {
                "type": "text", "role": "headline",
                "x": 100, "y": 80, "width": 600, "height": 100,
                "color": "#222222", "text_color": "#FFFFFF",
                "content": "Big Sale!", "font_size": 48, "z_index": 1,
            }},
            {"action_type": "add_element", "params": {
                "type": "shape", "role": "cta",
                "x": 300, "y": 250, "width": 200, "height": 60,
                "color": "#FF9800", "text_color": "#000000",
                "content": "Buy Now", "font_size": 20, "z_index": 2,
                "corner_radius": 8,
            }},
            {"action_type": "move_element", "params": {"id": "?", "x": 50, "y": 50}},  # errors gracefully
            {"action_type": "no_op",        "params": {}},
        ]

        for step_idx, action in enumerate(random_actions):
            _, reward, terminated, truncated, info = ep_env.step(action)
            done = terminated or truncated
            err  = info.get("action_error", "")
            print(
                f"  step {step_idx+1}: action={action['action_type']:18s}"
                f"  reward={reward:+.4f}"
                f"  elements={info['element_count']}"
                f"  done={done}"
                + (f"  [err: {err}]" if err else "")
            )
            if done:
                print(f"\n  Episode done.  Final reward = {reward:+.4f}")
                bd = info.get("reward_breakdown", {})
                if bd:
                    print_reward(bd)
                break

    print("  Demo complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketCanvas-Env demo")
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Target design prompt (default: generic email banner prompt)",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip saving the PNG output",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PNG file path (default: demo_banner.png)",
    )
    parser.add_argument(
        "--skip-episode", action="store_true",
        help="Skip the random-action gymnasium episode demo",
    )
    main(parser.parse_args())
