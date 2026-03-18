"""
demo_llm.py — MarketCanvas-Env × Groq LLM demo
===============================================
Connects the Groq free API (Llama 3.3 70B) to the MarketCanvas environment
using the same tool interface exposed by the MCP server.  The LLM autonomously
interprets a design prompt and calls tools to build the banner, step by step.

Image placeholders are filled with AI-generated images via the Hugging Face
Inference API (FLUX.1-schnell model — free tier, no card required).

Setup
-----
    # .env file (or real environment variables):
    GROQ_API_KEY=<key>            # free at https://console.groq.com
    HUGGINGFACE_API_KEY=<token>   # free at https://huggingface.co/settings/tokens

Usage
-----
    python demo_llm.py
    python demo_llm.py --prompt "A bold product launch banner with headline and CTA"
    python demo_llm.py --max-turns 20
    python demo_llm.py --no-render
    python demo_llm.py --output my_llm_banner.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

# Ensure Unicode renders correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from groq import Groq
except ImportError:
    print(
        "ERROR: 'groq' package not found.\n"
        "Install it with:  pip install groq\n"
        "Then set your free API key:  export GROQ_API_KEY=<key>",
        file=sys.stderr,
    )
    sys.exit(1)

from marketcanvas import MarketCanvasEnv, TargetSpec

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL        = "llama-3.3-70b-versatile"   # free tier on Groq
DEFAULT_PROMPT = (
    "Create an email banner with headline 'Summer Sale', "
    "sub-headline 'Up to 50% off everything', "
    "body copy 'Limited time offer. Shop today.', "
    "a 'Shop Now' call-to-action button, and a supporting image."
)
W, H = 800, 600

# ---------------------------------------------------------------------------
# MCP-compatible tool handlers (call the env directly)
# ---------------------------------------------------------------------------

def _tool_get_canvas_state(env: MarketCanvasEnv, _params: dict) -> dict:
    return env.get_semantic_state()


def _tool_execute_action(env: MarketCanvasEnv, params: dict) -> dict:
    action_type = params.get("action_type", "no_op")
    action_params = params.get("params", {})
    return env.apply_action(action_type, action_params)


def _tool_get_current_reward(env: MarketCanvasEnv, _params: dict) -> dict:
    return env.compute_reward()


def _tool_reset_environment(env: MarketCanvasEnv, params: dict) -> dict:
    opts = {"target_prompt": params["target_prompt"]} if params.get("target_prompt") else None
    env.reset(options=opts)
    return {
        "success": True,
        "target_prompt": env.target_prompt,
        "required_roles": env.spec.required_roles,
    }


def _tool_list_actions(_env: MarketCanvasEnv, _params: dict) -> dict:
    from mcp_server import ACTION_CATALOGUE
    return ACTION_CATALOGUE


_HANDLERS = {
    "get_canvas_state":   _tool_get_canvas_state,
    "execute_action":     _tool_execute_action,
    "get_current_reward": _tool_get_current_reward,
    "reset_environment":  _tool_reset_environment,
    "list_actions":       _tool_list_actions,
}

# ---------------------------------------------------------------------------
# Groq tool schemas (mirrors the MCP server tool definitions)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_canvas_state",
            "description": (
                "Return the complete current state of the design canvas. "
                "Each element has id, type, role, position, size, z_index, "
                "colours, content, and overlap information."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_action",
            "description": (
                "Apply one action to the canvas. Use action_type to choose "
                "the operation and params for its arguments. "
                "Call list_actions first to see all available action types."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "description": (
                            "One of: add_element, move_element, resize_element, "
                            "change_color, change_text_color, change_content, "
                            "set_z_index, set_font_size, set_opacity, "
                            "remove_element, change_role, no_op"
                        ),
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            "Action-specific parameters. For add_element include: "
                            "type ('text'|'shape'|'image'), role, x, y, width, height, "
                            "z_index, color (hex), text_color (hex), content, font_size. "
                            "For move_element: id, x, y. For resize_element: id, width, height."
                        ),
                    },
                },
                "required": ["action_type", "params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_reward",
            "description": (
                "Compute and return the reward breakdown for the current canvas. "
                "Returns sub-scores for element_presence, wcag_contrast, "
                "layout_alignment, overlap_penalty, boundary_score, content_quality, "
                "and a normalised reward in [-1.0, 1.0]."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_actions",
            "description": "Return the full action catalogue with all valid action types and their parameter schemas.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are an expert graphic designer working on a {W}×{H} px design canvas.
Your goal is to create a banner that matches the user's design prompt.

Canvas rules:
- Canvas is {W}×{H} pixels; keep all elements within these bounds.
- z_index: background=0, images=1, text=2, buttons/badges=3.
- Always begin with a full-canvas background (role="background", x=0, y=0, width={W}, height={H}, z_index=0).
- Use high-contrast colour pairs for WCAG compliance (e.g. #FFD700 text on #1A237E background).
- Available element types: "text", "shape", "image".
- Available roles: "background", "headline", "subheadline", "body", "cta", "image_placeholder", "divider", "generic".

Image elements (type="image", role="image_placeholder"):
- Set the "content" field to a vivid, descriptive English phrase of what the image should show.
  Example: content="majestic lion roaring on a rocky cliff at golden hour"
- This description is sent to an AI image generator, so be specific and visual.
- Do NOT use generic words like "image" or "photo" — describe the actual subject.

Suggested workflow:
1. Call list_actions to confirm available operations.
2. Add all required elements using execute_action(action_type="add_element", ...).
3. Call get_current_reward to evaluate your design.
4. Optionally refine elements (move, resize, recolor) to improve the reward.
5. When satisfied, call get_current_reward one final time and stop.

Design brief follows in the user message."""

# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_llm_loop(
    env: MarketCanvasEnv,
    prompt: str,
    client: Groq,
    max_turns: int,
) -> list[dict]:
    """Drive the LLM tool-calling loop. Returns the full message history."""

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    print(f"\n{'─'*60}")
    print("  LLM agentic loop starting…")
    print(f"{'─'*60}")

    for turn in range(1, max_turns + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )

        msg = response.choices[0].message
        finish = response.choices[0].finish_reason

        # Append the assistant turn (may include tool_calls)
        messages.append(msg.model_dump(exclude_unset=True))

        if not msg.tool_calls:
            # LLM finished — print any final text
            if msg.content:
                print(f"\n  [LLM] {msg.content}")
            print(f"\n  Loop ended after {turn} turn(s)  (finish={finish})")
            break

        # Execute each tool call and feed results back
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            handler = _HANDLERS.get(fn_name)
            if handler:
                result = handler(env, fn_args)
            else:
                result = {"error": f"Unknown tool '{fn_name}'"}

            # Summarise for the console
            summary = _summarise_tool_result(fn_name, fn_args, result)
            print(f"  [turn {turn:2d}] {fn_name:<22s} → {summary}")

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(result),
            })

    return messages


def _summarise_tool_result(name: str, args: dict, result: dict | list) -> str:
    """One-line human-readable summary of a tool result."""
    if name == "execute_action":
        ok = result.get("success", False)
        el_id = args.get("params", {}).get("id", "?")
        role  = args.get("params", {}).get("role", "")
        atype = args.get("action_type", "?")
        status = "OK" if ok else f"ERR({result.get('error', '?')})"
        return f"{status}  {atype}  id={el_id}  role={role}"
    if name == "get_current_reward":
        reward = result.get("reward", "?")
        return f"reward={reward:+.4f}" if isinstance(reward, float) else str(reward)
    if name == "get_canvas_state":
        n = len(result.get("elements", []))
        return f"{n} elements on canvas"
    if name == "reset_environment":
        return f"OK  roles={result.get('required_roles', [])}"
    if name == "list_actions":
        return f"{len(result)} action types"
    return str(result)[:80]


# ---------------------------------------------------------------------------
# AI image compositing via Hugging Face Inference API (huggingface_hub SDK)
# ---------------------------------------------------------------------------

# Ordered list of free-tier text-to-image models to try.
_HF_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5",
]


def _composite_generated_images(env: MarketCanvasEnv, png_path: str) -> None:
    """Replace every image_placeholder on the saved PNG with an AI-generated image.

    Uses huggingface_hub.InferenceClient which handles endpoint routing and
    model warm-up automatically.  Tries each model in _HF_MODELS until one
    succeeds.

    Reads HUGGINGFACE_API_KEY (or HF_TOKEN) from the environment.
    Free token: https://huggingface.co/settings/tokens  (no card required)
    """
    try:
        from huggingface_hub import InferenceClient
        from PIL import Image as PILImage
    except ImportError as exc:
        print(f"  [SKIP] Missing dependency ({exc}) — skipping image generation.")
        print("         Install with: pip install huggingface_hub Pillow")
        return

    hf_token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "  [SKIP] HUGGINGFACE_API_KEY not set — skipping image generation.\n"
            "         Get a free token at https://huggingface.co/settings/tokens"
        )
        return

    state = env.get_semantic_state()
    image_elements = [
        el for el in state["elements"]
        if el.get("type") == "image" or el.get("role") == "image_placeholder"
    ]
    if not image_elements:
        return

    client = InferenceClient(token=hf_token)
    banner = PILImage.open(png_path).convert("RGBA")

    for el in image_elements:
        description = (el.get("content") or "").strip()
        if not description or description.lower() in ("image", "photo", ""):
            description = "abstract colorful artwork"

        w = max(1, int(el["width"]))
        h = max(1, int(el["height"]))
        x = int(el["x"])
        y = int(el["y"])

        print(f"  [IMG] Generating: \"{description[:60]}\" ({w}×{h})…")
        generated = None
        for model in _HF_MODELS:
            try:
                img = client.text_to_image(description, model=model)
                generated = img.convert("RGBA").resize((w, h), PILImage.LANCZOS)
                print(f"       → {model.split('/')[1]}  composited at ({x},{y})")
                break
            except Exception as exc:
                print(f"       {model.split('/')[1]} failed: {exc}")

        if generated:
            banner.paste(generated, (x, y))
        else:
            print("  [WARN] All models failed — placeholder kept.")

    banner.convert("RGB").save(png_path, format="PNG")
    print(f"  [IMG] Final banner saved to: {png_path}")


# ---------------------------------------------------------------------------
# Pretty-print helpers (same as demo.py)
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
    load_dotenv(override=True)  # Load from .env if present, but allow real env vars to override
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "ERROR: GROQ_API_KEY environment variable not set.\n"
            "Get a free key at https://console.groq.com",
            file=sys.stderr,
        )
        sys.exit(1)

    prompt = args.prompt or DEFAULT_PROMPT

    print("\n" + "═" * 60)
    print("  MarketCanvas-Env  —  LLM Demo  (Groq / Llama 3.3 70B)")
    print("═" * 60)
    print(f"\n  Model  : {MODEL}")
    print(f"  Prompt : {prompt}\n")

    # Build env
    spec = TargetSpec.from_prompt(prompt, W, H)
    print(f"  Parsed required roles: {spec.required_roles}")

    env = MarketCanvasEnv(
        target_spec=spec,
        canvas_width=W,
        canvas_height=H,
        max_steps=100,
    )
    env.reset()

    # Run agentic loop
    client = Groq(api_key=api_key)
    run_llm_loop(env, prompt, client, max_turns=args.max_turns)

    # Final state + reward
    print_state(env)
    breakdown = env.compute_reward()
    print_reward(breakdown)

    # Save PNG
    if not args.no_render:
        output = args.output or "llm_banner.png"
        try:
            saved = env.save_png(output)
            print(f"  Rendered canvas saved to: {saved}")
            _composite_generated_images(env, saved)
        except ImportError:
            print("  [SKIP] PNG rendering requires Pillow: pip install Pillow")
        except Exception as exc:
            print(f"  [WARN] Could not save PNG: {exc}")

    print("\n  Demo complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketCanvas-Env LLM demo via Groq")
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Design prompt for the LLM (default: Summer Sale banner)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=15,
        help="Maximum LLM turns before stopping (default: 15)",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip saving the PNG output",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PNG file path (default: llm_banner.png)",
    )
    main(parser.parse_args())
