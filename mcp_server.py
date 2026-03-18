"""
MarketCanvas MCP Server
=======================
Exposes the MarketCanvasEnv as a Model Context Protocol (MCP) server so that
LLM clients (e.g. Claude Desktop, Claude Code) can interact with the canvas
environment via structured tool calls.

Transport: stdio (default for local MCP servers)

Start the server
----------------
    python mcp_server.py

Or from Claude Desktop's config (mcpServers section in claude_desktop_config.json):
    {
      "mcpServers": {
        "marketcanvas": {
          "command": "python",
          "args": ["<absolute-path-to>/mcp_server.py"]
        }
      }
    }

Exposed tools
-------------
  get_canvas_state      → Full JSON DOM tree of the current canvas
  execute_action        → Apply a single action to the canvas
  get_current_reward    → Compute and return the reward breakdown
  reset_environment     → Reset to a blank canvas (optionally new prompt)
  render_canvas         → Save the canvas as a PNG and return the file path
  list_actions          → Return the catalogue of valid action types + schemas
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure the package is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    print(
        "WARNING: 'mcp' package not found.  Install it with: pip install mcp\n"
        "The MCP server will not start.",
        file=sys.stderr,
    )

from marketcanvas import MarketCanvasEnv, TargetSpec

# ---------------------------------------------------------------------------
# Global environment instance
# ---------------------------------------------------------------------------
_env = MarketCanvasEnv(
    target_prompt="Create a Summer Sale email banner with a headline, "
                  "a yellow CTA button, and good contrast",
    max_steps=100,   # generous limit for interactive LLM sessions
)

# ---------------------------------------------------------------------------
# Action schema catalogue (shown to LLMs via list_actions)
# ---------------------------------------------------------------------------
ACTION_CATALOGUE = {
    "add_element": {
        "description": "Add a new element to the canvas.",
        "params": {
            "type":          "('text'|'shape'|'image') — element type",
            "role":          "('headline'|'subheadline'|'body'|'cta'|'background'"
                             "|'image_placeholder'|'divider'|'generic')",
            "x":             "float — left edge in pixels",
            "y":             "float — top edge in pixels",
            "width":         "float — element width in pixels",
            "height":        "float — element height in pixels",
            "z_index":       "int — stacking order (higher = on top)",
            "color":         "str — fill hex colour e.g. '#FFD700'",
            "text_color":    "str — text hex colour e.g. '#FFFFFF'",
            "content":       "str — visible text or alt-text",
            "font_size":     "int — font size in points",
            "border_color":  "str — optional border hex colour",
            "border_width":  "int — border thickness px (0 = none)",
            "opacity":       "float — 0.0 to 1.0",
            "corner_radius": "int — rounded corner radius px",
            "id":            "str — optional fixed element ID",
        },
    },
    "move_element": {
        "description": "Move an element to an absolute canvas position.",
        "params": {"id": "str", "x": "float", "y": "float"},
    },
    "resize_element": {
        "description": "Resize an element.",
        "params": {"id": "str", "width": "float", "height": "float"},
    },
    "change_color": {
        "description": "Change the fill/background colour of an element.",
        "params": {"id": "str", "color": "str hex e.g. '#FF0000'"},
    },
    "change_text_color": {
        "description": "Change the text colour of an element.",
        "params": {"id": "str", "color": "str hex"},
    },
    "change_content": {
        "description": "Update the text content or alt-text of an element.",
        "params": {"id": "str", "content": "str"},
    },
    "set_z_index": {
        "description": "Set the stacking order of an element.",
        "params": {"id": "str", "z_index": "int"},
    },
    "set_font_size": {
        "description": "Set the font size (points) of a text element.",
        "params": {"id": "str", "font_size": "int"},
    },
    "set_opacity": {
        "description": "Set element opacity (0.0 transparent → 1.0 opaque).",
        "params": {"id": "str", "opacity": "float 0.0–1.0"},
    },
    "remove_element": {
        "description": "Remove an element from the canvas.",
        "params": {"id": "str"},
    },
    "change_role": {
        "description": "Change the semantic role of an element.",
        "params": {"id": "str", "role": "str"},
    },
    "no_op": {
        "description": "Do nothing (useful for padding episodes).",
        "params": {},
    },
}


# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------

def build_server() -> "Server":
    server = Server("marketcanvas-env")

    # ------------------------------------------------------------------ #
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="get_canvas_state",
                description=(
                    "Return the complete current state of the design canvas as a "
                    "JSON DOM tree.  Each element node contains its id, type, role, "
                    "position (x/y), size, z-index, colours, content, and overlap info."
                ),
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            types.Tool(
                name="execute_action",
                description=(
                    "Apply one action to the canvas.  Provide action_type (string) "
                    "and params (object).  See list_actions for the full action catalogue."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "description": "The action to perform (e.g. 'add_element', 'move_element').",
                        },
                        "params": {
                            "type": "object",
                            "description": "Action-specific parameters.",
                        },
                    },
                    "required": ["action_type"],
                },
            ),
            types.Tool(
                name="get_current_reward",
                description=(
                    "Compute and return the reward breakdown for the current canvas "
                    "state against the active target spec.  Returns a dict with "
                    "sub-scores (element_presence, wcag_contrast, layout_alignment, "
                    "overlap_penalty, boundary_score, content_quality) and the final "
                    "normalised reward in [-1.0, 1.0]."
                ),
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            types.Tool(
                name="reset_environment",
                description=(
                    "Clear the canvas and reset the episode counter.  Optionally "
                    "supply a new target_prompt to change the design objective."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "target_prompt": {
                            "type": "string",
                            "description": "Optional new design brief.",
                        }
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="render_canvas",
                description=(
                    "Render the current canvas to a PNG file and return its absolute "
                    "path.  Optionally specify output_path; defaults to a temp file."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_path": {
                            "type": "string",
                            "description": "Optional file path for the PNG output.",
                        }
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="list_actions",
                description=(
                    "Return the full action catalogue: every valid action_type "
                    "with its description and parameter schema."
                ),
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
        ]

    # ------------------------------------------------------------------ #
    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[types.TextContent]:
        arguments = arguments or {}

        if name == "get_canvas_state":
            state_json = _env.get_semantic_state_json()
            return [types.TextContent(type="text", text=state_json)]

        elif name == "execute_action":
            action_type = arguments.get("action_type", "no_op")
            params      = arguments.get("params", {})
            result      = _env.apply_action(action_type, params)
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_current_reward":
            breakdown = _env.compute_reward()
            return [types.TextContent(type="text", text=json.dumps(breakdown, indent=2))]

        elif name == "reset_environment":
            prompt = arguments.get("target_prompt")
            opts   = {"target_prompt": prompt} if prompt else None
            _env.reset(options=opts)
            msg = {
                "success": True,
                "target_prompt": _env.target_prompt,
                "required_roles": _env.spec.required_roles,
            }
            return [types.TextContent(type="text", text=json.dumps(msg, indent=2))]

        elif name == "render_canvas":
            output_path = arguments.get("output_path")
            if not output_path:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False, prefix="marketcanvas_"
                )
                output_path = tmp.name
                tmp.close()
            try:
                saved_path = _env.save_png(output_path)
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"success": True, "path": saved_path}, indent=2),
                )]
            except ImportError as exc:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"success": False, "error": str(exc)}, indent=2),
                )]

        elif name == "list_actions":
            return [types.TextContent(
                type="text",
                text=json.dumps(ACTION_CATALOGUE, indent=2),
            )]

        else:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: '{name}'"}, indent=2),
            )]

    return server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    if not _MCP_AVAILABLE:
        sys.exit(1)

    import asyncio
    asyncio.run(main())
