"""
MarketCanvasEnv — a Gymnasium-compatible RL environment.

State / Observation
-------------------
The observation is a dict with two keys:
  "semantic"  : JSON string (UTF-8) of the canvas DOM tree
  "pixels"    : uint8 RGB array of shape (H, W, 3)  (if render_mode="rgb_array")
                or None otherwise

Action Space (High-Level Semantic)
-----------------------------------
Actions are passed as Python dicts (not a fixed Gymnasium Box/Discrete space)
because the set of valid element IDs is dynamic.  The agent selects an
action_type plus a params dict:

  {"action_type": "add_element",     "params": { type, role, x, y, w, h, color, ... }}
  {"action_type": "move_element",    "params": { id, x, y }}
  {"action_type": "resize_element",  "params": { id, width, height }}
  {"action_type": "change_color",    "params": { id, color }}
  {"action_type": "change_text_color","params": { id, color }}
  {"action_type": "change_content",  "params": { id, content }}
  {"action_type": "set_z_index",     "params": { id, z_index }}
  {"action_type": "remove_element",  "params": { id }}
  {"action_type": "change_role",     "params": { id, role }}
  {"action_type": "no_op",           "params": {} }

Gymnasium compliance
--------------------
  env = MarketCanvasEnv(target_prompt="...", max_steps=20)
  obs, info = env.reset()
  obs, reward, terminated, truncated, info = env.step(action_dict)

Reward is 0.0 at every intermediate step; the terminal reward is computed
by RewardFunction at the end of the episode.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

from .canvas import Canvas, CanvasError
from .reward import RewardFunction, TargetSpec
from .renderer import CanvasRenderer

# ---------------------------------------------------------------------------
# Action type registry
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = frozenset([
    "add_element",
    "move_element",
    "resize_element",
    "change_color",
    "change_text_color",
    "change_content",
    "set_z_index",
    "set_font_size",
    "set_opacity",
    "remove_element",
    "change_role",
    "no_op",
])


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MarketCanvasEnv:
    """
    Gymnasium-style environment wrapping the 2D canvas.

    Parameters
    ----------
    target_prompt : str
        Free-text design brief.  Parsed by TargetSpec.from_prompt().
    target_spec : TargetSpec, optional
        Explicit structured spec.  If provided, *target_prompt* is ignored.
    canvas_width, canvas_height : int
        Canvas resolution.
    max_steps : int
        Maximum number of actions per episode before truncation.
    render_mode : str or None
        "rgb_array" to include pixel observations; None for semantic-only.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        target_prompt: str = "Create a Summer Sale email banner",
        target_spec: Optional[TargetSpec] = None,
        canvas_width: int = 800,
        canvas_height: int = 600,
        max_steps: int = 30,
        render_mode: Optional[str] = None,
    ) -> None:
        self.canvas_width  = canvas_width
        self.canvas_height = canvas_height
        self.max_steps     = max_steps
        self.render_mode   = render_mode

        self.target_prompt = target_prompt
        self.spec = target_spec or TargetSpec.from_prompt(
            target_prompt, canvas_width, canvas_height
        )

        self._canvas   = Canvas(canvas_width, canvas_height)
        self._reward_fn = RewardFunction()
        self._renderer  = None   # lazy-init when needed
        self._step_count = 0
        self._last_reward_breakdown = None

        # Gymnasium spaces (optional; used when gymnasium is installed)
        if _GYM_AVAILABLE and _NUMPY_AVAILABLE:
            self.observation_space = spaces.Dict({
                "semantic": spaces.Text(max_length=65536),
                "pixels": spaces.Box(
                    low=0, high=255,
                    shape=(canvas_height, canvas_width, 3),
                    dtype="uint8",
                ) if render_mode == "rgb_array" else spaces.Discrete(1),
            })
            self.action_space = spaces.Text(max_length=4096)   # JSON-encoded action

    # ------------------------------------------------------------------ #
    # Core Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """Reset the environment to an empty canvas."""
        self._canvas = Canvas(self.canvas_width, self.canvas_height)
        self._step_count = 0
        self._last_reward_breakdown = None

        if options and "target_prompt" in options:
            self.target_prompt = options["target_prompt"]
            self.spec = TargetSpec.from_prompt(
                self.target_prompt, self.canvas_width, self.canvas_height
            )

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: dict) -> Tuple[dict, float, bool, bool, dict]:
        """Apply one action and return (obs, reward, terminated, truncated, info).

        *action* should be a dict:
            {"action_type": str, "params": dict}

        At every non-terminal step reward = 0.0.
        At the terminal step (done or truncated) the full episode reward is
        returned.
        """
        self._step_count += 1
        error_msg = None

        try:
            action_type = action.get("action_type", "no_op")
            params      = action.get("params", {})
            self._apply_action(action_type, params)
        except (CanvasError, ValueError, KeyError, TypeError) as exc:
            error_msg = str(exc)

        terminated = False
        truncated  = self._step_count >= self.max_steps
        done       = terminated or truncated

        reward = 0.0
        if done:
            breakdown = self._reward_fn(self._canvas, self.spec)
            self._last_reward_breakdown = breakdown
            reward = breakdown.normalised

        obs  = self._get_obs()
        info = self._get_info()
        if error_msg:
            info["action_error"] = error_msg

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional["np.ndarray"]:
        """Return an RGB pixel array or None."""
        if self.render_mode == "rgb_array":
            renderer = self._get_renderer()
            return renderer.to_numpy(self._canvas)
        return None

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Direct canvas / reward access (used by MCP server and demo)
    # ------------------------------------------------------------------ #

    @property
    def canvas(self) -> Canvas:
        return self._canvas

    def get_semantic_state(self) -> dict:
        return self._canvas.to_semantic_state()

    def get_semantic_state_json(self) -> str:
        return self._canvas.to_json()

    def compute_reward(self) -> dict:
        """Compute reward for the current canvas state without ending the episode."""
        bd = self._reward_fn(self._canvas, self.spec)
        self._last_reward_breakdown = bd
        return bd.to_dict()

    def save_png(self, path: str) -> str:
        renderer = self._get_renderer()
        return renderer.save(self._canvas, path)

    def apply_action(self, action_type: str, params: dict) -> dict:
        """Public action dispatch (also used by MCP server)."""
        if action_type not in VALID_ACTION_TYPES:
            return {"success": False, "error": f"Unknown action_type '{action_type}'"}
        try:
            result = self._apply_action(action_type, params)
            return {"success": True, "result": result}
        except (CanvasError, ValueError, KeyError, TypeError) as exc:
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Action dispatch
    # ------------------------------------------------------------------ #

    def _apply_action(self, action_type: str, params: dict) -> Any:
        c = self._canvas

        if action_type == "no_op":
            return None

        elif action_type == "add_element":
            el = c.add_element(
                type=params.get("type", "shape"),
                role=params.get("role", "generic"),
                x=float(params.get("x", 0)),
                y=float(params.get("y", 0)),
                width=float(params.get("width", 100)),
                height=float(params.get("height", 50)),
                z_index=int(params.get("z_index", 0)),
                color=params.get("color", "#FFFFFF"),
                text_color=params.get("text_color", "#000000"),
                content=params.get("content", ""),
                font_size=int(params.get("font_size", 16)),
                border_color=params.get("border_color", ""),
                border_width=int(params.get("border_width", 0)),
                opacity=float(params.get("opacity", 1.0)),
                corner_radius=int(params.get("corner_radius", 0)),
                element_id=params.get("id"),
            )
            return el.to_dict()

        elif action_type == "move_element":
            el = c.move_element(params["id"], float(params["x"]), float(params["y"]))
            return el.to_dict()

        elif action_type == "resize_element":
            el = c.resize_element(
                params["id"], float(params["width"]), float(params["height"])
            )
            return el.to_dict()

        elif action_type == "change_color":
            el = c.change_color(params["id"], params["color"])
            return el.to_dict()

        elif action_type == "change_text_color":
            el = c.change_text_color(params["id"], params["color"])
            return el.to_dict()

        elif action_type == "change_content":
            el = c.change_content(params["id"], params["content"])
            return el.to_dict()

        elif action_type == "set_z_index":
            el = c.set_z_index(params["id"], int(params["z_index"]))
            return el.to_dict()

        elif action_type == "set_font_size":
            el = c.set_font_size(params["id"], int(params["font_size"]))
            return el.to_dict()

        elif action_type == "set_opacity":
            el = c.set_opacity(params["id"], float(params["opacity"]))
            return el.to_dict()

        elif action_type == "remove_element":
            c.remove_element(params["id"])
            return {"removed": params["id"]}

        elif action_type == "change_role":
            el = c.change_role(params["id"], params["role"])
            return el.to_dict()

        else:
            raise ValueError(f"Unknown action_type: '{action_type}'")

    # ------------------------------------------------------------------ #
    # Observation / info helpers
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> dict:
        obs: dict = {"semantic": self.get_semantic_state_json()}
        if self.render_mode == "rgb_array":
            renderer = self._get_renderer()
            obs["pixels"] = renderer.to_numpy(self._canvas)
        return obs

    def _get_info(self) -> dict:
        info: dict = {
            "step": self._step_count,
            "element_count": len(self._canvas.elements),
            "target_prompt": self.target_prompt,
            "required_roles": self.spec.required_roles,
        }
        if self._last_reward_breakdown is not None:
            info["reward_breakdown"] = self._last_reward_breakdown.to_dict()
        return info

    def _get_renderer(self) -> CanvasRenderer:
        if self._renderer is None:
            self._renderer = CanvasRenderer()
        return self._renderer
