"""
MarketCanvas-Env: A minimalist deterministic 2D design canvas RL environment
with MCP server integration for LLM-driven agent training.
"""

from .elements import CanvasElement, ElementType
from .canvas import Canvas
from .environment import MarketCanvasEnv
from .reward import RewardFunction, TargetSpec

__all__ = [
    "CanvasElement",
    "ElementType",
    "Canvas",
    "MarketCanvasEnv",
    "RewardFunction",
    "TargetSpec",
]
