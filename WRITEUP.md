# MarketCanvas-Env — Design Write-Up

## 1. State & Observation Space

### Choice: Semantic (JSON DOM) + Optional Pixel Array

The primary observation is a **JSON accessibility tree** — an ordered list of
element nodes, each carrying its full property set plus derived spatial
relationships (`overlaps_with`, `within_bounds`).  This mirrors the Document
Object Model browsers expose to screen-readers and, crucially, is the format
modern LLMs reason over most fluently.

**Why semantic-first?**

| Concern | Semantic JSON | Raw Pixels |
|---|---|---|
| LLM readability | Native | Requires vision encoder |
| RL policy gradient | Low-dim → fast convergence | High-dim → slow |
| Determinism | Exact floats | Anti-aliasing noise |
| Element identity | Element `id` preserved | Requires object detection |
| Data efficiency | O(n_elements) | O(W × H × 3) |

**Optional pixel array** (`render_mode="rgb_array"`) is included as a bonus
path for future multimodal VLM training.  The rendered PNG also serves as a
human-readable audit of what the agent produced.

The observation dict structure is:

```json
{
  "canvas": { "width": 800, "height": 600, "element_count": 8 },
  "elements": [
    {
      "id": "bg", "type": "shape", "role": "background",
      "x": 0, "y": 0, "width": 800, "height": 600,
      "z_index": 0, "color": "#1A237E", "text_color": "#FFFFFF",
      "content": "", "font_size": 16, "opacity": 1.0,
      "within_bounds": true, "overlaps_with": []
    },
    ...
  ]
}
```

---

## 2. Action Space

### Choice: High-Level Semantic API

Two action spaces were considered.  The trade-offs are:

| Dimension | Low-Level (computer use) | High-Level (semantic) |
|---|---|---|
| Expressiveness | Any pixel operation | Only defined intents |
| LLM alignment | Unnatural for LLMs | Natural tool-call API |
| Sample efficiency | Very low (millions of steps) | High (dozens of steps) |
| Transfer to real apps | High (mouse/keyboard = universal) | Low (app-specific schema) |
| Reward density | Extremely sparse | Moderate |

For this assignment the **high-level semantic space** was chosen because:

1. The task (design a banner) maps cleanly to a small vocabulary of intents.
2. LLMs using MCP tool-calling can act directly without wrapping a mouse.
3. Training convergence is tractable on a single machine.

The twelve action types are:

```
add_element, move_element, resize_element,
change_color, change_text_color, change_content,
set_z_index, set_font_size, set_opacity,
remove_element, change_role, no_op
```

A hybrid approach is possible: expose both, train a high-level policy for
curriculum warm-up, then fine-tune with low-level actions for pixel-accurate
positioning.

---

## 3. Reward Function

The reward is a scalar in **[-1.0, 1.0]** computed at episode end.
It is the weighted sum of five sub-scores:

### Sub-scores

| Component | Weight | What it measures |
|---|---|---|
| `element_presence` | +0.35 | Fraction of required semantic roles present |
| `wcag_contrast` | +0.25 | WCAG 2.1 AA pass rate across text elements |
| `layout_alignment` | +0.20 | Fraction of non-bg elements centred horizontally |
| `overlap_penalty` | −0.30 | Normalised overlapping area between foreground elements |
| `boundary_score` | +0.10 | Fraction of elements fully within canvas bounds |
| `content_quality` | +0.10 | Non-empty content on key roles (headline, CTA, body) |

**WCAG contrast** uses the official relative-luminance formula:

```
L  = 0.2126·R_lin + 0.7152·G_lin + 0.0722·B_lin
CR = (L_lighter + 0.05) / (L_darker + 0.05)
```

AA threshold: 4.5:1 for normal text, 3:1 for large text (≥ 18 pt).

**Overlap penalty** computes pairwise intersection areas for foreground
(non-background) elements; significant overlaps (> 10 % of the smaller
element's area) are accumulated and normalised by canvas area.

The raw score `[-0.30, 1.00]` is linearly mapped to `[-1.0, 1.0]`.

### Potential Reward-Hacking Loopholes

1. **Role stuffing at a single point.**  The agent can satisfy
   `element_presence` by adding all required roles as 1×1 elements at (0,0).
   The overlap penalty only fires if the overlap exceeds 10 % of the *smaller*
   element's area — infinitesimal elements barely penalise overlap.
   *Mitigation:* add a minimum-size constraint per role (e.g., headline ≥ 200 px
   wide) or a "coverage" score rewarding elements that fill a meaningful canvas
   fraction.

2. **Invisible text passes WCAG.**  An element with `text_color == color`
   passes both presence and WCAG checks because the contrast code reads the
   computed colour, not whether text is actually readable.
   *Mitigation:* assert `text_color ≠ color`, or compute contrast against the
   element's own fill even when they match.

3. **Centred overlap stack.**  Eight centred elements stacked on top of each
   other scores full alignment credit but only incurs overlap penalty for
   foreground pairs.  Centering is not penalised when elements obscure each
   other.
   *Mitigation:* weight alignment by unique non-overlapping visible area.

4. **Empty content.**  The agent can add a `headline` shape with `content=""`
   and still score `element_presence`.  `content_quality` only adds +0.10 for
   non-empty key elements, not fully compensating.
   *Mitigation:* make `content_quality` a hard gate on `element_presence`.

5. **Contrasting border vs. fill.**  The WCAG check compares `text_color`
   against the element's own fill colour, since text is rendered on its own
   box.  (An earlier version compared against the element *beneath* in
   z-order, which was exploitable by placing text atop a high-contrast
   background while using an illegible fill.)

6. **Role relabelling.**  The `change_role` action lets the agent relabel
   any existing element (e.g., turn a generic shape into a "headline") to
   satisfy `element_presence` without creating a proper element.
   *Mitigation:* gate `element_presence` on minimum-size or content
   requirements per role, or restrict `change_role` to a curated set of
   valid transitions.

---

## 4. Scaling to 10,000 Parallel PPO Rollouts with a VLM

### Anticipated Infrastructure Bottlenecks

**A. VLM inference throughput**

A typical 7B–70B VLM processes 1–20 tokens/s on a single GPU.  At 10 k
parallel rollouts, naive one-at-a-time inference would create a wall-clock
bottleneck of many hours per update step.

*Solution:* Deploy the VLM as a **batched inference service** (vLLM,
TensorRT-LLM) behind a load balancer.  Use continuous batching with a
request queue; rollout workers push observations and pull actions
asynchronously.  With 8× A100s and a 7B model, ~500 tok/s throughput is
achievable, but 10 k concurrent environments will still require careful
back-pressure control.

**B. Observation serialisation overhead**

Sending a JSON string per step per worker at 10 k scale means ~10 MB/s of
JSON throughput per second, plus potential Python pickle overhead when using
multiprocessing.

*Solution:* Switch to a compact binary format (MessagePack, Flatbuffers, or
Arrow IPC).  Maintain a shared-memory ring buffer using `multiprocessing.
shared_memory` or Ray Plasma store so workers do not copy observations.

**C. Environment CPU parallelism**

Python's GIL limits single-process parallelism.  The canvas engine is
CPU-bound (PIL rendering, reward computation).

*Solution:*
- Use `multiprocessing` or **Ray** actors for worker parallelism.
- Disable PIL rendering for non-visual rollouts; only compute pixels for the
  VLM's vision encoder (every Nth step, or on episode end).
- Vectorise the reward function with NumPy batch operations across all workers
  in one call.

**D. Reward function latency**

The WCAG contrast check and pairwise overlap computation are O(n²) in the
number of elements.  At 10 k rollouts × 30 steps × ~10 elements = 3 M
evaluations per update, this becomes significant.

*Solution:* Pre-compute a KD-tree or spatial grid per canvas state.  Cache
luminance values; recompute only on colour-change actions.  For PPO, defer
reward computation to episode end only (current design already does this).

**E. PPO update bottleneck**

Collecting 10 k rollouts × 30 steps = 300 k experiences per update then
running a PPO gradient step requires moving all observations to GPU.  Full
pixel arrays (800×600×3 × 300 k = ~400 GB) are infeasible.

*Solution:* Use **semantic-only observations** for the policy (low-dimension),
keeping pixel arrays only for the value head or for a separate VLM call.  A
two-tower architecture (semantic policy + visual critic) reduces bandwidth by
~1 000×.

### Proposed Redesign for Scale

```
┌─────────────────────────────────────────────────┐
│  10 k Canvas Env Workers  (Ray actors, no GIL)  │
│  semantic JSON obs  →  shared Arrow buffer       │
└────────────────────┬────────────────────────────┘
                     │ async batched obs
                     ▼
┌─────────────────────────────────────────────────┐
│  VLM Inference Cluster  (vLLM, 8× A100)         │
│  continuous batching, LoRA adapter for policy   │
└────────────────────┬────────────────────────────┘
                     │ action tokens  →  parsed JSON action dicts
                     ▼
┌─────────────────────────────────────────────────┐
│  PPO Trainer  (Ray RLlib or custom)             │
│  semantic-only policy  +  visual critic (VLM)  │
│  gradient checkpointing, mixed precision        │
└─────────────────────────────────────────────────┘
```

Key redesign decisions:
- **Async rollout collection** with Ray, decoupled from gradient updates.
- **Semantic-only policy network** (lightweight MLP or small transformer) for
  action selection; VLM invoked only for the value baseline.
- **Pixel rendering on demand** — render once per episode end for logging and
  reward verification; never during rollout.
- **Stateless environment workers** — episode state serialised as a dict;
  workers are restarted fresh each episode to prevent memory leaks.
- **Reward shaping at sub-step granularity** (delta reward per action) to
  reduce variance, rather than computing reward only at episode end.
