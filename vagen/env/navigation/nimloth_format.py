"""Shared Nimloth navigation prompt/action format for VAGEN legacy.

This mirrors the Nimloth format used by the newer VAGEN navigation stack while
keeping the legacy environment import layout (``vagen.env.navigation``).
"""

from __future__ import annotations

ACTION_NAMES = (
    "move_forward",
    "move_backward",
    "move_right",
    "move_left",
    "turn_right",
    "turn_left",
    "look_up",
    "look_down",
)

ACTION_TO_IDX = {name: idx for idx, name in enumerate(ACTION_NAMES)}
IDX_TO_ACTION = {idx: name for name, idx in ACTION_TO_IDX.items()}
ACTION_TOKEN = {name: f"<|action_({idx})|>" for name, idx in ACTION_TO_IDX.items()}

SPECIAL_TOKENS = ["<|latent_state|>", "<|action_start|>", "<|action_end|>"] + [
    ACTION_TOKEN[name] for name in ACTION_NAMES
]

_ACTION_IDX_LEGEND = (
    "where idx is one of: 0=move_forward, 1=move_backward, 2=move_right, 3=move_left, "
    "4=turn_right, 5=turn_left, 6=look_up, 7=look_down."
)

NIMLOTH_ACTION_BLOCK = "<|latent_state|><|action_start|><|action_(idx)|><|action_end|>"

NIMLOTH_FORMAT_BODY = (
    "<think>...</think>"
    f"{NIMLOTH_ACTION_BLOCK}\n"
    f"{_ACTION_IDX_LEGEND}"
)

NIMLOTH_FORMAT_INSTRUCTION = f"Respond in this format:\n{NIMLOTH_FORMAT_BODY}"

NIMLOTH_EVAL_FORMAT_INSTRUCTION = (
    "You can optionally think first, then give your action. " + NIMLOTH_FORMAT_INSTRUCTION
)

NIMLOTH_WM_FORMAT_BODY = (
    "<observation>...</observation><think>...</think>"
    f"{NIMLOTH_ACTION_BLOCK}"
    "<prediction>...</prediction>\n"
    f"{_ACTION_IDX_LEGEND}"
)

NIMLOTH_WM_FORMAT_INSTRUCTION = (
    "You need to describe your observation, think, give your action, then predict "
    f"what you will see next. Respond in this format:\n{NIMLOTH_WM_FORMAT_BODY}"
)
