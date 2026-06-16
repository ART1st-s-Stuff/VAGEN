"""Shared Nimloth navigation prompt/action format (SFT1 v1).

Single source of truth for:
  - convert_sft1_rollouts_to_nimloth.py
  - train_sft1_qwen25vl.py
  - VAGEN prompt_format=nimloth eval/rollout
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

NIMLOTH_FORMAT_BODY = (
    "<think>...</think><|latent_state|>"
    "<|action_start|><|action_(idx)|><|action_end|>\n"
    "where idx is one of: 0=move_forward, 1=move_backward, 2=move_right, 3=move_left, "
    "4=turn_right, 5=turn_left, 6=look_up, 7=look_down."
)

NIMLOTH_FORMAT_INSTRUCTION = f"Respond in this format:\n{NIMLOTH_FORMAT_BODY}"

NIMLOTH_EVAL_FORMAT_INSTRUCTION = (
    "You can optionally think first, then give your action. " + NIMLOTH_FORMAT_INSTRUCTION
)
