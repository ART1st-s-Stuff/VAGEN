"""
Response parsing and reward utilities for the navigation environment.

Formats:
  - free_think: <think>...</think><action>...</action>
  - wm: <observation>...</observation><think>...</think><action>...</action><prediction>...</prediction>
  - no_think: <action>...</action>
  - eval_mode: only requires <action>...</action> (everything else optional, lenient)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .nimloth_format import ACTION_TOKEN_TO_NAME, action_block


# ---------------------------------------------------------------------------
# Parse patterns
# ---------------------------------------------------------------------------

_PARSE_PATTERNS = {
    "free_think": r"<think>(.*?)</think>\s*<action>(.*?)</action>",
    "wm": (
        r"<observation>(.*?)</observation>\s*"
        r"<think>(.*?)</think>\s*"
        r"<action>(.*?)</action>\s*"
        r"<prediction>(.*?)</prediction>"
    ),
    "no_think": r"^\s*<action>(.*?)</action>\s*$",  # strict: entire response must be <action>...</action>
    "eval_mode": r"<action>(.*?)</action>",  # lenient: first <action> anywhere
}

_NIMLOTH_ACTION_TOKEN_RE = re.compile(r"<\|action_\([^|>]*\)\|>")
_NIMLOTH_LATENT_TOKEN_RE = re.compile(r"<\|latent_state(?:_[^|>]*)?\|>")
_NIMLOTH_ENVELOPE_RE = re.compile(r"^\s*<think>(.*?)</think>(.*?)\s*$", re.DOTALL)


def parse_response(
    response: str,
    prompt_format: str = "free_think",
    action_sep: str = "|",
    max_actions: int = 5,
    latent_token_count: int = 1,
) -> Dict[str, Any]:
    """Parse an LLM response and extract actions.

    Returns dict with keys:
        llm_raw_response, actions, format_correct,
        and optional think/observation/prediction text.
    """
    result: Dict[str, Any] = {"llm_raw_response": response, "actions": [], "format_correct": False}

    if prompt_format == "nimloth":
        return _parse_nimloth_response(
            response,
            max_actions=max_actions,
            latent_token_count=latent_token_count,
        )

    pattern = _PARSE_PATTERNS.get(prompt_format)
    if pattern is None:
        raise ValueError(f"Unknown prompt_format: {prompt_format}")
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return result

    # Extract named sections based on format
    if prompt_format == "free_think":
        result["think"] = match.group(1).strip()
        action_text = match.group(2).strip()
    elif prompt_format == "wm":
        result["observation"] = match.group(1).strip()
        result["think"] = match.group(2).strip()
        action_text = match.group(3).strip()
        result["prediction"] = match.group(4).strip()
    else:
        # no_think / eval — single capture group
        action_text = match.group(1).strip()

    result["format_correct"] = True
    actions = [a.strip().lower() for a in action_text.split(action_sep) if a.strip()][:max_actions]
    result["actions"] = actions
    return result


def _parse_nimloth_response(
    response: str,
    *,
    max_actions: int,
    latent_token_count: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "llm_raw_response": response,
        "actions": [],
        "format_correct": False,
    }
    if max_actions != 1:
        raise ValueError("prompt_format=nimloth requires exactly one action")
    envelope = _NIMLOTH_ENVELOPE_RE.fullmatch(response)
    if envelope is None or not envelope.group(1).strip():
        return result
    response_tail = envelope.group(2).strip()
    known_tokens = list(ACTION_TOKEN_TO_NAME)
    found_tokens = _NIMLOTH_ACTION_TOKEN_RE.findall(response_tail)
    if len(found_tokens) != 1 or found_tokens[0] not in ACTION_TOKEN_TO_NAME:
        return result
    expected_block = action_block(
        latent_token_count=latent_token_count,
        action=found_tokens[0],
    )
    if response_tail != expected_block:
        return result
    if len(_NIMLOTH_LATENT_TOKEN_RE.findall(response_tail)) != latent_token_count:
        return result
    result["actions"] = [ACTION_TOKEN_TO_NAME[found_tokens[0]]]
    result["format_correct"] = True
    return result


def compute_reward(
    parsed: Dict[str, Any],
    valid_actions: List[str],
    success: bool,
    format_reward: float = 0.5,
    per_turn_format_reward: float = 0.0,
    success_reward: float = 10.0,
    is_format_correct_so_far: bool = True,
) -> float:
    """Compute step reward following ViewSuite pattern.

    - per_turn_format_reward: given every step if format is correct this turn
    - format_reward: given at episode end only if ALL turns had correct format
    - success_reward: given when goal is reached
    """
    reward = 0.0

    # Per-turn format bonus (given each step if this turn's format is correct)
    if parsed["format_correct"] and valid_actions:
        reward += per_turn_format_reward

    # Success bonus
    if success:
        reward += success_reward
        # End-of-episode format bonus (only if all turns were correct)
        if is_format_correct_so_far and parsed["format_correct"]:
            reward += format_reward

    return reward
