"""Nimloth latent-query/action-token contract for navigation."""

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
ACTION_TOKEN_TO_NAME = {
    f"<|action_({index})|>": name for index, name in enumerate(ACTION_NAMES)
}


def latent_state_tokens(count: int) -> tuple[str, ...]:
    """Return the exact K-slot inject block used by the checkpoint."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("latent_token_count must be a positive int")
    if count == 1:
        return ("<|latent_state|>",)
    return (
        "<|latent_state|>",
        *(f"<|latent_state_{index}|>" for index in range(1, count)),
    )


def action_block(*, latent_token_count: int, action: str = "<|action_(idx)|>") -> str:
    return (
        "".join(latent_state_tokens(latent_token_count))
        + "<|action_start|>"
        + action
        + "<|action_end|>"
    )


__all__ = [
    "ACTION_NAMES",
    "ACTION_TOKEN_TO_NAME",
    "action_block",
    "latent_state_tokens",
]
