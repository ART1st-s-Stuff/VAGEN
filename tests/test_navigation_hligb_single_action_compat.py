"""Golden compatibility checks for the hligb step10 single-action policy."""

import hashlib

from vagen.env.navigation.prompt import (
    HLIGB_SINGLE_ACTION_MODE,
    action_template,
    format_prompt,
    init_observation_template,
    system_prompt,
)
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP, parse_grounding_worldmodeling


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_hligb_single_action_prompt_matches_source_eval_golden() -> None:
    common = {
        "max_actions_per_step": 1,
        "action_sep": ",",
    }
    outputs = [
        system_prompt(format=HLIGB_SINGLE_ACTION_MODE, **common),
        format_prompt[HLIGB_SINGLE_ACTION_MODE](add_example=True, **common),
        format_prompt[HLIGB_SINGLE_ACTION_MODE](add_example=False, **common),
        init_observation_template(
            observation="<image>",
            instruction="Go to the vase.",
            format=HLIGB_SINGLE_ACTION_MODE,
        ),
        action_template(
            valid_action=["moveahead"],
            observation="<image>",
            reward=0.5,
            done=False,
            instruction="Go to the vase.",
            env_feedback="Last action is executed successfully.",
            format=HLIGB_SINGLE_ACTION_MODE,
        ),
    ]

    # Generated from committed VAGEN HEAD 8839a2a, which env job 479522
    # loaded before the later explicit-one-action working-tree edit.
    assert [_sha256(value) for value in outputs] == [
        "ff5846aed6b8ab6ec60d1b18d943fa4f0a9dc9190d7bcba623f5122e3aadeff6",
        "56ba893d1bd00065f062fd75c3074e7d02f85dea6fe7cbd95be82b0e2398624e",
        "9cf7c11b5f3723f963c31776c4f08b61557661a687e647a6a8a96e15e6d86a03",
        "337f1a4a0a67bae733e7eafe5e661804539a8f67d327938a81c67ecb33ce4d19",
        "a438d6d7a00638413f96fc1459870145b6be4eda3df770e2c1ddfd3fc949e6ff",
    ]
    rendered_system = outputs[0] + "\n" + outputs[1]
    assert _sha256(rendered_system) == "ee38bc0c257422734b55e3b301b3c767f95ad76a74e682c35705a6c1d37c900f"
    assert "You can take multiple actions at a time" in rendered_system
    assert "You can take up to 1 action(s)" in rendered_system
    assert "Choose exactly one action" not in rendered_system


def test_hligb_single_action_mode_uses_source_response_parser() -> None:
    assert PARSE_FUNC_MAP[HLIGB_SINGLE_ACTION_MODE] is parse_grounding_worldmodeling
