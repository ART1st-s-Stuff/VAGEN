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

    # Generated from the exact source files used by eval job 479904.
    assert [_sha256(value) for value in outputs] == [
        "6d62fef7025b03b219c2af7e175eb964e23e7d206e81f7c3145f562659b67b26",
        "00bc2dd7e2c45f55ca440b14366958d9c8c7565635df2c970f7d540c0aca8dbb",
        "c75fbb3c756cd180064f8a5790ed4bceb61fe5b02999c5004a87293cbb4ac064",
        "337f1a4a0a67bae733e7eafe5e661804539a8f67d327938a81c67ecb33ce4d19",
        "a438d6d7a00638413f96fc1459870145b6be4eda3df770e2c1ddfd3fc949e6ff",
    ]


def test_hligb_single_action_mode_uses_source_response_parser() -> None:
    assert PARSE_FUNC_MAP[HLIGB_SINGLE_ACTION_MODE] is parse_grounding_worldmodeling
