from vagen.env.navigation.env_config import NavigationEnvConfig
from vagen.env.navigation.prompt import (
    SOURCE_EVAL_MODE,
    action_template,
    format_prompt,
    init_observation_template,
    system_prompt,
)
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP


EXPECTED_SYSTEM = """You are a home robot and perform navigation tasks according to instructions.
Actions you can take: move_forward, move_backward, move_right, move_left, turn_right, turn_left, look_up, look_down.
move_forward: Move forward by some distance
move_backward: Move backward by some distance
move_right: Move rightward by some distance
move_left: Move leftward by some distance
turn_right: Rotate to the right by 90 degrees
turn_left: Rotate to the left by 90 degrees
look_up: Tilt the camera upward by 30 degrees
look_down: Tilt the camera downward by 30 degrees
The instruction will be provided in the first observation. Look at the image carefully and navigate to complete the instruction.

Hints:
1. Choose exactly one valid action for the current step. Do not combine actions.
2. If the target object is far away, move toward it one step at a time across multiple turns.
3. If you seem to be stuck, use one action such as look_down, turn_left, or turn_right to inspect another view.
4. Output the action only inside the required <action>...</action> XML tag.

You must take exactly one action in each response. Do not output multiple actions and do not use '|'.
You can optionally think first, then give your action. Respond in this format:
<think>...</think><action>some_action</action>"""


def test_source_eval_prompt_matches_step60_transcript_format():
    actual = system_prompt(format=SOURCE_EVAL_MODE, max_actions_per_step=1, action_sep="|")
    actual += "\n\n" + format_prompt[SOURCE_EVAL_MODE](
        max_actions_per_step=1,
        action_sep="|",
        add_example=True,
    )
    assert actual == EXPECTED_SYSTEM

    assert init_observation_template(
        observation="<image>", instruction="navigate", format=SOURCE_EVAL_MODE
    ) == (
        "[Initial Observation]:\n<image>\nHuman Instruction: navigate\n"
        "Decide your next action(s)."
    )
    assert action_template(
        valid_action=["move_forward"],
        observation="<image>",
        reward=0.01,
        done=False,
        env_feedback="Last action is executed successfully.",
        format=SOURCE_EVAL_MODE,
    ) == (
        "After your action, the extracted valid action is ['move_forward'].\n"
        "The environment feedback is: Last action is executed successfully.\n"
        "reward: 0.01\n"
        "done: False\n"
        "After that, the observation is:\n<image>\n"
        "Decide your next action(s)."
    )


def test_source_eval_parser_and_explicit_environment_values():
    parsed = PARSE_FUNC_MAP[SOURCE_EVAL_MODE](
        "<think>inspect</think><action>TURN_LEFT</action>",
        action_sep="|",
        max_actions=1,
    )
    assert parsed["format_correct"] is True
    assert parsed["actions"] == ["turn_left"]

    cfg = NavigationEnvConfig(
        prompt_format=SOURCE_EVAL_MODE,
        max_actions_per_step=1,
        action_sep="|",
        example_count=0,
        step_length=0.3,
        success_threshold=1.0,
        format_reward=0.0,
        per_turn_format_reward=0.01,
        success_reward=1.0,
    )
    assert cfg.step_length == 0.3
    assert cfg.success_threshold == 1.0
    assert cfg.format_reward == 0.0
    assert cfg.per_turn_format_reward == 0.01
    assert cfg.success_reward == 1.0
