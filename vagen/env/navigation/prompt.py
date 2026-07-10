# Prompt templates for the legacy AI2-THOR navigation environment.
from vagen.env.navigation.nimloth_format import (
    NIMLOTH_ACTION_BLOCK,
    NIMLOTH_EVAL_FORMAT_INSTRUCTION,
    NIMLOTH_WM_FORMAT_INSTRUCTION,
)

SOURCE_EVAL_MODE = "source_eval_mode"


FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><answer>...</answer>",
        "example": """<think>I can see the target ahead-left, so I should move one step toward it.</think><answer>moveahead</answer>""",
    },
    "no_think": {
        "description": "You should provide only your answer.",
        "format": "<answer>...</answer>",
        "example": """<answer>moveahead</answer>""",
    },
    "eval_mode": {
        "description": "You can optionally think first, then give your action.",
        "format": "<think>...</think><action>...</action>",
        "example": """<think>I should move one step toward the target.</think><action>moveahead</action>""",
    },
    SOURCE_EVAL_MODE: {
        "description": "You can optionally think first, then give your action. Respond in this format:",
        "format": "<think>...</think><action>some_action</action>",
        "example": "",
    },
    "grounding": {
        "description": "You should first give your thought process with your observation and reasoning, and finally your answer.\nThe observation should be described in detail about what you see in the environment.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "example": """<think><observation>The target is ahead-left.</observation><reasoning>I should move one step toward it.</reasoning></think><answer>moveahead</answer>""",
    },
    "worldmodeling": {
        "description": "You should first give your thought process with reasoning and prediction of next state, then your answer.\nThe prediction should describe what you expect to see after your actions are executed.",
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": """<think><reasoning>I should turn right to inspect the room.</reasoning><prediction>I expect to see another part of the room.</prediction></think><answer>rotateright</answer>""",
    },
    "grounding_worldmodeling": {
        "description": "You should first give your thought process with your observation, reasoning, and prediction of next state, then your answer.\nBoth the observation and prediction should describe what you see or expect to see in the environment.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": """<think><observation>The target is not visible.</observation><reasoning>I should rotate to search.</reasoning><prediction>I expect to see a new view of the room.</prediction></think><answer>rotateright</answer>""",
    },
    "nimloth": {
        "description": NIMLOTH_EVAL_FORMAT_INSTRUCTION,
        "format": f"<think>...</think>{NIMLOTH_ACTION_BLOCK}",
        "example": f"<think>I should move one step toward the visible target.</think>{NIMLOTH_ACTION_BLOCK.replace('(idx)', '(0)')}",
    },
    "nimloth_wm": {
        "description": NIMLOTH_WM_FORMAT_INSTRUCTION,
        "format": f"<observation>...</observation><think>...</think>{NIMLOTH_ACTION_BLOCK}<prediction>...</prediction>",
        "example": f"<observation>The target is ahead-left.</observation><think>I should move forward one step and reassess.</think>{NIMLOTH_ACTION_BLOCK.replace('(idx)', '(0)')}<prediction>I expect to be closer to the target.</prediction>",
    },
}

_BASE_SYSTEM_PROMPT = """\
You are a home robot and perform navigation tasks according to instructions.
Actions you can take: moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown.
moveahead: Move forward by some distance
moveback: Move backward by some distance
moveright: Move rightward by some distance
moveleft: Move leftward by some distance
rotateright: Rotate to the right by 90 degrees
rotateleft: Rotate to the left by 90 degrees
lookup: Tilt the camera upward by 30 degrees
lookdown: Tilt the camera downward by 30 degrees
Rewards:
Format correct: +0.5
Achieve the human instruction: +10.0
The instruction will be provided in the first observation. Look at the image carefully and navigate to complete the instruction."""

_SINGLE_ACTION_HINTS = """\
Hints:
1. Choose exactly one valid action for the current step. Do not combine actions.
2. If the target object is far away, move toward it one step at a time across multiple turns.
3. If you seem to be stuck, use one action such as lookdown, rotateleft, or rotateright to inspect another view.
4. Output the action only inside the required action field for the selected format."""

_SOURCE_EVAL_BASE_SYSTEM_PROMPT = """\
You are a home robot and perform navigation tasks according to instructions.
Actions you can take: move_forward, move_backward, move_right, move_left, turn_right, turn_left, look_up, look_down.
move_forward: Move forward by some distance
move_backward: Move backward by some distance
move_right: Move rightward by some distance
move_left: Move leftward by some distance
turn_right: Rotate to the right by 90 degrees
turn_left: Rotate to the left by 90 degrees
look_up: Tilt the camera upward by 30 degrees
look_down: Tilt the camera downward by 30 degrees
The instruction will be provided in the first observation. Look at the image carefully and navigate to complete the instruction."""

_SOURCE_EVAL_SINGLE_ACTION_HINTS = """\
Hints:
1. Choose exactly one valid action for the current step. Do not combine actions.
2. If the target object is far away, move toward it one step at a time across multiple turns.
3. If you seem to be stuck, use one action such as look_down, turn_left, or turn_right to inspect another view.
4. Output the action only inside the required <action>...</action> XML tag."""

_NIMLOTH_SINGLE_ACTION_HINTS = """\
Hints:
1. Choose exactly one valid action for the current step. Do not combine actions.
2. If the target object is far away, move toward it one step at a time across multiple turns.
3. If you seem to be stuck, use one action such as lookdown, rotateleft, or rotateright to inspect another view.
4. Output exactly one action index token between <|action_start|> and <|action_end|>."""

_MULTI_ACTION_HINTS = """\
Hints:
1. You can take multiple actions at a time. If the target object is far away, you may call actions such as moveahead or moveleft multiple times.
2. If you seem to be stuck, you can lookdown to see if there is any object above or below you, and you can rotate to inspect another view."""


def system_prompt(**kwargs):
    selected_format = kwargs.get("format", kwargs.get("format_name", "default"))
    max_actions_per_step = kwargs.get("max_actions_per_step", 5)

    if selected_format == SOURCE_EVAL_MODE:
        return "\n\n".join([_SOURCE_EVAL_BASE_SYSTEM_PROMPT, _SOURCE_EVAL_SINGLE_ACTION_HINTS])

    parts = [_BASE_SYSTEM_PROMPT]
    if selected_format in ("nimloth", "nimloth_wm"):
        parts.append(_NIMLOTH_SINGLE_ACTION_HINTS)
    else:
        parts.append(_SINGLE_ACTION_HINTS if max_actions_per_step <= 1 else _MULTI_ACTION_HINTS)
    return "\n\n".join(parts)


def init_observation_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    instruction = kwargs.get("instruction", "No instruction provided.")
    suffix = "Decide your next action(s)." if kwargs.get("format") == SOURCE_EVAL_MODE else "Decide your next action."
    return f"""[Initial Observation]:
{observation}
Human Instruction: {instruction}
{suffix}"""


def action_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    valid_action = kwargs.get("valid_action", "No valid action provided.")
    env_feedback = kwargs.get("env_feedback", "No environment feedback provided.")
    reward = kwargs.get("reward", "No reward provided.")
    done = kwargs.get("done", "No done status provided.")
    if kwargs.get("format") == SOURCE_EVAL_MODE:
        return f"""After your action, the extracted valid action is {valid_action}.
The environment feedback is: {env_feedback}
reward: {reward}
done: {done}
After that, the observation is:
{observation}
Decide your next action(s)."""
    return f"""After your answer, the extracted valid action is {valid_action}.
The environment feedback is: {env_feedback}
reward: {reward}
done: {done}
After that, the observation is:
{observation}
Decide your next action."""


def format_prompt_generator(format_type):
    """Generate a per-format instruction for the navigation task."""

    def prompt_function(**kwargs):
        max_actions_per_step = kwargs.get("max_actions_per_step", 5)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", True)

        if format_type in ("nimloth", "nimloth_wm") and max_actions_per_step > 1:
            raise ValueError(f"prompt_format={format_type} only supports max_actions_per_step=1")

        if format_type not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format_type: {format_type}")
        config = FORMAT_CONFIGS[format_type]

        if format_type == SOURCE_EVAL_MODE:
            if max_actions_per_step <= 1:
                action_count_instruction = "You must take exactly one action in each response. Do not output multiple actions and do not use '|'."
                action_example = "some_action"
            else:
                action_count_instruction = f"You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'."
                action_example = f"action1{action_sep} action2{action_sep} ..."
            return (
                f"{action_count_instruction}\n"
                "You can optionally think first, then give your action. Respond in this format:\n"
                f"<think>...</think><action>{action_example}</action>"
            )

        if max_actions_per_step <= 1:
            action_count_instruction = "You must take exactly one action in each response. Do not output multiple actions and do not use '|'."
        else:
            action_count_instruction = f"You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'."

        base_prompt = f"""{action_count_instruction}
{config["description"]}
Your response should be in the format of:
{config["format"]}"""

        if add_example:
            example_text = config["example"].format(action_sep=action_sep)
            return base_prompt + "\n" + f"e.g. {example_text}"

        return base_prompt

    return prompt_function


format_prompt = {ft: format_prompt_generator(ft) for ft in FORMAT_CONFIGS}
# Nimloth baseline config uses the short alias `wm` for world modeling.
format_prompt["wm"] = format_prompt["worldmodeling"]


if __name__ == "__main__":
    max_actions_per_step = 1
    action_sep = ","

    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(system_prompt(format=key, max_actions_per_step=max_actions_per_step))
        print(func(max_actions_per_step=max_actions_per_step, action_sep=action_sep))
        print("\n" + "=" * 50 + "\n")
