# Prompt templates for the legacy AI2-THOR navigation environment.
from vagen.env.navigation.nimloth_format import (
    NIMLOTH_EVAL_FORMAT_INSTRUCTION,
    NIMLOTH_WM_FORMAT_INSTRUCTION,
)

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
    "eval_mode": {
        "description": "You can optionally think first, then give your action.",
        "format": "<think>...</think><action>...</action>",
        "example": """<think>I should move one step toward the target.</think><action>moveahead</action>"""
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
        "format": "<think>...</think><|latent_state|><|action_start|><|action_(idx)|><|action_end|>",
        "example": """<think>I should move one step toward the visible target.</think><|latent_state|><|action_start|><|action_(0)|><|action_end|>""",
    },
    "nimloth_wm": {
        "description": NIMLOTH_WM_FORMAT_INSTRUCTION,
        "format": "<observation>...</observation><think>...</think><|latent_state|><|action_start|><|action_(idx)|><|action_end|><prediction>...</prediction>",
        "example": """<observation>The target is ahead-left.</observation><think>I should move forward one step and reassess.</think><|latent_state|><|action_start|><|action_(0)|><|action_end|><prediction>I expect to be closer to the target.</prediction>""",
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

_NIMLOTH_SINGLE_ACTION_HINTS = """\
Hints:
1. Choose exactly one valid action for the current step. Do not combine actions.
2. If the target object is far away, move toward it one step at a time across multiple turns.
3. If you seem to be stuck, use one action such as lookdown, rotateleft, or rotateright to inspect another view.
4. Output exactly one action index token between <|action_start|> and <|action_end|>."""

_NIMLOTH_MULTI_ACTION_HINTS = """\
Hints:
1. You can take multiple actions at a time. If the target object is far away, you may output repeated action index tokens such as move_forward multiple times.
2. Output 1 to the configured maximum number of action index tokens between <|action_start|> and <|action_end|>; the actions will be executed in order.
3. If you seem to be stuck, use actions such as look_down, turn_left, or turn_right to inspect another view."""

_MULTI_ACTION_HINTS = """\
Hints:
1. You can take multiple actions at a time. If the target object is far away, you may call actions such as moveahead or moveleft multiple times.
2. If you seem to be stuck, you can lookdown to see if there is any object above or below you, and you can rotate to inspect another view."""


def system_prompt(**kwargs):
    selected_format = kwargs.get("format", kwargs.get("format_name", "default"))
    max_actions_per_step = kwargs.get("max_actions_per_step", 5)

    parts = [_BASE_SYSTEM_PROMPT]
    if selected_format in ("nimloth", "nimloth_wm"):
        parts.append(_NIMLOTH_SINGLE_ACTION_HINTS if max_actions_per_step <= 1 else _NIMLOTH_MULTI_ACTION_HINTS)
    else:
        parts.append(_SINGLE_ACTION_HINTS if max_actions_per_step <= 1 else _MULTI_ACTION_HINTS)
    return "\n\n".join(parts)


def init_observation_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    instruction = kwargs.get("instruction", "No instruction provided.")
    return f"""[Initial Observation]:
{observation}
Human Instruction: {instruction}
Decide your next action."""


def action_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    valid_action = kwargs.get("valid_action", "No valid action provided.")
    env_feedback = kwargs.get("env_feedback", "No environment feedback provided.")
    reward = kwargs.get("reward", "No reward provided.")
    done = kwargs.get("done", "No done status provided.")
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

        if format_type not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format_type: {format_type}")
        config = FORMAT_CONFIGS[format_type]
        response_format = config["format"]
        description = config["description"]

        if format_type in ("nimloth", "nimloth_wm"):
            if max_actions_per_step <= 1:
                action_count_instruction = "You must take exactly one action in each response. Output exactly one action index token between <|action_start|> and <|action_end|>."
            else:
                action_count_instruction = f"You can take up to {max_actions_per_step} action(s) at a time. Output 1 to {max_actions_per_step} action index token(s) between <|action_start|> and <|action_end|>; no separator is required."
                if format_type == "nimloth":
                    description = "You can optionally think first, then give your ordered action token(s)."
                    response_format = "<think>...</think><|latent_state|><|action_start|><|action_(idx)|>[<|action_(idx)|>...]<|action_end|>"
                else:
                    description = "You need to describe your observation, think, give your ordered action token(s), then predict what you will see next."
                    response_format = "<observation>...</observation><think>...</think><|latent_state|><|action_start|><|action_(idx)|>[<|action_(idx)|>...]<|action_end|><prediction>...</prediction>"
        elif max_actions_per_step <= 1:
            action_count_instruction = "You must take exactly one action in each response. Do not output multiple actions and do not use '|'."
        else:
            action_count_instruction = f"You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'."

        base_prompt = f"""{action_count_instruction}
{description}
Your response should be in the format of:
{response_format}"""

        if add_example:
            if max_actions_per_step > 1 and format_type == "nimloth":
                example_text = "<think>I should move forward twice and then reassess.</think><|latent_state|><|action_start|><|action_(0)|><|action_(0)|><|action_end|>"
            elif max_actions_per_step > 1 and format_type == "nimloth_wm":
                example_text = "<observation>The target is ahead.</observation><think>I should move forward twice.</think><|latent_state|><|action_start|><|action_(0)|><|action_(0)|><|action_end|><prediction>I expect to be closer to the target.</prediction>"
            else:
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
