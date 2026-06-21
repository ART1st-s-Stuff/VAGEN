import re
from typing import Dict, List
import json

from vagen.env.navigation.nimloth_format import IDX_TO_ACTION


_NIMLOTH_ACTION_RE = re.compile(r"<\|action_\((\d+)\)\|>")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_OBS_RE = re.compile(r"<observation>(.*?)</observation>", re.DOTALL)
_PRED_RE = re.compile(r"<prediction>(.*?)</prediction>", re.DOTALL)
_LATENT_STATE_TOKEN = "<|latent_state|>"
_ACTION_START_TOKEN = "<|action_start|>"
_ACTION_END_TOKEN = "<|action_end|>"


def _extract_nimloth_actions(block: str, max_actions: int) -> List[str]:
    indices = [int(m.group(1)) for m in _NIMLOTH_ACTION_RE.finditer(block)]
    if not indices:
        return []

    actions: List[str] = []
    for idx in indices[:max_actions]:
        action = IDX_TO_ACTION.get(idx)
        if action is None:
            return []
        actions.append(action)
    return actions


def parse_nimloth(response: str, special_token_list=None, action_sep=',', max_actions=1) -> Dict:
    """Parse Nimloth latent/action-token navigation responses.

    Expected action block:
    ``<|latent_state|><|action_start|><|action_(idx)|><|action_end|>``.
    ``action_sep`` and ``special_token_list`` are accepted for API compatibility
    with the legacy parser map.
    """
    response = response.replace("<image>", "")
    result = {
        "llm_raw_response": response,
        "llm_response": "",
        "think_content": "",
        "action_content": "",
        "actions": [],
        "format_correct": False,
    }

    think_match = _THINK_RE.search(response)
    if think_match:
        result["think_content"] = think_match.group(1).strip()

    action_start_idx = response.find(_ACTION_START_TOKEN)
    action_end_idx = response.find(_ACTION_END_TOKEN)
    if action_start_idx < 0 or action_end_idx < 0 or action_start_idx >= action_end_idx:
        return result

    latent_idx = response.rfind(_LATENT_STATE_TOKEN, 0, action_start_idx)
    if latent_idx < 0:
        return result

    block = response[action_start_idx + len(_ACTION_START_TOKEN):action_end_idx]
    actions = _extract_nimloth_actions(block, max_actions)
    if not actions:
        return result

    result["actions"] = actions
    result["action_content"] = block.strip()
    result["format_correct"] = True
    result["llm_response"] = (
        f"<think>{result['think_content']}</think>"
        f"{_LATENT_STATE_TOKEN}{_ACTION_START_TOKEN}{result['action_content']}{_ACTION_END_TOKEN}"
    )
    return result


def parse_nimloth_wm(response: str, special_token_list=None, action_sep=',', max_actions=1) -> Dict:
    """Parse strict Nimloth world-modeling navigation responses."""
    response = response.replace("<image>", "")
    result = parse_nimloth(response, special_token_list=special_token_list, action_sep=action_sep, max_actions=max_actions)
    if not result["format_correct"]:
        result.update({
            "observation_content": "",
            "prediction_content": "",
        })
        return result

    obs_match = _OBS_RE.search(response)
    think_match = _THINK_RE.search(response)
    pred_match = _PRED_RE.search(response)
    action_start_idx = response.find(_ACTION_START_TOKEN)
    action_end_idx = response.find(_ACTION_END_TOKEN)
    latent_idx = response.rfind(_LATENT_STATE_TOKEN, 0, action_start_idx)

    if not (obs_match and think_match and pred_match):
        result["format_correct"] = False
        result["actions"] = []
        return result
    if not (obs_match.start() < think_match.start() < latent_idx < action_start_idx < action_end_idx < pred_match.start()):
        result["format_correct"] = False
        result["actions"] = []
        return result

    result["observation_content"] = obs_match.group(1).strip()
    result["think_content"] = think_match.group(1).strip()
    result["prediction_content"] = pred_match.group(1).strip()
    result["llm_response"] = response.strip()
    return result


def parse_freethink(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <think>...</think><answer>...</answer>
    
    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with <think> and <answer> tags
    - think_content: the content inside <think> tag
    - action_content: the content inside <answer> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    response = response.replace("<image>","")
    #Pattern to check for content strictly in the format <think>...</think><answer>...</answer>
    strict_pattern = r'^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    
    
    # Pattern to extract content from think and answer tags
    extraction_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    format_correct = strict_match is not None
    
    if not strict_match:
        think_content, action_content, actions = "", "", []
    else:
        think_content, action_content = match.group(1), match.group(2)
        if special_token_list is not None:
            for special_token in special_token_list: # remove all special tokens in responses to forbid confusion in training
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<think>" + think_content.strip() + "</think>" + "<answer>" + action_content.strip() + "</answer>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_no_think(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <answer>...</answer>
    
    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with <answer> tag
    - think_content: empty string (no think content in this format)
    - action_content: the content inside <answer> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    response = response.replace("<image>","")
    # Pattern to check for content strictly in the format <answer>...</answer>
    strict_pattern = r'^\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None
    
    # Pattern to extract content from answer tag
    extraction_pattern = r'<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    #format_correct = match is not None
    
    if not strict_match:
        think_content, action_content, actions = "", "", []
    else:
        action_content = match.group(1)
        think_content = ""  # No think content in this format
        if special_token_list is not None:
            for special_token in special_token_list:
                action_content = action_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<answer>" + action_content.strip() + "</answer>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_grounding(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>
    
    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with all tags
    - observation_content: the content inside <observation> tag
    - think_content: the entire content inside <think> tag
    - reasoning_content: the content inside <reasoning> tag
    - action_content: the content inside <answer> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    response = response.replace("<image>","")
    # Pattern to check for content strictly in the expected format
    strict_pattern = r'^\s*<think>\s*<observation>(.*?)</observation>\s*<reasoning>(.*?)</reasoning>\s*</think>\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None
    
    # Pattern to extract content from tags
    extraction_pattern = r'<think>\s*<observation>(.*?)</observation>\s*<reasoning>(.*?)</reasoning>\s*</think>\s*<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    
    if not match:
        observation_content, reasoning_content, action_content, actions = "", "", "", []
        think_content = ""
    else:
        observation_content = match.group(1)
        reasoning_content = match.group(2)
        action_content = match.group(3)
        think_content = "<observation>" + observation_content + "</observation><reasoning>" + reasoning_content + "</reasoning>"
        
        if special_token_list is not None:
            for special_token in special_token_list:
                observation_content = observation_content.replace(special_token, "").strip()
                reasoning_content = reasoning_content.replace(special_token, "").strip()
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
                
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)
    
    # Reconstruct the cleaned llm_response
    llm_response = "<think>" + think_content.strip() + "</think>" + "<answer>" + action_content.strip() + "</answer>"
    
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "observation_content": observation_content,
        "think_content": think_content,
        "reasoning_content": reasoning_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_worldmodeling(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>
    
    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with all tags
    - think_content: the entire content inside <think> tag
    - reasoning_content: the content inside <reasoning> tag
    - prediction_content: the content inside <prediction> tag
    - action_content: the content inside <answer> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    response = response.replace("<image>","")
    # Pattern to check for content strictly in the expected format
    strict_pattern = r'^\s*<think>\s*<reasoning>(.*?)</reasoning>\s*<prediction>(.*?)</prediction>\s*</think>\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None
    
    # Pattern to extract content from tags
    extraction_pattern = r'<think>\s*<reasoning>(.*?)</reasoning>\s*<prediction>(.*?)</prediction>\s*</think>\s*<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    
    if not match:
        reasoning_content, prediction_content, action_content, actions = "", "", "", []
        think_content = ""
    else:
        reasoning_content = match.group(1)
        prediction_content = match.group(2)
        action_content = match.group(3)
        think_content = "<reasoning>" + reasoning_content + "</reasoning><prediction>" + prediction_content + "</prediction>"
        
        if special_token_list is not None:
            for special_token in special_token_list:
                reasoning_content = reasoning_content.replace(special_token, "").strip()
                prediction_content = prediction_content.replace(special_token, "").strip()
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
                
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)
    
    # Reconstruct the cleaned llm_response
    llm_response = "<think>" + think_content.strip() + "</think>" + "<answer>" + action_content.strip() + "</answer>"
    
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "reasoning_content": reasoning_content,
        "prediction_content": prediction_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_grounding_worldmodeling(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>
    
    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with all tags
    - observation_content: the content inside <observation> tag
    - reasoning_content: the content inside <reasoning> tag
    - prediction_content: the content inside <prediction> tag
    - think_content: the entire content inside <think> tag
    - action_content: the content inside <answer> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    response = response.replace("<image>","")
    # Pattern to check for content strictly in the expected format
    strict_pattern = r'^\s*<think>\s*<observation>(.*?)</observation>\s*<reasoning>(.*?)</reasoning>\s*<prediction>(.*?)</prediction>\s*</think>\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None
    
    # Pattern to extract content from tags
    extraction_pattern = r'<think>\s*<observation>(.*?)</observation>\s*<reasoning>(.*?)</reasoning>\s*<prediction>(.*?)</prediction>\s*</think>\s*<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    
    if not match:
        observation_content, reasoning_content, prediction_content, action_content, actions = "", "", "", "", []
        think_content = ""
    else:
        observation_content = match.group(1)
        reasoning_content = match.group(2)
        prediction_content = match.group(3)
        action_content = match.group(4)
        think_content = "<observation>" + observation_content + "</observation><reasoning>" + reasoning_content + "</reasoning><prediction>" + prediction_content + "</prediction>"
        
        if special_token_list is not None:
            for special_token in special_token_list:
                observation_content = observation_content.replace(special_token, "").strip()
                reasoning_content = reasoning_content.replace(special_token, "").strip()
                prediction_content = prediction_content.replace(special_token, "").strip()
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
                
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)
    
    # Reconstruct the cleaned llm_response
    llm_response = "<think>" + think_content.strip() + "</think>" + "<answer>" + action_content.strip() + "</answer>"
    
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "observation_content": observation_content,
        "reasoning_content": reasoning_content,
        "prediction_content": prediction_content,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }
    
PARSE_FUNC_MAP = {
    "free_think": parse_freethink,
    "no_think": parse_no_think,
    "grounding": parse_grounding,
    "worldmodeling": parse_worldmodeling,
    "wm": parse_worldmodeling,
    "grounding_worldmodeling": parse_grounding_worldmodeling,
    "nimloth": parse_nimloth,
    "nimloth_wm": parse_nimloth_wm,
    "grounding_structured": parse_grounding,
    "worldmodeling_structured": parse_worldmodeling,
    "grounding_worldmodeling_structured": parse_grounding_worldmodeling,
    "grounding_symbolic": parse_grounding,
    "worldmodeling_symbolic": parse_worldmodeling,
    "grounding_worldmodeling_symbolic": parse_grounding_worldmodeling,
}
