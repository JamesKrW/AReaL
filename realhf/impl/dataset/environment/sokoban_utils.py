import re
from typing import List, Tuple, Dict

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
    strict_pattern = r'^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    extraction_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    format_correct = strict_match is not None

    if not strict_match or not match:
        think_content, action_content, actions = "", "", []
    else:
        think_content, action_content = match.group(1), match.group(2)
        if special_token_list is not None:
            for special_token in special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
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

def get_system_prompt(max_actions_per_step: int = 3, action_sep: str = ',') -> str:
    """
    Returns the Sokoban freethink system prompt with example.
    """
    return f"""You are a Sokoban solver.
Sokoban Quick Guide
Goal: Push all boxes onto targets.
Symbols:
# Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target
Rules:
1. Push boxes (can't pull).
2. Avoid walls.
Actions you can take: Left, Down, Right, Up.
You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.
You should first give your reasoning, and then your answer.
Your response should be in the format of:
<think>...</think><answer>...</answer>
e.g. <think>I need to go down then push the box down to the target.</think><answer>Down{action_sep}Down</answer>
"""

def get_observation_prompt(observation: str, max_actions_per_step: int = 3, action_sep: str = ',') -> str:
    """
    Returns the observation prompt for each step.
    """
    return f"""[Observation]:
{observation}
Decide your next action(s).
You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.
Respond in the format: <think>...</think><answer>...</answer>
"""

def parse_actions_from_response(response: str, max_actions: int = 3, action_sep: str = ',') -> Tuple[List[str], bool]:
    """
    Parses the <think>...</think><answer>...</answer> part and returns up to max_actions actions.
    Returns (actions, format_correct)
    """
    result = parse_freethink(response, action_sep=action_sep, max_actions=max_actions)
    valid_actions = {"Up", "Down", "Left", "Right"}
    actions = [a for a in result["actions"] if a in valid_actions]
    format_correct = result["format_correct"] and len(actions) > 0
    return actions, format_correct 