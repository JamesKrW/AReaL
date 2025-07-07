import re
from typing import List, Tuple

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
    Parses the <answer>...</answer> part and returns up to max_actions actions.
    Returns (actions, format_correct)
    """
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    format_correct = think_match is not None and answer_match is not None
    actions = []
    if answer_match:
        answer_str = answer_match.group(1).strip()
        actions = [a.strip().capitalize() for a in answer_str.split(action_sep) if a.strip()]
        actions = actions[:max_actions]
        # Only allow valid actions
        valid_actions = {"Up", "Down", "Left", "Right"}
        actions = [a for a in actions if a in valid_actions]
        format_correct = format_correct and len(actions) > 0
    else:
        format_correct = False
    return actions, format_correct 