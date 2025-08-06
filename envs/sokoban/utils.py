import re
from typing import Dict, List

def parse_free_think(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """
    Parse free_think format response: <think>...</think><answer>...</answer>
    
    Args:
        response: Raw LLM response string
        action_sep: Separator between actions
        max_actions: Maximum number of actions to extract
    
    Returns:
        Dict containing parsed components and validation info
    """
    # Pattern to match <think>...</think><answer>...</answer>
    pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)
    
    format_correct = match is not None
    
    if not match:
        think_content = ""
        action_content = ""
        actions = []
    else:
        think_content = match.group(1).strip()
        action_content = match.group(2).strip()
        
        # Split actions by separator and clean them
        actions = [action.strip().lower() for action in action_content.split(action_sep) if action.strip()]
        
        # Limit to max_actions
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = action_sep.join(actions)
    
    # Reconstruct formatted response
    llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>"
    
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }