
from dataclasses import dataclass

@dataclass
class SokobanEnvConfig:
    """Configuration for Sokoban environment"""
    dim_room: tuple = (6, 6)  # Room dimensions (height, width)
    max_steps: int = 100      # Maximum steps per episode
    num_boxes: int = 1        # Number of boxes in the room
    render_mode: str = "text" # "text" or "vision"
    max_actions_per_step: int = 3  # Max actions per step
    action_sep: str = ","     # Separator between actions
    image_placeholder: str = "<image>"  # Placeholder for vision mode