import numpy as np
from PIL import Image
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
from areal.envs.sokoban.env_config import SokobanEnvConfig
from areal.envs.sokoban.prompt import system_prompt, init_observation_template, action_template, format_prompt
from areal.envs.sokoban.utils import parse_free_think

class SokobanEnv:
    """Simplified Sokoban Environment"""
    
    # Grid symbols for text rendering
    GRID_LOOKUP = {
        0: " # ",  # wall
        1: " _ ",  # floor
        2: " O ",  # target
        3: " √ ",  # box on target
        4: " X ",  # box
        5: " P ",  # player
        6: " S ",  # player on target
    }

    # Action mapping
    ACTION_LOOKUP = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
    }

    def __init__(self, config: SokobanEnvConfig):
        """Initialize Sokoban environment with given config"""
        self.config = config
        self.env = GymSokobanEnv(
            dim_room=self.config.dim_room,
            max_steps=self.config.max_steps,
            num_boxes=self.config.num_boxes,
        )
        self.total_reward = 0
        self.valid_actions = []

    def reset(self, seed=None):
        """Reset environment and return initial observation"""
        obs = self.env.reset()
        self.total_reward = 0
        self.valid_actions = []
        
        return self._render(init_obs=True), {}

    def step(self, action_str: str):
        """
        Execute action(s) from string input
        
        Args:
            action_str: String in format <think>...</think><answer>...</answer>
            
        Returns:
            observation, reward, done, info (standard gym format)
        """
        # Parse the action string
        parsed = parse_free_think(
            response=action_str,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step
        )
        
        action_list = parsed['actions']
        prev_player_pos = self.env.player_position.copy()
        
        # Initialize metrics
        metrics = {
            "turn_metrics": {
                "action_is_valid": len(action_list) > 0 and parsed["format_correct"],
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            }
        }
        
        reward = 0
        self.valid_actions = []
        done = False
        info = {}
        info.update(parsed)
        
        # Execute each action in the list
        for action in action_list:
            if action in self.ACTION_LOOKUP:
                action_int = self.ACTION_LOOKUP[action]
                _, step_reward, step_done, _ = self.env.step(action_int)
                reward += step_reward
                self.valid_actions.append(action)
                
                # Check if puzzle is solved
                if self._is_success():
                    done = True
                    metrics['traj_metrics']['success'] = True
                    break
            else:
                metrics['turn_metrics']['action_is_valid'] = False
                break
        
        # Check if action was effective (player moved)
        metrics['turn_metrics']['action_is_effective'] = not np.array_equal(
            prev_player_pos, self.env.player_position
        )
        
        info["metrics"] = metrics
        self.total_reward += reward
        
        return self._render(init_obs=False), reward, done, info

    def close(self):
        """Close the environment"""
        self.env.close()

    def get_system_prompt(self):
        """Get system prompt with format instructions"""
        format_prompt_str = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=True
        )
        return system_prompt() + "\n" + format_prompt_str

    def _render(self, init_obs=True):
        """Render current state as text or vision"""
        multi_modal_data = None
        
        # Get format prompt (without example for observations)
        format_prompt_str = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False
        )
        
        if self.config.render_mode == 'vision':
            # Vision mode: use image placeholder
            img_str = self.config.image_placeholder
            rgb_array = self.env.render(mode='rgb_array')
            multi_modal_data = {
                self.config.image_placeholder: [self._numpy_to_pil(rgb_array)]
            }
        else:
            # Text mode: convert grid to text representation
            img_str = self._grid_to_text()
        
        # Generate observation string
        if init_obs:
            obs_str = init_observation_template(img_str) + "\n" + format_prompt_str
        else:
            obs_str = action_template(self.valid_actions, img_str) + "\n" + format_prompt_str
        
        if multi_modal_data is not None:
            return {
                "obs_str": obs_str,
                "multi_modal_data": multi_modal_data,
            }
        else:
            return {
                "obs_str": obs_str,
            }

    def _grid_to_text(self):
        """Convert room state to text representation"""
        # Handle player on target case
        room_state = np.where(
            (self.env.room_state == 5) & (self.env.room_fixed == 2), 
            6, 
            self.env.room_state
        )
        
        # Convert grid to text using lookup table
        text_rows = []
        for row in room_state:
            text_row = "".join(self.GRID_LOOKUP.get(cell, "?") for cell in row)
            text_rows.append(text_row)
        
        return "\n".join(text_rows)

    def _numpy_to_pil(self, numpy_array):
        """Convert numpy array to PIL Image"""
        if numpy_array.shape[-1] == 3:
            return Image.fromarray(numpy_array, mode='RGB')
        else:
            raise ValueError(f"Unsupported channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")

    def _is_success(self):
        """Check if puzzle is solved (all boxes on targets)"""
        return self.env.boxes_on_target == self.env.num_boxes


if __name__ == "__main__":
    # Test the environment
    config = SokobanEnvConfig(
        render_mode='text',  # or 'vision'
        num_boxes=1,
        dim_room=(5, 5),
        max_actions_per_step=2
    )
    
    env = SokobanEnv(config)
    
    print("System Prompt:")
    print(env.get_system_prompt())
    print("\n" + "="*50 + "\n")
    
    # Reset and start game
    obs, info = env.reset()
    print("Initial Observation:")
    print(obs["obs_str"])
    
    # Test game loop
    step_count = 0
    while step_count < 5:  # Test a few steps
        print(f"\nStep {step_count + 1}:")
        action_input = input("Enter action string (or 'quit'): ")
        
        if action_input.lower() == 'quit':
            break
            
        # If user doesn't provide full format, add it
        if not action_input.startswith('<think>'):
            action_input = f"<think>Moving towards the goal.</think><answer>{action_input}</answer>"
        
        obs, reward, done, info = env.step(action_input)
        print(f"Observation:\n{obs['obs_str']}")
        print(f"Reward: {reward}, Done: {done}")
        print(f"Valid actions: {info.get('actions', [])}")
        
        if done:
            print("Puzzle solved!")
            break
            
        step_count += 1
    
    print(f"\nTotal reward: {env.total_reward}")
    env.close()