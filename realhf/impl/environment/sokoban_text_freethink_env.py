# Copyright 2025 Ant Group Inc.

import asyncio
import numpy as np
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
from realhf.api.core.env_api import EnvironmentService, register_environment
from realhf.impl.environment.sokoban_utils import get_system_prompt, get_observation_prompt, parse_actions_from_response

class SokobanTextFreethinkEnv(EnvironmentService):
    def __init__(self, dim_room=(6, 6), num_boxes=3, max_steps=100, max_actions_per_turn=3, action_sep=",", format_reward=0.5):
        self.dim_room = dim_room
        self.num_boxes = num_boxes
        self.max_steps = max_steps
        self.max_actions_per_turn = max_actions_per_turn
        self.action_sep = action_sep
        self.format_reward = format_reward
        self.env = GymSokobanEnv(dim_room=dim_room, max_steps=max_steps, num_boxes=num_boxes)
        self.last_obs = None
        self.last_info = {}
        self.system_prompt = get_system_prompt(max_actions_per_step=max_actions_per_turn, action_sep=action_sep)

    async def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        
        # 🔧 FIX: Use room_state instead of the rendered image
        if hasattr(self.env, 'room_state'):
            room_state = self.env.room_state
        elif hasattr(self.env, '_room_state'):
            room_state = self.env._room_state
        elif hasattr(self.env, 'get_room_state'):
            room_state = self.env.get_room_state()
        else:
            # Fallback: try to access room_state through different possible attributes
            room_state = getattr(self.env, 'room_state', obs)
        
        self.last_obs = self._room_state_to_text(room_state)
        self.last_info = {}
        obs_prompt = get_observation_prompt(self.last_obs, self.max_actions_per_turn, self.action_sep)
        prompt = self.system_prompt + "\n" + obs_prompt
        
        # 🐛 DEBUG: Print lengths
        print(f"🐛 ENV DEBUG: system_prompt length = {len(self.system_prompt)} chars")
        print(f"🐛 ENV DEBUG: obs_text length = {len(self.last_obs)} chars")
        print(f"🐛 ENV DEBUG: obs_prompt length = {len(obs_prompt)} chars")
        print(f"🐛 ENV DEBUG: final prompt length = {len(prompt)} chars")
        print(f"🐛 ENV DEBUG: obs_text content: {repr(self.last_obs[:200])}")  # Show first 200 chars
        print(f"🐛 ENV DEBUG: room_state shape: {room_state.shape if hasattr(room_state, 'shape') else 'no shape'}")
        print(f"🐛 ENV DEBUG: room_state type: {type(room_state)}")
        
        return prompt, self.last_info

    async def step(self, action):
        # action: (qid, [response_str])
        _, responses = action
        response = responses[0]
        actions, format_correct = parse_actions_from_response(response, max_actions=self.max_actions_per_turn, action_sep=self.action_sep)
        total_reward = 0.0
        done = False
        info = {"actions": actions, "format_correct": format_correct}
        for act in actions:
            act_idx = self._action_to_idx(act)
            obs, reward, done, step_info = self.env.step(act_idx)
            total_reward += reward
            if done and step_info.get("all_boxes_on_target", False):
                total_reward = 10.0
                break
        if format_correct:
            total_reward += self.format_reward
            info["is_format_rewarded"] = True
        else:
            info["is_format_rewarded"] = False
        
        # 🔧 FIX: Use room_state instead of the rendered image  
        if hasattr(self.env, 'room_state'):
            room_state = self.env.room_state
        elif hasattr(self.env, '_room_state'):
            room_state = self.env._room_state
        elif hasattr(self.env, 'get_room_state'):
            room_state = self.env.get_room_state()
        else:
            room_state = getattr(self.env, 'room_state', obs)
            
        self.last_obs = self._room_state_to_text(room_state)
        self.last_info = info
        prompt = get_observation_prompt(self.last_obs, self.max_actions_per_turn, self.action_sep)
        return prompt, total_reward, done, info

    def _room_state_to_text(self, room_state):
        """Convert room state to text representation"""
        # Room state should be the logical 2D grid (e.g., 6x6 for dim_room=(6,6))
        lookup = {
            0: "#",  # wall
            1: " ",  # floor  
            2: ".",  # target
            3: "*",  # box on target
            4: "$",  # box
            5: "@",  # player
            6: "+",  # player on target
        }
        
        # Convert to list if it's a numpy array
        if hasattr(room_state, 'tolist'):
            state_list = room_state.tolist()
        else:
            state_list = room_state
            
        # Handle the case where room_state is the logical grid
        if len(state_list) == self.dim_room[0]:  # Should be 6x6 for our case
            def cell_to_char(cell):
                if isinstance(cell, list):
                    # If cell is still a list, take the first element or flatten
                    return ''.join(str(lookup.get(int(c), "?")) for c in cell)
                return lookup.get(int(cell), "?")
            return "\n".join(''.join(cell_to_char(cell) for cell in row) for row in state_list)
        else:
            # Fallback for unexpected formats
            print(f"🐛 ENV WARNING: Unexpected room_state format, shape: {np.array(state_list).shape}")
            return "# # # # # #\n# @ $ . # #\n# # # # # #\n# # # # # #\n# # # # # #\n# # # # # #"

    def _obs_to_text(self, obs):
        """Legacy method - should not be used anymore"""
        print(f"🐛 ENV WARNING: _obs_to_text called with obs shape: {obs.shape if hasattr(obs, 'shape') else 'no shape'}")
        return self._room_state_to_text(obs)

    def _action_to_idx(self, act):
        # Map text action to gym action index
        action_map = {"Up": 0, "Down": 1, "Left": 2, "Right": 3}
        return action_map.get(act, 0)

register_environment("sokoban-text-freethink", SokobanTextFreethinkEnv) 