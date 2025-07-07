# Copyright 2025 Ant Group Inc.

import asyncio
import numpy as np
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
from realhf.api.core.env_api import EnvironmentService, register_environment
from realhf.impl.agent.sokoban_utils import get_system_prompt, get_observation_prompt, parse_actions_from_response

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
        self.last_obs = self._obs_to_text(obs)
        self.last_info = {}
        prompt = self.system_prompt + "\n" + get_observation_prompt(self.last_obs, self.max_actions_per_turn, self.action_sep)
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
        self.last_obs = self._obs_to_text(obs)
        self.last_info = info
        prompt = get_observation_prompt(self.last_obs, self.max_actions_per_turn, self.action_sep)
        return prompt, total_reward, done, info

    def _obs_to_text(self, obs):
        # obs is a numpy array
        lookup = {
            0: "#",  # wall
            1: " ",  # floor
            2: ".",  # target
            3: "*",  # box on target
            4: "$",  # box
            5: "@",  # player
            6: "+",  # player on target
        }
        return "\n".join("".join(lookup.get(cell, "?") for cell in row) for row in obs)

    def _action_to_idx(self, act):
        # Map text action to gym action index
        action_map = {"Up": 0, "Down": 1, "Left": 2, "Right": 3}
        return action_map.get(act, 0)

register_environment("sokoban-text-freethink", SokobanTextFreethinkEnv) 