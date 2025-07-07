# Licensed under the Apache License, Version 2.0 (the "License").

import copy
import dataclasses
import os
from typing import Any, Dict, List, Tuple

import realhf.base.logging as logging
from realhf.api.cli_args import ModelTrainEvalConfig, PPOMATHExperimentOptions
from realhf.api.core.config import (
    AgentAbstraction,
    EnvServiceAbstraction,
    ModelInterfaceAbstraction,
)
from realhf.api.core.model_api import GenerationHyperparameters
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.base import constants
from realhf.experiments.async_exp.async_rl_exp import AsyncRLExperimentConfig
from realhf.experiments.common.ppo_math_exp import PPOMATHConfig

logger = logging.getLogger("Async PPO Sokoban exp", "colored")


@dataclasses.dataclass
class AsyncPPOSokobanConfig(AsyncRLExperimentConfig, PPOMATHConfig):
    """Configuration for Async PPO Sokoban experiments.
    
    This configuration is specifically designed for Sokoban RL training,
    where the environment generates prompts dynamically without requiring
    a static dataset path.
    """

    @property
    def agent(self) -> AgentAbstraction:
        return AgentAbstraction(
            "multi-turn-agent",
            args=dict(
                gconfig=self.generation_config,
                tokenizer_path=self.actor.path,
                num_turns=3,  # Default number of turns for sokoban
            ),
        )

    @property
    def env(self) -> EnvServiceAbstraction:
        return EnvServiceAbstraction(
            "sokoban-text-freethink",
            args=dict(
                dim_room=(6, 6),  # Default room size
                num_boxes=3,       # Default number of boxes
                max_steps=100,     # Default max steps per episode
                max_actions_per_turn=3,  # Default actions per turn
                action_sep=",",    # Action separator
                format_reward=0.5, # Format reward for correct action format
            ),
        )

    @property
    def gen_backend_args(self) -> Any:
        return self.actor.sglang

    @property
    def generation_config(self) -> GenerationHyperparameters:
        return GenerationHyperparameters(**dataclasses.asdict(self.ppo.gen)).new(n=self.group_size)

    @property
    def rpcs(self):
        rpcs = super(AsyncPPOSokobanConfig, self).rpcs
        rpcs["actor_gen"].output_keys = (
            *rpcs["actor_gen"].output_keys,
            "packed_prompts",
            "version_start",
            "version_end",
            "rewards",
            "birth_time",
        )
        rpcs["actor_train"].input_keys = (
            *rpcs["actor_train"].input_keys,
            "version_start",
            "version_end",
        )
        # Revert the effect of fuse_rew_ref, because we don't have the reward RPC in async experiments.
        if "ref_inf" in rpcs:
            actor_interface = rpcs["actor_train"].interface_impl
            rpcs["ref_inf"].interface_impl = copy.deepcopy(actor_interface)
            rpcs["ref_inf"].interface_impl.args["enable_save"] = False
            rpcs["ref_inf"].input_keys = ("packed_input_ids",)
            rpcs["ref_inf"].output_keys = ("packed_ref_logprobs",)
        if "rew_inf" in rpcs:
            rpcs.pop("rew_inf")
        if self.no_training:
            rpcs["actor_train"].interface_impl = ModelInterfaceAbstraction("null")
            rpcs["actor_gen"].interface_impl = ModelInterfaceAbstraction("null")
            if "actor_inf" in rpcs:
                rpcs["actor_inf"].interface_impl = ModelInterfaceAbstraction("null")
        return rpcs

    @property
    def models(self) -> Dict[str, ModelTrainEvalConfig]:
        models = super().models
        if "reward" in models:
            models.pop("reward")
        return models

    @property
    def allocations(self):
        allocations = super(AsyncPPOSokobanConfig, self).allocations
        if "rew_inf" in allocations:
            allocations.pop("rew_inf")
        return allocations


register_quickstart_exp("async-ppo-sokoban", AsyncPPOSokobanConfig) 