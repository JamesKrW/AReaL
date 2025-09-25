from areal.api.cli_args import GRPOConfig
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class EnvSpec:
    """One logical environment family to expand into N data points."""
    name: str                    # key in REGISTERED_ENVS, e.g., "GymProxyNoTool"
    n_envs: int                  # e.g., 10000
    split: str                   # "train" | "valid"
    seed: int = 0                # per-env base seed
    config: Dict[str, Any] = field(default_factory=dict)  # env-specific cfg


@dataclass
class AgentGRPOConfig(GRPOConfig):
    envs: List[EnvSpec] = field(default_factory=list)
    max_turns: int = 5