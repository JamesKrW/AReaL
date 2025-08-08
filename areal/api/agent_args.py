from .cli_args import GRPOConfig, load_expr_config
from dataclasses import asdict, dataclass, field
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class EnvSpec:
    """One logical environment family to expand into N data points."""
    name: str                    # e.g., "sokoban"
    n_envs: int                  # e.g., 10000
    split: str                   # "train" | "valid"
    seed: int = 0                # per-env base seed
    config: Dict[str, Any] = field(default_factory=dict)  # env-specific cfg


@dataclass
class AgentGRPOConfig(GRPOConfig):
    envs: List[EnvSpec] = field(default_factory=list)