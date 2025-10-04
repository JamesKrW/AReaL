from areal.api.cli_args import GRPOConfig
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class EnvSpec:
    """One logical environment family to expand into N data points."""
    # Key in REGISTERED_ENVS, e.g., "GymProxyNoTool"
    name: str
    # How many concrete instances to materialize from this spec
    n_envs: int
    # Which dataset split this spec belongs to ("train" | "valid" | ...)
    split: str
    tag_id: int = 0
    # Environment-specific configuration passed through untouched
    config: Dict[str, Any] = field(default_factory=dict)
    # Seed directive: [base] | [min, max] | [min, max, limit]
    # 1-element list: fixed base seed
    # 2-element list: for each env, uniformly random sample a seed in [min, max]
    # 3-element list: as above, but each seed occur at most 'limit' times
    seed: List[int] = field(default_factory=lambda: [0])
    # Optional explicit per-instance seeds; must be longer than n_envs
    seed_list: Optional[List[int]] = None
    max_turns: int =1


@dataclass
class AgentGRPOConfig(GRPOConfig):
    fileroot: str = "./data"
    envs: List[EnvSpec] = field(default_factory=list)
