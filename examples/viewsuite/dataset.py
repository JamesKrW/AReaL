import hashlib
from typing import List

from torch.utils.data import Dataset

from agent_args import EnvSpec


def _deterministic_seed(base_seed: int, per_env_base_seed: int, idx: int) -> int:
    """
    Generate a deterministic 32-bit seed from (base_seed, per_env_base_seed, idx).
    This ensures reproducibility regardless of RNG state.
    """
    h = hashlib.blake2b(
        f"{base_seed}|{per_env_base_seed}|{idx}".encode("utf-8"),
        digest_size=4,  # 4 bytes → 32-bit integer
    ).digest()
    return int.from_bytes(h, "little")


class AgenticDataset(Dataset):
    """
    Expands a list of EnvSpec into individual environment instances, with deterministic seeds,
    sharded across distributed workers.
    """

    def __init__(
        self,
        env_specs: List[EnvSpec],
        base_seed: int,
        rank: int,
        world_size: int,
    ):
        assert world_size >= 1 and 0 <= rank < world_size, "Invalid rank/world_size"
        self.items = []

        for spec in env_specs:
            # Only process envs for this rank
            for i in range(spec.n_envs):
                if i % world_size != rank:
                    continue
                env_seed = _deterministic_seed(base_seed, spec.seed, i)
                self.items.append(
                    {
                        "name": spec.name,
                        "seed": env_seed,
                        "config": spec.config,
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_env_dataset(
    envs_config: List[EnvSpec],
    split: str,
    base_seed: int,
    rank: int,
    world_size: int,
) -> AgenticDataset:
    """
    Filter EnvSpec list by split, and return a sharded AgenticDataset.

    Args:
        envs_config: List of EnvSpec objects (train + valid + other splits).
        split: "train" or "valid".
        base_seed: Global base seed.
        rank: Distributed rank.
        world_size: Total number of processes.

    Returns:
        AgenticDataset containing only data for the given split and rank.
    """
    split_specs = [spec for spec in envs_config if spec.split == split]
    return AgenticDataset(split_specs, base_seed, rank, world_size)