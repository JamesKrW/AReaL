import hashlib
import random
from typing import List, Optional, Sequence

from torch.utils.data import Dataset

from agent_args import EnvSpec


# Upper bound used for RNG sampling when only a base seed is provided
MAX_INT32 = 2 ** 31 - 1


def _make_rng_seed(base_seed: int, spec: EnvSpec, spec_idx: int, hint: str) -> int:
    """Expand the global seed into a deterministic per-spec RNG seed."""
    payload = f"{base_seed}|{spec_idx}|{spec.name}|{spec.split}|{hint}"
    h = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little")


def _coerce_to_int_list(values: Optional[Sequence]) -> Optional[List[int]]:
    """Return a list of ints if `values` is a proper sequence, otherwise None."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError("seed_list must be a sequence of integers, not string")
    coerced = [int(v) for v in values]
    return coerced


def _normalize_seed_directive(seed_field) -> List[int]:
    """Normalise the seed directive into a list of ints, defaulting to [0]."""
    if seed_field is None:
        return [0]
    if isinstance(seed_field, (int, float)):
        return [int(seed_field)]
    if isinstance(seed_field, Sequence) and not isinstance(seed_field, (str, bytes)):
        coerced = [int(v) for v in seed_field]
        return coerced if coerced else [0]
    raise TypeError("seed must be an integer or a sequence of integers")


def _generate_from_len_one(
    rng: random.Random,
    n_envs: int,
) -> List[int]:
    """len(seed)==1 → sample uniformly from the full 32-bit range."""
    return [rng.randrange(0, MAX_INT32 + 1) for _ in range(n_envs)]


def _generate_from_len_two(
    rng: random.Random,
    n_envs: int,
    minimum: int,
    maximum: int,
) -> List[int]:
    """len(seed)==2 → sample from the inclusive [min, max] range."""
    if maximum < minimum:
        raise ValueError("seed[1] must be >= seed[0] when len(seed) == 2")
    if minimum == maximum:
        return [minimum] * n_envs
    return [rng.randrange(minimum, maximum + 1) for _ in range(n_envs)]


def _generate_from_len_three(
    rng: random.Random,
    n_envs: int,
    minimum: int,
    maximum: int,
    limit: int,
) -> List[int]:
    """len(seed)==3 → sample from [min, max] but cap occurrences per value."""
    if maximum < minimum:
        raise ValueError("seed[1] must be >= seed[0] when len(seed) == 3")
    if limit <= 0:
        raise ValueError("seed[2] must be a positive integer when len(seed) == 3")
    range_size = maximum - minimum + 1
    if range_size <= 0:
        raise ValueError("seed range must contain at least one value")
    if range_size * limit < n_envs:
        raise ValueError(
            "seed range with given limit cannot supply enough unique seeds for n_envs"
        )

    if limit == 1:
        population = range(minimum, maximum + 1)
        return rng.sample(population, n_envs)

    counts = {}
    seeds = []
    while len(seeds) < n_envs:
        candidate = rng.randint(minimum, maximum)
        count = counts.get(candidate, 0)
        if count >= limit:
            continue
        counts[candidate] = count + 1
        seeds.append(candidate)
    return seeds


def _generate_seeds_for_spec(
    spec: EnvSpec,
    base_seed: int,
    spec_idx: int,
) -> List[int]:
    """Generate `n_envs` seeds using either seed_list or seed directive rules."""
    explicit_list = _coerce_to_int_list(spec.seed_list)
    if explicit_list is not None:
        if len(explicit_list) <= spec.n_envs:
            raise ValueError(
                f"seed_list for env '{spec.name}' must contain more than n_envs values"
            )
        return explicit_list[: spec.n_envs]

    directive = _normalize_seed_directive(spec.seed)
    length = len(directive)
    if length == 0:
        directive = [0]
        length = 1

    rng_seed = _make_rng_seed(base_seed, spec, spec_idx, f"seed-{directive}")
    rng = random.Random(rng_seed)

    if length == 1:
        return _generate_from_len_one(rng, spec.n_envs)
    if length == 2:
        return _generate_from_len_two(rng, spec.n_envs, directive[0], directive[1])
    if length == 3:
        return _generate_from_len_three(rng, spec.n_envs, directive[0], directive[1], directive[2])

    raise ValueError(
        "seed directive must be of length 1, 2, or 3 when seed_list is not provided"
    )


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

        for spec_idx, spec in enumerate(env_specs):
            seeds = _generate_seeds_for_spec(
                spec,
                base_seed,
                spec_idx,
            )
            for i, env_seed in enumerate(seeds):
                if i % world_size != rank:
                    continue
                # Each record contains env metadata and the resolved RNG seed
                tag_id = int(getattr(spec, "tag_id", 0))
                self.items.append(
                    {
                        "name": spec.name,
                        "seed": env_seed,
                        "config": spec.config,
                        "max_turns": spec.max_turns,
                        "tag_id": tag_id,
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
    return AgenticDataset(
        split_specs,
        base_seed,
        rank,
        world_size,
    )
