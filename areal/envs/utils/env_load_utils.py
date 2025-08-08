# env_loader.py
from __future__ import annotations
from dataclasses import is_dataclass, fields
from typing import Any, Dict, Tuple


import importlib
import pkgutil
import threading
import areal.envs  

from areal.envs.registry import REGISTERED_ENVS, REGISTERED_ENVCONFIGS

# Guard so we only scan once
__ENV_SCAN_DONE = False
__ENV_SCAN_LOCK = threading.Lock()

def ensure_envs_registered():
    """
    Import all submodules under areal.envs (recursively) to trigger @env_class/@config_class
    side effects and populate the registries.
    """
    global __ENV_SCAN_DONE
    if __ENV_SCAN_DONE:
        return
    with __ENV_SCAN_LOCK:
        if __ENV_SCAN_DONE:
            return

        # Recursively walk all packages/modules under areal.envs
        for finder, name, ispkg in pkgutil.walk_packages(areal.envs.__path__, areal.envs.__name__ + "."):
            # print(name)
            try:
                importlib.import_module(name)
            except Exception as e:
                # You might want to log instead of raising, so one bad env doesn't kill the run
                print(f"[env-scan] Failed to import {name}: {e}")
                pass

        __ENV_SCAN_DONE = True


def _coerce_config(cfg_cls, cfg_dict: Dict[str, Any]):
    """
    Convert a plain dictionary into a dataclass config instance.

    - Supports converting tuple strings like "(6, 6)" → (6, 6).
    - Ignores unknown fields not present in the dataclass.
    """
    if cfg_dict is None:
        cfg_dict = {}

    if not is_dataclass(cfg_cls):
        # If cfg_cls is a regular class (not a dataclass), instantiate directly.
        return cfg_cls(**cfg_dict)

    kwargs = {}
    valid_fields = {f.name: f.type for f in fields(cfg_cls)}
    for k, v in cfg_dict.items():
        if k not in valid_fields:
            continue
        # Parse strings like "(a, b)" into tuples of ints
        if isinstance(v, str) and v.strip().startswith("(") and v.strip().endswith(")"):
            body = v.strip()[1:-1]
            parts = [p.strip() for p in body.split(",") if p.strip() != ""]
            try:
                v = tuple(int(p) for p in parts)
            except Exception:
                pass  # Keep the original value if parsing fails
        kwargs[k] = v
    return cfg_cls(**kwargs)

def load_env_from_registry(data: dict):
    # NEW: make sure we've imported all submodules at least once
    ensure_envs_registered()

    name = data.get("name")
    seed = data.get("seed", None)
    cfg_dict = data.get("config", {})
    if name is None:
        raise ValueError("`data['name']` is required.")

    if name not in REGISTERED_ENVS:
        raise ValueError(f"Env '{name}' not found. Known envs: {list(REGISTERED_ENVS.keys())}")
    if name not in REGISTERED_ENVCONFIGS:
        raise ValueError(
            f"Config class for '{name}' not found. Known configs: {list(REGISTERED_ENVCONFIGS.keys())}"
        )

    cfg_cls = REGISTERED_ENVCONFIGS[name]
    env_cls = REGISTERED_ENVS[name]

    cfg = _coerce_config(cfg_cls, cfg_dict)
    env = env_cls(cfg)
    return env, seed
    



if __name__ == "__main__":
    # Example usage
    example_data = {
        "name": "sokoban",
        "seed": 12345,
        "config": {"dim_room": "(6, 6)", "num_boxes": 1, "render_mode": "text"}
    }
    env, seed = load_env_from_registry(example_data)
    print(f"Loaded environment: {env}, with seed: {seed}")
    obs,_=env.reset(seed=seed)
    print(obs)