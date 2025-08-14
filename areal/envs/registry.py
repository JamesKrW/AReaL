from typing import Type, Any, Dict

REGISTERED_ENVS: Dict[str, Type[Any]] = {}
REGISTERED_ENVCONFIGS: Dict[str, Type[Any]] = {}

def env_class(name: str, *, overwrite: bool = False):
    def _deco(cls: Type[Any]) -> Type[Any]:
        if not overwrite and name in REGISTERED_ENVS:
            raise ValueError(f"Env '{name}' already exists: {REGISTERED_ENVS[name]}")
        REGISTERED_ENVS[name] = cls
        return cls
    return _deco

def config_class(name: str, *, overwrite: bool = False):
    def _deco(cls: Type[Any]) -> Type[Any]:
        if not overwrite and name in REGISTERED_ENVCONFIGS:
            raise ValueError(f"Config '{name}' already exists: {REGISTERED_ENVCONFIGS[name]}")
        REGISTERED_ENVCONFIGS[name] = cls
        return cls
    return _deco
