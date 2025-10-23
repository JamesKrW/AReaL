# areal/launcher/ray_abs.py
# Simplified Ray launcher: all entry scripts (trainer, sglang, vllm) are passed as ABSOLUTE paths.
# No automatic guessing or joining.

import importlib.util
import os
import sys
import time
import re
from functools import partial
from typing import Callable, Dict, List, Optional

import ray
import ray.exceptions
from ray.runtime_env import RuntimeEnv
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from areal.api.alloc_mode import AllocationMode, AllocationType
from areal.api.cli_args import (
    ClusterSpecConfig, LauncherConfig, RecoverConfig, SGLangConfig,
    parse_cli_args, to_structured_cfg, vLLMConfig,
)
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names
from areal.utils.launcher import (
    JobException, JobState, get_env_vars, validate_config_for_distributed_launcher,
    wait_llm_server_addrs,
)
from areal.utils.ray import get_placement_group_master_ip_and_port

logger = logging.getLogger("RayLauncher")

RAY_WAIT_CHECK_TIME_INTERVAL = 5
DEFAULT_MAIN_FUNC_NAME = "main"
RAY_LAUNCHER = None
RECOVER_TIME_INTERVAL = 10


def _assert_abs(p: str, what: str):
    if not os.path.isabs(p):
        raise ValueError(f"{what} must be absolute: {p}")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{what} not found: {p}")


def run_func(file_path, func_name, *args, **kwargs):
    """Load a module by absolute path and run func_name."""
    file_path = os.path.abspath(file_path)
    _assert_abs(file_path, "Entry file")

    module_name = file_path.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, func_name)(*args, **kwargs)


class RayLauncher:
    def __init__(self, exp_name, trial_name, fileroot):
        self.exp_name = exp_name
        self.trial_name = trial_name
        self.fileroot = fileroot
        self.jobs = {}
        self.placement_groups = {}

    def submit(
        self, job_name, file_path, func_name, args,
        gpus, cpus, mem, env_vars=None,
        placement_group=None, bundle_index=-1, kwargs=None
    ):
        kwargs = kwargs or {}
        _assert_abs(file_path, f"{job_name} entry")
        runtime_env = RuntimeEnv(env_vars=env_vars or {})
        scheduling = (
            PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_index,
                placement_group_capture_child_tasks=True,
            ) if placement_group else "DEFAULT"
        )
        future = ray.remote(
            num_cpus=cpus, num_gpus=gpus,
            memory=mem * 1024 * 1024,
            runtime_env=runtime_env,
            scheduling_strategy=scheduling,
        )(run_func).remote(file_path, func_name, *args, **kwargs)
        self.jobs[job_name] = future
        return future

    def submit_array(
        self, job_name, file_path, func_name, count, nodes,
        list_args, gpus_per_task, cpus_per_task, mem_per_task,
        list_kwargs=None, env_vars=None, env_hook=None,
    ):
        list_kwargs = list_kwargs or [{} for _ in range(count)]
        _assert_abs(file_path, f"{job_name} entry")
        tasks_per_node = count // nodes
        gpus_per_node = gpus_per_task * tasks_per_node
        cpus_per_node = cpus_per_task * tasks_per_node
        mem_per_node = mem_per_task * tasks_per_node

        if job_name not in self.placement_groups:
            bundles = [
                {"CPU": cpus_per_node, "GPU": gpus_per_node, "memory": mem_per_node * 1024 * 1024}
            ] * nodes
            pg = ray.util.placement_group(bundles=bundles, strategy="PACK")
            ray.get(pg.ready(), timeout=60)
            self.placement_groups[job_name] = pg
        else:
            pg = self.placement_groups[job_name]

        extra_envs = env_hook(pg) if env_hook else [{} for _ in range(count)]
        futures = []
        for i in range(count):
            env = dict(env_vars or {})
            env.update(extra_envs[i])
            futures.append(
                self.submit(
                    f"{job_name}:{i}", file_path, func_name, list_args[i],
                    gpus_per_task, cpus_per_task, mem_per_task,
                    env_vars=env, placement_group=pg, bundle_index=i // tasks_per_node,
                    kwargs=list_kwargs[i],
                )
            )
        return futures

    def stop_all(self, force=False, pattern=None):
        for name, fut in list(self.jobs.items()):
            if not pattern or re.search(pattern, name):
                try:
                    ray.cancel(fut, force=force)
                except Exception as e:
                    logger.error(f"Cancel {name} failed: {e}")
                self.jobs.pop(name, None)

    def wait(self):
        while self.jobs:
            finished = []
            for name, fut in list(self.jobs.items()):
                try:
                    ray.get(fut, timeout=0.2)
                    finished.append(name)
                except ray.exceptions.GetTimeoutError:
                    continue
                except ray.exceptions.RayTaskError as e:
                    logger.error(f"{name} failed: {e}")
                    finished.append(name)
            for f in finished:
                self.jobs.pop(f, None)
            time.sleep(RAY_WAIT_CHECK_TIME_INTERVAL)


def main():
    ray.init()
    cfg, _ = parse_cli_args(sys.argv[1:])
    ray_main(cfg)


def ray_main(cfg, run_id=0):
    cfg.launcher = to_structured_cfg(cfg.launcher, LauncherConfig)
    cfg.cluster = to_structured_cfg(cfg.cluster, ClusterSpecConfig)
    cfg.recover = to_structured_cfg(cfg.recover, RecoverConfig)
    validate_config_for_distributed_launcher(cfg)

    name_resolve.reconfigure(cfg.cluster.name_resolve)
    name_resolve.clear_subtree(names.trial_root(cfg.experiment_name, cfg.trial_name))

    launcher = RayLauncher(cfg.experiment_name, cfg.trial_name, cfg.cluster.fileroot)
    alloc = AllocationMode.from_str(cfg.allocation_mode)

    # From CLI: trainer_entry, sglang_entry, vllm_entry passed explicitly
    trainer_entry = os.environ.get("AREAL_TRAINER_ENTRY")
    sglang_entry = os.environ.get("AREAL_SGLANG_ENTRY")
    vllm_entry = os.environ.get("AREAL_VLLM_ENTRY")
    if not trainer_entry:
        trainer_entry = sys.argv[1]
    _assert_abs(trainer_entry, "Trainer entry")

    n_nodes = cfg.cluster.n_nodes
    n_gpus_per_node = cfg.cluster.n_gpus_per_node

    # -------------------- LLM servers --------------------
    sglang_addrs, vllm_addrs = [], []
    n_sglang_nodes = n_vllm_nodes = 0

    if alloc.gen_backend == "sglang":
        cfg.sglang = to_structured_cfg(cfg.sglang, SGLangConfig)
        n_sglang_servers = alloc.gen.dp_size
        n_sglang_nodes = alloc.gen.world_size // n_gpus_per_node
        node_group_size = max(1, alloc.gen_instance_size // n_gpus_per_node)
        cross_nodes = alloc.gen_instance_size > n_gpus_per_node

        base_seed = cfg.sglang.random_seed
        sglang_args = [
            [[*sys.argv[1:], f"sglang.random_seed={base_seed + i}"]] for i in range(n_sglang_nodes)
        ]
        _assert_abs(sglang_entry, "SGLang entry")

        def sglang_env(pg: PlacementGroup) -> List[Dict]:
            addrs, ports = [], []
            for i in range(0, n_sglang_nodes):
                host, port = get_placement_group_master_ip_and_port(pg, i)
                addrs.append(host)
                ports.append(port)
            envs = []
            for i in range(n_sglang_nodes):
                envs.append(dict(
                    AREAL_SGLANG_MULTI_NODE_RANK=str(i),
                    AREAL_SGLANG_MULTI_NODE_MASTER_ADDR=addrs[0],
                    AREAL_SGLANG_MULTI_NODE_MASTER_PORT=str(ports[0]),
                ))
            return envs

        launcher.submit_array(
            "llm_server", sglang_entry, DEFAULT_MAIN_FUNC_NAME,
            count=n_sglang_nodes, nodes=n_sglang_nodes, list_args=sglang_args,
            gpus_per_task=n_gpus_per_node,
            cpus_per_task=cfg.launcher.inference_server_cpus_per_gpu * n_gpus_per_node,
            mem_per_task=cfg.launcher.inference_server_mem_per_gpu * n_gpus_per_node,
            env_vars=get_env_vars(cfg.cluster.cluster_name, cfg.launcher.inference_server_env_vars),
            env_hook=sglang_env if cross_nodes else None,
        )
        sglang_addrs = wait_llm_server_addrs(cfg.experiment_name, cfg.trial_name, n_sglang_servers)

    elif alloc.gen_backend == "vllm":
        cfg.vllm = to_structured_cfg(cfg.vllm, vLLMConfig)
        _assert_abs(vllm_entry, "vLLM entry")
        vllm_tp = alloc.gen.tp_size
        n_vllm_servers = alloc.gen.dp_size
        n_vllm_nodes = alloc.gen.world_size // n_gpus_per_node
        base_seed = cfg.vllm.seed
        vllm_args = [[[*sys.argv[1:], f"vllm.seed={base_seed + i}"]] for i in range(n_vllm_servers)]

        launcher.submit_array(
            "llm_server", vllm_entry, DEFAULT_MAIN_FUNC_NAME,
            count=n_vllm_servers, nodes=n_vllm_nodes, list_args=vllm_args,
            gpus_per_task=vllm_tp,
            cpus_per_task=cfg.launcher.inference_server_cpus_per_gpu * vllm_tp,
            mem_per_task=cfg.launcher.inference_server_mem_per_gpu * vllm_tp,
            env_vars=get_env_vars(cfg.cluster.cluster_name, cfg.launcher.inference_server_env_vars),
        )
        vllm_addrs = wait_llm_server_addrs(cfg.experiment_name, cfg.trial_name, n_vllm_servers)

    # -------------------- Trainers --------------------
    if alloc.type_ == AllocationType.DECOUPLED_EVAL:
        trainer_n_nodes = 1
        gpus_per_task = 0
    else:
        trainer_n_nodes = n_nodes - (n_sglang_nodes if alloc.gen_backend == "sglang" else n_vllm_nodes)
        gpus_per_task = 1

    trainer_args = [[sys.argv[2:]] for _ in range(trainer_n_nodes * cfg.cluster.n_gpus_per_node)]

    if alloc.type_ != AllocationType.LLM_SERVER_ONLY:
        llm_addrs = sglang_addrs if alloc.gen_backend == "sglang" else vllm_addrs

        def torch_env(pg: PlacementGroup) -> List[Dict]:
            host, port = get_placement_group_master_ip_and_port(pg)
            envs = []
            for i in range(trainer_n_nodes * cfg.cluster.n_gpus_per_node):
                envs.append(dict(
                    RANK=str(i), WORLD_SIZE=str(trainer_n_nodes * cfg.cluster.n_gpus_per_node),
                    LOCAL_RANK="0", MASTER_ADDR=host, MASTER_PORT=str(port),
                ))
            return envs

        env_base = dict(
            AREAL_LLM_SERVER_ADDRS=",".join(llm_addrs),
            AREAL_RECOVER_RUN=str(int(run_id > 0)),
        )
        if alloc.gen_backend == "sglang":
            env_base["NCCL_CUMEM_ENABLE"] = "0"
            env_base["NCCL_NVLS_ENABLE"] = "0"

        launcher.submit_array(
            "trainer", trainer_entry, DEFAULT_MAIN_FUNC_NAME,
            count=trainer_n_nodes * cfg.cluster.n_gpus_per_node,
            nodes=trainer_n_nodes, list_args=trainer_args,
            gpus_per_task=gpus_per_task,
            cpus_per_task=cfg.launcher.trainer_cpus_per_gpu,
            mem_per_task=cfg.launcher.trainer_mem_per_gpu,
            env_vars=dict(**get_env_vars(cfg.cluster.cluster_name, cfg.launcher.trainer_env_vars), **env_base),
            env_hook=torch_env,
        )

    launcher.wait()


if __name__ == "__main__":
    # Example:
    # AREAL_SGLANG_ENTRY=/abs/path/sglang_server.py \
    # AREAL_VLLM_ENTRY=/abs/path/vllm_server.py \
    # python -m areal.launcher.ray_abs \
    #   /abs/path/train_v1.py \
    #   --config /abs/path/viewsuite_active_explore_7b_ray.yaml
    main()
