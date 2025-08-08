# multi_turn_agent_env_workflow.py
import asyncio
import os
import uuid

import colorama
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import LLMRequest
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors
from realhf.base import logging

from areal.envs.utils.env_load_utils import load_env_from_registry  # Unified env loading

logger = logging.getLogger("Multi-Turn AgentEnv workflow")


class MultiTurnAgentEnvWorkflow(RolloutWorkflow):
    """
    Multi-turn workflow driven by an environment.

    - The environment provides the system prompt and initial observation (text mode only).
    - The LLM generates an action string; env.step(...) returns the next observation, reward, and done flag.
    - Rewards are taken directly from the environment (no discounting).
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int,
        dump_dir: str | None = None,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def _run_one_episode(self, engine: InferenceEngine, data: dict, rid: str):
        """
        Run a single environment episode with multi-turn LLM interaction.

        Args:
            data: Example:
                {'name': 'sokoban', 'seed': 691527629, 'config': {...}}
            rid: Unique run ID for logging/tracking.

        Returns:
            Tuple:
                - TensorDict rollout data
                - Prompt string
                - Final completion string
                - Cumulative reward
                - Sequence length
        """
        env, seed = load_env_from_registry(data)

        # Rollout accumulators
        seq, logprobs, loss_mask, versions = [], [], [], []
        cumulative_reward = 0.0
        t = 0

        try:
            # Reset environment and get initial observation
            init_obs, _ = env.reset(seed=seed)
            if "multi_modal_data" in init_obs and init_obs["multi_modal_data"] is not None:
                raise NotImplementedError("This workflow currently supports text-mode envs only.")

            sys_prompt = env.get_system_prompt()
            obs_str = init_obs["obs_str"]

            # Conversation context: system + first observation as user
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": obs_str},
            ]

            done = False
            last_completion_text = ""
            input_ids = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            while not done and t < self.max_turns:
                # Tokenize current conversation and generate assistant reply

                req = LLMRequest(
                    rid=rid,
                    input_ids=input_ids,
                    gconfig=self.gconfig.new(n_samples=1),
                )
                resp = await engine.agenerate(req)

                # Decode model output into an action string
                completion_text = self.tokenizer.decode(resp.output_tokens, skip_special_tokens=True)
                last_completion_text = completion_text

                # Append token-level info to rollout arrays
                input_len = len(resp.input_tokens) - len(seq)
                assert len(seq) == 0 or resp.input_tokens[:-input_len] == seq, (
                    seq,
                    resp.input_tokens[:-input_len],
                    len(seq),
                    len(resp.input_tokens[:-input_len]),
                )
                seq += resp.input_tokens[-input_len:] + resp.output_tokens
                logprobs += [0.0] * input_len + resp.output_logprobs
                loss_mask += [0] * input_len + [1] * resp.output_len
                versions += [-1] * input_len + resp.output_versions

                # Step the environment with the generated action string
                next_obs, step_reward, done, info = env.step(completion_text)
                cumulative_reward += float(step_reward)


                if done:
                    break
                input_ids+=resp.output_tokens
                obs_str = next_obs["obs_str"]
                new_messages = [{"role": "assistant", "content": "some random message."}]
                s1 = self.tokenizer.apply_chat_template(new_messages, tokenize=True)
                new_messages += [
                    {
                        "role": "user",
                        "content": obs_str,
                    }
                ]
                s2 = self.tokenizer.apply_chat_template(
                    new_messages, tokenize=True, add_generation_prompt=True
                )
                input_ids += s2[len(s1) :]
                if input_ids[-1] != self.tokenizer.eos_token_id:
                    input_ids += [self.tokenizer.eos_token_id]
                t += 1

            # Pack rollout data into a TensorDict
            res = dict(
                input_ids=torch.tensor(seq),
                logprobs=torch.tensor(logprobs),
                loss_mask=torch.tensor(loss_mask),
                versions=torch.tensor(versions),
                rewards=torch.tensor(float(cumulative_reward)),
                attention_mask=torch.ones(len(seq), dtype=torch.bool),
            )
            res = {k: v.unsqueeze(0) for k, v in res.items()}

            # Reconstruct prompt string and last completion for logging
            total_str = self.tokenizer.decode(
                input_ids
            )

            return (
                TensorDict(res, batch_size=[1]),
                total_str,
                cumulative_reward,
                len(seq),
            )
        finally:
            try:
                env.close()
            except Exception:
                pass

    async def arun_episode(self, engine: InferenceEngine, data: dict):
        """
        Public API to run one episode with potentially multiple samples.
        """
        rid = uuid.uuid4().hex
        tasks = [
            self._run_one_episode(engine, data, rid)
            for _ in range(self.gconfig.n_samples)
        ]
        results = await asyncio.gather(*tasks)

        # Optional dump to disk
        if self.dump_dir is not None:
            version = engine.get_version()
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex
            with open(os.path.join(self.dump_dir, str(version), f"{qid}.txt"), "a") as f:
                n_samples = self.gconfig.n_samples
                for i, (_, total_str, cumulative_reward, len_seq) in enumerate(results):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {len_seq}, cumulative reward: {cumulative_reward}.",
                            f"Total string is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{total_str}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    f.write(info + "\n")

        td_list = [res[0] for res in results]
        return concat_padded_tensors(td_list)
