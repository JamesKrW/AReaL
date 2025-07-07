# Copyright 2025 Ant Group Inc.

import asyncio
from datetime import datetime
from typing import List
import torch
from realhf.api.core.agent_api import Agent, register_agent
from realhf.api.core.data_api import SequenceSample, load_hf_tokenizer
from realhf.api.core.env_api import EnvironmentService
from realhf.api.core.model_api import BundledGenerationOutputs

class MultiTurnAgent(Agent):
    def __init__(self, gconfig, tokenizer_path, num_turns=20):
        self.gconfig = gconfig.new(n=1)
        self.tokenizer = load_hf_tokenizer(tokenizer_path)
        self.num_turns = num_turns

    async def collect_trajectory(
        self,
        prompt: SequenceSample,
        env: EnvironmentService,
        obs_queue: asyncio.Queue,
        act_queue: asyncio.Queue,
    ) -> List[SequenceSample]:
        # Reset environment and get initial observation
        obs, _ = await env.reset()
        assert isinstance(obs, str), f"Environment must return string obs, got {type(obs)}"
        
        # Generate a unique qid for this trajectory
        qid = f"trajectory_{int(datetime.now().timestamp() * 1000)}"
        birth_time = int(datetime.now().timestamp() * 1000)
        
        # Tokenize the initial observation
        token_ids = self.tokenizer.encode(obs, add_special_tokens=True)
        
        all_rewards = []
        x = dict(
            keys=[
                "packed_input_ids",
                "prompt_mask",
                "packed_logprobs",
                "seq_no_eos_mask",
                "packed_prompts",
                "rewards",
                "birth_time",
            ],
            ids=[qid],
            dtypes=dict(
                packed_prompts=torch.long,
                packed_input_ids=torch.long,
                prompt_mask=torch.bool,
                seq_no_eos_mask=torch.bool,
                packed_logprobs=torch.float32,
                rewards=torch.float32,
                birth_time=torch.long,
            ),
            trailing_shapes=dict(
                packed_input_ids=(),
                prompt_mask=(),
                seq_no_eos_mask=(),
                packed_prompts=(),
                packed_logprobs=(),
                rewards=(),
                birth_time=(),
            ),
            seqlens=dict(
                packed_input_ids=[[]],
                packed_logprobs=[[]],
                packed_prompts=[[len(token_ids)]],
                prompt_mask=[[]],
                seq_no_eos_mask=[[1 for _ in range(self.num_turns)]],
                rewards=[[1 for _ in range(self.num_turns)]],
                birth_time=[[1]],
            ),
            data=dict(
                packed_prompts=token_ids,
                packed_logprobs=[],
                packed_input_ids=[],
                seq_no_eos_mask=[],
                rewards=[],
                birth_time=torch.tensor([birth_time], dtype=torch.long),
                prompt_mask=[],
            ),
        )
        
        current_obs = obs
        for turn in range(self.num_turns):
            # Send current observation to model for generation
            await obs_queue.put((qid, token_ids, self.gconfig))
            act: BundledGenerationOutputs = await act_queue.get()
            
            # Decode the generated response
            seq_strs = self.tokenizer.batch_decode(
                act.seqs,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
            prompt_str = self.tokenizer.batch_decode(
                [act.prompt_ids],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )[0]
            
            # Extract the response (everything after the prompt)
            response = seq_strs[0][len(prompt_str):].strip()
            
            # Send response to environment
            obs, reward, done, info = await env.step((qid, [response]))
            all_rewards.append(reward)
            
            # Record the generation data
            x["data"]["packed_input_ids"].extend(list(act.seqs[0]))
            x["data"]["packed_logprobs"].extend(list(act.logprobs[0]))
            x["data"]["seq_no_eos_mask"].append(act.no_eos[0])
            x["data"]["prompt_mask"].extend(
                [1] * act.prompt_len + [0] * (act.seqlens[0] - act.prompt_len)
            )
            x["seqlens"]["packed_input_ids"][0].append(act.seqlens[0])
            x["seqlens"]["packed_logprobs"][0].append(act.seqlens[0] - 1)
            x["seqlens"]["prompt_mask"][0].append(act.seqlens[0])
            x["data"]["rewards"].append(reward)
            
            if done:
                break
                
            # Update token_ids for next turn (use the full sequence)
            token_ids = list(act.seqs[0])
            current_obs = obs
        
        # Convert all data to tensors
        for k in x["keys"]:
            if not isinstance(x["data"][k], torch.Tensor):
                x["data"][k] = torch.tensor(x["data"][k], dtype=x["dtypes"][k])
        
        x = SequenceSample(**x)
        return [x]

    def log_rewards_to_file(
        self,
        qid: str,
        prompt: str,
        prompt_len: int,
        answers: List[str],
        seqlens: List[int],
        rewards: List[float],
        success: List[bool],
        version_starts: List[int],
        version_ends: List[int],
    ):
        group_size = len(answers)

        for group_idx in range(group_size):
            # NOTE: we can ensure that only one process is logging this query id
            gen_file_path = os.path.join(
                self.answer_save_path,
                str(version_starts[group_idx]),
                f"{qid}.txt",
            )
            os.makedirs(os.path.dirname(gen_file_path), exist_ok=True)

            version_start = version_starts[group_idx]
            version_end = version_ends[group_idx]
            reward = rewards[group_idx]
            answer = answers[group_idx]
            seqlen = seqlens[group_idx]
            with open(gen_file_path, "a") as _f:
                info = "\n".join(
                    [
                        f"idx: {group_idx + 1} / {group_size}, seqlen: {seqlen}, "
                        f"head version: {version_start}, tail version: {version_end}.",
                        f"reward is {reward}, prompt is {colorama.Fore.YELLOW + colorama.Style.DIM}{prompt}{colorama.Style.RESET_ALL}",
                        f"sequence is: {colorama.Fore.YELLOW + colorama.Style.DIM}{answer}{colorama.Style.RESET_ALL}.",
                    ]
                )
                _f.write(info + "\n")

            train_pass_monitor_file_path = os.path.join(
                self.answer_save_path,
                str(version_starts[group_idx]),
                f"{qid}.jsonl",
            )
            os.makedirs(os.path.dirname(train_pass_monitor_file_path), exist_ok=True)

            with open(train_pass_monitor_file_path, "a") as monitor_file:
                monitor_file.write(
                    json.dumps(
                        {
                            "version_start": int(version_start),
                            "version_end": int(version_end),
                            "success": bool(success),
                            "prompt_len": prompt_len,
                            "answer_len": seqlen - prompt_len,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
register_agent("multi-turn-agent", MultiTurnAgent) 