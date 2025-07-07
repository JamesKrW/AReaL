# Copyright 2025 Ant Group Inc.

import asyncio
from datetime import datetime
from typing import List
import torch
from realhf.api.core.agent_api import Agent, register_agent
from realhf.api.core.data_api import SequenceSample, load_hf_tokenizer
from realhf.api.core.env_api import EnvironmentService
from realhf.api.core.model_api import BundledGenerationOutputs

class SokobanTextMultiTurnAgent(Agent):
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
        await env.reset()
        assert prompt.bs == 1
        prompt_token_ids = prompt.data["packed_prompts"].cpu().numpy().tolist()
        qid = prompt.ids[0]
        birth_time = int(datetime.now().timestamp() * 1000)
        token_ids = prompt_token_ids
        all_rewards = []
        all_actions = []
        all_success = []
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
                packed_prompts=[[len(prompt_token_ids)]],
                prompt_mask=[[]],
                seq_no_eos_mask=[[1 for _ in range(self.num_turns)]],
                rewards=[[1 for _ in range(self.num_turns)]],
                birth_time=[[1]],
            ),
            data=dict(
                packed_prompts=list(prompt_token_ids),
                packed_logprobs=[],
                packed_input_ids=[],
                seq_no_eos_mask=[],
                rewards=[],
                birth_time=torch.tensor([birth_time], dtype=torch.long),
                prompt_mask=[],
            ),
        )
        for turn in range(self.num_turns):
            await obs_queue.put((qid, token_ids, self.gconfig))
            act: BundledGenerationOutputs = await act_queue.get()
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
            actions = [seq_str.split(prompt_str)[1].strip() for seq_str in seq_strs]
            all_actions.extend(actions)
            _, reward, done, info = await env.step((qid, actions))
            all_rewards.append(reward)
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
            # Feedback: always freethink, no hints, just continue
            token_ids = list(act.seqs[0])
        for k in x["keys"]:
            if not isinstance(x["data"][k], torch.Tensor):
                x["data"][k] = torch.tensor(x["data"][k], dtype=x["dtypes"][k])
        x = SequenceSample(**x)
        return [x]

register_agent("sokoban-text-multiturn", SokobanTextMultiTurnAgent) 