# Copyright 2025 Ant Group Inc.

import asyncio
import uuid
from datetime import datetime
from typing import List
import torch
from realhf.api.core.agent_api import Agent, register_agent
from realhf.api.core.data_api import SequenceSample, load_hf_tokenizer
from realhf.api.core.env_api import EnvironmentService
from realhf.api.core.model_api import BundledGenerationOutputs

class MultiTurnAgent(Agent):
    def __init__(self, gconfig, tokenizer_path, num_turns=20):
        print(f"🔧 MultiTurnAgent.__init__: gconfig={gconfig}, tokenizer_path={tokenizer_path}, num_turns={num_turns}")
        self.gconfig = gconfig.new(n=1)
        print(f"🔧 MultiTurnAgent.__init__: Created gconfig with n=1: {self.gconfig}")
        self.tokenizer = load_hf_tokenizer(tokenizer_path)
        print(f"🔧 MultiTurnAgent.__init__: Loaded tokenizer from {tokenizer_path}")
        self.num_turns = num_turns
        print(f"✅ MultiTurnAgent initialized successfully!")

    async def collect_trajectory(
        self,
        prompt: SequenceSample,
        env: EnvironmentService,
        obs_queue: asyncio.Queue,
        act_queue: asyncio.Queue,
    ) -> List[SequenceSample]:
        print(f"🚀 collect_trajectory: Starting trajectory collection")
        print(f"🚀 collect_trajectory: prompt={prompt}, env={env}")
        print(f"🚀 collect_trajectory: obs_queue={obs_queue}, act_queue={act_queue}")
        
        # Reset environment and get initial observation
        print(f"🌍 collect_trajectory: Calling env.reset()...")
        obs, _ = await env.reset()
        print(f"🌍 collect_trajectory: Got observation from env.reset(): {obs[:100]}..." if isinstance(obs, str) and len(obs) > 100 else f"🌍 collect_trajectory: Got observation: {obs}")
        assert isinstance(obs, str), f"Environment must return string obs, got {type(obs)}"
        
        # Generate a unique qid for this trajectory using UUID + timestamp
        base_qid = f"traj_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp() * 1000)}"
        birth_time = int(datetime.now().timestamp() * 1000)
        print(f"🆔 collect_trajectory: Generated base qid={base_qid}, birth_time={birth_time}")
        
        # Tokenize the initial observation
        print(f"🔤 collect_trajectory: Tokenizing observation...")
        print(f"🔤 collect_trajectory: obs type: {type(obs)}, obs length: {len(obs)}")
        print(f"🔤 collect_trajectory: obs content preview (first 500 chars): {repr(obs[:500])}")
        print(f"🔤 collect_trajectory: obs content preview (last 500 chars): {repr(obs[-500:])}")
        
        token_ids = self.tokenizer.encode(obs, add_special_tokens=True)
        print(f"🔤 collect_trajectory: Tokenized to {len(token_ids)} tokens: {token_ids[:20]}..." if len(token_ids) > 20 else f"🔤 collect_trajectory: Tokenized to {len(token_ids)} tokens: {token_ids}")
        
        all_rewards = []
        print(f"📊 collect_trajectory: Initializing data structure...")
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
            ids=[base_qid],
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
        print(f"📊 collect_trajectory: Data structure initialized with {len(x['keys'])} keys")
        
        current_obs = obs
        print(f"🔄 collect_trajectory: Starting {self.num_turns} turns loop")
        for turn in range(self.num_turns):
            print(f"🔄 collect_trajectory: ===== TURN {turn + 1}/{self.num_turns} =====")
            print(f"🔄 collect_trajectory: Current obs length: {len(current_obs) if isinstance(current_obs, str) else 'N/A'}")
            
            # Create a unique qid for this turn
            qid = f"{base_qid}"
            print(f"🆔 collect_trajectory: Using qid={qid} for turn {turn + 1}")
            
            # Send current observation to model for generation
            print(f"📤 collect_trajectory: Putting data to obs_queue (qid={qid}, tokens={len(token_ids)}, gconfig={self.gconfig})")
            await obs_queue.put((qid, token_ids, self.gconfig))
            print(f"📤 collect_trajectory: Successfully put data to obs_queue")
            
            print(f"📥 collect_trajectory: Waiting for response from act_queue...")
            try:
                # Wait for response with timeout
                act: BundledGenerationOutputs = await asyncio.wait_for(act_queue.get(), timeout=300.0)
                print(f"📥 collect_trajectory: Got response from act_queue: {type(act)}")
                print(f"📥 collect_trajectory: Response details - seqs shape: {act.seqs.shape if hasattr(act.seqs, 'shape') else len(act.seqs)}")
                print(f"📥 collect_trajectory: Response details - logprobs shape: {act.logprobs.shape if hasattr(act.logprobs, 'shape') else len(act.logprobs)}")
            except asyncio.TimeoutError:
                print(f"⚠️ collect_trajectory: Timeout waiting for response from act_queue for qid={qid}")
                raise
            except Exception as e:
                print(f"❌ collect_trajectory: Error waiting for response: {e}")
                raise
            
            # Decode the generated response
            print(f"🔤 collect_trajectory: Decoding generated response...")
            seq_strs = self.tokenizer.batch_decode(
                act.seqs,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
            print(f"🔤 collect_trajectory: Decoded {len(seq_strs)} sequences")
            
            prompt_str = self.tokenizer.batch_decode(
                [act.prompt_ids],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )[0]
            print(f"🔤 collect_trajectory: Decoded prompt: {prompt_str[:100]}..." if len(prompt_str) > 100 else f"🔤 collect_trajectory: Decoded prompt: {prompt_str}")
            
            # Extract the response (everything after the prompt)
            response = seq_strs[0][len(prompt_str):].strip()
            print(f"🔤 collect_trajectory: Extracted response: {response[:100]}..." if len(response) > 100 else f"🔤 collect_trajectory: Extracted response: {response}")
            
            # Send response to environment
            print(f"🌍 collect_trajectory: Sending response to environment...")
            obs, reward, done, info = await env.step((qid, [response]))
            print(f"🌍 collect_trajectory: Got environment step result - reward={reward}, done={done}")
            print(f"🌍 collect_trajectory: New obs: {obs[:100]}..." if isinstance(obs, str) and len(obs) > 100 else f"🌍 collect_trajectory: New obs: {obs}")
            
            all_rewards.append(reward)
            print(f"🏆 collect_trajectory: Added reward {reward}, total rewards so far: {all_rewards}")
            
            # Record the generation data
            print(f"📊 collect_trajectory: Recording generation data...")
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
            print(f"📊 collect_trajectory: Generation data recorded for turn {turn + 1}")
            
            if done:
                print(f"🏁 collect_trajectory: Episode done after turn {turn + 1}")
                break
                
            # Update token_ids for next turn (use the full sequence)
            token_ids = list(act.seqs[0])
            current_obs = obs
            print(f"🔄 collect_trajectory: Updated token_ids for next turn, new length: {len(token_ids)}")
        
        print(f"🔄 collect_trajectory: Finished turns loop")
        
        # Convert all data to tensors
        print(f"🔧 collect_trajectory: Converting data to tensors...")
        for k in x["keys"]:
            if not isinstance(x["data"][k], torch.Tensor):
                print(f"🔧 collect_trajectory: Converting {k} to tensor (current type: {type(x['data'][k])})")
                x["data"][k] = torch.tensor(x["data"][k], dtype=x["dtypes"][k])
        
        print(f"🔧 collect_trajectory: Creating SequenceSample...")
        x = SequenceSample(**x)
        print(f"✅ collect_trajectory: Successfully created trajectory with {len(all_rewards)} rewards: {all_rewards}")
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