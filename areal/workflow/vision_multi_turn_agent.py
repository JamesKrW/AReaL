# vision_multi_turn_agent_env_workflow.py
import asyncio
import os
import uuid
from typing import Any, Dict, List

import colorama
import torch
from PIL import Image
from tensordict import TensorDict
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.workflow_api import RolloutWorkflow
from areal.dataset.clevr_count_70k import convert_image
from areal.envs.utils.env_load_utils import (
    load_env_from_registry,  # Unified env loading
)
from areal.utils.data import concat_padded_tensors
from areal.utils.image import image2base64
from realhf.base import logging

logger = logging.getLogger("Vision Multi-Turn AgentEnv workflow")


def convert_placeholder_to_image_token(
    text_content: str, image_placeholder, processor
) -> str:
    # replace image placeholder with <image> or <|vision_start|><|image_pad|><|vision_end|>
    if "qwen" in processor.image_processor.image_processor_type.lower():
        image_token = "<|vision_start|><|image_pad|><|vision_end|>"
    else:
        image_token = processor.image_token if processor is not None else "<image>"
    return text_content.replace(image_placeholder, image_token)


def get_images_from_multi_modal_data(
    multi_modal_data: Dict[str, Any],
    image_placeholder: str = "<image>",
) -> List[Image.Image]:
    """
    Extract a list of PIL images from env.obs['multi_modal_data'] for a given placeholder.
    """
    image_list = multi_modal_data[image_placeholder]
    images: List[Image.Image] = []
    for img in image_list:
        images.append(img if isinstance(img, Image.Image) else convert_image(img))
    return images


class VisionMultiTurnAgentEnvWorkflow(RolloutWorkflow):
    """
    Multi-turn workflow driven by an environment with vision support.

    - The environment provides the system prompt and initial observation (with multi-modal data).
    - The LLM generates an action string; env.step(...) returns the next observation, reward, and done flag.
    - Supports both text and images in observations.
    - obs_str contains placeholders (e.g., <image>) that indicate where images should be inserted.
    - multi_modal_data contains the actual image data corresponding to these placeholders.
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        processor: AutoProcessor,
        max_turns: int,
        image_placeholder: str = "<image>",
        dump_dir: str | None = None,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_turns = max_turns
        self.image_placeholder = image_placeholder
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def _run_one_episode(
        self, engine: InferenceEngine, data: dict, rid: str
    ) -> TensorDict:
        # ensure env can close
        env, seed = load_env_from_registry(data)
        try:
            init_obs, _ = env.reset(seed=seed)
            sys_prompt = env.get_system_prompt()

            all_images: List[Image.Image] = []

            # Rollout accumulators (we will append deltas only)
            input_ids: List[int] = []  # running prompt ids (what we send to the engine)
            logprobs: List[float] = (
                []
            )  # per-token logprobs (assistant tokens; user tokens are zeros)
            loss_mask: List[int] = []  # 1 for assistant tokens to train; 0 otherwise
            versions: List[int] = []  # fill with -1
            cumulative_reward = 0.0
            t = 0

            # Visual segments (only for newly added images per user turn)
            pv_segs: List[torch.Tensor] = []
            thw_segs: List[torch.Tensor] = []

            # -------------------- Initialize image and text --------------------
            new_messages = [
                {
                    "role": "system",
                    "content": convert_placeholder_to_image_token(
                        sys_prompt, self.image_placeholder, self.processor
                    ),
                },
                {
                    "role": "user",
                    "content": convert_placeholder_to_image_token(
                        init_obs["obs_str"], self.image_placeholder, self.processor
                    ),
                },
            ]

            new_images = get_images_from_multi_modal_data(
                init_obs.get("multi_modal_data", {}), self.image_placeholder
            )

            new_text = self.processor.tokenizer.apply_chat_template(
                tokenize=False, add_generation_prompt=True, conversation=new_messages
            )

            processed_input = self.processor(
                images=new_images if new_images else None,
                text=new_text,
                padding=False,
                return_tensors="pt",
            )

            cur_input_ids = processed_input["input_ids"].tolist()[0]
            input_ids.extend(cur_input_ids)
            logprobs.extend([0.0] * len(cur_input_ids))  # user segment: no logprobs
            loss_mask.extend([0] * len(cur_input_ids))  # user segment: not trained
            versions.extend([-1] * len(cur_input_ids))

            # Record visual tensors only for the newly added images of this turn
            if new_images:
                pv, thw = (
                    processed_input["pixel_values"],
                    processed_input["image_grid_thw"],
                )
                pv_segs.append(pv)
                thw_segs.append(thw)
                all_images.extend(new_images)

            fixed_message = {
                "role": "assistant",
                "content": "random messages",
            }
            fixed_token_id_len = len(
                self.processor.tokenizer.apply_chat_template(
                    tokenize=True,
                    add_generation_prompt=False,
                    conversation=[fixed_message],
                )
            )
            # -------------------- Main loop: assistant -> env -> next user (delta) --------------------
            while t < self.max_turns:
                # ASSISTANT generation: call engine with current incremental prompt_ids and all_images
                img_b64 = image2base64(all_images) if len(all_images) > 0 else []

                req = ModelRequest(
                    rid=rid,
                    input_ids=input_ids,
                    image_data=img_b64,
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                )

                resp = await engine.agenerate(req)

                # Append assistant outputs
                input_ids += resp.output_tokens
                logprobs += resp.output_logprobs
                loss_mask += [1] * len(resp.output_tokens)
                versions += resp.output_versions

                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None and (
                    len(resp.output_tokens) == 0 or input_ids[-1] != eos_id
                ):
                    input_ids.append(eos_id)
                    logprobs.append(0.0)
                    loss_mask.append(0)
                    versions.append(-1)

                # Decode assistant text for env step
                assistant_text = self.tokenizer.decode(
                    resp.output_tokens, skip_special_tokens=True
                )

                # Step environment
                next_obs, r, done, info = env.step(assistant_text)
                cumulative_reward += float(r)
                if done or t + 1 >= self.max_turns:
                    break

                new_messages = [
                    fixed_message,
                    {
                        "role": "user",
                        "content": convert_placeholder_to_image_token(
                            next_obs["obs_str"], self.image_placeholder, self.processor
                        ),
                    },
                ]

                new_images = get_images_from_multi_modal_data(
                    init_obs.get("multi_modal_data", {}), self.image_placeholder
                )

                new_text = self.processor.tokenizer.apply_chat_template(
                    tokenize=False,
                    add_generation_prompt=True,
                    conversation=new_messages,
                )

                processed_input = self.processor(
                    images=new_images if new_images else None,
                    text=new_text,
                    padding=False,
                    return_tensors="pt",
                )

                # Delta is the suffix beyond the previously materialized prompt length
                cur_input_ids = processed_input["input_ids"].tolist()[0][
                    fixed_token_id_len:
                ]
                input_ids += cur_input_ids
                logprobs += [0.0] * len(cur_input_ids)  # user segment: no logprobs
                loss_mask += [0] * len(cur_input_ids)  # user segment: not trained
                versions += [-1] * len(cur_input_ids)

                # Visual tensors only for newly added images of this user turn
                if new_images:
                    pv, thw = (
                        processed_input["pixel_values"],
                        processed_input["image_grid_thw"],
                    )
                    pv_segs.append(pv)
                    thw_segs.append(thw)
                    all_images.extend(new_images)

                t += 1

            # -------------------- Pack results --------------------
            L = len(input_ids)
            res = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.ones(1, L, dtype=torch.bool),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.long).unsqueeze(0),
                "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
                "versions": torch.tensor(versions, dtype=torch.long).unsqueeze(0),
                "rewards": torch.tensor(float(cumulative_reward)).unsqueeze(0),
            }
            multi_modal_input = {}
            if len(pv_segs) > 0:
                multi_modal_input["pixel_values"] = torch.cat(pv_segs, dim=0)
            if len(thw_segs) > 0:
                multi_modal_input["image_grid_thw"] = torch.cat(thw_segs, dim=0)
            if multi_modal_input:
                num_image_pad_tokens = (
                    (res["input_ids"] == self.processor.image_token_id).sum().item()
                )
                num_pixel_features = multi_modal_input["pixel_values"].shape[0] // (
                    self.processor.image_processor.merge_size**2
                )
                assert num_image_pad_tokens == num_pixel_features, (
                    f"Mismatch: input_ids has {num_image_pad_tokens} <|image_pad|> tokens, "
                    f"but pixel_values has {num_pixel_features} features"
                )
                res["multi_modal_input"] = [multi_modal_input]
            total_str = self.tokenizer.decode(input_ids)
            return (
                TensorDict(res, batch_size=[1]),
                total_str,
                cumulative_reward,
                len(input_ids),
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
            with open(
                os.path.join(self.dump_dir, str(version), f"{qid}.txt"), "a"
            ) as f:
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
