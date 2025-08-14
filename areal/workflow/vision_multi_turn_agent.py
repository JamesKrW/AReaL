# vision_multi_turn_agent_env_workflow.py
import asyncio
import os
import uuid
from typing import Dict, List, Any, Tuple

import colorama
import torch
from tensordict import TensorDict
from transformers import AutoProcessor, PreTrainedTokenizerFast
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import VLMRequest,LLMRequest
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data_pad import concat_padded_tensors
from areal.utils.image import image2base64
from realhf.base import logging
from areal.dataset.clevr_count_70k import convert_image
from areal.envs.utils.env_load_utils import load_env_from_registry  # Unified env loading

logger = logging.getLogger("Vision Multi-Turn AgentEnv workflow")

def _count_img_segments(text: str, processor: AutoProcessor) -> int:
    import re
    iproc = getattr(processor, "image_processor", None)
    if iproc and hasattr(iproc, "image_processor_type") and "qwen" in str(iproc.image_processor_type).lower():
        return len(re.findall(r"<\|vision_start\|>", text))
    image_tok = getattr(processor, "image_token", "<image>")
    return text.count(image_tok)


def processor_output(
    processor: AutoProcessor,
    text_with_placeholders: str,
    new_images: List[Image.Image],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    
    if not new_images:
        return None, None
    k = _count_img_segments(text_with_placeholders, processor)
    if k != len(new_images):
        raise ValueError(f"mismatch between place holders: {k} and images: {len(new_images)}")
    out = processor(text=text_with_placeholders, images=new_images, padding=False, return_tensors="pt")
    return out


def extract_images_from_multimodal_data(
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

def get_text_with_image_token(text_content: str, image_placeholder, processor) -> str:
    if "qwen" in processor.image_processor.image_processor_type.lower():
        image_token = "<|vision_start|><|image_pad|><|vision_end|>"
    else:
        image_token = processor.image_token if processor is not None else "<image>"
    return text_content.replace(image_placeholder, image_token)
    

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


    async def _run_one_episode(self, engine: InferenceEngine, data: dict, rid: str) -> TensorDict:
        # ensure env can close
        env, seed = load_env_from_registry(data)
        try:
            init_obs, _ = env.reset(seed=seed)
            sys_prompt = env.get_system_prompt()

            # Running media state (all images encountered so far, in order)
            all_images: List[Image.Image] = []

            # Rollout accumulators (we will append deltas only)
            input_ids: List[int] = []     # running prompt ids (what we send to the engine)
            logprobs:  List[float] = []   # per-token logprobs (assistant tokens; user tokens are zeros)
            loss_mask: List[int] = []     # 1 for assistant tokens to train; 0 otherwise
            versions:  List[int] = []     # fill with -1
            cumulative_reward = 0.0
            t = 0

            # Visual segments (only for newly added images per user turn)
            pv_segs: List[torch.Tensor] = []
            thw_segs: List[torch.Tensor] = []

            # Helper: processor encode to ids with optional images
            def _proc_ids(text: str, images: List[Image.Image]) -> List[int]:
                out = self.processor(text=text, images=images if images else None,
                                    padding=False, return_tensors="pt")
                return out["input_ids"][0].tolist()

            # Helper: get visual tensors only for new images of THIS user turn
            def _proc_visual_for_new_imgs(text_with_placeholders: str,
                                        new_images: List[Image.Image]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
                if not new_images:
                    return None, None
                out = self.processor(text=text_with_placeholders, images=new_images,
                                    padding=False, return_tensors="pt")
                pv  = out.get("pixel_values", None)
                thw = out.get("image_grid_thw", None)
                return pv, thw

            # -------------------- Initialize messages and initial user delta --------------------
            messages = [
                {"role": "system", "content": get_text_with_image_token(sys_prompt, self.image_placeholder, self.processor)},
                {"role":"user","content":get_text_with_image_token(init_obs["obs_str"], self.image_placeholder, self.processor)}
            ]

            
            init_new_imgs  = extract_images_from_multimodal_data(init_obs.get("multi_modal_data", {}), self.image_placeholder)

            
            curr_prompt_text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            curr_ids = _proc_ids(curr_prompt_text, all_images + init_new_imgs)

            # Delta for the initial USER segment
            
            input_ids.extend(curr_ids)
            logprobs.extend([0.0] * len(curr_ids))   # user segment: no logprobs
            loss_mask.extend([0] * len(curr_ids))    # user segment: not trained
            versions.extend([-1] * len(curr_ids))

            # Record visual tensors only for the newly added images of this turn
            if init_new_imgs:
                pv, thw = _proc_visual_for_new_imgs(curr_prompt_text, init_new_imgs)
                if pv is not None:
                    pv_segs.append(pv)
                if thw is not None:
                    thw_segs.append(thw)
                # Now these images become part of the running media state
                all_images.extend(init_new_imgs)

            # -------------------- Main loop: assistant -> env -> next user (delta) --------------------
            while t < self.max_turns:
                # ASSISTANT generation: call engine with current incremental prompt_ids and all_images
                img_b64 = image2base64(all_images) if len(all_images) > 0 else None
                if img_b64 is not None:
                    req = VLMRequest(
                        rid=rid,
                        input_ids=input_ids,     # incremental; DO NOT re-encode the whole prompt
                        image_data=img_b64,
                        gconfig=self.gconfig.new(n_samples=1),
                    )
                else:
                    req = LLMRequest(
                        rid=rid,
                        input_ids=input_ids,
                        gconfig=self.gconfig.new(n_samples=1),
                    )

                resp = await engine.agenerate(req)
               
                
                # Append assistant outputs
                input_ids+=resp.output_tokens
                logprobs+=resp.output_logprobs
                loss_mask+=[1] * len(resp.output_tokens) 
                versions+=resp.output_versions
                
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None and (len(resp.output_tokens) == 0 or input_ids[-1] != eos_id):
                    input_ids.append(eos_id)
                    logprobs.append(0.0)
                    loss_mask.append(0)
                    versions.append(-1)

                # Decode assistant text for env step
                assistant_text = self.tokenizer.decode(resp.output_tokens, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": assistant_text})

                # Step environment
                next_obs, r, done, _ = env.step(assistant_text)
                cumulative_reward += float(r)
                if done:
                    break

                # NEXT USER turn: build delta (add only new text part)
                next_user_text = get_text_with_image_token(next_obs["obs_str"], self.image_placeholder, self.processor)
                new_imgs = extract_images_from_multimodal_data(next_obs.get("multi_modal_data", {}), self.image_placeholder)

                # prev prompt before adding the new user (gen prompt ON)
                prev_len = len(input_ids)  # the length of the prompt we actually sent/hold
                prev_prompt_text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # (Optional sanity) The ids below should match input_ids; we avoid recomputation to keep it incremental.

                # Append the new user to messages and compute current prompt ids (locally) to take delta
                messages.append({"role": "user", "content": next_user_text})
                curr_prompt_text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                curr_ids = _proc_ids(curr_prompt_text, all_images + new_imgs)

                # Delta is the suffix beyond the previously materialized prompt length
                delta_ids = curr_ids[prev_len:]
                input_ids += delta_ids
                logprobs += [0.0] * len(delta_ids)  # user segment: no logprobs
                loss_mask += [0] * len(delta_ids)   # user segment: not trained
                versions += [-1] * len(delta_ids)

                # Visual tensors only for newly added images of this user turn
                if new_imgs:
                    pv, thw = _proc_visual_for_new_imgs(next_user_text, new_imgs)
                    if pv is not None:
                        pv_segs.append(pv)
                    if thw is not None:
                        thw_segs.append(thw)
                    all_images.extend(new_imgs)

                t += 1

            # -------------------- Pack results --------------------
            L = len(input_ids)
            res: Dict[str, torch.Tensor] = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.ones(1, L, dtype=torch.bool),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.long).unsqueeze(0),
                "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
                "versions": torch.tensor(versions, dtype=torch.long).unsqueeze(0),
                "rewards": torch.tensor(float(cumulative_reward)).unsqueeze(0),
            }

            if len(pv_segs) > 0:
                # per-turn vision is concatenated on dim=0 → add batch on dim=0
                res["pixel_values"] = torch.cat(pv_segs, dim=0).unsqueeze(0)      # (1, sum_tokens_over_new_imgs, D)
            if len(thw_segs) > 0:
                res["image_grid_thw"] = torch.cat(thw_segs, dim=0).unsqueeze(0)   # (1, sum_new_images, 3)

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