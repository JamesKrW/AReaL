# vision_multi_turn_agent_env_workflow.py (async GymImageEnv version; VLM+LLM compatible)
import asyncio
import os
import uuid
from typing import Any, Dict, List, Tuple, Optional

import colorama
import torch
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.workflow_api import RolloutWorkflow
from areal.dataset.clevr_count_70k import convert_image
from areal.utils.data import concat_padded_tensors
from areal.utils.image import image2base64
from realhf.base import logging
from view_suite.gym.gym_image_env import GymImageEnv
from areal.viewsuite.registry import REGISTERED_ENVS
import traceback
logger = logging.getLogger("Vision Multi-Turn AgentEnv workflow")


# ---------------------- Helpers ----------------------

def _is_vlm_processor(proc: Optional[AutoProcessor]) -> bool:
    """Heuristically determine if `proc` can handle images."""
    if proc is None:
        return False
    return hasattr(proc, "image_processor")

def convert_placeholder_to_image_token(
    text_content: str,
    image_placeholder: str,
    processor: Optional[AutoProcessor],
    for_vlm: bool,
) -> str:
    """
    Replace <image> placeholder based on whether we're running a VLM turn.
    - If VLM (images present & processor is vision-capable):
        - Qwen-style: <|vision_start|><|image_pad|><|vision_end|>
        - Else: processor.image_token if available else "<image>"
    - If LLM/text-only turn: normalize to harmless marker.
    """
    if for_vlm and _is_vlm_processor(processor):
        iproc = getattr(processor, "image_processor", None)
        if iproc and getattr(iproc, "image_processor_type", "").lower().find("qwen") >= 0:
            image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        else:
            image_token = getattr(processor, "image_token", "<image>")
    else:
        image_token = "[image]"
    return text_content.replace(image_placeholder, image_token)

def get_images_from_multi_modal_input(
    multi_modal_input: Optional[Dict[str, Any]],
    image_placeholder: str = "<image>",
) -> List[Image.Image]:
    """Extract PIL images from obs['multi_modal_input'][image_placeholder]."""
    if not multi_modal_input:
        return []
    image_list = multi_modal_input.get(image_placeholder, [])
    images: List[Image.Image] = []
    for img in image_list:
        if isinstance(img, Image.Image):
            images.append(img)
        else:
            images.append(convert_image(img))
    return images

def _extract_success_flag(info: Optional[Dict[str, Any]]) -> float:
    """Best-effort extraction of a boolean success indicator."""
    if not info:
        return 0.0
    for key in ("success", "is_success", "solved"):
        if key in info:
            try:
                return 1.0 if bool(info[key]) else 0.0
            except Exception:
                continue
    return 0.0


class VisionMultiTurnAgentEnvWorkflow(RolloutWorkflow):
    """
    Multi-turn workflow that supports BOTH VLM (vision) and LLM (text-only).
    - If new images exist and processor is vision-capable -> VLM path (processor)
    - Else -> LLM path (tokenizer only)
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        processor: Optional[AutoProcessor] = None,
        image_placeholder: str = "<image>",
        dump_dir: Optional[str] = None,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_placeholder = image_placeholder
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    def _robust_image_token_id(self) -> Optional[int]:
        """Best-effort retrieval of image token id for Qwen-like VLMs."""
        tok_id = None
        if self.processor is not None:
            tok_id = getattr(self.processor, "image_token_id", None) \
                     or getattr(self.processor, "image_pad_token_id", None)
        if tok_id is None:
            try:
                tok_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                if isinstance(tok_id, list):
                    tok_id = tok_id[0]
            except Exception:
                tok_id = None
        return tok_id

    def _merge_size(self) -> int:
        """Fetch spatial merge size from the processor if available, else 1."""
        if self.processor is not None and getattr(self.processor, "image_processor", None) is not None:
            return int(getattr(self.processor.image_processor, "merge_size", 1))
        return 1

    async def _run_one_episode(
        self, engine: InferenceEngine, data: dict, rid: str
    ) -> Tuple[Dict[str, Any], str, float, int]:
        seed = data["seed"]
        max_turns = data.get("max_turns", 1)
        env: GymImageEnv = REGISTERED_ENVS[data["name"]](data["config"])

        try:
            # ===== Init =====
            init_obs, _ = await env.reset(seed=seed)
            sys_prompt = await env.system_prompt()

            all_images: List[Image.Image] = []

            # Rollout accumulators
            input_ids: List[int] = []
            logprobs: List[float] = []
            loss_mask: List[int] = []
            versions: List[int] = []
            cumulative_reward = 0.0
            t = 0
            last_info: Dict[str, Any] = {}

            # Visual feature segments per user turn (only NEW images each time)
            pv_segs: List[torch.Tensor] = []
            thw_segs: List[torch.Tensor] = []

            # ----- Seed conversation with system + initial user -----
            new_images = get_images_from_multi_modal_input(
                init_obs.get("multi_modal_input"), self.image_placeholder
            )
            for_vlm_turn = len(new_images) > 0 and _is_vlm_processor(self.processor)

            new_messages = [
                {
                    "role": "system",
                    "content": convert_placeholder_to_image_token(
                        sys_prompt["obs_str"], self.image_placeholder, self.processor, for_vlm_turn
                    ),
                },
                {
                    "role": "user",
                    "content": convert_placeholder_to_image_token(
                        init_obs["obs_str"], self.image_placeholder, self.processor, for_vlm_turn
                    ),
                },
            ]

            # Build prompt encoding for the seed messages
            if for_vlm_turn:
                
                text_for_proc = self.tokenizer.apply_chat_template(
                    conversation=new_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                processed_input = self.processor(
                    images=new_images if new_images else None,
                    text=text_for_proc,
                    padding=False,
                    return_tensors="pt",
                )
                cur_input_ids = processed_input["input_ids"].tolist()[0]
                input_ids.extend(cur_input_ids)
                logprobs.extend([0.0] * len(cur_input_ids))
                loss_mask.extend([0] * len(cur_input_ids))
                versions.extend([-1] * len(cur_input_ids))

                if new_images:
                    pv = processed_input["pixel_values"]
                    thw = processed_input.get("image_grid_thw", None)
                    pv_segs.append(pv)
                    if thw is not None:
                        thw_segs.append(thw)
                    all_images.extend(new_images)
            else:
                # LLM path: text-only
                
                new_text = self.tokenizer.apply_chat_template(
                    conversation=new_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
               
                cur_input_ids = self.tokenizer(new_text, return_tensors="pt")["input_ids"][0].tolist()
                input_ids.extend(cur_input_ids)
                logprobs.extend([0.0] * len(cur_input_ids))
                loss_mask.extend([0] * len(cur_input_ids))
                versions.extend([-1] * len(cur_input_ids))

            # Fixed assistant message to measure prefix length for slicing the next user delta.
            fixed_message = {"role": "assistant", "content": "__PREFIX__"}
            
            fixed_token_ids = self.tokenizer.apply_chat_template(
                conversation=[fixed_message],
                tokenize=True,
                add_generation_prompt=False,
            )
            
            fixed_token_id_len = len(fixed_token_ids.tolist() if hasattr(fixed_token_ids, "tolist") else list(fixed_token_ids))

            # ===== Main loop =====
            while t < max_turns:
                img_b64 = image2base64(all_images) if all_images else []

                req = ModelRequest(
                    rid=rid,
                    input_ids=input_ids,
                    image_data=img_b64,
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                )
                resp = await engine.agenerate(req)

                # Append assistant output (trainable region)
                input_ids += resp.output_tokens
                logprobs += resp.output_logprobs
                loss_mask += [1] * len(resp.output_tokens)
                versions += resp.output_versions

                # Ensure EOS
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None and (len(resp.output_tokens) == 0 or input_ids[-1] != eos_id):
                    input_ids.append(eos_id)
                    logprobs.append(0.0)
                    loss_mask.append(0)
                    versions.append(-1)

                # Step env (ASYNC)
                assistant_text = self.tokenizer.decode(resp.output_tokens, skip_special_tokens=True).strip()
                next_obs, r, done, info = await env.step(assistant_text)
                cumulative_reward += float(r)
                last_info = info or {}
                if done or (t + 1) >= max_turns:
                    break

                # Build next user delta
                next_images = get_images_from_multi_modal_input(
                    next_obs.get("multi_modal_input"), self.image_placeholder
                )
                for_vlm_turn = len(next_images) > 0 and _is_vlm_processor(self.processor)

                new_messages = [
                    fixed_message,
                    {
                        "role": "user",
                        "content": convert_placeholder_to_image_token(
                            next_obs["obs_str"], self.image_placeholder, self.processor, for_vlm_turn
                        ),
                    },
                ]

                if for_vlm_turn:
                    
                    text_for_proc = self.tokenizer.apply_chat_template(
                        conversation=new_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                   
                    processed_input = self.processor(
                        images=next_images if next_images else None,
                        text=text_for_proc,
                        padding=False,
                        return_tensors="pt",
                    )
                    cur_ids_full = processed_input["input_ids"].tolist()[0]
                    cur_input_ids = cur_ids_full[fixed_token_id_len:]

                    input_ids += cur_input_ids
                    logprobs += [0.0] * len(cur_input_ids)
                    loss_mask += [0] * len(cur_input_ids)
                    versions += [-1] * len(cur_input_ids)

                    if next_images:
                        pv = processed_input["pixel_values"]
                        thw = processed_input.get("image_grid_thw", None)
                        pv_segs.append(pv)
                        if thw is not None:
                            thw_segs.append(thw)
                        all_images.extend(next_images)
                else:
                    new_text = self.tokenizer.apply_chat_template(
                        conversation=new_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                   
                    cur_ids_full = self.tokenizer(new_text, return_tensors="pt")["input_ids"][0].tolist()
                    cur_input_ids = cur_ids_full[fixed_token_id_len:]

                    input_ids += cur_input_ids
                    logprobs += [0.0] * len(cur_input_ids)
                    loss_mask += [0] * len(cur_input_ids)
                    versions += [-1] * len(cur_input_ids)

                t += 1

            # ===== Pack outputs =====
            L = len(input_ids)
            success_flag = _extract_success_flag(last_info)
            res: Dict[str, Any] = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.ones(1, L, dtype=torch.long),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.long).unsqueeze(0),
                "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
                "versions": torch.tensor(versions, dtype=torch.long).unsqueeze(0),
                "rewards": torch.tensor(float(cumulative_reward)).unsqueeze(0),
                "success": torch.tensor(success_flag, dtype=torch.float32).unsqueeze(0),
            }
            tag_id = int(data.get("tag_id", -1))
            res["tag_id"] = torch.tensor([tag_id], dtype=torch.long)

            multi_modal_input: Dict[str, torch.Tensor] = {}
            if pv_segs:
                multi_modal_input["pixel_values"] = torch.cat(pv_segs, dim=0)
            if thw_segs:
                multi_modal_input["image_grid_thw"] = torch.cat(thw_segs, dim=0)

            if multi_modal_input:
                # Consistency check: #<image_pad> == #features / merge^2
                image_token_id = self._robust_image_token_id()
                if image_token_id is not None:
                    merge_sz = self._merge_size()
                    num_image_pad_tokens = int((res["input_ids"] == image_token_id).sum().item())
                    num_pixel_features = int(multi_modal_input["pixel_values"].shape[0] // (merge_sz ** 2))
                    assert num_image_pad_tokens == num_pixel_features, (
                        f"Mismatch: #<image_pad>={num_image_pad_tokens} vs "
                        f"#features/merge^2={num_pixel_features}. "
                        f"Use processor(..., images=...) for VLM turns and ensure merge_size matches."
                    )
                res["multi_modal_input"] = [multi_modal_input]

            total_str = self.tokenizer.decode(input_ids)
            return res, total_str, cumulative_reward, len(input_ids)
        except Exception as e:
            logger.error(f"Error during episode rollout: {e}", exc_info=True)
            raise
        finally:
            try:
                await env.close()
            except Exception:
                pass

    async def arun_episode(self, engine: InferenceEngine, data: dict):
        """Run one episode with potentially multiple samples (async)."""
        rid = uuid.uuid4().hex
        tasks = [self._run_one_episode(engine, data, rid) for _ in range(self.gconfig.n_samples)]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as exc:
            logger.error("Episode failed", exc_info=True)
            return None

        # Optional dump to disk
        if self.dump_dir is not None:
            version = engine.get_version()
            out_dir = os.path.join(self.dump_dir, str(version))
            os.makedirs(out_dir, exist_ok=True)
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex
            dump_path = os.path.join(out_dir, f"{qid}.txt")
            with open(dump_path, "a") as f:
                n_samples = self.gconfig.n_samples
                for i, (_, total_str, cumulative_reward, len_seq) in enumerate(results):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {len_seq}, cumulative reward: {cumulative_reward}.",
                            "Total string is \n"
                            + colorama.Fore.YELLOW
                            + colorama.Style.DIM
                            + total_str
                            + colorama.Style.RESET_ALL,
                        ]
                    )
                    f.write(info + "\n")

        td_list = [result[0] for result in results]
        return concat_padded_tensors(td_list)
