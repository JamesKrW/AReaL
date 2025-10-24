# vision_multi_turn_agent_env_workflow.py
# Async multi-turn GymImageEnv workflow for VLM/LLM; single-point validation; Qwen2.5-VL friendly.

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
from areal.utils import logging, stats_tracker
from view_suite.gym.gym_image_env import GymImageEnv
from areal.viewsuite.registry import REGISTERED_ENVS

logger = logging.getLogger("Vision Multi-Turn AgentEnv workflow")


# ---------------------- Helpers ----------------------
def _is_vlm_processor(proc: Optional[AutoProcessor]) -> bool:
    """Whether the processor is vision-capable."""
    return proc is not None and hasattr(proc, "image_processor")

def convert_placeholder_to_image_token(
    text: str,
    placeholder: str,
    processor: Optional[AutoProcessor],
    is_vlm_turn: bool,
) -> str:
    """
    Replace <image> placeholder for VLM turns; use a harmless marker for text-only turns.
    - Qwen-style VLM: "<|vision_start|><|image_pad|><|vision_end|>"
    """
    if is_vlm_turn and _is_vlm_processor(processor):
        iproc = getattr(processor, "image_processor", None)
        if iproc and str(getattr(iproc, "image_processor_type", "")).lower().find("qwen") >= 0:
            tok = "<|vision_start|><|image_pad|><|vision_end|>"
        else:
            tok = getattr(processor, "image_token", "<image>")
    else:
        tok = "[image]"
    return text.replace(placeholder, tok)

def get_images_from_multi_modal_input(
    multi_modal_input: Optional[Dict[str, Any]],
    placeholder: str = "<image>",
) -> List[Image.Image]:
    """Extract PIL images from multi_modal_input[placeholder]."""
    if not multi_modal_input:
        return []
    items = multi_modal_input.get(placeholder, [])
    out: List[Image.Image] = []
    for it in items:
        out.append(it if isinstance(it, Image.Image) else convert_image(it))
    return out

def _extract_success_flag(info: Optional[Dict[str, Any]]) -> float:
    """Extract a boolean success flag if present."""
    if not info:
        return 0.0
    for k in ("success", "is_success", "solved"):
        if k in info:
            return 1.0 if bool(info[k]) else 0.0
    return 0.0


class VisionMultiTurnAgentEnvWorkflow(RolloutWorkflow):
    """
    Multi-turn workflow that supports BOTH VLM (vision) and LLM (text-only).

    Rules:
    - If the turn has NEW images and processor is vision-capable -> encode that user turn with `processor`.
    - Otherwise -> encode with `tokenizer` only.
    - Generation is always done by the engine (which sees accumulated input_ids + base64 images).
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        processor: Optional[AutoProcessor] = None,
        image_placeholder: str = "<image>",
        dump_dir: Optional[str] = None,
        rollout_stat_scope: str = "rollout",
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_placeholder = image_placeholder
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        if self.dump_dir and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    # ---------------------- Single-point final validation ----------------------
    def _final_validate(self, result: Dict[str, Any]) -> None:
        """
        Validate ONLY once right before returning.

        For Qwen2.5-VL:
        - Let image_grid_thw be shape (n_img, 3) with rows [t, h, w] in *unmerged* grid.
        - Let merge = processor.image_processor.merge_size.
        - Then #<|image_pad|> tokens in all vision spans MUST equal sum(t*h*w) / merge^2.

        Also ensure each vision span contains at least one <|image_pad|>.
        If no multi_modal_input: assert there is no <|image_pad|> in the masked region.
        """
        input_ids_tensor = result["input_ids"][0]
        attention_mask_tensor = result["attention_mask"][0]
        input_ids = input_ids_tensor.tolist()
        attention_mask = attention_mask_tensor.tolist()
        masked = [tid for tid, m in zip(input_ids, attention_mask) if m == 1]

        vs_id = int(self.tokenizer.convert_tokens_to_ids("<|vision_start|>"))
        ve_id = int(self.tokenizer.convert_tokens_to_ids("<|vision_end|>"))

        # Prefer model.config.image_token_id, then processor.image_token_id.
        # No try/except here on purpose — fail fast if missing.
        if hasattr(getattr(getattr(self, "processor", None), "image_processor", None), "merge_size"):
            merge_size = int(self.processor.image_processor.merge_size)
        else:
            merge_size = 1

        if hasattr(getattr(getattr(self, "tokenizer", None), "model", None), "config") and \
        hasattr(self.tokenizer.model.config, "image_token_id"):
            ip_id = int(self.tokenizer.model.config.image_token_id)  # if you attached tokenizer to model elsewhere, ignore this
        elif hasattr(self, "processor") and hasattr(self.processor, "image_token_id"):
            ip_id = int(self.processor.image_token_id)
        else:
            # Last resort: model.model.config if available on engine’s model object (common in HF)
            # If you don’t have self.model here, comment this out and keep only processor.image_token_id.
            ip_id = int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))  # will explode if wrong; acceptable per your rule

        if "multi_modal_input" in result:
            multi = result["multi_modal_input"][0]
            assert "image_grid_thw" in multi, "image_grid_thw missing in multi_modal_input"
            image_grid_thw = multi["image_grid_thw"]  # (n_img, 3)
            assert image_grid_thw.ndim == 2 and image_grid_thw.shape[1] == 3, "image_grid_thw malformed"

            # Count <|image_pad|> inside all vision spans
            i = 0
            image_pad_in_spans = 0
            while True:
                try:
                    s = masked.index(vs_id, i)
                except ValueError:
                    break
                e = masked.index(ve_id, s + 1)  # let it raise if malformed
                seg = masked[s + 1 : e]
                assert ip_id in seg, "vision segment has no <|image_pad|>"
                image_pad_in_spans += seg.count(ip_id)
                i = e + 1

            # Expected: sum(t*h*w)/merge^2
            merge = 1
            if self.processor is not None and hasattr(self.processor, "image_processor"):
                merge = int(getattr(self.processor.image_processor, "merge_size", 1))

            thw = image_grid_thw.long()
            total_patches = int((thw[:, 0] * thw[:, 1] * thw[:, 2]).sum().item())
            expected_image_pad = total_patches // (merge * merge)

            assert image_pad_in_spans == expected_image_pad, (
                f"pad mismatch: {image_pad_in_spans} vs {expected_image_pad} "
                f"(sum t*h*w={total_patches}, merge={merge})"
            )
        else:
            assert ip_id not in masked, "<|image_pad|> appears but multi_modal_input is missing"

    # ---------------------- Episode core ----------------------
    async def _run_one_episode(
        self, engine: InferenceEngine, data: dict, rid: str
    ) -> Tuple[Dict[str, Any], str, float, int]:
        seed = data["seed"]
        max_turns = data.get("max_turns", 1)
        env: GymImageEnv = REGISTERED_ENVS[data["name"]](data["config"])

        # ---- Reset and seed turn ----
        init_obs, _ = await env.reset(seed=seed)
        sys_prompt = await env.system_prompt()

        all_images: List[Image.Image] = []

        input_ids: List[int] = []
        logprobs: List[float] = []
        loss_mask: List[int] = []
        versions: List[int] = []
        cumulative_reward = 0.0
        t = 0
        last_info: Dict[str, Any] = {}

        pixel_values_segments: List[torch.Tensor] = []
        image_grid_thw_segments: List[torch.Tensor] = []

        new_images = get_images_from_multi_modal_input(init_obs.get("multi_modal_input"), self.image_placeholder)
        is_vlm_turn = len(new_images) > 0 and _is_vlm_processor(self.processor)

        messages = [
            {
                "role": "system",
                "content": convert_placeholder_to_image_token(
                    sys_prompt["obs_str"], self.image_placeholder, self.processor, is_vlm_turn
                ),
            },
            {
                "role": "user",
                "content": convert_placeholder_to_image_token(
                    init_obs["obs_str"], self.image_placeholder, self.processor, is_vlm_turn
                ),
            },
        ]

        if is_vlm_turn:
            text = self.tokenizer.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=True
            )
            processed = self.processor(images=new_images, text=text, padding=False, return_tensors="pt")
            ids = processed["input_ids"].tolist()[0]
            input_ids += ids
            logprobs += [0.0] * len(ids)
            loss_mask += [0] * len(ids)
            versions += [-1] * len(ids)

            if new_images:
                pixel_values_segments.append(processed["pixel_values"])
                if "image_grid_thw" in processed:
                    image_grid_thw_segments.append(processed["image_grid_thw"])
                all_images += new_images
        else:
            text = self.tokenizer.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=True
            )
            ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
            input_ids += ids
            logprobs += [0.0] * len(ids)
            loss_mask += [0] * len(ids)
            versions += [-1] * len(ids)

        # ---- Multi-turn loop ----
        while t < max_turns:
            req = ModelRequest(
                rid=rid,
                input_ids=input_ids,
                image_data=image2base64(all_images) if all_images else [],
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            resp = await engine.agenerate(req)

            input_ids += resp.output_tokens
            logprobs += resp.output_logprobs
            loss_mask += [1] * len(resp.output_tokens)
            versions += resp.output_versions

            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and (len(resp.output_tokens) == 0 or input_ids[-1] != eos_id):
                input_ids.append(eos_id)
                logprobs.append(0.0)
                loss_mask.append(0)
                versions.append(-1)

            assistant_text = self.tokenizer.decode(resp.output_tokens, skip_special_tokens=True).strip()
            next_obs, r, done, info = await env.step(assistant_text)
            cumulative_reward += float(r)
            last_info = info or {}
            if done or (t + 1) >= max_turns:
                break

            next_images = get_images_from_multi_modal_input(next_obs.get("multi_modal_input"), self.image_placeholder)
            is_vlm_turn = len(next_images) > 0 and _is_vlm_processor(self.processor)

            fixed = {"role": "assistant", "content": "__PREFIX__"}
            user = {
                "role": "user",
                "content": convert_placeholder_to_image_token(
                    next_obs["obs_str"], self.image_placeholder, self.processor, is_vlm_turn
                ),
            }

            if is_vlm_turn:
                # Use processor both times to measure prefix safely (do not cut visual tokens).
                s1_text = self.tokenizer.apply_chat_template(
                    conversation=[fixed], tokenize=False, add_generation_prompt=False
                )
                s1_proc = self.processor(images=None, text=s1_text, padding=False, return_tensors="pt")
                fixed_len = int(s1_proc["input_ids"].shape[-1])

                s2_text = self.tokenizer.apply_chat_template(
                    conversation=[fixed, user], tokenize=False, add_generation_prompt=True
                )
                s2_proc = self.processor(images=next_images, text=s2_text, padding=False, return_tensors="pt")

                full = s2_proc["input_ids"][0].tolist()
                delta = full[fixed_len:]

                input_ids += delta
                logprobs += [0.0] * len(delta)
                loss_mask += [0] * len(delta)
                versions += [-1] * len(delta)

                if next_images:
                    pixel_values_segments.append(s2_proc["pixel_values"])
                    if "image_grid_thw" in s2_proc:
                        image_grid_thw_segments.append(s2_proc["image_grid_thw"])
                    all_images += next_images
            else:
                s1 = self.tokenizer.apply_chat_template(
                    conversation=[fixed], tokenize=True, add_generation_prompt=False
                )
                s2 = self.tokenizer.apply_chat_template(
                    conversation=[fixed, user], tokenize=True, add_generation_prompt=True
                )
                s1_ids = s1.tolist() if hasattr(s1, "tolist") else list(s1)
                s2_ids = s2.tolist() if hasattr(s2, "tolist") else list(s2)
                delta = s2_ids[len(s1_ids) :]

                input_ids += delta
                logprobs += [0.0] * len(delta)
                loss_mask += [0] * len(delta)
                versions += [-1] * len(delta)

            t += 1

        # ---- Pack output dict ----
        L = len(input_ids)
        result: Dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones(1, L, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(versions, dtype=torch.long).unsqueeze(0),
            "rewards": torch.tensor(float(cumulative_reward)).unsqueeze(0),
            "success": torch.tensor(_extract_success_flag(last_info), dtype=torch.float32).unsqueeze(0),
            "tag_id": torch.tensor([int(data.get("tag_id", -1))], dtype=torch.long),
        }
        multi_modal_input: Dict[str, torch.Tensor] = {}
        if pixel_values_segments:
            multi_modal_input["pixel_values"] = torch.cat(pixel_values_segments, dim=0)
        if image_grid_thw_segments:
            multi_modal_input["image_grid_thw"] = torch.cat(image_grid_thw_segments, dim=0)
        if multi_modal_input:
            result["multi_modal_input"] = [multi_modal_input]

        # ---- Single final validation (raises on failure) ----
        self._final_validate(result)

        total_str = self.tokenizer.decode(input_ids)
        await env.close()
        return result, total_str, cumulative_reward, len(input_ids)

    async def arun_episode(self, engine: InferenceEngine, data: dict):
        """Public API: run one episode with possibly multiple samples (async)."""
        rid = uuid.uuid4().hex
        try:
            results = await asyncio.gather(
                *[self._run_one_episode(engine, data, rid) for _ in range(self.gconfig.n_samples)],
                return_exceptions=False,
            )
        except Exception:
            logger.error("Episode failed", exc_info=True)
            return None

        # Optional dump
        if self.dump_dir is not None:
            out_dir = os.path.join(self.dump_dir, str(engine.get_version()))
            os.makedirs(out_dir, exist_ok=True)
            qid = next((data.get(k) for k in ["query_id", "id", "qid"] if data.get(k) is not None), uuid.uuid4().hex)
            with open(os.path.join(out_dir, f"{qid}.txt"), "a") as f:
                for i, (_, total_str, reward, seqlen) in enumerate(results):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {self.gconfig.n_samples}, seqlen: {seqlen}, cumulative reward: {reward}.",
                            "Total string is \n"
                            + colorama.Fore.YELLOW
                            + colorama.Style.DIM
                            + total_str
                            + colorama.Style.RESET_ALL,
                        ]
                    )
                    f.write(info + "\n")

        # Log rewards and concatenate tensors
        td_list = []
        for result, _, reward, _ in results:
            td_list.append(result)
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)
        return concat_padded_tensors(td_list)
