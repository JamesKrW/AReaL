# vision_multi_turn_agent_env_workflow.py (async GymImageEnv version; VLM+LLM compatible)
import asyncio, os, uuid, colorama, torch
from typing import Any, Dict, List, Tuple, Optional
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

logger = logging.getLogger("Vision Multi-Turn AgentEnv workflow")


# ---------------------- Helpers ----------------------
def _is_vlm_processor(proc: Optional[AutoProcessor]) -> bool:
    return proc is not None and hasattr(proc, "image_processor")

def convert_placeholder_to_image_token(txt: str, placeholder: str, processor: Optional[AutoProcessor], vlm: bool) -> str:
    """
    Replace <image> placeholder for VLM turns; use a harmless marker for text-only turns.
    """
    if vlm and _is_vlm_processor(processor):
        iproc = getattr(processor, "image_processor", None)
        tok = "<|vision_start|><|image_pad|><|vision_end|>" if (iproc and str(getattr(iproc, "image_processor_type", "")).lower().find("qwen") >= 0) else getattr(processor, "image_token", "<image>")
    else:
        tok = "[image]"
    return txt.replace(placeholder, tok)

def get_images_from_multi_modal_input(multi_modal_input: Optional[Dict[str, Any]], placeholder: str = "<image>") -> List[Image.Image]:
    """
    Extract PIL images from multi_modal_input[placeholder].
    """
    if not multi_modal_input:
        return []
    imgs = multi_modal_input.get(placeholder, [])
    return [img if isinstance(img, Image.Image) else convert_image(img) for img in imgs]

def _extract_success_flag(info: Optional[Dict[str, Any]]) -> float:
    """
    Best-effort extraction of a boolean success indicator.
    """
    if not info:
        return 0.0
    for k in ("success", "is_success", "solved"):
        if k in info:
            return 1.0 if bool(info[k]) else 0.0
    return 0.0


class VisionMultiTurnAgentEnvWorkflow(RolloutWorkflow):
    """
    Multi-turn workflow that supports BOTH VLM (vision) and LLM (text-only).
    - If new images exist and processor is vision-capable -> VLM path (processor)
    - Else -> LLM path (tokenizer only)
    """

    def __init__(self,
                 gconfig: GenerationHyperparameters,
                 tokenizer: PreTrainedTokenizerFast,
                 processor: Optional[AutoProcessor] = None,
                 image_placeholder: str = "<image>",
                 dump_dir: Optional[str] = None):
        self.gconfig, self.tokenizer, self.processor = gconfig, tokenizer, processor
        self.image_placeholder, self.dump_dir = image_placeholder, dump_dir
        if self.dump_dir and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    # ---------------------- Final (single-point) validation ----------------------
    def _final_validate(self, res: Dict[str, Any]) -> None:
        """
        Single, unified validation invoked right before returning:
        - If multi_modal_input exists:
            * Assert #<image_pad> == pixel_features / merge_size^2
            * For each [vision_start..vision_end) in the masked region, assert it contains at least one <image_pad>
        - Else: assert no <image_pad> appears in the masked region
        """
        input_ids = res["input_ids"][0]
        attention_mask = res["attention_mask"][0]

        image_pad_id = int(getattr(self.processor, "image_token_id")) if (self.processor is not None and hasattr(self.processor, "image_token_id")) else int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        vision_start_id = int(self.tokenizer.convert_tokens_to_ids("<|vision_start|>"))
        vision_end_id = int(self.tokenizer.convert_tokens_to_ids("<|vision_end|>"))

        if "multi_modal_input" in res:
            multi_modal_input = res["multi_modal_input"][0]
            assert "pixel_values" in multi_modal_input, "pixel_values missing in multi_modal_input"

            merge_size = int(self.processor.image_processor.merge_size) if (self.processor is not None and hasattr(self.processor, "image_processor")) else 1
            num_image_pad = int((input_ids == image_pad_id).sum().item())
            num_pixel_features = int(multi_modal_input["pixel_values"].shape[0] // (merge_size ** 2))
            if num_image_pad != num_pixel_features:
                logger.error(f"Final validation: #<image_pad>({num_image_pad}) != #features/merge^2({num_pixel_features})")
                assert False, "image_pad count mismatch"

            # Structure check across the masked region.
            seq = [int(t) for t, m in zip(input_ids.tolist(), attention_mask.tolist()) if m == 1]
            i = 0
            while True:
                try:
                    s = seq.index(vision_start_id, i)
                except ValueError:
                    break
                try:
                    e = seq.index(vision_end_id, s + 1)
                except ValueError:
                    logger.error(f"Final validation: vision_start at {s} has no closing vision_end")
                    assert False, "vision segment not closed"
                if image_pad_id not in seq[s + 1:e]:
                    logger.error(f"Final validation: no <image_pad> between start={s} and end={e}")
                    assert False, "vision segment missing image_pad"
                i = e + 1
        else:
            if int((input_ids == image_pad_id).sum().item()) != 0:
                logger.error("Final validation: <image_pad> appears but multi_modal_input is missing")
                assert False, "image_pad present without multi_modal_input"

    # ---------------------- Episode ----------------------
    async def _run_one_episode(self, engine: InferenceEngine, data: dict, rid: str) -> Tuple[Dict[str, Any], str, float, int]:
        seed, max_turns = data["seed"], data.get("max_turns", 1)
        env: GymImageEnv = REGISTERED_ENVS[data["name"]](data["config"])

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
        pixel_values_segs: List[torch.Tensor] = []
        image_grid_thw_segs: List[torch.Tensor] = []

        # ----- Seed turn -----
        new_images = get_images_from_multi_modal_input(init_obs.get("multi_modal_input"), self.image_placeholder)
        is_vlm_turn = len(new_images) > 0 and _is_vlm_processor(self.processor)
        messages = [
            {"role": "system", "content": convert_placeholder_to_image_token(sys_prompt["obs_str"], self.image_placeholder, self.processor, is_vlm_turn)},
            {"role": "user", "content": convert_placeholder_to_image_token(init_obs["obs_str"], self.image_placeholder, self.processor, is_vlm_turn)},
        ]
        if is_vlm_turn:
            text = self.tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
            processed = self.processor(images=new_images, text=text, padding=False, return_tensors="pt")
            seq = processed["input_ids"].tolist()[0]
            input_ids += seq
            logprobs += [0.0] * len(seq)
            loss_mask += [0] * len(seq)
            versions += [-1] * len(seq)
            if new_images:
                pixel_values_segs.append(processed["pixel_values"])
                if "image_grid_thw" in processed:
                    image_grid_thw_segs.append(processed["image_grid_thw"])
                all_images += new_images
        else:
            text = self.tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
            seq = self.tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
            input_ids += seq
            logprobs += [0.0] * len(seq)
            loss_mask += [0] * len(seq)
            versions += [-1] * len(seq)

        # ----- Turns loop -----
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
            user = {"role": "user", "content": convert_placeholder_to_image_token(next_obs["obs_str"], self.image_placeholder, self.processor, is_vlm_turn)}

            if is_vlm_turn:
                # processor -> processor delta alignment to avoid cutting visual tokens
                s1_text = self.tokenizer.apply_chat_template(conversation=[fixed], tokenize=False, add_generation_prompt=False)
                s1_proc = self.processor(images=None, text=s1_text, padding=False, return_tensors="pt")
                fixed_len = int(s1_proc["input_ids"].shape[-1])

                s2_text = self.tokenizer.apply_chat_template(conversation=[fixed, user], tokenize=False, add_generation_prompt=True)
                s2_proc = self.processor(images=next_images, text=s2_text, padding=False, return_tensors="pt")

                full = s2_proc["input_ids"][0].tolist()
                delta = full[fixed_len:]

                input_ids += delta
                logprobs += [0.0] * len(delta)
                loss_mask += [0] * len(delta)
                versions += [-1] * len(delta)
                if next_images:
                    pixel_values_segs.append(s2_proc["pixel_values"])
                    if "image_grid_thw" in s2_proc:
                        image_grid_thw_segs.append(s2_proc["image_grid_thw"])
                    all_images += next_images
            else:
                # tokenizer -> tokenizer delta alignment
                s1 = self.tokenizer.apply_chat_template(conversation=[fixed], tokenize=True, add_generation_prompt=False)
                s2 = self.tokenizer.apply_chat_template(conversation=[fixed, user], tokenize=True, add_generation_prompt=True)
                s1_ids = s1.tolist() if hasattr(s1, "tolist") else list(s1)
                s2_ids = s2.tolist() if hasattr(s2, "tolist") else list(s2)
                delta = s2_ids[len(s1_ids):]
                input_ids += delta
                logprobs += [0.0] * len(delta)
                loss_mask += [0] * len(delta)
                versions += [-1] * len(delta)

            t += 1

        # ----- Pack -----
        length = len(input_ids)
        result: Dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(versions, dtype=torch.long).unsqueeze(0),
            "rewards": torch.tensor(float(cumulative_reward)).unsqueeze(0),
            "success": torch.tensor(_extract_success_flag(last_info), dtype=torch.float32).unsqueeze(0),
            "tag_id": torch.tensor([int(data.get("tag_id", -1))], dtype=torch.long),
        }
        multi_modal_input_dict: Dict[str, torch.Tensor] = {}
        if pixel_values_segs:
            multi_modal_input_dict["pixel_values"] = torch.cat(pixel_values_segs, dim=0)
        if image_grid_thw_segs:
            multi_modal_input_dict["image_grid_thw"] = torch.cat(image_grid_thw_segs, dim=0)
        if multi_modal_input_dict:
            result["multi_modal_input"] = [multi_modal_input_dict]

        # ----- Final validation (log + assert). No try/except here. -----
        self._final_validate(result)

        total_str = self.tokenizer.decode(input_ids)
        await env.close()
        return result, total_str, cumulative_reward, len(input_ids)

    async def arun_episode(self, engine: InferenceEngine, data: dict):
        rid = uuid.uuid4().hex
        try:
            results = await asyncio.gather(
                *[self._run_one_episode(engine, data, rid) for _ in range(self.gconfig.n_samples)],
                return_exceptions=False
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
                    info = "\n".join([
                        f"idx: {i+1} / {self.gconfig.n_samples}, seqlen: {seqlen}, cumulative reward: {reward}.",
                        "Total string is \n" + colorama.Fore.YELLOW + colorama.Style.DIM + total_str + colorama.Style.RESET_ALL,
                    ])
                    f.write(info + "\n")

        td_list = [r[0] for r in results]
        return concat_padded_tensors(td_list)
