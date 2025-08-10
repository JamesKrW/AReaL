# vision_multi_turn_agent_env_workflow.py
import asyncio
import os
import uuid
from typing import Dict, List, Any

import colorama
import torch
from tensordict import TensorDict
from transformers import AutoProcessor, PreTrainedTokenizerFast
from PIL import Image

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import VLMRequest
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors
from areal.utils.image import image2base64
from realhf.base import logging

from areal.envs.utils.env_load_utils import load_env_from_registry  # Unified env loading

logger = logging.getLogger("Vision Multi-Turn AgentEnv workflow")


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

    def _extract_images_from_multimodal_data(self, multi_modal_data: Dict[str, Any]) -> List[Image.Image]:
        """
        Extract PIL Images from multi_modal_data using the specified image placeholder.
        
        Args:
            multi_modal_data: Dictionary with placeholder keys and data lists as values
            
        Returns:
            List of PIL Images corresponding to self.image_placeholder
        """
        if not multi_modal_data or self.image_placeholder not in multi_modal_data:
            return []
        
        image_list = multi_modal_data[self.image_placeholder]
        if not isinstance(image_list, list):
            logger.warning(f"Expected list for {self.image_placeholder}, got {type(image_list)}")
            return []
        
        images = []
        for img in image_list:
            if isinstance(img, Image.Image):
                images.append(img)
            else:
                logger.warning(f"Unexpected image type: {type(img)}, expected PIL.Image")
        
        return images

    def _construct_messages_with_images(self, text_content: str, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Construct messages list for the processor, handling image placeholders.
        
        Args:
            text_content: Text with image placeholders
            images: List of PIL Images
            
        Returns:
            Messages list compatible with the processor
        """
        # Count placeholders in text
        placeholder_count = text_content.count(self.image_placeholder)
        
        if placeholder_count != len(images):
            logger.warning(
                f"Mismatch: {placeholder_count} placeholders but {len(images)} images. "
                f"Adjusting to use min({placeholder_count}, {len(images)}) images."
            )
        
        # Use the minimum of available images and placeholders
        num_images_to_use = min(placeholder_count, len(images))
        images_to_use = images[:num_images_to_use]
        
        # For now, we'll construct a simple message structure
        # This might need adjustment based on your specific processor requirements
        message = {
            "role": "user",
            "content": text_content,
            "images": images_to_use  # Some processors expect images in the message
        }
        
        return [message]

    async def _run_one_episode(self, engine: InferenceEngine, data: dict, rid: str):
        """
        Run a single environment episode with multi-turn LLM interaction supporting vision.

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
        
        # Track accumulated images across turns for proper tensor construction
        all_images = []

        try:
            # Reset environment and get initial observation
            init_obs, _ = env.reset(seed=seed)
            
            sys_prompt = env.get_system_prompt()
            obs_str = init_obs["obs_str"]
            multi_modal_data = init_obs.get("multi_modal_data", {})
            
            # Extract images from the initial observation
            current_images = self._extract_images_from_multimodal_data(multi_modal_data)
            all_images.extend(current_images)
            
            # Construct conversation context with vision support
            messages = [
                {"role": "system", "content": sys_prompt}
            ]
            
            # Add user message with images
            if current_images:
                user_message = {
                    "role": "user", 
                    "content": obs_str,
                    "images": current_images
                }
            else:
                user_message = {"role": "user", "content": obs_str}
            
            messages.append(user_message)

            done = False
            
            # Process the initial input
            if all_images:
                processed_input = self.processor(
                    images=all_images,
                    text=messages,
                    padding=False,
                    return_tensors="pt",
                )
                input_ids = processed_input["input_ids"].tolist()[0]
                pixel_values = processed_input["pixel_values"]
                image_grid_thw = processed_input["image_grid_thw"]
                byte_images = image2base64(all_images)
            else:
                # Fallback to text-only processing
                input_ids = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                pixel_values = None
                image_grid_thw = None
                byte_images = None

            while not done and t < self.max_turns:
                # Generate assistant reply
                if byte_images is not None:
                    req = VLMRequest(
                        rid=rid,
                        input_ids=input_ids,
                        image_data=byte_images,
                        gconfig=self.gconfig.new(n_samples=1),
                    )
                else:
                    # Fallback to LLMRequest for text-only
                    from areal.api.io_struct import LLMRequest
                    req = LLMRequest(
                        rid=rid,
                        input_ids=input_ids,
                        gconfig=self.gconfig.new(n_samples=1),
                    )
                
                resp = await engine.agenerate(req)

                # Decode model output into an action string
                completion_text = self.tokenizer.decode(resp.output_tokens, skip_special_tokens=True)

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

                # Prepare for next turn
                input_ids += resp.output_tokens
                if input_ids[-1] != self.tokenizer.eos_token_id:
                    input_ids += [self.tokenizer.eos_token_id]
                
                # Get new observation
                obs_str = next_obs["obs_str"]
                new_multi_modal_data = next_obs.get("multi_modal_data", {})
                new_images = self._extract_images_from_multimodal_data(new_multi_modal_data)
                
                # Add new images to the accumulated set
                all_images.extend(new_images)
                
                # Construct the next user message
                if new_images:
                    # If there are new images, we need to reprocess everything
                    # This is necessary because the processor needs to see all images together
                    messages.append({"role": "assistant", "content": completion_text})
                    messages.append({
                        "role": "user",
                        "content": obs_str,
                        "images": new_images
                    })
                    
                    # Reprocess with all accumulated images
                    processed_input = self.processor(
                        images=all_images,
                        text=messages,
                        padding=False,
                        return_tensors="pt",
                    )
                    input_ids = processed_input["input_ids"].tolist()[0]
                    pixel_values = processed_input["pixel_values"] 
                    image_grid_thw = processed_input["image_grid_thw"]
                    byte_images = image2base64(all_images)
                else:
                    # Text-only turn, use the simpler approach
                    new_messages = [{"role": "assistant", "content": "some random message."}]
                    s1 = self.tokenizer.apply_chat_template(new_messages, tokenize=True)
                    new_messages += [{"role": "user", "content": obs_str}]
                    s2 = self.tokenizer.apply_chat_template(
                        new_messages, tokenize=True, add_generation_prompt=True
                    )
                    input_ids += s2[len(s1):]
                
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
            
            # Add vision-specific data if available
            if pixel_values is not None:
                res["pixel_values"] = pixel_values
                res["image_grid_thw"] = image_grid_thw
            
            res = {k: v.unsqueeze(0) for k, v in res.items()}

            total_str = self.tokenizer.decode(seq)

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