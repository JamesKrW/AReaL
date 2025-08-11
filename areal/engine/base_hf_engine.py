from __future__ import annotations
import gc
import os
import time
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from tensordict import TensorDict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import FinetuneSpec, TrainEngine
from areal.utils.data import (
    MicroBatchList,
    amend_position_ids,
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    pad_mb_list,
    reorder_list,
    split_padded_tensor_dict_into_mb_list,
    unpack_sequence,
    unsqueeze_mb_list,
)
from areal.utils.fsdp import get_cosine_schedule_with_warmup
from areal.utils.model import (
    VALID_VISION_MODELS,
    disable_dropout_in_model,
    is_qwen2_vl_model,
)
from realhf.api.core.data_api import load_hf_processor_and_tokenizer, load_hf_tokenizer
from realhf.base import constants, logging

logger = logging.getLogger("Base HF Engine")

import torch
from tensordict import TensorDict
from typing import Optional
from torch import distributed as dist  # adjust if you wrap torch.distributed

# --- These must exist in your codebase; we just reference them here ---
# from areal.engine.utils import is_qwen2_vl_model, amend_position_ids
# from areal.engine.pack import split_padded_tensor_dict_into_mb_list, pack_tensor_dict, pad_mb_list, unsqueeze_mb_list
# from areal.engine.types import MicroBatchList

def _take_vision_2d(mb: dict, key: str) -> Optional[torch.Tensor]:
    """
    Normalize a batched vision tensor for one micro-batch into 2D/N-first and slice to true length.
    Expected inputs in mb (after unsqueeze_mb_list):
      - key:         (1, N_max, ...)  or (N_max, ...)  or (D,) for single row
      - key_lengths: (1,) per micro-batch (optional)
    Returns:
      - (N_i, ...) 2D tensor sliced to true length, or None if key absent.
    """
    if key not in mb:
        return None
    v = mb[key]
    if not isinstance(v, torch.Tensor) or v.dim() < 1:
        return v

    # Squeeze batch if present: (1, N_max, ...) -> (N_max, ...)
    if v.dim() >= 2 and v.size(0) == 1:
        v = v.squeeze(0)

    # Single vector case: (D,) -> (1, D)
    if v.dim() == 1:
        v = v.unsqueeze(0)

    len_key = f"{key}_lengths"
    if len_key in mb and isinstance(mb[len_key], torch.Tensor) and mb[len_key].numel() >= 1:
        n = int(mb[len_key].view(-1)[0].item())
        if v.size(0) >= n:
            v = v[:n]

    return v


def _force_rope_inputs_2d(mb: dict, key: str, need_last_dim: int | None = None) -> Optional[torch.Tensor]:
    """
    Stronger variant used right before get_rope_index:
      - squeeze leading batch
      - convert (D,) -> (1, D)
      - slice by *_lengths if available
      - validate last dim if need_last_dim is set
    """
    x = _take_vision_2d(mb, key)
    if x is None:
        return None
    if need_last_dim is not None:
        assert x.dim() == 2 and x.size(-1) == need_last_dim, f"{key} must be (N,{need_last_dim}), got {list(x.shape)}"
    mb[key] = x
    return x

class BaseHFEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.optimizer_config = config.optimizer

        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer
        self.tokenizer: PreTrainedTokenizerFast
        self.processor: AutoProcessor | None = None
        # huggingface model config
        self.model_config: PretrainedConfig
        self._version: int = 0

        # initialization
        self.initialized = False
        self.own_global_group = False
        self._parallelism_group: dist.ProcessGroup
        self.weight_update_group_initialized = False

        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        self.is_vision_model = self.model_config.model_type in VALID_VISION_MODELS

        self.world_size = int(os.environ["WORLD_SIZE"])

    def set_version(self, version: int):
        self._version = version

    def get_version(self) -> int:
        return self._version

    def train(self, mode: bool = True):
        assert self.model is not None
        self.model.train(mode=mode)
        return self

    @property
    def parallelism_group(self) -> dist.ProcessGroup:
        assert self.initialized
        return self._parallelism_group

    def create_process_group(self):
        # Required by NCCL weight update group for SGLang
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        if not dist.is_initialized():
            # TODO: Handle the condition when WORLD_SIZE and RANK is not set in launcher
            # NOTE: device_id **SHOULD NOT** be passed into init_process_group,
            # otherwise initializing the NCCL weight update group will be wrong!
            dist.init_process_group(
                backend="nccl",
                timeout=constants.NCCL_DEFAULT_TIMEOUT,
            )
            self.own_global_group = True
        self._parallelism_group = dist.new_group()

    def create_device_model(self):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(int(os.environ["LOCAL_RANK"]))

        dtype = getattr(torch, self.config.dtype)

        if self.is_vision_model:
            if dtype == torch.float16:
                raise ValueError(
                    "Vision models do not support float16 dtype. Please use bfloat16."
                )
            if self.config.init_from_scratch:
                raise ValueError(
                    "Vision models do not support initialization from scratch. Please use a pretrained model."
                )
            self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
                self.config.path
            )

            tik = time.perf_counter()
            with torch.device("cuda"):
                model = AutoModelForImageTextToText.from_pretrained(
                    pretrained_model_name_or_path=self.config.path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    attn_implementation=self.config.attn_impl,
                )
                if self.config.disable_dropout:
                    disable_dropout_in_model(model)
        else:
            self.tokenizer = load_hf_tokenizer(self.config.path)
            tik = time.perf_counter()
            with torch.device("cuda"):
                if self.config.init_from_scratch:
                    # initialize scratch model from config
                    # NOTE: VLM cannot directly load state dict using this
                    # random initialized model, so otherwise we call
                    # from_pretrained rather than loading weights into this random model.
                    model = AutoModelForCausalLM.from_config(
                        self.model_config,
                        torch_dtype=dtype,
                        attn_implementation=self.config.attn_impl,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        pretrained_model_name_or_path=self.config.path,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        attn_implementation=self.config.attn_impl,
                    )
                if self.config.disable_dropout:
                    disable_dropout_in_model(model)

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        logger.info(f"Model creation and loading time: {time.perf_counter() - tik}")
        self.model = model

    def create_optimizer(self, ft_spec: FinetuneSpec):
        if self.optimizer_config is None:
            return
        assert self.model is not None
        # Set up optimizer
        tik = time.perf_counter()
        assert (
            self.optimizer_config.type == "adam"
        ), "Only AdamW optimizer is supported in this engine."
        lr = self.optimizer_config.lr
        weight_decay = self.optimizer_config.weight_decay
        beta1 = self.optimizer_config.beta1
        beta2 = self.optimizer_config.beta2
        eps = self.optimizer_config.eps

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=eps,
        )
        total_train_steps = ft_spec.total_train_steps
        num_warmup_steps = int(
            self.optimizer_config.warmup_steps_proportion * total_train_steps
        )

        if self.optimizer_config.lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
                total_train_steps,
                min_lr_ratio=self.optimizer_config.min_lr_ratio,
            )
        elif self.optimizer_config.lr_scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
                total_train_steps,
            )
        elif self.optimizer_config.lr_scheduler_type == "constant":
            self.lr_scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
            )
        else:
            raise ValueError(
                f"Unknown lr scheduler type {self.optimizer_config.lr_scheduler_type}"
            )
        logger.info(f"Create optimizer time: {time.perf_counter() - tik}")

    def destroy(self):
        """Destroy the engine and release GPU memory."""
        del self.optimizer
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        dist.destroy_process_group(self.parallelism_group)
        if self.own_global_group:
            dist.destroy_process_group()
        self.initialized = False

    def save_optimizer_state(self, path: str):
        # Save FSDP sharded state dict on each rank
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        state_dict = self.optimizer.state_dict()
        torch.save(state_dict, shard_path)
        dist.barrier(device_ids=[self.device.index])

    def load_optimizer_state(self, path: str):
        # Load FSDP sharded state dict
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        optimizer_state_dict = torch.load(shard_path, weights_only=False)
        self.optimizer.load_state_dict(optimizer_state_dict)
        dist.barrier(device_ids=[self.device.index])

    def step_lr_scheduler(self):
        assert self.lr_scheduler is not None
        self.lr_scheduler.step()

    


    def prepare_mb_list(self, input_: TensorDict) -> "MicroBatchList":
        """
        Patched prepare_mb_list:
        - Works with variable-length vision inputs batched as (B, N_max, ...)
        - Uses *_lengths (shape B) to slice per-micro-batch vision tensors to 2D before Qwen get_rope_index
        - Ensures input_ids/attention_mask are [1, L] when calling get_rope_index
        - Finally converts position_ids to [3, 1, L] for HF Qwen models
        """
        assert "attention_mask" in input_ and "input_ids" in input_

        if self.is_vision_model:
            assert ("pixel_values" in input_ or "pixel_values_flat" in input_), \
                "For vision-language models, pixel_values or pixel_values_flat must be present in input_"
            assert ("image_grid_thw" in input_ or "image_grid_thw_flat" in input_), \
                "For vision-language models, image_grid_thw or image_grid_thw_flat must be present in input_"

        # Normalize to TensorDict
        if isinstance(input_, dict):
            input_ = TensorDict(input_, batch_size=[input_["input_ids"].shape[0]])

        need_qwen_rope = self.is_qwen2_vl_model(self.model_config.model_type) \
            if hasattr(self, "is_qwen2_vl_model") else is_qwen2_vl_model(self.model_config.model_type)

        # Non-Qwen: keep original behavior (compute generic position ids early)
        if not need_qwen_rope:
            if hasattr(self, "amend_position_ids"):
                input_ = self.amend_position_ids(input_)
            else:
                input_ = amend_position_ids(input_)

        # 1) split / pack / pad
        mb_list = split_padded_tensor_dict_into_mb_list(input_, self.config.mb_spec)
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
        )
        

        # 2) HF expects [1, L]
        mb_list = unsqueeze_mb_list(mb_list)

        # 3) Qwen2.5-VL: per-micro-batch get_rope_index AFTER slicing vision to 2D
        if need_qwen_rope:
            for col in (mb_list.mbs, mb_list.padded_mbs):
                for mb in col:
                    # Ensure input_ids / attention_mask are [1, L]
                    if isinstance(mb["input_ids"], torch.Tensor) and mb["input_ids"].dim() == 1:
                        mb["input_ids"] = mb["input_ids"].unsqueeze(0)
                    assert mb["input_ids"].dim() == 2, f"input_ids must be [1, L], got {list(mb['input_ids'].shape)}"

                    attn_mask = mb.get("attention_mask", None)
                    if isinstance(attn_mask, torch.Tensor) and attn_mask.dim() == 1:
                        attn_mask = attn_mask.unsqueeze(0)
                    if isinstance(attn_mask, dict):
                        attn_mask = None

                    # Slice vision to 2D/N-first using *_lengths
                    _force_rope_inputs_2d(mb, "image_grid_thw", need_last_dim=3)        # (N_img, 3)
                    _force_rope_inputs_2d(mb, "video_grid_thw", need_last_dim=3)        # (N_vid, 3)
                    _force_rope_inputs_2d(mb, "second_per_grid_ts", need_last_dim=None) # (N_vid,) or (N_vid,1)

                    # Call Qwen's rope builder
                    position_ids, _ = self.model.model.get_rope_index(
                        input_ids=mb["input_ids"],                   # [1, L]
                        image_grid_thw=mb.get("image_grid_thw"),     # (N_img, 3) or None
                        video_grid_thw=mb.get("video_grid_thw"),     # (N_vid, 3) or None
                        second_per_grid_ts=mb.get("second_per_grid_ts"),  # 1D or None
                        attention_mask=attn_mask,                    # [1, L] or None
                    )
                    # Temporarily store as [1, L, 3]; convert to [3, 1, L] before forward
                    mb["position_ids"] = torch.einsum("ijk->jki", position_ids)

        # 4) Convert tensordict → plain dict
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)

        # 5) Final fixes & shape adjustments right before model forward
        for mb in mb_list.mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
            mb["cu_seqlens_q"] = mb["cu_seqlens_k"] = mb["cu_seqlens"]
            mb["use_cache"] = False
            mb["attention_mask"] = dict(full_attention=None)
            if need_qwen_rope and "position_ids" in mb:
                # [1, L, 3] -> [3, 1, L]
                mb["position_ids"] = torch.einsum("ijk->kij", mb["position_ids"])

        for mb in mb_list.padded_mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
            mb["cu_seqlens_q"] = mb["cu_seqlens_k"] = mb["cu_seqlens"]
            mb["use_cache"] = False
            mb["attention_mask"] = dict(full_attention=None)
            if need_qwen_rope and "position_ids" in mb:
                mb["position_ids"] = torch.einsum("ijk->kij", mb["position_ids"])

        return mb_list

    def train_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> Dict[str, float]:
        """Train on a batch using gradient accumulation."""
        input_ = input_.to(self.device)
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        self.optimizer.zero_grad()
        mb_list = self.prepare_mb_list(input_)

        total_loss_weight = torch.tensor(
            sum([loss_weight_fn(mb) for mb in mb_list.mbs]), dtype=torch.float32
        )
        assert total_loss_weight != 0
        dist.all_reduce(total_loss_weight)

        # Process microbatches with gradient accumulation
        for i, (pad_length, padded_mb_input, mb_input) in enumerate(
            zip(mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs)
        ):
            outputs = self.model(**padded_mb_input)

            logits = outputs.logits.squeeze(0)
            logits = logits[:-pad_length] if pad_length > 0 else logits
            loss = loss_fn(logits, mb_input)

            loss_scale = loss_weight_fn(mb_input) / total_loss_weight

            # Scale loss for accumulation
            # Revert gradient averaging across dp ranks
            # FIXME: should be DP size
            loss_scale *= self.world_size

            loss *= loss_scale
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.optimizer_config.gradient_clipping,
            norm_type=2.0,
            error_if_nonfinite=False,
            foreach=None,
        )
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad()
            update_successful = False
        else:
            self.optimizer.step()
            update_successful = True

        current_lr = self.lr_scheduler.get_last_lr()[0]
        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
        )

    @torch.no_grad()
    def eval_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> torch.Tensor | None:
        """Evaluate on a batch."""
        input_ = input_.to(self.device)
        mb_list = self.prepare_mb_list(input_)
        total_loss_weight = torch.tensor(
            sum([loss_weight_fn(mb) for mb in mb_list.mbs]), dtype=torch.float32
        )
        assert total_loss_weight != 0

        total_loss = 0.0
        total_weight = 0.0

        for pad_length, padded_mb_input, mb_input in zip(
            mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
        ):
            outputs = self.model(**padded_mb_input)
            logits = outputs.logits.squeeze(0)
            logits = logits[:-pad_length] if pad_length > 0 else logits
            loss = loss_fn(logits, mb_input)

            # Simple weight calculation (could be improved)
            loss_scale = loss_weight_fn(mb_input) / total_loss_weight
            total_loss += loss.item() * loss_scale
            total_weight += loss_scale

        return torch.tensor(total_loss / total_weight)

    @torch.no_grad()
    def forward(
        self,
        input_: TensorDict,
        output_seqlens: List[int] | None = None,
        post_hook: Callable[[torch.Tensor, TensorDict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Forward pass with optional post-processing."""
        input_ = input_.to(self.device)
        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
        mb_list = self.prepare_mb_list(input_)

        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()

        results = []
        for pad_length, padded_mb_input, mb_input in zip(
            mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
        ):

            outputs = self.model(**padded_mb_input)
            logits = outputs.logits.squeeze(0)
            logits = logits[:-pad_length] if pad_length > 0 else logits

            if post_hook:
                result = post_hook(logits, mb_input)
                results.append(result)
            else:
                results.append(logits)

        res = aggregate_fn(results)
        output_seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
        unpacked = unpack_sequence(res, lens=output_seqlens, dim=0)
        reordered = reorder_list(unpacked, mb_list.backward_indices)
        return pad_and_stack_tensors_along_first_dim(reordered)
