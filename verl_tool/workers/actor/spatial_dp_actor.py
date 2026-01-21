# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Spatial Linking Data Parallel PPO Actor.

This module extends the DataParallelPPOActor to support spatial linking models
by passing refer_boxes to the model forward pass during GRPO training.

The spatial linking model requires refer_boxes to enhance <|box_end|> tokens
with cross-attention to image patches within the bounding boxes.
"""

import logging
import os
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import ActorConfig

__all__ = ["SpatialDataParallelPPOActor"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def extract_refer_boxes(micro_batch: dict) -> Optional[List[Optional[torch.Tensor]]]:
    """
    Extract refer_boxes from micro_batch for spatial linking.
    
    The refer_boxes should be a list of tensors, one per batch item,
    each of shape [N, 4] where N is the number of boxes (typically 3:
    person, object, interaction).
    
    Args:
        micro_batch: Dictionary containing batch data
        
    Returns:
        List of refer_boxes tensors or None if not present
    """
    if "refer_boxes" not in micro_batch:
        return None
    
    refer_boxes = micro_batch["refer_boxes"]
    
    # Handle different input formats
    if refer_boxes is None:
        return None
    
    # If it's already a list of tensors, return as is
    if isinstance(refer_boxes, list):
        return refer_boxes
    
    # If it's a numpy array of objects (from non_tensor_batch)
    if hasattr(refer_boxes, 'tolist'):
        boxes_list = refer_boxes.tolist()
        result = []
        for boxes in boxes_list:
            if boxes is None:
                result.append(None)
            elif isinstance(boxes, torch.Tensor):
                result.append(boxes)
            else:
                # Convert to tensor
                result.append(torch.tensor(boxes, dtype=torch.float32))
        return result
    
    # If it's a single tensor [batch, max_boxes, 4]
    if isinstance(refer_boxes, torch.Tensor):
        if refer_boxes.dim() == 3:
            return [refer_boxes[i] for i in range(refer_boxes.shape[0])]
        return [refer_boxes]
    
    return None


class SpatialDataParallelPPOActor(DataParallelPPOActor):
    """
    FSDP DataParallel PPO Actor with Spatial Linking support.
    
    This actor extends the base DataParallelPPOActor to:
    1. Extract refer_boxes from micro_batch during forward pass
    2. Pass refer_boxes to the spatial linking model
    3. Handle spatial linking specific data flow
    
    The spatial linking model enhances <|box_end|> tokens with cross-attention
    to image patches within bounding boxes, enabling better region-based reasoning.
    
    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module (SpatialLinkingInteractionModel)
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
        use_spatial_linking (bool): Whether spatial linking is enabled. Defaults to True.
    """

    def __init__(
        self, 
        config: ActorConfig, 
        actor_module: nn.Module, 
        actor_optimizer: torch.optim.Optimizer = None,
        use_spatial_linking: bool = True,
    ):
        super().__init__(config, actor_module, actor_optimizer)
        self.use_spatial_linking = use_spatial_linking
        
        # Check if model supports spatial linking
        self._has_spatial_linking = self._check_spatial_linking_support()
        
        if self.use_spatial_linking and not self._has_spatial_linking:
            logger.warning(
                "Spatial linking enabled but model does not have spatial_linking module. "
                "refer_boxes will be ignored."
            )
    
    def _check_spatial_linking_support(self) -> bool:
        """Check if the actor module supports spatial linking."""
        # Navigate through FSDP/PEFT wrappers to find the base model
        model = self.actor_module
        
        if hasattr(model, '_fsdp_wrapped_module'):
            model = model._fsdp_wrapped_module
        if hasattr(model, 'base_model'):
            model = model.base_model
        if hasattr(model, 'model'):
            model = model.model
        
        return hasattr(model, 'spatial_linking')
    
    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a micro batch with spatial linking support.
        
        This extends the parent method to extract and pass refer_boxes
        to the spatial linking model during forward pass.
        
        Args:
            micro_batch: Dictionary containing batch data including optional refer_boxes
            temperature: Temperature for logits scaling
            calculate_entropy: Whether to compute entropy
            
        Returns:
            Tuple of (entropy, log_probs) tensors
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])
        
        # Extract refer_boxes for spatial linking
        refer_boxes = None
        if self.use_spatial_linking and self._has_spatial_linking:
            refer_boxes = extract_refer_boxes(micro_batch)
            if refer_boxes is not None:
                # Move refer_boxes to the correct device
                device = micro_batch["input_ids"].device
                refer_boxes = [
                    box.to(device) if box is not None and isinstance(box, torch.Tensor) else box
                    for box in refer_boxes
                ]

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo
                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                
                # Add refer_boxes for spatial linking
                if refer_boxes is not None:
                    extra_args["refer_boxes"] = refer_boxes

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)
                    logits_rmpad.div_(temperature)

                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                
                # Add refer_boxes for spatial linking
                if refer_boxes is not None:
                    extra_args["refer_boxes"] = refer_boxes

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    @GPUMemoryLogger(role="spatial dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """
        Compute the log probability with spatial linking support.
        
        This method extends the parent to include refer_boxes in the data selection.
        
        Args:
            data (DataProto): DataProto containing input data and optionally refer_boxes
            calculate_entropy: Whether to compute entropy
            
        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_refer_boxes = "refer_boxes" in data.non_tensor_batch.keys()
        
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if has_refer_boxes:
            non_tensor_select_keys.append("refer_boxes")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.batch.split(micro_batch_size)
            # Handle non_tensor_batch splitting for refer_boxes
            if has_refer_boxes:
                refer_boxes_all = data.non_tensor_batch.get("refer_boxes", None)
                if refer_boxes_all is not None:
                    # Split refer_boxes along batch dimension
                    num_batches = len(micro_batches)
                    batch_size = len(refer_boxes_all)
                    boxes_per_batch = batch_size // num_batches
                    refer_boxes_splits = [
                        refer_boxes_all[i * boxes_per_batch : (i + 1) * boxes_per_batch]
                        for i in range(num_batches)
                    ]

        all_entropy = []
        all_log_probs = []

        for idx, micro_batch in enumerate(micro_batches):
            # Convert to dict and add refer_boxes
            if isinstance(micro_batch, dict):
                micro_batch_dict = micro_batch
            else:
                micro_batch_dict = {k: v for k, v in micro_batch.items()}
            
            # Add multi_modal_inputs if present
            if has_multi_modal_inputs and not use_dynamic_bsz:
                multi_modal_inputs_all = data.non_tensor_batch.get("multi_modal_inputs", None)
                if multi_modal_inputs_all is not None:
                    batch_size = len(multi_modal_inputs_all)
                    num_batches = len(micro_batches)
                    items_per_batch = batch_size // num_batches
                    micro_batch_dict["multi_modal_inputs"] = multi_modal_inputs_all[
                        idx * items_per_batch : (idx + 1) * items_per_batch
                    ]
            
            # Add refer_boxes if present
            if has_refer_boxes and not use_dynamic_bsz:
                if refer_boxes_all is not None:
                    micro_batch_dict["refer_boxes"] = refer_boxes_splits[idx]

            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    micro_batch_dict, temperature=temperature, calculate_entropy=calculate_entropy
                )
            all_log_probs.append(log_probs)
            if calculate_entropy:
                all_entropy.append(entropy)

        log_probs = torch.concat(all_log_probs, dim=0)

        if use_dynamic_bsz:
            batch_idx = sum(batch_idx_list, [])
            log_probs = restore_dynamic_batch(log_probs, batch_idx)

        if calculate_entropy:
            entropy = torch.concat(all_entropy, dim=0)
            if use_dynamic_bsz:
                entropy = restore_dynamic_batch(entropy, batch_idx)
            return log_probs, entropy

        return log_probs

    @GPUMemoryLogger(role="spatial dp actor update", logger=logger)
    def update_policy(self, data: DataProto):
        """
        Update policy with spatial linking support.
        
        This extends the parent to handle refer_boxes in the training loop.
        """
        self.actor_module.train()

        use_dynamic_bsz = self.config.use_dynamic_bsz
        temperature = data.meta_info["temperature"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_refer_boxes = "refer_boxes" in data.non_tensor_batch.keys()

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "response_mask",
        ]
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if has_refer_boxes:
            non_tensor_select_keys.append("refer_boxes")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Use parent's update_policy implementation for the actual update
        # The _forward_micro_batch override will handle refer_boxes
        return super().update_policy(data)
