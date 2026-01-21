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
Spatial Linking FSDP Workers for GRPO Training.

This module extends the base ActorRolloutRefWorker to support spatial linking
models during GRPO training. The key changes are:

1. Custom model loading to use SpatialLinkingInteractionModel
2. Support for refer_boxes in the data pipeline
3. Spatial linking module parameter management (freeze/unfreeze)

Usage:
    In main_ppo.py, import this worker when use_spatial_linking=True:
    
    if config.actor_rollout_ref.model.get("use_spatial_linking", False):
        from verl_tool.workers.spatial_fsdp_workers import SpatialActorRolloutRefWorker
        actor_rollout_cls = SpatialActorRolloutRefWorker
"""

import logging
import os
import warnings
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
)
from verl.utils.model import update_model_config
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.fs import copy_to_local
from verl.utils.py_functional import convert_to_regular_types

from .spatial_model_loader import (
    load_spatial_linking_model,
    apply_lora_to_spatial_model,
    is_spatial_linking_model,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_vl_model_vision_tower(vl_model_instance):
    """
    Util to extract Vision Tower from a VL model instance.
    Handles both standard and spatial linking models.
    """
    if hasattr(vl_model_instance, "model") and hasattr(vl_model_instance.model, "visual"):
        return vl_model_instance.model.visual
    elif hasattr(vl_model_instance, "visual"):
        return vl_model_instance.visual
    return None


class SpatialActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    FSDP Worker with Spatial Linking support for GRPO training.
    
    This worker extends the base ActorRolloutRefWorker to:
    1. Load SpatialLinkingInteractionModel instead of standard Qwen3-VL
    2. Handle spatial linking module parameters (freeze/unfreeze)
    3. Support refer_boxes in the training pipeline
    
    The spatial linking model enhances <|box_end|> tokens with cross-attention
    to image patches within bounding boxes, enabling better region-based reasoning.
    """
    
    def _build_model_optimizer(
        self,
        model_path: str,
        fsdp_config,
        optim_config,
        override_model_config: dict,
        use_remove_padding: bool = False,
        use_fused_kernels: bool = False,
        enable_gradient_checkpointing: bool = False,
        trust_remote_code: bool = True,
        use_liger: bool = False,
        role: str = "actor",
        enable_activation_offload: bool = False,
    ) -> Tuple:
        """
        Build model and optimizer with spatial linking support.
        
        This overrides the base method to:
        1. Check if use_spatial_linking is enabled
        2. Load SpatialLinkingInteractionModel if enabled
        3. Apply appropriate freezing/training settings
        """
        use_spatial_linking = self.config.model.get("use_spatial_linking", False)
        
        if not use_spatial_linking:
            # Fall back to parent implementation
            return super()._build_model_optimizer(
                model_path=model_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=enable_gradient_checkpointing,
                trust_remote_code=trust_remote_code,
                use_liger=use_liger,
                role=role,
                enable_activation_offload=enable_activation_offload,
            )
        
        # Use spatial linking model
        if self.rank == 0:
            print(f"[{role}] Building SpatialLinkingInteractionModel...")
        
        from verl.utils import hf_tokenizer, hf_processor
        from verl.models.transformers.monkey_patch import apply_monkey_patch
        from transformers import AutoConfig, GenerationConfig
        
        local_path = model_path
        
        # Load tokenizer and processor
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
        
        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template
        
        # Determine torch dtype
        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        # Load model config for overrides
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        
        # Handle VL model attention implementation
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"
        
        # Load generation config if available
        try:
            self.generation_config = GenerationConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        except Exception:
            self.generation_config = None
        
        # Apply config overrides
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")
        
        # Initialize context for model creation
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
        )
        
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Load spatial linking model
            actor_module = load_spatial_linking_model(
                local_path=local_path,
                config=self.config,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=trust_remote_code,
            )
            
            if self.rank == 0:
                print(f"[{role}] SpatialLinkingInteractionModel loaded successfully")
                if hasattr(actor_module, 'get_trainable_params_info'):
                    params_info = actor_module.get_trainable_params_info()
                    print(f"[{role}] Spatial linking params: {params_info.get('spatial_linking_params', 'N/A')}")
            
            # Apply Liger kernel if enabled
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=actor_module)
            
            # Apply monkey patches
            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )
            
            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )
            
            # Convert to correct dtype
            actor_module.to(torch_dtype)
            
            # Enable gradient checkpointing if requested
            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        # Apply LoRA if configured
        if self._is_lora:
            if self.rank == 0:
                print(f"[{role}] Applying LoRA to spatial linking model")
            
            actor_module.enable_input_require_grads()
            
            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                if self.rank == 0:
                    print(f"[{role}] Loading pre-trained LoRA adapter from: {lora_adapter_path}")
                
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))
                actor_module = PeftModel.from_pretrained(actor_module, local_adapter_path, is_trainable=True)
                peft_config = actor_module.peft_config["default"]
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM
            else:
                # Create new LoRA config
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "exclude_modules": convert_to_regular_types(self.config.model.exclude_modules),
                    "bias": "none",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
        
        # Handle vision tower freezing
        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print(f"[{role}] Vision tower is set to not trainable")
        
        # Handle spatial linking freezing
        freeze_spatial_linking = self.config.model.get("freeze_spatial_linking", False)
        if freeze_spatial_linking:
            # Find and freeze spatial linking parameters
            base_model = actor_module
            if hasattr(actor_module, 'base_model'):
                base_model = actor_module.base_model
            if hasattr(base_model, 'model'):
                base_model = base_model.model
            
            if hasattr(base_model, 'spatial_linking'):
                for param in base_model.spatial_linking.parameters():
                    param.requires_grad = False
                if self.rank == 0:
                    print(f"[{role}] Spatial linking module is frozen")
        else:
            if self.rank == 0:
                print(f"[{role}] Spatial linking module is trainable")
        
        # Now continue with FSDP wrapping (same as parent)
        # This follows the same pattern as the base class
        from verl.utils.fsdp_utils import (
            fsdp_version,
            apply_fsdp2,
            fsdp2_load_full_state_dict,
            get_shard_placement_fn,
            MixedPrecisionPolicy,
            CPUOffloadPolicy,
        )
        from verl.workers.config.optimizer import build_optimizer
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
        
        fsdp_strategy = fsdp_config.get("strategy", "fsdp")
        param_dtype = PrecisionType.to_dtype(fsdp_config.get("param_dtype", "bf16"))
        reduce_dtype = PrecisionType.to_dtype(fsdp_config.get("reduce_dtype", "fp32"))
        
        # Get FSDP mesh
        fsdp_size = fsdp_config.get("fsdp_size", -1)
        if fsdp_size < 0 or fsdp_size >= self.world_size:
            fsdp_mesh = self.device_mesh
        else:
            fsdp_mesh = self.device_mesh["fsdp"]
        
        # Apply FSDP
        if fsdp_strategy == "fsdp":
            # FSDP1 implementation
            from torch.distributed.fsdp import CPUOffload
            
            sharding_strategy = ShardingStrategy.FULL_SHARD if self.device_mesh.ndim == 1 else ShardingStrategy.HYBRID_SHARD
            mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            
            auto_wrap_policy = get_fsdp_wrap_policy(model=actor_module, config=fsdp_config.get("wrap_policy", {}))
            cpu_offload = None
            if fsdp_config.get("offload_policy", False):
                cpu_offload = CPUOffload(offload_params=True)
                self._is_offload_param = True
            
            actor_module_fsdp = FSDP(
                actor_module,
                auto_wrap_policy=auto_wrap_policy,
                device_id=self.local_rank,
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
                cpu_offload=cpu_offload,
            )
        elif fsdp_strategy == "fsdp2":
            # FSDP2 implementation
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            cpu_offload = None
            if role == "actor" and fsdp_config.get("offload_policy", False):
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.get("reshard_after_forward", True),
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise ValueError(f"Unknown FSDP strategy: {fsdp_strategy}")
        
        # Build optimizer and scheduler
        actor_optimizer = None
        actor_lr_scheduler = None
        if optim_config is not None:
            actor_optimizer, actor_lr_scheduler = build_optimizer(
                model=actor_module_fsdp,
                optim_config=optim_config,
            )
        
        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config


class SpatialAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker, SpatialActorRolloutRefWorker):
    """
    Async version of SpatialActorRolloutRefWorker for async rollout mode.
    
    This combines the async rollout capabilities with spatial linking support.
    """
    pass
