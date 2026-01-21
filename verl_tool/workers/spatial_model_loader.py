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
Spatial Linking Model Loader for GRPO Training.

This module provides utilities for loading the SpatialLinkingInteractionModel
for use in verl-tool's GRPO training pipeline.

The spatial linking model extends Qwen3-VL with:
- Cross-attention from <|box_end|> tokens to image patches within bounding boxes
- Three-region representation: person, object, interaction (union)
- Residual addition to preserve token semantics
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig

# Add spatial_linking_training to path if not already present
SPATIAL_LINKING_PATH = Path(__file__).parent.parent.parent.parent / "spatial_linking_training"
if str(SPATIAL_LINKING_PATH) not in sys.path:
    sys.path.insert(0, str(SPATIAL_LINKING_PATH))

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def load_spatial_linking_model(
    local_path: str,
    config: Union[DictConfig, Dict[str, Any]],
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    trust_remote_code: bool = True,
    device_map: Optional[str] = None,
    **kwargs,
) -> "SpatialLinkingInteractionModel":
    """
    Load the SpatialLinkingInteractionModel for GRPO training.
    
    This function loads the spatial linking model which extends Qwen3-VL with:
    - Cross-attention module linking <|box_end|> tokens to image patches
    - Support for three-region representation (person, object, interaction)
    
    Args:
        local_path: Path to the base model (e.g., Qwen/Qwen3-VL-8B-Instruct)
        config: Configuration dictionary or DictConfig with model settings
        torch_dtype: Data type for model weights
        attn_implementation: Attention implementation to use
        trust_remote_code: Whether to trust remote code
        device_map: Device map for model placement
        **kwargs: Additional arguments passed to from_pretrained
        
    Returns:
        SpatialLinkingInteractionModel instance
    """
    # Import spatial linking model
    try:
        from spatial_linking_training.models.spatial_model import SpatialLinkingInteractionModel
    except ImportError as e:
        raise ImportError(
            f"Failed to import SpatialLinkingInteractionModel. "
            f"Make sure spatial_linking_training is in the Python path. "
            f"Error: {e}"
        )
    
    # Convert DictConfig to dict if needed
    if hasattr(config, 'model'):
        model_config = config.model if isinstance(config.model, dict) else dict(config.model)
    else:
        model_config = config if isinstance(config, dict) else dict(config)
    
    logger.info(f"Loading SpatialLinkingInteractionModel from: {local_path}")
    
    # Load HF config first
    hf_config = AutoConfig.from_pretrained(
        local_path,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    
    # Load the spatial linking model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SpatialLinkingInteractionModel.from_pretrained(
            pretrained_model_name_or_path=local_path,
            torch_dtype=torch_dtype,
            config=hf_config,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            device_map=device_map,
            **kwargs,
        )
    
    # Optionally load spatial linking weights from a separate checkpoint
    spatial_linking_checkpoint = model_config.get("spatial_linking_checkpoint")
    if spatial_linking_checkpoint and os.path.exists(spatial_linking_checkpoint):
        logger.info(f"Loading spatial linking weights from: {spatial_linking_checkpoint}")
        state_dict = torch.load(spatial_linking_checkpoint, map_location="cpu")
        
        # Handle different checkpoint formats
        if "spatial_linking" in state_dict:
            spatial_state = state_dict["spatial_linking"]
        elif any(k.startswith("spatial_linking.") for k in state_dict.keys()):
            spatial_state = {
                k.replace("spatial_linking.", ""): v 
                for k, v in state_dict.items() 
                if k.startswith("spatial_linking.")
            }
        else:
            spatial_state = state_dict
        
        model.spatial_linking.load_state_dict(spatial_state, strict=False)
        logger.info("Spatial linking weights loaded successfully")
    
    # Configure freezing behavior
    freeze_spatial_linking = model_config.get("freeze_spatial_linking", False)
    freeze_vision = model_config.get("freeze_vision_tower", True)
    freeze_llm = model_config.get("freeze_llm", False)  # Usually False since we train with LoRA
    
    if freeze_spatial_linking:
        logger.info("Freezing spatial linking module")
        for param in model.spatial_linking.parameters():
            param.requires_grad = False
    else:
        logger.info("Spatial linking module is trainable")
        for param in model.spatial_linking.parameters():
            param.requires_grad = True
    
    # Log trainable params info
    if hasattr(model, 'get_trainable_params_info'):
        params_info = model.get_trainable_params_info()
        logger.info(f"Spatial linking params: {params_info.get('spatial_linking_params', 'N/A')}")
    
    return model


def apply_lora_to_spatial_model(
    model: "SpatialLinkingInteractionModel",
    lora_config: Union[DictConfig, Dict[str, Any], LoraConfig],
    lora_adapter_path: Optional[str] = None,
) -> "SpatialLinkingInteractionModel":
    """
    Apply LoRA adapters to the spatial linking model.
    
    Args:
        model: SpatialLinkingInteractionModel instance
        lora_config: LoRA configuration (dict or LoraConfig)
        lora_adapter_path: Optional path to pre-trained LoRA adapters
        
    Returns:
        Model with LoRA adapters applied
    """
    model.enable_input_require_grads()
    
    if lora_adapter_path is not None and os.path.exists(lora_adapter_path):
        logger.info(f"Loading pre-trained LoRA adapters from: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=True)
        peft_config = model.peft_config["default"]
        if isinstance(peft_config.task_type, str):
            peft_config.task_type = TaskType.CAUSAL_LM
    else:
        # Create new LoRA config
        if isinstance(lora_config, LoraConfig):
            peft_lora_config = lora_config
        else:
            lora_dict = lora_config if isinstance(lora_config, dict) else dict(lora_config)
            peft_lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_dict.get("lora_rank", 64),
                lora_alpha=lora_dict.get("lora_alpha", 128),
                target_modules=lora_dict.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                exclude_modules=lora_dict.get("exclude_modules", None),
                lora_dropout=lora_dict.get("lora_dropout", 0.05),
                bias="none",
            )
        
        logger.info(f"Applying new LoRA config: rank={peft_lora_config.r}, alpha={peft_lora_config.lora_alpha}")
        model = get_peft_model(model, peft_lora_config)
    
    return model


def get_spatial_linking_model_class():
    """
    Get the SpatialLinkingInteractionModel class.
    
    This is useful for type checking and isinstance checks.
    
    Returns:
        SpatialLinkingInteractionModel class
    """
    try:
        from spatial_linking_training.models.spatial_model import SpatialLinkingInteractionModel
        return SpatialLinkingInteractionModel
    except ImportError:
        return None


def is_spatial_linking_model(model) -> bool:
    """
    Check if a model is a SpatialLinkingInteractionModel.
    
    Args:
        model: Model to check
        
    Returns:
        True if model is a SpatialLinkingInteractionModel
    """
    model_class = get_spatial_linking_model_class()
    if model_class is None:
        return False
    
    # Handle PEFT/FSDP wrapped models
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
    if hasattr(base_model, 'model'):
        base_model = base_model.model
    if hasattr(base_model, '_fsdp_wrapped_module'):
        base_model = base_model._fsdp_wrapped_module
    
    return isinstance(base_model, model_class) or hasattr(base_model, 'spatial_linking')
