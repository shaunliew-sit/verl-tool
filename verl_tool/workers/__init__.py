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
Worker modules for verl-tool GRPO training.

Includes spatial linking support for HOI detection training.
"""

from .spatial_model_loader import (
    load_spatial_linking_model,
    apply_lora_to_spatial_model,
    is_spatial_linking_model,
    get_spatial_linking_model_class,
)

from .spatial_fsdp_workers import (
    SpatialActorRolloutRefWorker,
    SpatialAsyncActorRolloutRefWorker,
)

__all__ = [
    # Spatial model loader
    "load_spatial_linking_model",
    "apply_lora_to_spatial_model",
    "is_spatial_linking_model",
    "get_spatial_linking_model_class",
    # Spatial FSDP workers
    "SpatialActorRolloutRefWorker",
    "SpatialAsyncActorRolloutRefWorker",
]
