#!/bin/bash
# HOI Detection RL Training Script with Spatial Linking Support
# 
# This script trains a SpatialLinkingInteractionModel for HOI detection using GRPO.
# The spatial linking module enhances <|box_end|> tokens with cross-attention to
# image patches within bounding boxes, enabling better region-based reasoning.
#
# KEY FEATURES:
# 1. Uses SpatialLinkingInteractionModel instead of standard Qwen3-VL
# 2. Supports loading pre-trained spatial linking weights
# 3. Passes refer_boxes through the data pipeline to the model
# 4. LoRA training for efficient fine-tuning
#
# DATA FORMAT:
# The training parquet should include a 'refer_boxes' column with format:
# [
#   [x1, y1, x2, y2],  # person box (0-1000 normalized)
#   [x1, y1, x2, y2],  # object box
#   [x1, y1, x2, y2],  # interaction box (union of person+object)
# ]
#
# Usage (8 GPUs - default):
#   bash examples/train/hoi/train_hoi_spatial_grpo.sh
#
# Usage with SFT checkpoint (recommended):
#   MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct \
#   LORA_ADAPTER_PATH=/path/to/sft-spatial-linking-model \
#   SPATIAL_LINKING_CKPT=/path/to/sft-spatial-linking-model/spatial_linking.pt \
#   bash examples/train/hoi/train_hoi_spatial_grpo.sh
#
# Example with actual checkpoint path:
#   MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct \
#   LORA_ADAPTER_PATH=/workspace/spatial_linking_training/outputs/sft-spatial-linking-model \
#   SPATIAL_LINKING_CKPT=/workspace/spatial_linking_training/outputs/sft-spatial-linking-model/spatial_linking.pt \
#   bash examples/train/hoi/train_hoi_spatial_grpo.sh
#
# Usage (4 GPUs - H100 80GB recommended):
#   N_GPUS=4 BATCH_SIZE=64 TP_SIZE=2 GPU_MEM_UTIL=0.7 DO_OFFLOAD=True \
#   MAX_PROMPT_LEN=8192 MAX_RESPONSE_LEN=4096 MAX_BATCHED_TOKENS=8000 \
#   bash examples/train/hoi/train_hoi_spatial_grpo.sh
#
# Prerequisites:
#   - Run data preparation with refer_boxes and multi-tool:
#     python examples/data_preprocess/hoi/prepare_hoi_multitool.py --local_dir data/hoi/train_data_multitool
#   - Ensure spatial_linking_training module is accessible

set -x

# Add spatial_linking_training to Python path for Ray workers
# This is required because Ray workers don't inherit sys.path modifications
export PYTHONPATH="${PYTHONPATH:-}:/workspace/spatial_linking_training"
export SPATIAL_LINKING_PATH="/workspace/spatial_linking_training"

# Dataset configuration
# NOTE: Use multi-tool spatial-enabled dataset with refer_boxes column
dataset_name=${DATASET_NAME:-hoi/train_data_multitool}
train_data=[$(pwd)/data/${dataset_name}/train.parquet]
val_data=[$(pwd)/data/${dataset_name}/val.parquet]

# Model configuration
# Base model (will be wrapped with SpatialLinkingInteractionModel)
model_name=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}

# Spatial linking checkpoint (optional - loads pre-trained spatial linking weights)
spatial_linking_checkpoint=${SPATIAL_LINKING_CKPT:-null}

# LoRA adapter path (optional - loads pre-trained LoRA adapter from SFT)
lora_adapter_path=${LORA_ADAPTER_PATH:-null}

# Spatial linking training configuration
use_spatial_linking=${USE_SPATIAL_LINKING:-True}
freeze_spatial_linking=${FREEZE_SPATIAL:-False}  # Set to True to train only LoRA
freeze_vision_tower=${FREEZE_VISION:-True}  # Recommended: freeze vision tower

echo "Using model: $model_name"
echo "Spatial linking enabled: $use_spatial_linking"
echo "Spatial linking checkpoint: $spatial_linking_checkpoint"
echo "LoRA adapter path: $lora_adapter_path"
echo "Freeze spatial linking: $freeze_spatial_linking"
echo "Freeze vision tower: $freeze_vision_tower"

# RL algorithm configuration
rl_alg=grpo  # grpo or gae(ppo)
n_gpus_per_node=${N_GPUS:-8}  # Set via env var: N_GPUS=4 for 4 GPUs
n_nodes=1
n=8  # Number of samples per prompt (for GRPO group normalization)
batch_size=${BATCH_SIZE:-64}  # Reduced from 128 to prevent OOM with vision models
ppo_mini_batch_size=${BATCH_SIZE:-64}

# Sequence length configuration - Adjusted for memory constraints
max_prompt_length=${MAX_PROMPT_LEN:-8192}  # Reduced from 16384 to prevent OOM
max_response_length=${MAX_RESPONSE_LEN:-4096}  # Reduced from 8192 to prevent OOM
max_action_length=2048
max_obs_length=2048  # Reduced from 4096
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)

# Sampling configuration
temperature=1.0
top_p=1.0

# Agent configuration
enable_agent=True  # Enable agent for tool use
action_stop_tokens='</tool_call>'
max_turns=3  # Maximum tool interaction turns

# Training configuration
strategy="fsdp2"
kl_loss_coef=0.001
kl_coef=0.001
entropy_coeff=0.01
kl_loss_type=low_var_kl
lr=5e-7

# LoRA configuration (important for spatial linking training)
# Note: Use specific language model modules instead of all-linear because
# vLLM only supports LoRA on language model layers, not vision encoder
lora_rank=${LORA_RANK:-64}
lora_alpha=${LORA_ALPHA:-128}
# Qwen3-VL language model layers (attention + MLP)
# Note: For vLLM compatibility, only target language model layers (not vision encoder)
# Format follows Hydra list syntax like trainer.logger=['console','wandb']
target_modules=${TARGET_MODULES:-"['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']"}

# Reward manager
reward_manager=hoi_reward_v2

# GPU and memory configuration
# Reduced memory settings to prevent OOM during multimodal embedding operations
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
tensor_model_parallel_size=${TP_SIZE:-2}
gpu_memory_utilization=${GPU_MEM_UTIL:-0.70}  # Balanced setting for GPU memory utilization
do_offload=${DO_OFFLOAD:-False}
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1
additional_eos_token_ids=[151645]  # <|im_end|> token id
mask_observations=True
enable_mtrl=True
max_num_batched_tokens=${MAX_BATCHED_TOKENS:-6144}  # Increased from 4096 for better throughput
max_num_seqs=${MAX_NUM_SEQS:-128}  # Keep at 128 to limit concurrent sequences
max_concurrent_trajectories=${MAX_CONCURRENT_TRAJ:-96}  # Increased from 64 for better parallelism

# Run name
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="hoi-spatial-grpo"

if [ "$enable_agent" = "True" ]; then
    run_name="${reward_manager}-${strategy}-spatial-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}-${run_name_postfix}"
else
    run_name="${reward_manager}-${strategy}-spatial-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}-${run_name_postfix}"
fi

export VERL_RUN_ID=$run_name
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
rollout_mode='async'

# Create temp file for action tokens
action_stop_tokens_file="$(pwd)$(mktemp)"
mkdir -p $(dirname $action_stop_tokens_file)
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

# Start tool server
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation

# Use hoi_detector tool type (multi-tool: zoom_in, zoom_out, detect_objects)
python -m verl_tool.servers.serve --host $host --port $port --tool_type "hoi_detector" --workers_per_tool 4 &
server_pid=$!

echo "Tool Server (pid=$server_pid) started at $tool_server_url"
echo "Using HOI Detector tools: zoom_in, zoom_out, detect_objects (multi-tool)"
echo ""
echo "===== Spatial Linking GRPO Training ====="
echo "Key features:"
echo "  1. SpatialLinkingInteractionModel with cross-attention"
echo "  2. refer_boxes for region-based reasoning"
echo "  3. LoRA rank: ${lora_rank}, alpha: ${lora_alpha}"
echo "  4. Spatial linking frozen: ${freeze_spatial_linking}"
echo "============================================"
echo ""

# Wait for server to start
sleep 5

# Build spatial linking config arguments
spatial_config=""
if [ "$use_spatial_linking" = "True" ]; then
    spatial_config="
        actor_rollout_ref.model.use_spatial_linking=True
        actor_rollout_ref.model.freeze_spatial_linking=$freeze_spatial_linking
        actor_rollout_ref.model.freeze_vision_tower=$freeze_vision_tower
        actor_rollout_ref.model.lora_rank=$lora_rank
        actor_rollout_ref.model.lora_alpha=$lora_alpha
        actor_rollout_ref.model.target_modules=$target_modules
    "
    
    if [ "$spatial_linking_checkpoint" != "null" ] && [ -f "$spatial_linking_checkpoint" ]; then
        spatial_config="$spatial_config
            actor_rollout_ref.model.spatial_linking_checkpoint=$spatial_linking_checkpoint
        "
        echo "Will load spatial linking weights from: $spatial_linking_checkpoint"
    fi
    
    if [ "$lora_adapter_path" != "null" ] && [ -d "$lora_adapter_path" ]; then
        spatial_config="$spatial_config
            actor_rollout_ref.model.lora_adapter_path=$lora_adapter_path
        "
        echo "Will load LoRA adapter from: $lora_adapter_path"
    fi
fi

# Run training
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=250 \
    data.dataloader_num_workers=4 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.max_concurrent_trajectories=$max_concurrent_trajectories \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=$max_num_seqs \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.disable_rollout_lora=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=hoi_spatial_grpo \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.rollout_data_dir=$(pwd)/verl_step_records/$run_name \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=200 \
    $spatial_config

# Cleanup
pkill -P -9 $server_pid
kill -9 $server_pid
