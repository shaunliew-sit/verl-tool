#!/bin/bash
# HOI Detection RL Training Script V2 for Qwen3-VL
# 
# IMPROVEMENTS OVER V1:
# 1. Uses hoi_reward_v2 with verb-first scoring (fixes "riding" vs "holding" issue)
# 2. Increased total training steps from 100 to 200
# 3. Lower learning rate for more stable training
# 4. Added gradient clipping
#
# Usage (8 GPUs - default):
#   bash examples/train/hoi/train_hoi_qwen3vl_v2.sh
#
# Usage (4 GPUs - H100 80GB recommended):
#   N_GPUS=4 BATCH_SIZE=64 TP_SIZE=2 GPU_MEM_UTIL=0.7 DO_OFFLOAD=True \
#   MAX_PROMPT_LEN=8192 MAX_RESPONSE_LEN=4096 MAX_BATCHED_TOKENS=8000 \
#   bash examples/train/hoi/train_hoi_qwen3vl_v2.sh
#
# Prerequisites:
#   - Run data preparation first: python examples/data_preprocess/hoi/prepare_hoi.py
#
# GPU Memory Estimation (Qwen3-VL-4B):
#   - Model weights (bf16): ~8GB
#   - Optimizer states: ~16GB (sharded via FSDP)
#   - KV cache (vLLM): ~20-40GB depending on sequence length
#   - Activations: Variable
#   
#   For 4x H100 (80GB each = 320GB total):
#   - Recommended: Use offloading and reduced sequence lengths
#   - Expected memory usage: ~60-70GB per GPU

set -x

# Dataset configuration
dataset_name=hoi/train_data
train_data=[$(pwd)/data/${dataset_name}/train.parquet]
val_data=[$(pwd)/data/${dataset_name}/val.parquet]

# Model configuration
# Use SFT checkpoint if available, otherwise use base model
# Set MODEL_PATH environment variable to use SFT checkpoint:
#   MODEL_PATH=./checkpoints/hoi_sft/hoi-sft-qwen3vl-4b/latest bash examples/train/hoi/train_hoi_qwen3vl_v2.sh
model_name=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}

# Alternatively use a larger model:
# model_name=Qwen/Qwen3-VL-8B-Instruct

echo "Using model: $model_name"

# RL algorithm configuration
rl_alg=grpo  # grpo or gae(ppo)
n_gpus_per_node=${N_GPUS:-8}  # Set via env var: N_GPUS=4 for 4 GPUs
n_nodes=1
n=8  # Number of samples per prompt (for GRPO group normalization)
batch_size=${BATCH_SIZE:-128}  # Reduce for fewer GPUs: BATCH_SIZE=64
ppo_mini_batch_size=${BATCH_SIZE:-128}

# Sequence length configuration - Adjust for memory constraints
max_prompt_length=${MAX_PROMPT_LEN:-16384}  # Reduce if OOM: MAX_PROMPT_LEN=8192
max_response_length=${MAX_RESPONSE_LEN:-8192}
max_action_length=2048
max_obs_length=4096
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
kl_loss_coef=0.001  # V2: Small KL penalty to prevent divergence
kl_coef=0.001
entropy_coeff=0.01  # V2: Encourage exploration
kl_loss_type=low_var_kl
lr=5e-7  # V2: Lower learning rate for stability

# ===== V2 KEY CHANGE: Use verb-first scoring reward =====
reward_manager=hoi_reward_v2

# GPU and memory configuration
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
tensor_model_parallel_size=${TP_SIZE:-2}
gpu_memory_utilization=${GPU_MEM_UTIL:-0.7}
do_offload=${DO_OFFLOAD:-False}
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1
additional_eos_token_ids=[151645]  # <|im_end|> token id
mask_observations=True
enable_mtrl=True
max_num_batched_tokens=${MAX_BATCHED_TOKENS:-10000}

# Run name
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="hoi-detection-v2"

if [ "$enable_agent" = "True" ]; then
    run_name="${reward_manager}-${strategy}-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}-${run_name_postfix}"
else
    run_name="${reward_manager}-${strategy}-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}-${run_name_postfix}"
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

# Use hoi_detector tool type
python -m verl_tool.servers.serve --host $host --port $port --tool_type "hoi_detector" --workers_per_tool 4 &
server_pid=$!

echo "Tool Server (pid=$server_pid) started at $tool_server_url"
echo "Using HOI Detector tool: zoom_in (aligned with SFT/Chain-of-Focus)"
echo ""
echo "===== V2 Training with Verb-First Scoring ====="
echo "Key improvements:"
echo "  1. Verb must match for any positive reward (referring task)"
echo "  2. Lower learning rate (${lr}) for stability"
echo "  3. Small KL penalty to prevent divergence"
echo "  4. 200 total training steps (vs 100 in v1)"
echo "================================================"
echo ""

# Wait for server to start
sleep 5

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
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
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
    actor_rollout_ref.agent.max_concurrent_trajectories=128 \
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
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
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
    trainer.project_name=$reward_manager \
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

# Cleanup
pkill -P -9 $server_pid
kill -9 $server_pid

