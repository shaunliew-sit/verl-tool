# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VerlTool** is a unified and easy-to-extend tool-agent training framework built on [verl](https://github.com/volcengine/verl) (Bytedance's RL framework). It specializes in reinforcement learning for LLM agents that use tools during multi-turn interactions.

**Key architectural paradigm**: Tool-as-environment with complete decoupling of actor rollout and environment interaction. Tool servers run independently from the RL training process, communicating via HTTP API.

**Technology stack**: verl (0.6.0), vLLM (0.11.0), PyTorch 2.6.0, Ray 2.43.0, Transformers 4.51.3, FastAPI 0.115.12, Hydra 1.3.2

**Current branch `hoi`**: Working on Human-Object Interaction detection using vision-language models (Qwen3-VL) with GRPO training.

## Installation & Setup

### UV Installation (Recommended)
```bash
# Initialize submodules
git submodule update --init --recursive

# Install with UV
uv sync
source .venv/bin/activate
uv pip install -e verl
uv pip install -e ".[vllm,acecoder,torl,search_tool]"
uv pip install "flash-attn==2.8.3" --no-build-isolation
```

### Conda Installation (Alternative)
```bash
git submodule update --init --recursive
conda create --name verl-tool-env python=3.10
conda activate verl-tool-env
pip install -e verl
pip install -e ".[vllm,acecoder,torl,search_tool]"
pip install "flash-attn==2.8.3" --no-build-isolation
```

### Optional Dependencies
Available in [pyproject.toml](pyproject.toml):
- `[vllm]` - vLLM inference engine
- `[tool_browser]` - Browser tool (mini_webarena)
- `[acecoder]` - Code completion with AceCoder
- `[torl]` - Tool-integrated reasoning with math verification
- `[search_tool]` - Google/Bing search tools
- `[sql_tool]` - SQL database tools
- `[mcp_tool]` - Model Context Protocol tools
- `[python_code_dep]` - Scientific computing dependencies

### Megatron Backend (Optional)
```bash
uv pip install megatron-core
uv pip install --no-build-isolation transformer-engine[pytorch]
```
Then set `strategy="megatron"` in training scripts instead of `"fsdp"`.

## Training Commands

### Data Preprocessing
```bash
python examples/data_preprocess/deepmath.py \
    --data_source zwhe99/DeepMath-103K \
    --local_dir data/deepmath_torl \
    --sys_prompt_style torl
```

### Single-Node Training
Each training recipe has its own script in [examples/train/](examples/train/):
```bash
# Math tool-integrated reasoning (ToRL)
bash examples/train/math_tir/train_1.5b_grpo.sh

# Search-R1
bash examples/train/search_r1/train_3b.sh

# SQL tool use
bash examples/train/skysql/train_7b.sh

# AceCoder (code completion)
bash examples/train/acecoder/train_with_tool.sh
```

### Multi-Node Training (SLURM)
```bash
sbatch --account ${your_account} --nodes 2 \
    examples/train/math_tir/train_7b_grpo_multi_node_slurm.sh \
    ${container_path} ${mount_path} ${workdir_inside_container}
```

### Training Configuration via Hydra Overrides
Training scripts use Hydra configuration. Key parameters are passed as command-line overrides:

```bash
python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_data \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.agent.enable_agent=True \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_turns=10 \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    reward_model.reward_manager=torl \
    trainer.experiment_name=$run_name
```

**Key configuration patterns:**
- `algorithm.adv_estimator`: Algorithm choice (`grpo`, `gae` for PPO, `dapo`)
- `actor_rollout_ref.agent.*`: Agent-specific settings (tool server URL, max turns, stop tokens)
- `actor_rollout_ref.actor.strategy`: Distributed strategy (`fsdp`, `fsdp2`, `megatron`)
- `reward_model.reward_manager`: Reward function (`torl`, `acecoder`, `hoi_reward_v2`, etc.)
- `trainer.rollout_data_dir`: Training logs location

### Training Logs
During training, generated responses and rewards are recorded in:
```
verl_step_records/$run_name/
```

For multi-node training, logs may be in: `{checkpoint_dir}/step_records/`

### Memory Optimization Tips
From [assets/docs/training_guide.md](assets/docs/training_guide.md):

**Low VRAM GPUs:**
- Set `do_offload=True`, `enforce_eager=True`, `tensor_parallel_size=1`
- Set `use_dynamic_bsz=False` and use small `ppo_micro_batch_size_per_gpu`

**High VRAM GPUs:**
- Set `do_offload=False` and `use_dynamic_bsz=True` to speed up training

**If VLLM generation gets stuck:**
- Lower `workers_per_tool` and reduce `gpu_memory_utilization`

**If CPU OOM during rollout:**
- Set `do_offload=False` and lower `gpu_memory_utilization`

**If GPU OOM on rank-0 during loading:**
- Change `strategy="fsdp2"` to `strategy="fsdp"`

## Evaluation Commands

### Start Tool Server + API Service
```bash
# 1. Start tool server
host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve \
    --host $host --port $port \
    --tool_type "python_code" \
    --workers_per_tool 32 \
    --done_if_invalid True \
    --silent True &
server_pid=$!

# 2. Start OpenAI-compatible API service
model_path=VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step
action_stop_tokens_file=$(mktemp)
echo -n '```output' > $action_stop_tokens_file

python eval_service/app.py \
    --host 0.0.0.0 \
    --port 5000 \
    --tool-server-url $tool_server_url \
    --model $model_path \
    --max_turns 4 \
    --action_stop_tokens $action_stop_tokens_file \
    --tensor-parallel-size 1 \
    --num-models 1
```

Or use the integrated script:
```bash
bash eval_service/scripts/start_api_service.sh &
```

### Test API Service
```bash
model_name=VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step
test_task=math  # or code
test_type=chat_completion  # or completion
base_url=http://localhost:5000

python eval_service/test/test_api.py \
    --model_name $model_name \
    --test_task $test_task \
    --test_type $test_type \
    --base_url $base_url
```

### Benchmark Evaluation
Benchmarks are in [benchmarks/](benchmarks/) (submodules):
- `math-evaluation-harness` - Math problems (GSM8K, MATH, AIME, AMC)
- `bigcodebench` - Code generation
- `evalplus` - HumanEval, MBPP
- `LiveCodeBench` - Live code evaluation

See [assets/docs/evaluation.md](assets/docs/evaluation.md) and [benchmarks/README.md](benchmarks/README.md) for setup.

## Code Architecture

### Core Training Flow

**Entry point:** [verl_tool/trainer/main_ppo.py](verl_tool/trainer/main_ppo.py)
- Handles PPO, GRPO, and DAPO training via Hydra configuration
- Launches Ray-based distributed training

**Key inheritance chain:**
```
verl.ActorRolloutRefWorker
  ↓ (inherit)
verl_tool.workers.fsdp_workers.AgentActorRolloutRefWorker
```

The `AgentActorRolloutRefWorker` in [verl_tool/workers/fsdp_workers.py](verl_tool/workers/fsdp_workers.py):
- Overrides `generate_sequences()` to enable agent behavior
- Delegates multi-turn tool interactions to `AgentActorManager.run_llm_loop()`

**Multi-turn interaction manager:** [verl_tool/llm_agent/manager.py](verl_tool/llm_agent/manager.py)
- `AgentActorManager` handles the agent loop: model generates action → tool server returns observation → repeat until done
- Manages environment state storage/reload per trajectory

**Agent configuration:** [verl_tool/llm_agent/config.py](verl_tool/llm_agent/config.py)
- `AgentActorConfig` defines all agent parameters
- Set via `actor_rollout_ref.agent.{param_name}=value` in training scripts

### Tool System

**Tool server launcher:** [verl_tool/servers/serve.py](verl_tool/servers/serve.py)
```bash
python -m verl_tool.servers.serve \
    --host localhost --port 5000 \
    --tool_type "python_code,bash_terminal" \
    --workers_per_tool 512 \
    --use_ray=True \
    --max_concurrent_requests=8192
```

**Tool implementations:** [verl_tool/servers/tools/](verl_tool/servers/tools/)

20+ tools available:
- Code execution: `python_code.py`, `ipython_code.py`, `piston.py`
- Web: `text_browser.py`, `bash_terminal.py`
- Search: `google_search.py`, `bing_search.py`, `search_retrieval.py`
- Vision: `pixel_reasoner.py`, `audio_crop.py`
- Database: `sql.py`, `hoi_detector.py`
- Integration: `mcp_interface.py` (Model Context Protocol)

**Tool request/response format:**

Request to tool server:
```json
{
    "trajectory_ids": ["traj_1", "traj_2"],
    "actions": ["action_1", "action_2"],
    "finish": [false, true],
    "is_last_step": [false, false]
}
```

Response from tool server:
```json
{
    "observations": ["obs_1", "obs_2"],
    "dones": [false, true],
    "valids": [true, false]
}
```

**Tool flow:**
1. Tool server receives request with actions
2. For each action, tries to parse using all active tools' `parse_action()` method
3. Matching tool's `get_observations()` executes the action and returns observation
4. If no tool matches and `done_if_invalid=True`, trajectory is marked done

**Testing tools:**
```bash
# Test python_code tool
python -m verl_tool.servers.tests.test_python_code_tool python \
    --url=http://localhost:5000/get_observation

# Test efficiency
python verl_tool/servers/tests/test_ipython_efficiency.py \
    --url=http://localhost:5000/get_observation \
    --requests=512 --concurrency=512
```

### Reward System

**Reward managers:** [verl_tool/workers/reward_manager/](verl_tool/workers/reward_manager/)

15+ specialized reward implementations:
- `torl.py` - Tool-integrated reasoning for math
- `acecoder.py` - Code completion with execution feedback
- `hoi_reward_v2.py` - HOI detection reward
- `search_r1_qa_em.py` - Search-based QA with exact match
- `gsm8k_code.py`, `sqlcoder.py`, `mcp_universe_eval.py`, etc.

**Reward manager selection:**
Set via `reward_model.reward_manager={name}` in training config.

**Custom reward managers:**
Each reward manager implements logic in its file. To debug rewards, check:
```
verl_step_records/$run_name/
```
for recorded responses, actions, observations, and computed rewards.

### Evaluation Service

**OpenAI-compatible API:** [eval_service/app.py](eval_service/app.py)
- FastAPI server that wraps model + tool server interaction
- Exposes OpenAI-like chat/completion endpoints
- Handles multi-turn tool calling internally

**Model service logic:** [eval_service/model_service.py](eval_service/model_service.py)
- Manages multiple vLLM instances for parallel inference
- Integrates with tool server to handle action/observation flow
- Returns final response after multi-turn interaction completes

**Configuration:** [eval_service/config.py](eval_service/config.py)
- Default parameters overridden in `eval_service/scripts/start_api_service.sh`

## Key Architectural Patterns

### Decoupling Strategy

VerlTool achieves decoupling of RL training and tool calling via inheritance:

1. **Inherit from verl's base worker:**
   ```python
   class AgentActorRolloutRefWorker(ActorRolloutRefWorker):
       ...
   ```

2. **Override `generate_sequences()`:**
   ```python
   def generate_sequences(self, prompts):
       if not self.agent_config.enable_agent:
           # Standard verl behavior
           output = self.rollout.generate_sequences(prompts=prompts)
       else:
           # Agent behavior with tool calling
           output = self.manager.run_llm_loop(prompts)
       return output
   ```

3. **Tool server runs independently:**
   - Tool server is a separate HTTP API process
   - Launched before training starts
   - Communicates via `tool_server_url` (e.g., `http://localhost:5000/get_observation`)

See [assets/docs/sync_design.md](assets/docs/sync_design.md) for detailed design.

### Multi-turn Interaction Pattern

**Action stop tokens:** Model generation stops at special tokens (e.g., `` ```output ``) that signal a tool call.

**Agent loop:**
1. Model generates text until action stop token
2. Extract action from generated text
3. Send action to tool server → receive observation
4. Append observation to prompt
5. Repeat until `max_turns` reached or model signals finish

**Environment state management:**
- Each trajectory has persistent environment state (e.g., Python REPL session, browser context)
- State stored per `trajectory_id`
- State cleaned up when trajectory finishes via special `finish` tool

**Configuration:**
- `max_turns`: Maximum interaction rounds (default: 10)
- `action_stop_tokens`: Token(s) that trigger tool calling (e.g., `` ```output ``)
- `enable_agent`: Enable/disable agent behavior
- `mask_observations`: Whether to mask observations in KL loss

### Async Rollout (Trajectory-level)

VerlTool supports asynchronous rollout for 2x+ speedup:
- Set `actor_rollout_ref.rollout.mode='async'` in training script
- See [assets/docs/asyncRL.md](assets/docs/asyncRL.md) for details

## Training Recipe Examples

See [examples/train/README.md](examples/train/README.md) for all recipes.

**Available recipes:**
- `math_tir/` - Math with tool-integrated reasoning
- `acecoder/` - Code completion with IPython
- `search_r1/` - Search-based reasoning
- `deepsearch/` - Web search (Google/Bing)
- `skysql/` - SQL database interaction
- `pixel_reasoner/` - Visual reasoning with image tools
- `swe/` - Software engineering tasks
- `mcp_universe/` - MCP-based tool evaluation
- `hoi/` - Human-Object Interaction (current branch)

## Important Development Notes

- **Training logs location:** `verl_step_records/$run_name/` contains all generated responses, actions, observations, and rewards per step
- **Reward manager logic:** Defined in `verl_tool/workers/reward_manager/{name}.py`
- **Agent configuration:** Set via `actor_rollout_ref.agent.{param}=value` in training command
- **verl submodule:** Core RL algorithms (PPO, GAE, FSDP, etc.) handled by verl framework (version 0.6.0)
- **LLaMA-Factory submodule:** Used for SFT (Supervised Fine-Tuning) before RL training
- **Python version:** Requires Python 3.10+
- **Environment variables:** Configure API keys in `.env` (see `.env.example`)
- **Git submodules:** Remember to run `git submodule update --init --recursive` after cloning

## Related Documentation

- [README.md](README.md) - Main project overview
- [assets/docs/install.md](assets/docs/install.md) - Installation guide
- [assets/docs/training_guide.md](assets/docs/training_guide.md) - Training guide
- [assets/docs/evaluation.md](assets/docs/evaluation.md) - Evaluation guide
- [assets/docs/sync_design.md](assets/docs/sync_design.md) - Synchronous rollout design
- [assets/docs/asyncRL.md](assets/docs/asyncRL.md) - Asynchronous rollout design
- [assets/docs/tool_server.md](assets/docs/tool_server.md) - Tool server implementation
- [assets/docs/agent_config.md](assets/docs/agent_config.md) - Agent configuration
- [assets/docs/DAPO.md](assets/docs/DAPO.md) - DAPO algorithm
- [assets/docs/training_results.md](assets/docs/training_results.md) - Training benchmarks
- [assets/docs/contributing.md](assets/docs/contributing.md) - Contributing guide
