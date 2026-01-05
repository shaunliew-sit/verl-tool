# HOI Agent Evaluation

Evaluation scripts for Human-Object Interaction (HOI) detection with tool-calling agents.

## Prerequisites

### Required Packages

Install dependencies for metrics computation:

```bash
# For METEOR score
pip install nltk
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# For BERTScore
pip install bert-score

# For CIDEr (optional, fallback metrics work without it)
pip install pycocoevalcap
```

### Required Servers

You need to start **two servers** before running evaluation:

1. **vLLM Server** - Serves the trained model for inference
2. **Tool Server** (optional) - Provides HOI detector tools (zoom_in, detect_objects, etc.)

> **Note**: The evaluation script can run tools locally if the tool server is not running, but this requires additional GPU memory for Grounding DINO.

## Available Checkpoints

The trained model saves checkpoints during training. Only specific steps save the full HuggingFace model (usable by vLLM):

| Step | Full Model? | Path |
|------|-------------|------|
| 100 | ✓ | `checkpoints/hoi_reward_v2/.../global_step_100/actor/huggingface` |
| 150 | ✓ | `checkpoints/hoi_reward_v2/.../global_step_150/actor/huggingface` |
| 200 | ✓ | `checkpoints/hoi_reward_v2/.../global_step_200/actor/huggingface` |

**Recommendation**: Use `global_step_200` (the latest/final checkpoint) for best performance.

## Quick Start

### Step 1: Start the vLLM Server

```bash
# Using 4 GPUs with tensor parallelism (recommended for faster inference)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/eval/hoi/start_vllm_server.sh \
    checkpoints/hoi_reward_v2/hoi_reward_v2-fsdp2-agent-qwen_qwen3-vl-4b-instruct-grpo-n8-b128-t1.0-lr5e-7-hoi-detection-v2/global_step_200/actor/huggingface \
    8000 \
    hoi-trained \
    4

# Or using single GPU
bash examples/eval/hoi/start_vllm_server.sh \
    checkpoints/hoi_reward_v2/hoi_reward_v2-fsdp2-agent-qwen_qwen3-vl-4b-instruct-grpo-n8-b128-t1.0-lr5e-7-hoi-detection-v2/global_step_200/actor/huggingface \
    8000 \
    hoi-trained \
    1
```

Wait for the server to start (you'll see "Uvicorn running on http://0.0.0.0:8000").

### Step 2: Start the Tool Server (Optional but Recommended)

The tool server provides HOI detector tools (zoom_in, zoom_out, detect_objects with Grounding DINO).

```bash
# In a separate terminal
cd /workspace/verl-tool

# Get host IP and random port
host=$(hostname -i | awk '{print $1}')
port=30100

# Start tool server on GPU 5 (or any available GPU not used by vLLM)
CUDA_VISIBLE_DEVICES=5 python -m verl_tool.servers.serve \
    --host $host \
    --port $port \
    --tool_type "hoi_detector" \
    --workers_per_tool 2

# Note the endpoint URL printed, e.g., http://10.0.0.1:30100
```

> **If you skip this step**: The evaluation script will try to run tools locally, which requires additional GPU memory for Grounding DINO model.

**When do you need the tool server?**
- If the model uses tools (zoom_in, detect_objects) during inference → **Recommended**
- If you're just testing model responses without tool execution → **Not needed**
- If you have limited GPU memory → **Recommended** (offloads Grounding DINO to separate GPU)

### Step 3: Verify Servers are Running

```bash
# Check vLLM server
curl http://localhost:8000/v1/models

# Check tool server (if started)
curl http://<tool_server_host>:<tool_server_port>/health
```

### Step 4: Run Evaluation

#### Option A: Using Shell Scripts (Recommended)

**Debug mode first (to verify everything works):**
```bash
# Test with 10 images and verbose output
MAX_IMAGES=10 VERBOSE=true bash examples/eval/hoi/run_hico_ground_agent.sh
```

**Full evaluation:**
```bash
# HICO Grounding (find person-object pairs for action)
bash examples/eval/hoi/run_hico_ground_agent.sh

# HICO Referring (predict action from bounding boxes)
BERTSCORE_GPU=4 bash examples/eval/hoi/run_hico_referring_agent.sh

# SWIG Grounding
bash examples/eval/hoi/run_swig_ground_agent.sh

# SWIG Referring
BERTSCORE_GPU=4 bash examples/eval/hoi/run_swig_referring_agent.sh
```

**With all options:**
```bash
MAX_IMAGES=100 VERBOSE=true WANDB=true CONCURRENCY=8 BERTSCORE_GPU=4 \
    bash examples/eval/hoi/run_hico_referring_agent.sh
```

#### Option B: Direct Python Command

```bash
# HICO Grounding
python examples/eval/hoi/eval_hoi_agent.py \
    --task grounding \
    --dataset hico \
    --ann-file data/benchmarks_simplified/hico_ground_test_simplified.json \
    --img-prefix data/hico_20160224_det/images/test2015 \
    --endpoint http://localhost:8000/v1 \
    --model hoi-trained \
    --concurrency 8 \
    --save-thinking \
    --output-dir results/hico_ground_agent

# HICO Referring
python examples/eval/hoi/eval_hoi_agent.py \
    --task referring \
    --dataset hico \
    --ann-file data/benchmarks_simplified/hico_action_referring_test_simplified.json \
    --img-prefix data/hico_20160224_det/images/test2015 \
    --endpoint http://localhost:8000/v1 \
    --model hoi-trained \
    --concurrency 8 \
    --bertscore-gpu 4 \
    --save-thinking \
    --output-dir results/hico_referring_agent
```

## Environment Variables

The shell scripts support these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENDPOINT` | `http://localhost:8000/v1` | vLLM server endpoint |
| `MODEL` | `hoi-trained` | Model name served by vLLM |
| `MAX_IMAGES` | (all) | Limit number of images to evaluate |
| `CONCURRENCY` | `8` | Number of concurrent async requests |
| `BERTSCORE_GPU` | `4` | GPU for BERTScore computation |
| `VERBOSE` | `false` | Enable verbose output (auto-enables SAVE_VIZ) |
| `SAVE_VIZ` | `false` | Save visualization images |
| `UNIQUE_RUN` | `true` | Append timestamp to output dir for unique runs |
| `WANDB` | `false` | Enable W&B logging |
| `OUTPUT_DIR` | `results/{task}_agent` | Output directory base name |

### Setting Environment Variables

You can set environment variables in multiple ways:

**Method 1: Inline (single command)**
```bash
MAX_IMAGES=100 VERBOSE=true WANDB=true bash examples/eval/hoi/run_hico_ground_agent.sh
```

**Method 2: Export separately (persists in shell session)**
```bash
export MAX_IMAGES=100
export VERBOSE=true
export WANDB=true
bash examples/eval/hoi/run_hico_ground_agent.sh
```

**Method 3: Multiple variables inline**
```bash
MAX_IMAGES=50 VERBOSE=true WANDB=false CONCURRENCY=4 \
    bash examples/eval/hoi/run_hico_ground_agent.sh
```

## Debugging Mode

For initial testing, limit the number of images and enable verbose output:

### Quick Debug Test (10 images with visualization)

```bash
# HICO Grounding - debug mode with visualization
MAX_IMAGES=10 VERBOSE=true bash examples/eval/hoi/run_hico_ground_agent.sh

# HICO Referring - debug mode with visualization
MAX_IMAGES=10 VERBOSE=true BERTSCORE_GPU=4 bash examples/eval/hoi/run_hico_referring_agent.sh

# SWIG Grounding - debug mode with visualization
MAX_IMAGES=10 VERBOSE=true bash examples/eval/hoi/run_swig_ground_agent.sh

# SWIG Referring - debug mode with visualization
MAX_IMAGES=10 VERBOSE=true BERTSCORE_GPU=4 bash examples/eval/hoi/run_swig_referring_agent.sh
```

**Output:** Each run creates a unique directory like `results/hico_ground_agent_20260105_033000/` with annotated visualization images in the `visualizations/` subfolder.

### With W&B Logging

```bash
# Enable W&B for experiment tracking
MAX_IMAGES=100 VERBOSE=true WANDB=true bash examples/eval/hoi/run_hico_ground_agent.sh
```

### Full Evaluation (All Images)

```bash
# Full HICO grounding evaluation (~20k samples)
WANDB=true bash examples/eval/hoi/run_hico_ground_agent.sh

# Full HICO referring evaluation (~33k samples)
WANDB=true BERTSCORE_GPU=4 bash examples/eval/hoi/run_hico_referring_agent.sh
```

## Checkpoint Selection

### Which Checkpoint to Use?

For **evaluation**, you typically use **one specific checkpoint** (not all of them):

- **Final evaluation**: Use `global_step_200` (latest, best performance)
- **Comparing training progress**: Evaluate multiple checkpoints (100, 150, 200)

### Full Checkpoint Paths

```
# Step 100
checkpoints/hoi_reward_v2/hoi_reward_v2-fsdp2-agent-qwen_qwen3-vl-4b-instruct-grpo-n8-b128-t1.0-lr5e-7-hoi-detection-v2/global_step_100/actor/huggingface

# Step 150
checkpoints/hoi_reward_v2/hoi_reward_v2-fsdp2-agent-qwen_qwen3-vl-4b-instruct-grpo-n8-b128-t1.0-lr5e-7-hoi-detection-v2/global_step_150/actor/huggingface

# Step 200 (recommended)
checkpoints/hoi_reward_v2/hoi_reward_v2-fsdp2-agent-qwen_qwen3-vl-4b-instruct-grpo-n8-b128-t1.0-lr5e-7-hoi-detection-v2/global_step_200/actor/huggingface
```

## Visualization

### Enabling Visualization

Visualization is automatically enabled when `VERBOSE=true`, or can be enabled separately:

```bash
# Auto-enable with verbose mode
VERBOSE=true bash examples/eval/hoi/run_hico_ground_agent.sh

# Enable visualization only
SAVE_VIZ=true bash examples/eval/hoi/run_hico_ground_agent.sh

# Both datasets with visualization
VERBOSE=true bash examples/eval/hoi/run_swig_referring_agent.sh
```

### What Gets Visualized

**Grounding Task:**
- Ground truth boxes in **green**
- Predicted person boxes in **red**
- Predicted object boxes in **blue**
- Labels and IoU information

**Referring Task:**
- Person box in **cyan**
- Object box in **magenta**
- Text overlay showing GT action vs Predicted action
- Green checkmark for correct, red X for incorrect

### Unique Run Directories

By default, each evaluation run creates a unique output directory with a timestamp:

```
results/hico_ground_agent_20260105_033000/
results/hico_ground_agent_20260105_041500/
```

To disable and overwrite previous results:

```bash
UNIQUE_RUN=false bash examples/eval/hoi/run_hico_ground_agent.sh
```

## Output Files

After evaluation, results are saved to the output directory:

```
results/hico_ground_agent_20260105_033000/
├── metrics.json              # Aggregate metrics (AR, AR@0.5, etc.)
├── per_sample_results.json   # All predictions with per-sample scores
├── thinking.jsonl            # Full agent reasoning logs (if --save-thinking)
├── tool_usage_stats.json     # Tool usage analysis
├── action_stats.json         # Per-action breakdown (referring only)
└── visualizations/           # Annotated images (if --save-viz or --verbose)
    ├── HICO_test2015_00000001_0.jpg
    ├── HICO_test2015_00000002_1.jpg
    └── ...
```

### Thinking Log Format

Each line in `thinking.jsonl` contains the full agent reasoning:

```json
{
  "sample_id": 0,
  "file_name": "HICO_test2015_00000001.jpg",
  "task_type": "grounding",
  "ground_truth": {"action": "sitting on", "object": "bench", "pairs": [...]},
  "conversation": [...],           // Full message history
  "thinking_blocks": [...],        // Extracted <think>...</think> content
  "tool_calls": [...],             // Tools used with args and results
  "num_turns": 3,
  "num_tool_calls": 2,
  "tools_used": ["detect_objects", "zoom_in"],
  "prediction": "...",
  "metrics": {"recall@0.5": 1.0}
}
```

## Metrics

### Grounding Task

| Metric | Description |
|--------|-------------|
| `AR` | Average Recall across IoU thresholds [0.5-0.95] |
| `AR@0.5` | Recall at IoU threshold 0.5 |
| `AR@0.75` | Recall at IoU threshold 0.75 |
| `ARs` | AR for small objects (<32² pixels) |
| `ARm` | AR for medium objects (32²-96² pixels) |
| `ARl` | AR for large objects (>96² pixels) |

### Referring Task

| Metric | Description |
|--------|-------------|
| `exact_match` | Exact match accuracy |
| `meteor` | METEOR score (semantic similarity) |
| `cider` | CIDEr score (consensus-based) |
| `bertscore_f1` | BERTScore F1 (neural semantic similarity) |

## GPU Allocation

For optimal performance with 8 GPUs:

```
GPUs 0-3: vLLM server (tensor parallelism)
GPUs 4-7: Available for BERTScore and other tasks
```

Example setup:

```bash
# Terminal 1: Start vLLM server on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/eval/hoi/start_vllm_server.sh \
    checkpoints/.../global_step_200/actor/huggingface 8000 hoi-trained 4

# Terminal 2: Run evaluation (BERTScore on GPU 4)
BERTSCORE_GPU=4 bash examples/eval/hoi/run_hico_referring_agent.sh
```

## Complete Workflow Example

Here's a complete example running HICO grounding evaluation:

```bash
# Terminal 1: Start vLLM server (GPUs 0-3)
cd /workspace/verl-tool
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/eval/hoi/start_vllm_server.sh \
    checkpoints/hoi_reward_v2/hoi_reward_v2-fsdp2-agent-qwen_qwen3-vl-4b-instruct-grpo-n8-b128-t1.0-lr5e-7-hoi-detection-v2/global_step_200/actor/huggingface \
    8000 \
    hoi-trained \
    4

# Terminal 2: (Optional) Start tool server (GPU 5)
cd /workspace/verl-tool
CUDA_VISIBLE_DEVICES=5 python -m verl_tool.servers.serve \
    --host $(hostname -i | awk '{print $1}') \
    --port 30100 \
    --tool_type "hoi_detector" \
    --workers_per_tool 2

# Terminal 3: Run evaluation
cd /workspace/verl-tool

# First, test with 10 images (creates unique timestamped directory)
MAX_IMAGES=10 VERBOSE=true bash examples/eval/hoi/run_hico_ground_agent.sh

# Check results (find latest run)
ls -lt results/ | grep hico_ground
# e.g., results/hico_ground_agent_20260105_033000/

# View metrics
cat results/hico_ground_agent_*/metrics.json

# View visualizations
ls results/hico_ground_agent_*/visualizations/

# If working, run full evaluation
WANDB=true bash examples/eval/hoi/run_hico_ground_agent.sh
```

## Troubleshooting

### vLLM Server Not Starting

1. Check if port 8000 is already in use: `lsof -i :8000`
2. Verify checkpoint path exists and contains `config.json`
3. Check GPU memory: `nvidia-smi`

### Low Performance

1. Increase concurrency: `CONCURRENCY=16`
2. Use tensor parallelism with more GPUs
3. Reduce `max-model-len` if OOM errors occur

### BERTScore Errors

1. Ensure the GPU is available: `CUDA_VISIBLE_DEVICES=4 python -c "import torch; print(torch.cuda.is_available())"`
2. Install bert-score: `pip install bert-score`

### Model Not Using Tools

If the model outputs boxes directly without using tools (`num_tool_calls: 0`), this indicates:
- The model wasn't trained to use tools for this specific task format
- This is a **training issue**, not an evaluation bug
- The model still provides outputs; evaluate them directly

### Coordinate Mismatch (0.0 AR/AP)

The evaluation script auto-detects and converts between:
- **1000x1000 normalized format** (model output)
- **Pixel coordinates** (ground truth)

If you still see 0.0 metrics, check:
1. Model is generating valid bounding boxes in the expected format
2. Boxes are for the correct objects (person + target object)
3. Run with `VERBOSE=true` to see model responses

### Repetitive/Long Responses

Some models may generate repetitive text. The script:
- Limits referring task outputs to 256 tokens
- Truncates before detected repetition patterns
- Extracts action phrases from verbose responses

If issues persist, this indicates a model quality/training problem.

## Known Limitations

1. **Tool Usage**: The current trained model may not actively use zoom_in/detect_objects tools during evaluation. This is expected behavior if the model wasn't specifically trained with tool-calling reward signals.

2. **Coordinate Systems**: The model outputs in 1000x1000 normalized format, which is automatically converted to pixel coordinates for evaluation.

3. **Grounding Performance**: Low AR scores may indicate the model needs more training iterations or different prompt engineering during training.

