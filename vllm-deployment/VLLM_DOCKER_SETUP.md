# Fixing CUDA PTX Error: "the provided PTX was compiled with an unsupported toolchain"

## Test Environment

This solution was tested on the following hardware:

- **GPUs**: 8x NVIDIA H200 (143GB VRAM each)
- **Driver Version**: 550.144.03
- **CUDA Version**: 12.4 (from driver, shown by `nvidia-smi`)
- **Base Image**: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
- **Models Tested**: Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct

**Note**: This solution should works for any NVIDIA GPU with CUDA 12.4 driver, not just H200.

## Problem

When running Qwen2.5-VL or Qwen3-VL models with vLLM, you may encounter this error.

**Commands that triggered the error:**

```bash
# Running Qwen2.5-VL-3B-Instruct
vllm serve Qwen/Qwen2.5-VL-3B-Instruct

# Running Qwen3-VL-8B-Instruct
vllm serve Qwen/Qwen3-VL-8B-Instruct
```

**Error encountered:**

```
torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain
```

**Root Cause:**
- Host GPU driver supports CUDA 12.4 (check with `nvidia-smi` - shows "CUDA Version: 12.4")
- vLLM wheels are compiled for CUDA 12.9
- Mismatch between driver CUDA version (12.4) and compiled PTX code (12.9)
- The PTX (Parallel Thread Execution) code in vLLM's pre-compiled wheels requires CUDA 12.9 runtime, but the host driver only supports up to CUDA 12.4

## Solution

Use a Docker container with CUDA 12.9 compatibility libraries (`cuda-compat-12-9`) to bridge the gap between CUDA 12.4 driver and CUDA 12.9 PTX code.

## Quick Setup

### 1. Build and Run Container

```bash
# Clone or download the Dockerfile.vllm and build_and_run_vllm.sh
./build_and_run_vllm.sh
```

This will:
- Build Docker image with CUDA 12.4.1 base image
- Install `cuda-compat-12-9` package
- Set up CUDA compatibility library path
- Start container interactively

### 2. Setup vLLM Inside Container

Once inside the container (you'll be dropped into bash), run:

```bash
# Create Python virtual environment
uv venv --python 3.12 --seed

# Activate venv
source .venv/bin/activate

# Install vLLM with automatic PyTorch backend selection
uv pip install vllm --torch-backend=auto
```

### 3. Run Your Model

Now you can run any model with vLLM:

**Qwen3-VL:**
```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt video 0 \
  --async-scheduling
```

**Note:** For image-only use cases (no video), use `--limit-mm-per-prompt.video 0` to reduce GPU memory usage:

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling
```

This disables video processing capabilities and reduces GPU memory requirements.

**Qwen2.5-VL:**
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

**Switch Models Easily:**
- Stop current vLLM (Ctrl+C)
- Run new model command
- No need to restart container!

### 4. Run in Background (Long-Running)

For long-running vLLM servers, use `screen` or `tmux` inside the container:

**Using screen:**
```bash
# Inside container
screen -S vllm

# Run vLLM command
vllm serve Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8000

# Detach: Ctrl+A then D
# Reattach: screen -r vllm
```

**Using tmux:**
```bash
# Inside container
tmux new -s vllm

# Run vLLM command
vllm serve Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8000

# Detach: Ctrl+B then D
# Reattach: tmux attach -t vllm
```

## Key Components

### Dockerfile.vllm

The Dockerfile includes:

1. **CUDA 12.4.1 base image** - Matches host driver version
2. **cuda-compat-12-9 package** - Provides compatibility libraries
3. **LD_LIBRARY_PATH setup** - Points to CUDA 12.9 compatibility libraries
4. **uv package manager** - Fast Python environment management

### Verification

After setup, verify CUDA version shows 12.9:

```bash
# Inside container
nvidia-smi
```

Should show: `CUDA Version: 12.9` (even though host driver is 12.4)

## Files Included

- `Dockerfile.vllm` - Dockerfile with CUDA compatibility setup
- `build_and_run_vllm.sh` - Build and run script (interactive mode)
- `.dockerignore` - Excludes large directories from build context

## Usage Example

```bash
# 1. Build and start container
./build_and_run_vllm.sh

# 2. Inside container, setup vLLM
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto

# 3. Run model
vllm serve Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8000

# 4. Test API (from another terminal)
curl http://localhost:8000/health
```

## Why This Works

- **cuda-compat-12-9**: Provides compatibility libraries that allow CUDA 12.9 PTX code to run on CUDA 12.4 driver
- **LD_LIBRARY_PATH**: Tells the system where to find these compatibility libraries
- **Result**: vLLM (compiled for CUDA 12.9) runs successfully on CUDA 12.4 driver

## Troubleshooting

**PTX Error Still Occurs:**
- Verify `cuda-compat-12-9` is installed: `apt list --installed | grep cuda-compat`
- Check `LD_LIBRARY_PATH`: `env | grep LD_LIBRARY_PATH` (should include `/usr/local/cuda-12.9/compat`)
- Verify CUDA version: `nvidia-smi` (should show 12.9)

**Container Exits Immediately:**
- The build script uses `-it` (interactive) mode
- If container exits, restart with: `docker start -ai vllm-server`

**Restarting a Stopped Container:**

If your container is stopped and you want to exec into it again:

```bash
# Check container status
docker ps -a | grep vllm-server

# Start stopped container (keeps it running in background)
docker start vllm-server

# Now you can exec into it
docker exec -it vllm-server bash
```

**Or restart interactively:**
```bash
# Start and attach interactively
docker start -ai vllm-server

# Or restart the container
docker restart vllm-server
docker exec -it vllm-server bash
```

**Note:** If the container was removed, you'll need to run `./build_and_run_vllm.sh` again to recreate it.

## Testing from Another Docker Container

Use Docker networks to connect containers:

**Quick setup:**
```bash
# 1. Create network
docker network create vllm-network

# 2. Connect running containers
docker network connect vllm-network vllm-server
docker network connect vllm-network your-test-container

# 3. Test from another container
docker run --rm --network vllm-network \
  curlimages/curl:latest \
  curl http://vllm-server:8000/health
```

**Python test example:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://vllm-server:8000/v1",  # Use container name as hostname
    api_key="dummy"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

**Note:** Containers on the same network can communicate using container names as hostnames. No port mapping needed for inter-container communication.

## Summary

This solution resolves the CUDA PTX compilation error by:
1. Using CUDA 12.4.1 base image (matches host driver)
2. Installing `cuda-compat-12-9` for compatibility
3. Setting `LD_LIBRARY_PATH` to use compatibility libraries
4. Allowing vLLM (CUDA 12.9) to run on CUDA 12.4 driver

The container is interactive, so you can easily switch between models (Qwen2.5-VL, Qwen3-VL, etc.) without rebuilding.

## References

- [vLLM Troubleshooting: CUDA PTX Error](https://docs.vllm.ai/en/latest/usage/troubleshooting/#cuda-error-the-provided-ptx-was-compiled-with-an-unsupported-toolchain) - Official vLLM documentation on resolving CUDA PTX compilation errors
- [vLLM Quickstart Guide](https://docs.vllm.ai/en/latest/getting_started/quickstart/#offline-batched-inference) - Official vLLM quickstart guide for offline batched inference
- [Qwen3-VL Documentation](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html#qwen3-vl-235b-a22b-instruct) - Official vLLM documentation for Qwen3-VL models
