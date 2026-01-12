# HOI RL Data Version Analysis

## Summary

**Current Data Version: NEW VERSION (Single Tool - Chain-of-Focus Aligned)**

The data in `data/hoi/train_data/` uses the **NEW VERSION** with only `zoom_in` tool, aligned with Chain-of-Focus methodology.

---

## Version Comparison

### OLD VERSION (Multi-Tool)
- **Tools**: `zoom_in`, `zoom_out`, `detect_objects`
- **System Prompt**: Includes all three tool definitions
- **Status**: Deprecated, not aligned with SFT training

### NEW VERSION (Single Tool) ✓
- **Tools**: `zoom_in` only
- **System Prompt**: Only `zoom_in` tool definition
- **Status**: Current version, aligned with Chain-of-Focus and SFT data

---

## Evidence from Code Analysis

### 1. Data Preparation Script (`prepare_hoi.py`)

**File**: `examples/data_preprocess/hoi/prepare_hoi.py`

**Lines 50-66** show the current system prompt:

```python
# =============================================================================
# System Prompt - Aligned with SFT data (Chain-of-Focus style)
# Only zoom_in tool - matching generate_hoi_cof_sft.py
# =============================================================================
SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name":"zoom_in","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d). Coordinates use 1000x1000 normalized format.","parameters":{"properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2] in 1000x1000 normalized format, where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"target_image":{"type":"number","description":"The index of the image to zoom in on. Use 1 for the main image."}},"required":["bbox_2d", "target_image"], "type":"object"},"args_format": "Format the arguments as a JSON object."}}
</tools>
```

**Key Points**:
- ✅ Comment explicitly states: "Only zoom_in tool"
- ✅ Only one tool definition in `<tools>` tag
- ✅ Matches `generate_hoi_cof_sft.py` (SFT data generator)

### 2. Tool Server Implementation (`hoi_detector.py`)

**File**: `verl_tool/servers/tools/hoi_detector.py`

**Lines 297-311** show the tool implementation:

```python
@register_tool
class HOIDetectorTool(BaseTool):
    """
    HOI Detector Tool for Human-Object Interaction Detection.
    
    Aligned with Chain-of-Focus methodology - provides only zoom_in tool:
    - zoom_in: Zoom into a specific region of the image to examine details
    
    Note: zoom_out and detect_objects removed to align with SFT training data.
    """
    tool_type = "hoi_detector"

    stop_tokens = ["</tool_call>"]
    # Only zoom_in supported - aligned with SFT/Chain-of-Focus
    valid_mcp_func_names = ['zoom_in', 'crop_image']
```

**Key Points**:
- ✅ Comment explicitly states: "provides only zoom_in tool"
- ✅ `valid_mcp_func_names` only includes `zoom_in` and `crop_image` (alias)
- ✅ Note: "zoom_out and detect_objects removed to align with SFT training data"

### 3. Comparison with Old Version

**Old Version Example** (from `prepare_sft_data.py` lines 35-94):

```python
SYSTEM_PROMPT = '''You are an expert at analyzing human-object interactions in images. You have access to the following tools to help with your analysis:

<tools>
[
    {
        "type": "function",
        "function": {
            "name": "zoom_in",
            ...
        }
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_out",  # ❌ Old version has this
            ...
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_objects",  # ❌ Old version has this
            ...
        }
    }
]
</tools>
```

**New Version** (current `prepare_hoi.py`):
- ✅ Only `zoom_in` tool
- ✅ Simpler, aligned with Chain-of-Focus paper

---

## How to Verify Data Version

### Method 1: Check System Prompt in Parquet File

```python
import pandas as pd

# Read parquet file
df = pd.read_parquet('data/hoi/train_data/train.parquet')

# Get first sample
first_row = df.iloc[0]
prompt = first_row['prompt']  # List of messages

# Extract system prompt
system_prompt = [m['content'] for m in prompt if m['role'] == 'system'][0]

# Check tools
has_zoom_in = 'zoom_in' in system_prompt
has_zoom_out = 'zoom_out' in system_prompt
has_detect_objects = 'detect_objects' in system_prompt

print(f"zoom_in: {has_zoom_in}")
print(f"zoom_out: {has_zoom_out}")
print(f"detect_objects: {has_detect_objects}")

# Version determination
if has_zoom_in and not has_zoom_out and not has_detect_objects:
    print("✓ NEW VERSION (Single tool)")
elif has_zoom_in and (has_zoom_out or has_detect_objects):
    print("⚠ OLD VERSION (Multi-tool)")
```

### Method 2: Use Check Script

```bash
python3 examples/train/hoi/check_data_version.py data/hoi/train_data/train.parquet
```

(Requires: `pip install pandas pyarrow`)

### Method 3: Inspect Code

Check `examples/data_preprocess/hoi/prepare_hoi.py`:
- Lines 50-66: System prompt definition
- Look for number of tools in `<tools>` tag
- Check comments for version indicators

---

## Why New Version?

### Alignment with Chain-of-Focus
- **Paper**: Chain-of-Focus uses single `zoom_in` tool
- **SFT Data**: `generate_hoi_cof_sft.py` generates SFT data with only `zoom_in`
- **Consistency**: RL data must match SFT data format

### Benefits
1. **Simpler**: Fewer tools = less confusion for model
2. **Focused**: Forces model to use zoom strategically
3. **Aligned**: Matches SFT training data exactly
4. **Proven**: Follows Chain-of-Focus methodology

---

## Migration Notes

If you have **OLD VERSION** data and need to regenerate:

```bash
# Regenerate RL data with new version
python examples/data_preprocess/hoi/prepare_hoi.py \
    --local_dir data/hoi/train_data \
    --seed 42
```

This will create NEW VERSION data with only `zoom_in` tool.

---

## Conclusion

✅ **Current data uses NEW VERSION (Single Tool)**

The data preparation script (`prepare_hoi.py`) and tool server (`hoi_detector.py`) both confirm:
- Only `zoom_in` tool is supported
- Aligned with Chain-of-Focus methodology
- Matches SFT training data format

If you need to verify your specific data files, use the check script or inspect the system prompt in the parquet files directly.
