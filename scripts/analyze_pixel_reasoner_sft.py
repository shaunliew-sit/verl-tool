#!/usr/bin/env python3
"""
Script to download and analyze the PixelReasoner-SFT-Data dataset.
Focus on understanding the tool calling patterns for HOI referring and grounding tasks.
"""

import json
from datasets import load_dataset
from collections import Counter
import re
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path("/workspace/verl-tool/data/pixel_reasoner/sft_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset():
    """Download the PixelReasoner-SFT-Data dataset from HuggingFace."""
    print("=" * 80)
    print("Downloading PixelReasoner-SFT-Data from HuggingFace...")
    print("=" * 80)
    
    dataset = load_dataset("TIGER-Lab/PixelReasoner-SFT-Data", split="train")
    print(f"Dataset loaded successfully! Total samples: {len(dataset)}")
    return dataset


def analyze_dataset_structure(dataset):
    """Analyze the overall structure of the dataset."""
    print("\n" + "=" * 80)
    print("DATASET STRUCTURE ANALYSIS")
    print("=" * 80)
    
    # Check columns
    print(f"\nColumns: {dataset.column_names}")
    
    # Sample sizes
    msg_lengths = [len(sample['message_list']) for sample in dataset]
    print(f"\nMessage list lengths - Min: {min(msg_lengths)}, Max: {max(msg_lengths)}, Avg: {sum(msg_lengths)/len(msg_lengths):.2f}")
    
    # Count message length distribution
    length_counter = Counter(msg_lengths)
    print("\nMessage length distribution:")
    for length, count in sorted(length_counter.items()):
        print(f"  {length} messages: {count} samples ({count/len(dataset)*100:.1f}%)")
    
    return length_counter


def extract_tool_info(sample):
    """Extract tool calling information from a sample."""
    messages = sample['message_list']
    
    tool_calls = []
    has_crop_tool = False
    has_select_frames = False
    
    for msg in messages:
        if msg['role'] == 'assistant':
            for content in msg['content']:
                if content.get('text'):
                    text = content['text']
                    # Find tool calls
                    tool_call_matches = re.findall(r'<tool_call>\s*({.*?})\s*</tool_call>', text, re.DOTALL)
                    for match in tool_call_matches:
                        try:
                            tool_call = json.loads(match)
                            tool_calls.append(tool_call)
                            if tool_call.get('name') == 'crop_image_normalized':
                                has_crop_tool = True
                            elif tool_call.get('name') == 'select_frames':
                                has_select_frames = True
                        except json.JSONDecodeError:
                            pass
    
    return {
        'tool_calls': tool_calls,
        'has_crop_tool': has_crop_tool,
        'has_select_frames': has_select_frames,
        'num_tool_calls': len(tool_calls)
    }


def analyze_tool_usage(dataset):
    """Analyze how tools are used in the dataset."""
    print("\n" + "=" * 80)
    print("TOOL USAGE ANALYSIS")
    print("=" * 80)
    
    crop_samples = []
    select_frames_samples = []
    no_tool_samples = []
    
    tool_call_counts = Counter()
    
    for i, sample in enumerate(dataset):
        tool_info = extract_tool_info(sample)
        tool_call_counts[tool_info['num_tool_calls']] += 1
        
        if tool_info['has_crop_tool']:
            crop_samples.append((i, sample, tool_info))
        if tool_info['has_select_frames']:
            select_frames_samples.append((i, sample, tool_info))
        if tool_info['num_tool_calls'] == 0:
            no_tool_samples.append((i, sample))
    
    print(f"\nSamples with crop_image_normalized: {len(crop_samples)}")
    print(f"Samples with select_frames: {len(select_frames_samples)}")
    print(f"Samples without tool calls: {len(no_tool_samples)}")
    
    print("\nTool call count distribution:")
    for count, num_samples in sorted(tool_call_counts.items()):
        print(f"  {count} tool calls: {num_samples} samples ({num_samples/len(dataset)*100:.1f}%)")
    
    return crop_samples, select_frames_samples, no_tool_samples


def format_sample_for_display(sample, max_text_len=2000):
    """Format a sample for readable display."""
    messages = sample['message_list']
    formatted = []
    
    for msg in messages:
        role = msg['role'].upper()
        content_parts = []
        
        for content in msg['content']:
            if content.get('image'):
                content_parts.append(f"[IMAGE: {content['image']}]")
            if content.get('video'):
                content_parts.append(f"[VIDEO: {content['video']}]")
            if content.get('text'):
                text = content['text']
                if len(text) > max_text_len:
                    text = text[:max_text_len] + f"\n... [TRUNCATED, total {len(content['text'])} chars]"
                content_parts.append(text)
        
        formatted.append(f"\n{'='*40}\n[{role}]\n{'='*40}\n" + "\n".join(content_parts))
    
    return "\n".join(formatted)


def show_important_examples(dataset, crop_samples, select_frames_samples, no_tool_samples):
    """Show important examples for HOI dataset reference."""
    print("\n" + "=" * 80)
    print("IMPORTANT EXAMPLES FOR HOI DATASET REFERENCE")
    print("=" * 80)
    
    examples_output = []
    
    # Example 1: System prompt with tool definitions
    print("\n\n### EXAMPLE 1: SYSTEM PROMPT WITH TOOL DEFINITIONS ###")
    print("-" * 60)
    sample = dataset[0]
    system_msg = sample['message_list'][0]
    system_text = system_msg['content'][0]['text']
    print(system_text)
    examples_output.append({
        'name': 'System Prompt with Tool Definitions',
        'content': system_text
    })
    
    # Example 2: Image crop tool usage
    if crop_samples:
        print("\n\n### EXAMPLE 2: IMAGE CROP TOOL USAGE ###")
        print("-" * 60)
        idx, sample, tool_info = crop_samples[0]
        print(f"Sample index: {idx}")
        print(f"Tool calls: {json.dumps(tool_info['tool_calls'], indent=2)}")
        print("\nFull conversation:")
        formatted = format_sample_for_display(sample)
        print(formatted)
        examples_output.append({
            'name': 'Image Crop Tool Usage',
            'index': idx,
            'tool_calls': tool_info['tool_calls'],
            'messages': sample['message_list']
        })
    
    # Example 3: Multi-step reasoning with multiple tool calls
    multi_tool_samples = [(i, s, t) for i, s, t in crop_samples if t['num_tool_calls'] >= 2]
    if multi_tool_samples:
        print("\n\n### EXAMPLE 3: MULTI-STEP REASONING WITH MULTIPLE TOOL CALLS ###")
        print("-" * 60)
        idx, sample, tool_info = multi_tool_samples[0]
        print(f"Sample index: {idx}")
        print(f"Number of tool calls: {tool_info['num_tool_calls']}")
        print(f"Tool calls: {json.dumps(tool_info['tool_calls'], indent=2)}")
        print("\nFull conversation:")
        formatted = format_sample_for_display(sample, max_text_len=3000)
        print(formatted)
        examples_output.append({
            'name': 'Multi-step Reasoning with Multiple Tool Calls',
            'index': idx,
            'num_tool_calls': tool_info['num_tool_calls'],
            'tool_calls': tool_info['tool_calls'],
            'messages': sample['message_list']
        })
    
    # Example 4: Video frame selection
    if select_frames_samples:
        print("\n\n### EXAMPLE 4: VIDEO FRAME SELECTION TOOL USAGE ###")
        print("-" * 60)
        idx, sample, tool_info = select_frames_samples[0]
        print(f"Sample index: {idx}")
        print(f"Tool calls: {json.dumps(tool_info['tool_calls'], indent=2)}")
        print("\nFull conversation:")
        formatted = format_sample_for_display(sample)
        print(formatted)
        examples_output.append({
            'name': 'Video Frame Selection Tool Usage',
            'index': idx,
            'tool_calls': tool_info['tool_calls'],
            'messages': sample['message_list']
        })
    
    # Example 5: Textual reasoning without tools (simpler samples)
    if no_tool_samples:
        print("\n\n### EXAMPLE 5: TEXTUAL REASONING WITHOUT TOOLS ###")
        print("-" * 60)
        idx, sample = no_tool_samples[0]
        print(f"Sample index: {idx}")
        print("\nFull conversation:")
        formatted = format_sample_for_display(sample)
        print(formatted)
        examples_output.append({
            'name': 'Textual Reasoning Without Tools',
            'index': idx,
            'messages': sample['message_list']
        })
    
    # Save examples to file
    with open(OUTPUT_DIR / 'important_examples.json', 'w') as f:
        json.dump(examples_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nExamples saved to {OUTPUT_DIR / 'important_examples.json'}")
    
    return examples_output


def extract_key_patterns(dataset, crop_samples):
    """Extract key patterns useful for HOI dataset construction."""
    print("\n" + "=" * 80)
    print("KEY PATTERNS FOR HOI DATASET CONSTRUCTION")
    print("=" * 80)
    
    patterns = {
        'tool_definition_template': None,
        'tool_call_format': None,
        'reasoning_patterns': [],
        'bbox_usage': []
    }
    
    # Extract tool definition template from system prompt
    sample = dataset[0]
    system_text = sample['message_list'][0]['content'][0]['text']
    
    # Extract the tools section
    tools_match = re.search(r'<tools>(.*?)</tools>', system_text, re.DOTALL)
    if tools_match:
        patterns['tool_definition_template'] = tools_match.group(1).strip()
    
    # Extract tool call format
    patterns['tool_call_format'] = """<tool_call>
{"name": "<function-name>", "arguments": <args-json-object>}
</tool_call>"""
    
    # Analyze bbox usage patterns from crop samples
    print("\n### BOUNDING BOX USAGE PATTERNS ###")
    for idx, sample, tool_info in crop_samples[:5]:
        for tool_call in tool_info['tool_calls']:
            if tool_call.get('name') == 'crop_image_normalized':
                args = tool_call.get('arguments', {})
                bbox = args.get('bbox_2d', [])
                target_image = args.get('target_image')
                patterns['bbox_usage'].append({
                    'bbox_2d': bbox,
                    'target_image': target_image
                })
                print(f"  Sample {idx}: bbox={bbox}, target_image={target_image}")
    
    # Save patterns
    with open(OUTPUT_DIR / 'key_patterns.json', 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"\nKey patterns saved to {OUTPUT_DIR / 'key_patterns.json'}")
    
    return patterns


def create_hoi_template():
    """Create a template for HOI referring/grounding dataset based on PixelReasoner patterns."""
    print("\n" + "=" * 80)
    print("HOI DATASET TEMPLATE")
    print("=" * 80)
    
    # HOI-specific tool definition
    hoi_tools = {
        "type": "function",
        "function": {
            "name": "crop_image_normalized",
            "description": "Zoom in on a specific region of the image to better analyze human-object interactions. Use this when you need to examine fine-grained details of interactions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "description": "Normalized coordinates [x1, y1, x2, y2] for bounding box of the region to zoom in. Values should be within [0.0, 1.0].",
                        "items": {"type": "number"}
                    },
                    "target_image": {
                        "type": "number",
                        "description": "The index of the image to crop. Index from 1 to the number of images."
                    }
                },
                "required": ["bbox_2d", "target_image"]
            }
        }
    }
    
    # HOI-specific referring tool
    hoi_refer_tool = {
        "type": "function",
        "function": {
            "name": "identify_interaction",
            "description": "Identify and localize a specific human-object interaction in the image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "human_bbox": {
                        "type": "array",
                        "description": "Normalized bounding box [x1, y1, x2, y2] of the human involved in the interaction.",
                        "items": {"type": "number"}
                    },
                    "object_bbox": {
                        "type": "array",
                        "description": "Normalized bounding box [x1, y1, x2, y2] of the object involved in the interaction.",
                        "items": {"type": "number"}
                    },
                    "interaction": {
                        "type": "string",
                        "description": "The type of interaction between human and object (e.g., 'holding', 'riding', 'eating')."
                    }
                },
                "required": ["human_bbox", "object_bbox", "interaction"]
            }
        }
    }
    
    # System prompt template for HOI
    system_prompt_template = f"""You are a helpful assistant specialized in analyzing human-object interactions in images.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{json.dumps(hoi_tools)}
{json.dumps(hoi_refer_tool)}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
    
    # Example conversation templates
    hoi_referring_example = {
        "task": "HOI Referring",
        "description": "Given an image and a description of a human-object interaction, locate the interaction.",
        "example": {
            "message_list": [
                {
                    "role": "system",
                    "content": [{"text": system_prompt_template, "image": None, "video": None}]
                },
                {
                    "role": "user", 
                    "content": [
                        {"text": "Find the person who is riding a bicycle in the image. Provide the bounding boxes for both the person and the bicycle.\n\nThink step by step and use tools if needed to examine details.", "image": None, "video": None},
                        {"image": "images/example.jpg", "text": None, "video": None}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"text": """Let me analyze the image to find the person riding a bicycle.

First, I'll examine the image to identify all humans and bicycles present.

<tool_call>
{"name": "crop_image_normalized", "arguments": {"bbox_2d": [0.3, 0.2, 0.7, 0.9], "target_image": 1}}
</tool_call>""", "image": None, "video": None}]
                },
                {
                    "role": "user",
                    "content": [
                        {"image": "images/example_cropped.jpg", "text": None, "video": None}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"text": """Looking at the cropped region, I can clearly see a person riding a bicycle.

The person is seated on the bicycle with their hands on the handlebars and feet on the pedals, indicating an active riding interaction.

<tool_call>
{"name": "identify_interaction", "arguments": {"human_bbox": [0.35, 0.25, 0.55, 0.85], "object_bbox": [0.30, 0.50, 0.60, 0.90], "interaction": "riding"}}
</tool_call>

Final answer:
- Human bounding box: [0.35, 0.25, 0.55, 0.85]
- Bicycle bounding box: [0.30, 0.50, 0.60, 0.90]
- Interaction: riding""", "image": None, "video": None}]
                }
            ]
        }
    }
    
    hoi_grounding_example = {
        "task": "HOI Grounding",
        "description": "Given an image and specific bounding boxes, identify what interaction is occurring.",
        "example": {
            "message_list": [
                {
                    "role": "system",
                    "content": [{"text": system_prompt_template, "image": None, "video": None}]
                },
                {
                    "role": "user",
                    "content": [
                        {"text": "Look at the person at bounding box [0.2, 0.3, 0.4, 0.8] and the object at [0.35, 0.5, 0.5, 0.75]. What interaction is happening between them?\n\nThink step by step.", "image": None, "video": None},
                        {"image": "images/example2.jpg", "text": None, "video": None}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"text": """Let me zoom in on the region containing the person and object to better understand their interaction.

<tool_call>
{"name": "crop_image_normalized", "arguments": {"bbox_2d": [0.15, 0.25, 0.55, 0.85], "target_image": 1}}
</tool_call>""", "image": None, "video": None}]
                },
                {
                    "role": "user",
                    "content": [
                        {"image": "images/example2_cropped.jpg", "text": None, "video": None}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"text": """Based on the zoomed-in view, I can clearly see the interaction:

1. The person in the specified region is positioned with their arm extended toward the object
2. The object appears to be a cup/mug
3. The person's hand is wrapped around the cup in a grasping position

The interaction is **holding** - the person is holding the cup.

<tool_call>
{"name": "identify_interaction", "arguments": {"human_bbox": [0.2, 0.3, 0.4, 0.8], "object_bbox": [0.35, 0.5, 0.5, 0.75], "interaction": "holding"}}
</tool_call>

\\boxed{holding}""", "image": None, "video": None}]
                }
            ]
        }
    }
    
    templates = {
        "system_prompt": system_prompt_template,
        "tools": [hoi_tools, hoi_refer_tool],
        "examples": [hoi_referring_example, hoi_grounding_example]
    }
    
    # Save template
    with open(OUTPUT_DIR / 'hoi_dataset_template.json', 'w') as f:
        json.dump(templates, f, indent=2)
    
    print("HOI Dataset Template created!")
    print(f"\nSaved to {OUTPUT_DIR / 'hoi_dataset_template.json'}")
    print("\n### System Prompt Template ###")
    print(system_prompt_template)
    
    return templates


def main():
    print("=" * 80)
    print("PIXELREASONER SFT DATA ANALYSIS")
    print("For HOI Referring and Grounding Dataset Preparation")
    print("=" * 80)
    
    # Download dataset
    dataset = download_dataset()
    
    # Analyze structure
    length_counter = analyze_dataset_structure(dataset)
    
    # Analyze tool usage
    crop_samples, select_frames_samples, no_tool_samples = analyze_tool_usage(dataset)
    
    # Show important examples
    examples = show_important_examples(dataset, crop_samples, select_frames_samples, no_tool_samples)
    
    # Extract key patterns
    patterns = extract_key_patterns(dataset, crop_samples)
    
    # Create HOI template
    hoi_template = create_hoi_template()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("  - important_examples.json")
    print("  - key_patterns.json") 
    print("  - hoi_dataset_template.json")
    
    # Summary
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS FOR HOI DATASET")
    print("=" * 80)
    print("""
1. TOOL DEFINITION FORMAT:
   - Tools are defined in JSON schema format within <tools></tools> tags
   - Each tool has: name, description, parameters (with type, properties, required)

2. TOOL CALL FORMAT:
   - Tool calls use <tool_call></tool_call> XML tags
   - Inside: JSON with "name" and "arguments" keys

3. MULTI-TURN REASONING:
   - Model can make tool calls, receive results, and continue reasoning
   - Supports iterative refinement through multiple crop/zoom operations

4. BBOX FORMAT:
   - Uses normalized coordinates [0.0, 1.0]
   - Format: [x1, y1, x2, y2] for crop_image_normalized

5. MESSAGE STRUCTURE:
   - Each message has: role (system/user/assistant), content (list of {text, image, video})
   - Images are referenced by path in 'image' field

6. FOR HOI:
   - Adapt crop_image_normalized for zooming into interaction regions
   - Add custom tools for identifying human, object, and interaction
   - Include step-by-step reasoning before final answer
   - Use \\boxed{} for final answers
""")


if __name__ == "__main__":
    main()

