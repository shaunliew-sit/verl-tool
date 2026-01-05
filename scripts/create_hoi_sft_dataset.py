#!/usr/bin/env python3
"""
Script to create a high-quality SFT dataset for HOI Referring and Grounding tasks.
Based on patterns learned from PixelReasoner-SFT-Data.

Key Patterns from PixelReasoner:
1. System prompt with tool definitions in <tools></tools> XML tags
2. Tool calls use <tool_call></tool_call> XML tags with JSON inside
3. Multi-turn conversations with tool execution results
4. Step-by-step reasoning before final answer
5. Final answer in \\boxed{} format
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


# ============================================================================
# HOI-SPECIFIC TOOL DEFINITIONS
# ============================================================================

ZOOM_IN_TOOL = {
    "type": "function",
    "function": {
        "name": "zoom_in",
        "description": "Zoom in on a specific region of the image to examine human-object interactions in detail.",
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "description": "Bounding box coordinates [x1, y1, x2, y2] in 1000x1000 normalized format.",
                    "items": {"type": "number"}
                },
                "target_image": {
                    "type": "number",
                    "description": "The index of the image to zoom in on. Use 1 for the main image."
                }
            },
            "required": ["bbox_2d", "target_image"]
        }
    }
}

ZOOM_OUT_TOOL = {
    "type": "function",
    "function": {
        "name": "zoom_out",
        "description": "Reset the view to the original full image.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_image": {
                    "type": "number",
                    "description": "The index of the image to reset. Use 1 for the main image."
                }
            },
            "required": ["target_image"]
        }
    }
}

DETECT_OBJECTS_TOOL = {
    "type": "function",
    "function": {
        "name": "detect_objects",
        "description": "Detect objects in the image using an object detector.",
        "parameters": {
            "type": "object",
            "properties": {
                "class_names": {
                    "type": "string",
                    "description": "Object classes to detect, separated by ' . ' (e.g., 'person . cup . chair')."
                },
                "target_image": {
                    "type": "number",
                    "description": "The index of the image to analyze. Use 1 for the main image."
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Minimum confidence for detections (0.0-1.0). Default: 0.25"
                }
            },
            "required": ["class_names", "target_image"]
        }
    }
}


def create_system_prompt(tools: List[dict]) -> str:
    """Create a system prompt with tool definitions following PixelReasoner format."""
    tools_json = "\n".join(json.dumps(t) for t in tools)
    
    return f"""You are a helpful assistant specialized in analyzing human-object interactions in images.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_json}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def create_tool_call(name: str, arguments: dict) -> str:
    """Create a tool call string following PixelReasoner format."""
    return f'<tool_call>{{"name": "{name}", "arguments": {json.dumps(arguments)}}}</tool_call>'


# ============================================================================
# REASONING TRAJECTORY TEMPLATES
# ============================================================================

class HOIReasoningTrajectory:
    """Generate multi-turn reasoning trajectories for HOI tasks."""
    
    def __init__(self, tools: List[dict] = None):
        self.tools = tools or [ZOOM_IN_TOOL, ZOOM_OUT_TOOL, DETECT_OBJECTS_TOOL]
        self.system_prompt = create_system_prompt(self.tools)
    
    def create_referring_trajectory(
        self,
        image_path: str,
        action: str,
        object_category: str,
        person_bbox: List[int],
        object_bbox: List[int],
        use_tools: bool = True,
        num_tool_calls: int = 1
    ) -> List[Dict]:
        """
        Create a referring task trajectory.
        Task: Given action + object description, find the interacting person and object.
        """
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": [{"text": self.system_prompt, "image": None}]
        })
        
        # User query
        user_query = f"""Find the person who is {action} {object_category} in this image. Provide bounding boxes for both the person and the {object_category}.

Guidelines: Analyze the image step by step. You may use zoom_in to examine specific regions or detect_objects to find candidates. Output bounding boxes in JSON format: [{{"bbox_2d": [x1, y1, x2, y2], "label": "person"}}, {{"bbox_2d": [x1, y1, x2, y2], "label": "{object_category}"}}]. Coordinates should be in 1000x1000 normalized format."""
        
        messages.append({
            "role": "user",
            "content": [
                {"text": user_query, "image": None},
                {"image": image_path, "text": None}
            ]
        })
        
        if use_tools:
            # First assistant turn with reasoning and tool call
            combined_bbox = self._get_combined_bbox(person_bbox, object_bbox)
            
            reasoning = f"""Let me analyze the image to find the person {action} {object_category}.

First, I'll examine the image to identify potential interaction regions."""
            
            tool_call = create_tool_call("zoom_in", {
                "bbox_2d": combined_bbox,
                "target_image": 1
            })
            
            messages.append({
                "role": "assistant",
                "content": [{"text": f"{reasoning}\n\n{tool_call}", "image": None}]
            })
            
            # Tool result (cropped image)
            messages.append({
                "role": "user",
                "content": [
                    {"image": f"{image_path.replace('.jpg', '_cropped.jpg')}", "text": "Here is the zoomed region:"}
                ]
            })
            
            # Final assistant response with answer
            final_response = f"""Looking at the zoomed region, I can clearly identify the human-object interaction:

1. I can see a person in the region who appears to be {action} a {object_category}.
2. The person's posture and position indicate active interaction with the {object_category}.

Based on my analysis, here are the bounding boxes:

[{{"bbox_2d": {person_bbox}, "label": "person"}}, {{"bbox_2d": {object_bbox}, "label": "{object_category}"}}]"""
            
            messages.append({
                "role": "assistant",
                "content": [{"text": final_response, "image": None}]
            })
        else:
            # Direct answer without tools
            direct_response = f"""Analyzing the image to find the person {action} {object_category}:

1. I can identify a person who is engaged in the "{action}" interaction with a {object_category}.
2. The person's body pose and hand position clearly indicate they are {action} the {object_category}.

Based on my analysis, here are the bounding boxes:

[{{"bbox_2d": {person_bbox}, "label": "person"}}, {{"bbox_2d": {object_bbox}, "label": "{object_category}"}}]"""
            
            messages.append({
                "role": "assistant",
                "content": [{"text": direct_response, "image": None}]
            })
        
        return messages
    
    def create_grounding_trajectory(
        self,
        image_path: str,
        person_bbox: List[int],
        object_bbox: List[int],
        object_category: str,
        action: str,
        use_tools: bool = True
    ) -> List[Dict]:
        """
        Create a grounding task trajectory.
        Task: Given person and object bboxes, identify the interaction type.
        """
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": [{"text": self.system_prompt, "image": None}]
        })
        
        # User query
        user_query = f"""Look at the person at bounding box {person_bbox} and the {object_category} at {object_bbox}. What interaction is happening between them?

Guidelines: Analyze the spatial relationship and poses to determine the interaction type. You may use zoom_in to examine details. Output your answer in the format: \\boxed{{interaction_type}}"""
        
        messages.append({
            "role": "user",
            "content": [
                {"text": user_query, "image": None},
                {"image": image_path, "text": None}
            ]
        })
        
        if use_tools:
            # Assistant turn with zoom
            combined_bbox = self._get_combined_bbox(person_bbox, object_bbox)
            
            reasoning = f"""Let me zoom in on the interaction region to better understand the relationship between the person and the {object_category}."""
            
            tool_call = create_tool_call("zoom_in", {
                "bbox_2d": combined_bbox,
                "target_image": 1
            })
            
            messages.append({
                "role": "assistant",
                "content": [{"text": f"{reasoning}\n\n{tool_call}", "image": None}]
            })
            
            # Tool result
            messages.append({
                "role": "user",
                "content": [
                    {"image": f"{image_path.replace('.jpg', '_cropped.jpg')}", "text": "Here is the zoomed region:"}
                ]
            })
            
            # Final response
            final_response = f"""Based on the zoomed view, I can clearly analyze the interaction:

1. The person's position relative to the {object_category} shows clear engagement
2. The pose and hand positions indicate a "{action}" interaction
3. The spatial proximity confirms direct physical interaction

The interaction is **{action}**.

\\boxed{{{action}}}"""
            
            messages.append({
                "role": "assistant",
                "content": [{"text": final_response, "image": None}]
            })
        else:
            # Direct answer
            direct_response = f"""Analyzing the interaction between the person and the {object_category}:

1. Looking at the person's pose at {person_bbox}
2. Examining the position of the {object_category} at {object_bbox}
3. The spatial relationship and body pose indicate the person is {action} the {object_category}

\\boxed{{{action}}}"""
            
            messages.append({
                "role": "assistant",
                "content": [{"text": direct_response, "image": None}]
            })
        
        return messages
    
    def create_detection_trajectory(
        self,
        image_path: str,
        action: str,
        object_category: str,
        hoi_pairs: List[Tuple[List[int], List[int]]],
        use_tools: bool = True
    ) -> List[Dict]:
        """
        Create an HOI detection trajectory.
        Task: Detect all instances of a specific human-object interaction.
        """
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": [{"text": self.system_prompt, "image": None}]
        })
        
        # User query
        user_query = f"""Detect all instances of "person {action} {object_category}" in this image.

Guidelines: Find all person-{object_category} pairs where the interaction is "{action}". Output each pair as: [{{"bbox_2d": [x1, y1, x2, y2], "label": "person"}}, {{"bbox_2d": [x1, y1, x2, y2], "label": "{object_category}"}}]. Coordinates in 1000x1000 format."""
        
        messages.append({
            "role": "user",
            "content": [
                {"text": user_query, "image": None},
                {"image": image_path, "text": None}
            ]
        })
        
        if use_tools:
            # Use detect_objects tool first
            reasoning = f"""Let me first detect all persons and {object_category}s in the image to identify potential interaction pairs."""
            
            tool_call = create_tool_call("detect_objects", {
                "class_names": f"person . {object_category}",
                "target_image": 1,
                "confidence_threshold": 0.3
            })
            
            messages.append({
                "role": "assistant",
                "content": [{"text": f"{reasoning}\n\n{tool_call}", "image": None}]
            })
            
            # Simulated detection results
            detection_results = {
                "detections": [
                    {"bbox_2d": pair[0], "label": "person", "confidence": 0.92}
                    for pair in hoi_pairs
                ] + [
                    {"bbox_2d": pair[1], "label": object_category, "confidence": 0.88}
                    for pair in hoi_pairs
                ]
            }
            
            messages.append({
                "role": "user",
                "content": [
                    {"text": f"Detection results:\n```json\n{json.dumps(detection_results, indent=2)}\n```", "image": None}
                ]
            })
            
            # Final analysis
            pairs_output = []
            for i, (person_bbox, obj_bbox) in enumerate(hoi_pairs):
                pairs_output.append(f"Pair {i+1}: Person at {person_bbox} is {action} {object_category} at {obj_bbox}")
            
            final_response = f"""Based on the detections, I can identify the following {action} interactions:

{chr(10).join(pairs_output)}

Final output:
"""
            for person_bbox, obj_bbox in hoi_pairs:
                final_response += f'\n[{{"bbox_2d": {person_bbox}, "label": "person"}}, {{"bbox_2d": {obj_bbox}, "label": "{object_category}"}}]'
            
            messages.append({
                "role": "assistant",
                "content": [{"text": final_response, "image": None}]
            })
        else:
            # Direct answer
            pairs_output = []
            for person_bbox, obj_bbox in hoi_pairs:
                pairs_output.append(f'[{{"bbox_2d": {person_bbox}, "label": "person"}}, {{"bbox_2d": {obj_bbox}, "label": "{object_category}"}}]')
            
            direct_response = f"""Analyzing the image for "{action} {object_category}" interactions:

I found {len(hoi_pairs)} instance(s) of this interaction:

{chr(10).join(pairs_output)}"""
            
            messages.append({
                "role": "assistant",
                "content": [{"text": direct_response, "image": None}]
            })
        
        return messages
    
    def _get_combined_bbox(self, bbox1: List[int], bbox2: List[int], padding: int = 50) -> List[int]:
        """Get a combined bounding box that covers both boxes with padding."""
        x1 = max(0, min(bbox1[0], bbox2[0]) - padding)
        y1 = max(0, min(bbox1[1], bbox2[1]) - padding)
        x2 = min(1000, max(bbox1[2], bbox2[2]) + padding)
        y2 = min(1000, max(bbox1[3], bbox2[3]) + padding)
        return [x1, y1, x2, y2]


# ============================================================================
# DATASET GENERATION FROM EXISTING HOI DATA
# ============================================================================

def load_hoi_data(parquet_path: str) -> pd.DataFrame:
    """Load existing HOI training data."""
    return pd.read_parquet(parquet_path)


def convert_to_python_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_python_native(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_python_native(value) for key, value in obj.items()}
    return obj


def generate_sft_dataset(
    hoi_data: pd.DataFrame,
    output_path: str,
    tool_usage_ratio: float = 0.7,  # 70% with tools, 30% without
    num_samples: Optional[int] = None
) -> List[Dict]:
    """
    Generate SFT dataset from HOI data.
    
    Following PixelReasoner pattern:
    - ~70% samples with tool calls (pixel-space reasoning)
    - ~30% samples without tool calls (textual reasoning)
    """
    generator = HOIReasoningTrajectory()
    sft_samples = []
    
    if num_samples:
        hoi_data = hoi_data.sample(n=min(num_samples, len(hoi_data)), random_state=42)
    
    for idx, row in hoi_data.iterrows():
        try:
            # Parse existing data
            extra_info = row['extra_info']
            action = extra_info.get('action', 'interacting with')
            object_category = extra_info.get('object_category', 'object')
            boxes = extra_info.get('boxes_1000', [])
            
            # Convert numpy types to Python native
            boxes = convert_to_python_native(boxes)
            
            if len(boxes) < 2:
                continue
            
            person_bbox = boxes[0]
            object_bbox = boxes[1]
            
            image_path = row['images'][0]['image'] if row['images'] else None
            if not image_path:
                continue
            
            # Decide if using tools based on ratio
            use_tools = random.random() < tool_usage_ratio
            
            # Randomly choose task type
            task_type = random.choice(['referring', 'grounding', 'detection'])
            
            if task_type == 'referring':
                messages = generator.create_referring_trajectory(
                    image_path=image_path,
                    action=action,
                    object_category=object_category,
                    person_bbox=person_bbox,
                    object_bbox=object_bbox,
                    use_tools=use_tools
                )
            elif task_type == 'grounding':
                messages = generator.create_grounding_trajectory(
                    image_path=image_path,
                    person_bbox=person_bbox,
                    object_bbox=object_bbox,
                    object_category=object_category,
                    action=action,
                    use_tools=use_tools
                )
            else:  # detection
                messages = generator.create_detection_trajectory(
                    image_path=image_path,
                    action=action,
                    object_category=object_category,
                    hoi_pairs=[(person_bbox, object_bbox)],
                    use_tools=use_tools
                )
            
            sft_samples.append({
                "message_list": messages,
                "qid": str(idx),
                "task_type": task_type,
                "use_tools": use_tools,
                "metadata": {
                    "action": action,
                    "object_category": object_category,
                    "source": "hoi_sft_generated"
                }
            })
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(sft_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(sft_samples)} SFT samples")
    print(f"Saved to {output_path}")
    
    return sft_samples


def create_example_trajectories() -> None:
    """Create example trajectories for reference."""
    generator = HOIReasoningTrajectory()
    
    examples = []
    
    # Example 1: Referring with tools
    example1 = generator.create_referring_trajectory(
        image_path="images/HICO_train2015_00001234.jpg",
        action="riding",
        object_category="bicycle",
        person_bbox=[200, 150, 450, 800],
        object_bbox=[180, 400, 500, 850],
        use_tools=True
    )
    examples.append({
        "name": "Referring Task with Tool Usage",
        "description": "Find person-object pair given action description",
        "messages": example1
    })
    
    # Example 2: Grounding with tools
    example2 = generator.create_grounding_trajectory(
        image_path="images/HICO_train2015_00002345.jpg",
        person_bbox=[100, 200, 350, 700],
        object_bbox=[320, 350, 450, 500],
        object_category="cup",
        action="holding",
        use_tools=True
    )
    examples.append({
        "name": "Grounding Task with Tool Usage",
        "description": "Identify interaction type given bounding boxes",
        "messages": example2
    })
    
    # Example 3: Detection with tools
    example3 = generator.create_detection_trajectory(
        image_path="images/HICO_train2015_00003456.jpg",
        action="eating",
        object_category="pizza",
        hoi_pairs=[
            ([50, 100, 300, 600], [250, 200, 400, 400]),
            ([500, 150, 750, 650], [680, 250, 850, 450])
        ],
        use_tools=True
    )
    examples.append({
        "name": "Detection Task with Tool Usage",
        "description": "Detect all instances of specific HOI",
        "messages": example3
    })
    
    # Example 4: Referring without tools (textual reasoning only)
    example4 = generator.create_referring_trajectory(
        image_path="images/HICO_train2015_00004567.jpg",
        action="sitting on",
        object_category="chair",
        person_bbox=[300, 100, 550, 700],
        object_bbox=[280, 400, 600, 750],
        use_tools=False
    )
    examples.append({
        "name": "Referring Task without Tools (Textual Reasoning)",
        "description": "Direct answer without tool usage for simpler cases",
        "messages": example4
    })
    
    output_path = Path("/workspace/verl-tool/data/pixel_reasoner/sft_analysis/hoi_sft_examples.json")
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"Example trajectories saved to {output_path}")
    
    # Print formatted examples
    print("\n" + "="*80)
    print("EXAMPLE HOI SFT TRAJECTORIES")
    print("="*80)
    
    for ex in examples:
        print(f"\n### {ex['name']} ###")
        print(f"Description: {ex['description']}")
        print("-" * 60)
        for msg in ex['messages']:
            role = msg['role'].upper()
            print(f"\n[{role}]")
            for content in msg['content']:
                if content.get('text'):
                    print(content['text'][:500] + "..." if len(content.get('text', '')) > 500 else content.get('text', ''))
                if content.get('image'):
                    print(f"[IMAGE: {content['image']}]")


if __name__ == "__main__":
    # Create example trajectories
    create_example_trajectories()
    
    # Generate SFT dataset from existing HOI data
    hoi_data_path = "/workspace/verl-tool/data/hoi/train_data/train.parquet"
    output_path = "/workspace/verl-tool/data/hoi/sft_train.json"
    
    if Path(hoi_data_path).exists():
        print("\n" + "="*80)
        print("GENERATING SFT DATASET FROM EXISTING HOI DATA")
        print("="*80)
        
        hoi_data = load_hoi_data(hoi_data_path)
        print(f"Loaded {len(hoi_data)} HOI samples")
        
        # Generate a subset for demo
        sft_samples = generate_sft_dataset(
            hoi_data,
            output_path=output_path,
            tool_usage_ratio=0.7,
            num_samples=100  # Generate 100 samples for demo
        )

