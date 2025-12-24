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
HOI Detector Tool for Human-Object Interaction Detection.

This module provides tools for HOI detection with RL training:
- zoom_in: Crop and zoom into a specific region of the image
- zoom_out: Reset to the original image view
- detect_objects: Use Grounding DINO for object detection

All coordinates use the 1000x1000 normalized format for Qwen3-VL compatibility.
"""

from .base import BaseTool, register_tool
import regex as re
import json
import asyncio
import concurrent.futures
from typing import Tuple, Union, List, Dict, Any, Optional
import os

import base64
import io
from PIL import Image
from pathlib import Path
from verl_tool.agent_loop.vision_utils import process_image

import torch
import logging

logger = logging.getLogger(__name__)

# Global cache for Grounding DINO model
_grounding_dino_model = None
_grounding_dino_processor = None
_model_device = None


def get_grounding_dino_model(device: str = None):
    """
    Lazily load Grounding DINO model (cached after first load).
    
    Args:
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        
    Returns:
        tuple: (model, processor) or (None, None) if not available
    """
    global _grounding_dino_model, _grounding_dino_processor, _model_device
    
    if _grounding_dino_model is not None:
        return _grounding_dino_model, _grounding_dino_processor
    
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _model_device = device
    
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        
        model_id = "IDEA-Research/grounding-dino-tiny"
        logger.info(f"Loading Grounding DINO model: {model_id}")
        
        _grounding_dino_processor = AutoProcessor.from_pretrained(model_id)
        _grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        _grounding_dino_model = _grounding_dino_model.to(device)
        _grounding_dino_model.eval()
        
        logger.info(f"Grounding DINO loaded successfully on {device}")
        return _grounding_dino_model, _grounding_dino_processor
        
    except ImportError as e:
        logger.warning(f"Grounding DINO not available: {e}")
        logger.warning("Install with: pip install transformers")
        return None, None
    except Exception as e:
        logger.warning(f"Failed to load Grounding DINO: {e}")
        return None, None


def detect_objects_with_grounding_dino(
    image: Image.Image,
    class_names: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25
) -> List[Dict[str, Any]]:
    """
    Detect objects using Grounding DINO.
    
    Args:
        image: PIL Image to process
        class_names: Classes to detect, separated by " . " (e.g., "person . cup")
        box_threshold: Confidence threshold for boxes
        text_threshold: Confidence threshold for text matching
        
    Returns:
        List of detections, each with 'label', 'bbox', 'confidence'
        bbox is in [0-1000] normalized format
    """
    model, processor = get_grounding_dino_model()
    
    if model is None:
        return []
    
    # Prepare input
    inputs = processor(images=image, text=class_names, return_tensors="pt")
    inputs = {k: v.to(_model_device) for k, v in inputs.items()}
    
    # Run detection
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    
    try:
        # Try newer API (transformers >= 4.40)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            target_sizes=target_sizes,
            threshold=box_threshold,
            text_threshold=text_threshold,
        )[0]
    except TypeError:
        try:
            # Try alternative API
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            # Fallback: manual post-processing
            logits = outputs.logits.sigmoid()[0]
            boxes = outputs.pred_boxes[0]
            
            scores = logits.max(dim=-1).values
            mask = scores > box_threshold
            
            filtered_scores = scores[mask]
            filtered_boxes = boxes[mask]
            
            img_h, img_w = image.size[::-1]
            scale = torch.tensor([img_w, img_h, img_w, img_h])
            filtered_boxes = filtered_boxes * scale
            
            class_list = [c.strip() for c in class_names.split('.')]
            label_indices = logits[mask].argmax(dim=-1)
            
            results = {
                "scores": filtered_scores,
                "labels": [class_list[min(idx.item(), len(class_list)-1)] for idx in label_indices],
                "boxes": filtered_boxes
            }
    
    detections = []
    img_width, img_height = image.size
    
    scores = results.get("scores", [])
    labels = results.get("labels", [])
    boxes = results.get("boxes", [])
    
    for score, label, box in zip(scores, labels, boxes):
        score_val = score.item() if hasattr(score, 'item') else float(score)
        
        if score_val < box_threshold:
            continue
        
        # Convert to [0-1000] normalized format
        if hasattr(box, 'tolist'):
            x1, y1, x2, y2 = box.tolist()
        else:
            x1, y1, x2, y2 = box
            
        norm_bbox = [
            int(x1 / img_width * 1000),
            int(y1 / img_height * 1000),
            int(x2 / img_width * 1000),
            int(y2 / img_height * 1000)
        ]
        
        detections.append({
            'label': label,
            'bbox': norm_bbox,
            'confidence': round(score_val, 3)
        })
    
    return detections


def crop(str_image, bbox_2d, padding=(0.1, 0.1)):
    """
    Crop the image based on the bounding box coordinates.
    Supports both pixel coordinates and normalized [0-1] coordinates.
    """
    if isinstance(str_image, list):
        str_image = str_image[0]
    if isinstance(str_image, Path) and str_image.exists() or \
        isinstance(str_image, str) and os.path.exists(str_image):
        image = Image.open(str_image)
    elif isinstance(str_image, Image.Image):
        image = str_image
    else:
        image = decode_image_url(str_image)
    
    img_x, img_y = image.size
    padding_tr = (600.0/img_x, 600.0/img_y)
    padding = (min(padding[0], padding_tr[0]), min(padding[1], padding_tr[1]))

    # Handle 1000-grid coordinates (convert to normalized)
    if all(coord > 1 for coord in bbox_2d):
        # Assume 1000-grid format
        normalized_bbox_2d = (
            float(bbox_2d[0])/1000 - padding[0],
            float(bbox_2d[1])/1000 - padding[1],
            float(bbox_2d[2])/1000 + padding[0],
            float(bbox_2d[3])/1000 + padding[1]
        )
    elif bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
        normalized_bbox_2d = (
            float(bbox_2d[0]) - padding[0],
            float(bbox_2d[1]) - padding[1],
            float(bbox_2d[2]) + padding[0],
            float(bbox_2d[3]) + padding[1]
        )
    else:
        normalized_bbox_2d = (
            float(bbox_2d[0])/img_x - padding[0],
            float(bbox_2d[1])/img_y - padding[1],
            float(bbox_2d[2])/img_x + padding[0],
            float(bbox_2d[3])/img_y + padding[1]
        )
    
    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
    normalized_x1 = min(max(0, normalized_x1), 1)
    normalized_y1 = min(max(0, normalized_y1), 1)
    normalized_x2 = min(max(0, normalized_x2), 1)
    normalized_y2 = min(max(0, normalized_y2), 1)
    
    cropped_img = image.crop((
        int(normalized_x1*img_x),
        int(normalized_y1*img_y),
        int(normalized_x2*img_x),
        int(normalized_y2*img_y)
    ))
    return cropped_img


def encode_image(img: Image.Image) -> str:
    buffered = io.BytesIO()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def decode_image(img_str):
    img_data = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_data))
    return img


def encode_image_url(img: Image.Image) -> str:
    encoded_img = encode_image(img)
    return f"data:image/jpeg;base64,{encoded_img}"


def decode_image_url(img_str):
    if img_str.startswith("data:image/jpeg;base64,"):
        img_str = img_str.split("data:image/jpeg;base64,")[1]
    return decode_image(img_str)


def rm_tree(pth: Path):
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


@register_tool
class HOIDetectorTool(BaseTool):
    """
    HOI Detector Tool for Human-Object Interaction Detection.
    
    Provides three tools:
    - zoom_in: Zoom into a specific region of the image
    - zoom_out: Reset to the original image view
    - detect_objects: Object detection using Grounding DINO
    """
    tool_type = "hoi_detector"

    stop_tokens = ["</tool_call>"]
    valid_mcp_func_names = ['zoom_in', 'zoom_out', 'detect_objects', 'crop_image']

    def __init__(self, num_workers=1):
        super().__init__(num_workers)
        self.image_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) + 4),
            thread_name_prefix="hoi_image_processor"
        )

    def get_usage_inst(self):
        return """HOI Detection tools:
- zoom_in: Zoom into a region. Args: bbox_2d=[x1,y1,x2,y2], target_image=1
- zoom_out: Reset to original image. Args: target_image=1
- detect_objects: Detect objects. Args: class_names="person . object", target_image=1"""
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string into an actual action and its contents.
        """
        try:
            call = json.loads(action.split('<tool_call>')[1].split('</tool_call>')[0])
            name = call.get('name', '')
            if name not in self.valid_mcp_func_names:
                return "", False
        except:
            return "", False
        
        return call, True

    def load_env(self, trajectory_id):
        """Load the environment for the given trajectory_id."""
        env = self.env_cache.get(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {
                    "turns": 0,
                },
                "previous_obs": [],
                "images": None,
                "original_images": None,  # Store original images for zoom_out
                "temporary_images": [],
                "temporary_image_folder": Path(f"tmp/hoi_images/{trajectory_id}"),
            }
            env['temporary_image_folder'].mkdir(parents=True, exist_ok=True)
        return env
    
    def update_env(self, trajectory_id, env, action, is_valid, extra_field, observation, **kwargs):
        """Update the environment for the given trajectory_id."""
        if isinstance(observation, dict) and 'image' in observation:
            if isinstance(observation['image'], str):
                env['images'].append(self.save_image_to_env(trajectory_id, observation['image']))
            elif isinstance(observation['image'], list):
                env['images'].extend([self.save_image_to_env(trajectory_id, img) for img in observation['image']])
        env["metadata"]["turns"] += 1
        env["previous_obs"].append({
            "action": action,
            "is_valid": is_valid,
            "observation": observation,
            "extra_field": extra_field,
            **kwargs
        })
    
    def delete_env(self, trajectory_id):
        """Delete the environment for the given trajectory_id."""
        env = self.env_cache.pop(trajectory_id, None)

    def save_image_to_env(self, trajectory_id, image: Union[Image.Image, str]) -> str:
        """Save the image to the environment for the given trajectory_id."""
        env = self.load_env(trajectory_id)
        env['temporary_images'].append(image)
        return image

    async def _process_single_image(self, img_source, bbox_2d):
        """Process a single image crop operation asynchronously."""
        def _crop_and_process():
            cropped_img = crop(img_source, bbox_2d)
            processed_img = process_image({"image": cropped_img})
            return processed_img
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.image_executor, _crop_and_process)

    async def conduct_zoom_in_action_async(self, parameters, env):
        """Execute the zoom-in action asynchronously."""
        valid = False
        missing_parameters = []
        if 'bbox_2d' not in parameters:
            missing_parameters.append('bbox_2d')
        if 'target_image' not in parameters:
            missing_parameters.append('target_image')
        
        try:
            parameters['target_image'] = int(parameters['target_image'])
        except:
            pass
        
        if missing_parameters:
            observation = f"Missing parameters: {', '.join(missing_parameters)}"
        elif not isinstance(parameters['bbox_2d'], list) or len(parameters['bbox_2d']) != 4:
            observation = "Invalid bbox_2d format. It should be a list of four numbers [x1, y1, x2, y2]."
        elif not isinstance(parameters['target_image'], int) or parameters['target_image'] <= 0 or parameters['target_image'] > len(env['images']):
            observation = f"Invalid target_image index. It should be an integer between 1 and {len(env['images'])}."
        else:
            try:
                previous_images = env['images']
                img_to_crop = previous_images[parameters['target_image']-1]
                
                processed_img = await self._process_single_image(img_to_crop, parameters['bbox_2d'])
                
                encoded_cropped_img = encode_image_url(processed_img)
                image_width, image_height = processed_img.size
                observation = {
                    'obs': f"Here is the zoomed-in image. (Image Size: {image_width}x{image_height})\n<image>",
                    'image': encoded_cropped_img,
                }
                valid = True
            except Exception as e:
                observation = f"Error processing image: {str(e)}"
                logger.error(f"Error processing zoom-in action: {str(e)}; parameters: {parameters}")
        
        return observation, valid

    async def conduct_zoom_out_action_async(self, parameters, env):
        """Execute the zoom-out action asynchronously (reset to original image)."""
        valid = False
        
        if 'target_image' not in parameters:
            parameters['target_image'] = 1
        
        try:
            parameters['target_image'] = int(parameters['target_image'])
        except:
            pass
        
        if not env.get('original_images'):
            observation = "No original images available to zoom out to."
        elif not isinstance(parameters['target_image'], int) or parameters['target_image'] <= 0 or parameters['target_image'] > len(env['original_images']):
            observation = f"Invalid target_image index. It should be an integer between 1 and {len(env['original_images'])}."
        else:
            try:
                original_img_source = env['original_images'][parameters['target_image']-1]
                
                # Load and process the original image
                if isinstance(original_img_source, (str, Path)) and os.path.exists(str(original_img_source)):
                    original_img = Image.open(original_img_source)
                elif isinstance(original_img_source, Image.Image):
                    original_img = original_img_source
                else:
                    original_img = decode_image_url(original_img_source)
                
                processed_img = process_image({"image": original_img})
                encoded_img = encode_image_url(processed_img)
                image_width, image_height = processed_img.size
                
                observation = {
                    'obs': f"Here is the original full image. (Image Size: {image_width}x{image_height})\n<image>",
                    'image': encoded_img,
                }
                valid = True
            except Exception as e:
                observation = f"Error processing zoom-out: {str(e)}"
                logger.error(f"Error processing zoom-out action: {str(e)}; parameters: {parameters}")
        
        return observation, valid

    async def conduct_detect_objects_action_async(self, parameters, env):
        """Execute object detection using Grounding DINO asynchronously."""
        valid = False
        missing_parameters = []
        
        if 'class_names' not in parameters:
            missing_parameters.append('class_names')
        if 'target_image' not in parameters:
            parameters['target_image'] = 1
        
        try:
            parameters['target_image'] = int(parameters['target_image'])
        except:
            pass
        
        confidence_threshold = parameters.get('confidence_threshold', 0.25)
        
        if missing_parameters:
            observation = f"Missing parameters: {', '.join(missing_parameters)}"
        elif not isinstance(parameters['target_image'], int) or parameters['target_image'] <= 0 or parameters['target_image'] > len(env['images']):
            observation = f"Invalid target_image index. It should be an integer between 1 and {len(env['images'])}."
        else:
            try:
                img_source = env['images'][parameters['target_image']-1]
                
                # Load image
                if isinstance(img_source, (str, Path)) and os.path.exists(str(img_source)):
                    image = Image.open(img_source).convert('RGB')
                elif isinstance(img_source, Image.Image):
                    image = img_source.convert('RGB')
                else:
                    image = decode_image_url(img_source).convert('RGB')
                
                # Run detection in thread pool
                def _detect():
                    return detect_objects_with_grounding_dino(
                        image,
                        parameters['class_names'],
                        box_threshold=confidence_threshold,
                        text_threshold=confidence_threshold
                    )
                
                loop = asyncio.get_event_loop()
                detections = await loop.run_in_executor(self.image_executor, _detect)
                
                if not detections:
                    observation = f"No objects detected for classes: {parameters['class_names']}"
                else:
                    result_lines = [f'Detected {len(detections)} objects:']
                    for det in detections:
                        bbox_str = f"[{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]"
                        result_lines.append(
                            f"- {det['label']}: bbox={bbox_str}, confidence={det['confidence']}"
                        )
                    observation = '\n'.join(result_lines)
                
                valid = True
                
            except Exception as e:
                observation = f"Error during detection: {str(e)}"
                logger.error(f"Error processing detect_objects action: {str(e)}; parameters: {parameters}")
        
        return observation, valid

    async def aget_observations(self, trajectory_ids: List[str], actions: List[str], extra_fields: List[Dict[str, Any]]):
        """Async version of get_observations for concurrent processing."""
        observations = []
        dones = []
        valids = []
        
        tasks = []
        for i, (trajectory_id, action, extra_field) in enumerate(zip(trajectory_ids, actions, extra_fields)):
            task = self._conduct_action_async(trajectory_id, action, extra_field)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                observations.append(f"Processing error: {str(result)}")
                dones.append(False)
                valids.append(False)
            else:
                obs, done, valid = result
                observations.append(obs)
                dones.append(done)
                valids.append(valid)
        
        return observations, dones, valids

    async def _conduct_action_async(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]):
        """Execute the parsed action asynchronously."""
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        # Initialize images from extra_field if not already set
        if env['images'] is None:
            env['images'] = [Path(x) if not x.startswith("data:image") else decode_image_url(x) for x in extra_field.get('images', [])]
            # Store original images for zoom_out functionality
            env['original_images'] = env['images'].copy()
        
        if not is_valid:
            observation = ""
            done = False
            valid = False
        else:
            done = False
            valid = True
            if 'arguments' not in parsed_action:
                observation = "Missing 'arguments' in the tool call."
                valid = False
            elif not isinstance(parsed_action['arguments'], dict):
                observation = f"'arguments' should be a dictionary, got {type(parsed_action['arguments'])}."
                valid = False
            elif parsed_action['name'] in ['zoom_in', 'crop_image']:
                try:
                    observation, valid = await self.conduct_zoom_in_action_async(parsed_action['arguments'], env)
                except Exception as e:
                    observation = f"Error processing zoom_in action: {str(e)}"
                    valid = False
            elif parsed_action['name'] == 'zoom_out':
                try:
                    observation, valid = await self.conduct_zoom_out_action_async(parsed_action['arguments'], env)
                except Exception as e:
                    observation = f"Error processing zoom_out action: {str(e)}"
                    valid = False
            elif parsed_action['name'] == 'detect_objects':
                try:
                    observation, valid = await self.conduct_detect_objects_action_async(parsed_action['arguments'], env)
                except Exception as e:
                    observation = f"Error processing detect_objects action: {str(e)}"
                    valid = False
            else:
                observation = f"Unknown action name: {parsed_action['name']}"
                valid = False

        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid

    def conduct_action(self, trajectory_id, action, extra_field):
        """Synchronous wrapper for backward compatibility."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._conduct_action_async(trajectory_id, action, extra_field))
        finally:
            loop.close()

    def __del__(self):
        """Cleanup when tool is destroyed."""
        if hasattr(self, 'image_executor'):
            self.image_executor.shutdown(wait=False)


def clear_grounding_dino_cache():
    """Clear the cached Grounding DINO model to free memory."""
    global _grounding_dino_model, _grounding_dino_processor, _model_device
    
    if _grounding_dino_model is not None:
        del _grounding_dino_model
        del _grounding_dino_processor
        _grounding_dino_model = None
        _grounding_dino_processor = None
        _model_device = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Grounding DINO model cache cleared")

