#!/usr/bin/env python3
"""
Test script to verify vLLM API server supports multimodal (image + text) inputs
for Qwen3-VL-30B-A3B-Thinking model.
"""

import base64
import sys
from pathlib import Path
from openai import OpenAI


def image_to_base64_url(image_path: str) -> str:
    """Convert local image to base64 data URL for API."""
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Determine image format from extension
    ext = Path(image_path).suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        mime_type = 'image/jpeg'
    elif ext == '.png':
        mime_type = 'image/png'
    else:
        mime_type = 'image/jpeg'  # Default

    return f"data:{mime_type};base64,{image_data}"


def test_vllm_multimodal(image_path: str, api_url: str = "http://localhost:18000/v1"):
    """Test vLLM API server with image + text prompt."""

    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        print("\nUsage: python test_vllm_multimodal.py <path_to_image>")
        return False

    print(f"Connecting to vLLM API server at {api_url}...")
    print(f"Testing with image: {image_path}")

    client = OpenAI(
        base_url=api_url,
        api_key="not-needed"
    )

    print("Connected successfully!")
    print("\n" + "=" * 60)
    print("MULTIMODAL TEST - Image + Text")
    print("=" * 60)

    # Convert image to base64
    print("\nConverting image to base64...")
    image_url = image_to_base64_url(image_path)
    print(f"Image size: {len(image_url)} characters")

    # Test prompt
    text_query = "What do you see in this image? Please describe it briefly."

    print(f"\nText query: {text_query}")

    # Prepare multimodal message (OpenAI format)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text_query}
            ]
        }
    ]

    print("\nSending request to vLLM server...")

    try:
        # Generate response
        response = client.chat.completions.create(
            model="Qwen/Qwen3-VL-30B-A3B-Thinking",
            messages=messages,
            temperature=0.7,
            max_tokens=200,
            top_p=0.9
        )

        # Print results
        print("\n" + "=" * 60)
        print("RESPONSE:")
        print("=" * 60)

        generated_text = response.choices[0].message.content
        print(f"\n{generated_text}\n")

        print("=" * 60)
        print("Response details:")
        print("=" * 60)
        print(f"  - Model: {response.model}")
        print(f"  - Finish reason: {response.choices[0].finish_reason}")
        print(f"  - Usage:")
        print(f"    - Prompt tokens: {response.usage.prompt_tokens}")
        print(f"    - Completion tokens: {response.usage.completion_tokens}")
        print(f"    - Total tokens: {response.usage.total_tokens}")

        print("\n" + "=" * 60)
        print("✅ MULTIMODAL TEST PASSED!")
        print("=" * 60)
        print("\nYour vLLM server correctly handles image + text inputs.")
        print("The generate_sft_with_teacher.py script should work!")

        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ MULTIMODAL TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}\n")

        import traceback
        traceback.print_exc()

        print("\nPossible issues:")
        print("  1. vLLM server not started with vision model support")
        print("  2. Model doesn't support multimodal inputs")
        print("  3. Image format not supported")
        print("  4. vLLM version too old (need vllm>=0.11.0 for Qwen3-VL)")

        return False


if __name__ == "__main__":
    # Default test image path (you can change this)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find a sample image in the HOI benchmark data
        default_paths = [
            "data/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg",
            "data/benchmarks_simplified/sample_image.jpg",
        ]

        image_path = None
        for path in default_paths:
            if Path(path).exists():
                image_path = path
                break

        if image_path is None:
            print("Error: No test image found!")
            print("\nUsage: python test_vllm_multimodal.py <path_to_image>")
            print("\nExample:")
            print("  python test_vllm_multimodal.py data/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg")
            sys.exit(1)

    # Run test
    test_vllm_multimodal(image_path)
