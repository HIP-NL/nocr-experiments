# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-genai>=1.56"
# ]
# ///

"""
Utrecht 1899 Tax Records - VLM Extraction Experiments

This script runs experiments using Google's Gemini Flash models to extract
structured data from historical handwritten tax records.

Experiments test:
- Different model versions (2.0-flash, 2.5-flash, 3.0-flash, lite variants)
- Few-shot vs zero-shot learning
- Different thinking budgets
"""

import itertools
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.types import Content, FileData, GenerateContentConfig, Part

# Configuration
BASE_DIR = Path(__file__).parent.parent
# BASE_DIR = Path("/Users/Rijpm101/repos/nocr-experiments")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions_test"
PREDICTIONS_DIR = RESULTS_DIR / "predictions_test2"
METADATA_DIR = RESULTS_DIR / "metadata"
METADATA_DIR = RESULTS_DIR / "metadata2"

# Ensure output directories exist
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Gemini client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Image files to process
IMAGE_FILES = [
    "NL-UtHUA_A376076_000033_l.jpg",
    "NL-UtHUA_A376076_000033_r.jpg",
    "NL-UtHUA_A376079_000005_l.jpg",
    "NL-UtHUA_A376079_000005_r.jpg",
]

# Models to test
MODELS = [
    # "models/gemini-2.0-flash-lite",
    # "models/gemini-2.0-flash",
    "models/gemini-2.5-flash-lite",
    # "models/gemini-2.5-flash",
    # "models/gemini-3.0-flash-preview",
]

# Thinking budgets to test
THINKING_BUDGETS = [0, 2000]


def upload_image(image_path):
    """Upload an image to Gemini and return a Part object."""
    file = client.files.upload(
        file=str(image_path), config=dict(mime_type="image/jpeg")
    )
    print(f"  Uploaded: {image_path.name}")
    return Part(file_data=FileData(mime_type=file.mime_type, file_uri=file.uri))


def load_ground_truth(image_name):
    """Load ground truth JSON for an image."""
    gt_path = DATA_DIR / "ground_truth" / f"{image_name.replace('.jpg', '.json')}"
    with open(gt_path, "r") as f:
        return json.load(f)


def load_prompt():
    """Load the task prompt."""
    with open(DATA_DIR / "prompt.txt", "r") as f:
        prompt = f.read()
    return "Perform the following task using step-by-step reasoning." + prompt


def get_model_short_name(model_name):
    """Extract short model name from full model path."""
    # models/gemini-2.5-flash-lite -> gemini-2.5-flash-lite
    return model_name.replace("models/", "").replace(":", "-")


def build_output_filename(image_name, model_name, strategy, thinking_budget, suffix=""):
    """Build standardized output filename."""
    image_base = image_name.replace(".jpg", "")
    model_short = get_model_short_name(model_name)
    thinking_str = f"thinking{thinking_budget}"

    parts = [image_base, model_short, strategy, thinking_str]
    filename = "__".join(parts) + suffix + ".json"
    return filename


def run_experiment(model_name, messages, image_name, strategy, thinking_budget):
    """Run a single experiment and save results."""
    try:
        print(f"  Running {strategy} (thinking={thinking_budget}) on {image_name}...")

        # Configure thinking
        thinking_config = None
        if thinking_budget > 0:
            thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)

        # Generate content
        response = client.models.generate_content(
            model=model_name,
            config=GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.9,
                thinking_config=thinking_config,
            ),
            contents=messages,
        )

        # Extract metadata
        metadata = {
            "candidates_token_count": response.usage_metadata.candidates_token_count,
            "thoughts_token_count": response.usage_metadata.thoughts_token_count,
            "prompt_token_count": response.usage_metadata.prompt_token_count,
            "total_token_count": response.usage_metadata.total_token_count,
        }

        # Save prediction
        if response.text:
            response_json = json.loads(response.text)

            pred_filename = build_output_filename(
                image_name, model_name, strategy, thinking_budget
            )
            meta_filename = build_output_filename(
                image_name, model_name, strategy, thinking_budget
            )

            with open(PREDICTIONS_DIR / pred_filename, "w") as f:
                json.dump(response_json, f, indent=4)

            with open(METADATA_DIR / meta_filename, "w") as f:
                json.dump(metadata, f, indent=4)

            print(f"  ✓ Saved: {pred_filename}")
        else:
            print(f"  ✗ Empty response")

    except Exception as e:
        print(f"  ✗ Error: {e}")


# Load prompt
prompt_text = load_prompt()
prompt_part = Part(text=prompt_text)

# Upload all images
print("\nUploading images...")
image_parts = {}
for image_file in IMAGE_FILES:
    image_path = DATA_DIR / "images" / image_file
    image_parts[image_file] = upload_image(image_path)

# Generate all combinations: 3 images as examples, 1 as target
combinations = list(itertools.combinations(range(len(IMAGE_FILES)), 3))

# Main experiment loop.
for model_name in MODELS:
    print(f"\n{'=' * 70}")
    print(f"Model: {model_name}")
    print(f"{'=' * 70}")

    for combination in combinations:
        example_indices = combination
        target_index = list(set(range(len(IMAGE_FILES))) - set(example_indices))[0]
        target_image = IMAGE_FILES[target_index]

        print(f"\nTarget: {target_image}")

        for thinking_budget in THINKING_BUDGETS:
            # Few-shot experiment
            messages_fewshot = []
            for idx in example_indices:
                example_image = IMAGE_FILES[idx]
                gt_response = load_ground_truth(example_image)
                gt_part = Part(text=json.dumps(gt_response, indent=4))

                messages_fewshot.extend(
                    [
                        Content(
                            role="user",
                            parts=[image_parts[example_image], prompt_part],
                        ),
                        Content(role="model", parts=[gt_part]),
                    ]
                )

            messages_fewshot.append(
                Content(role="user", parts=[image_parts[target_image], prompt_part])
            )

            run_experiment(
                model_name,
                messages_fewshot,
                target_image,
                "fewshot",
                thinking_budget,
            )

            # Zero-shot experiment
            messages_zeroshot = [
                Content(role="user", parts=[image_parts[target_image], prompt_part])
            ]

            run_experiment(
                model_name,
                messages_zeroshot,
                target_image,
                "zeroshot",
                thinking_budget,
            )

            # time.sleep(2)  # Rate limiting, uncomment if needed

print("\n" + "=" * 70)
print("Experiments complete!")
print(f"Predictions saved to: {PREDICTIONS_DIR}")
print(f"Metadata saved to: {METADATA_DIR}")
print("=" * 70)
