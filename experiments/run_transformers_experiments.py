# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers>=4.52.0",
#   "torch>=2.2.0",
#   "accelerate>=0.30.0",
#   "sentencepiece>=0.2.0",
#   "pillow>=10.0.0",
#   "torchvision>=0.17.0"
# ]
# ///

"""
Minimal version of `run_experiments.py` that:
- uses a local Transformers vision-language model,  only zero-shot and few-shot experiments

uv run experiments/run_transformers_experiments.py
uv run experiments/run_transformers_experiments.py --model google/gemma-3-4b-it
"""

import argparse
import itertools
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL = "google/gemma-4-E2B-it"

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions_transformers"
METADATA_DIR = RESULTS_DIR / "metadata"

# Ensure output directories exist
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Image files to process
IMAGE_FILES = [
    "NL-UtHUA_A376076_000033_l.jpg",
    "NL-UtHUA_A376076_000033_r.jpg",
    "NL-UtHUA_A376079_000005_l.jpg",
    "NL-UtHUA_A376079_000005_r.jpg",
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_torch_dtype(device):
    if device.type in {"cuda", "mps"}:
        return torch.bfloat16
    return torch.float32


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simple zero-shot and few-shot OCR experiments with a local Transformers model."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face model id. Defaults to a small Gemma-family model: "
            f"`{DEFAULT_MODEL}`."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. If 0, generation uses greedy decoding.",
    )
    return parser.parse_args()


def load_ground_truth(image_name):
    """Load ground truth JSON for an image."""
    gt_path = DATA_DIR / "ground_truth" / f"{image_name.replace('.jpg', '.json')}"
    with open(gt_path, "r") as f:
        return json.load(f)


def load_prompt():
    """Load the task prompt."""
    with open(DATA_DIR / "prompt.txt", "r") as f:
        prompt = f.read()
    return prompt.format(output_format="JSON")


def load_image(image_name):
    image_path = DATA_DIR / "images" / image_name
    return Image.open(image_path).convert("RGB")


def get_model_short_name(model_name):
    """Extract a filesystem-friendly model name."""
    return model_name.replace("models/", "").replace(":", "-").replace("/", "--")


def build_output_filename(
    image_name, model_name, strategy, thinking_budget=0, suffix="", ext="json"
):
    """Build standardized output filename."""
    image_base = image_name.replace(".jpg", "")
    model_short = get_model_short_name(model_name)
    thinking_str = f"thinking{thinking_budget}"
    parts = [image_base, model_short, strategy, thinking_str]
    return "__".join(parts) + suffix + "." + ext


def extract_json_from_text(text):
    """Best-effort extraction of JSON from model output."""
    text = text.strip()

    code_block_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    for candidate in (text,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    if array_match:
        return json.loads(array_match.group(0))

    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        return json.loads(object_match.group(0))

    raise json.JSONDecodeError("Could not extract valid JSON", text, 0)


def build_zeroshot_messages(prompt_text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"},
            ],
        }
    ]


def build_fewshot_messages(prompt_text, example_images, target_image):
    messages = []

    for example_image in example_images:
        gt_response = load_ground_truth(example_image)
        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": json.dumps(gt_response, indent=2)}
                    ],
                },
            ]
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"},
            ],
        }
    )
    return messages


def flatten_images_for_messages(example_images, target_image, strategy):
    if strategy == "zeroshot":
        return [load_image(target_image)]

    images = [load_image(image_name) for image_name in example_images]
    images.append(load_image(target_image))
    return images


def generate_response(
    processor,
    model,
    device,
    messages,
    images,
    max_new_tokens,
    temperature,
):
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(images=images, text=prompt, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature

    outputs = model.generate(**inputs, **generate_kwargs)
    generated_ids = outputs[0][input_len:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def run_experiment(
    processor,
    model,
    model_name,
    device,
    prompt_text,
    image_name,
    strategy,
    example_images,
    max_new_tokens,
    temperature,
):
    """Run a single experiment and save results."""
    try:
        print(f"  Running {strategy} on {image_name}...")

        if strategy == "fewshot":
            messages = build_fewshot_messages(prompt_text, example_images, image_name)
        else:
            messages = build_zeroshot_messages(prompt_text)

        images = flatten_images_for_messages(example_images, image_name, strategy)

        started_at = time.time()
        response_text = generate_response(
            processor,
            model,
            device,
            messages,
            images,
            max_new_tokens,
            temperature,
        )
        duration_seconds = time.time() - started_at

        metadata = {
            "model": model_name,
            "strategy": strategy,
            "image_name": image_name,
            "example_images": example_images,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "thinking_budget": 0,
            "duration_seconds": round(duration_seconds, 3),
            "raw_text_preview": response_text[:500],
        }

        pred_filename = build_output_filename(
            image_name,
            model_name,
            strategy,
            thinking_budget=0,
            ext="json",
        )
        meta_filename = build_output_filename(
            image_name,
            model_name,
            strategy,
            thinking_budget=0,
            suffix="__json",
            ext="json",
        )

        parsed_json = extract_json_from_text(response_text)

        with open(PREDICTIONS_DIR / pred_filename, "w") as f:
            json.dump(parsed_json, f, indent=4, ensure_ascii=False)

        with open(METADATA_DIR / meta_filename, "w") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        print(f"  ✓ Saved: {pred_filename}")

    except Exception as e:
        print(f"  ✗ Error during {strategy} on {image_name}: {e}")


args = parse_args()

device = get_device()
torch_dtype = get_torch_dtype(device)

print("Loading model...")
print(f"Using device: {device}")
print(f"Using dtype: {torch_dtype}")

processor = AutoProcessor.from_pretrained(args.model)
model = AutoModelForImageTextToText.from_pretrained(
    args.model,
    torch_dtype=torch_dtype,
)
model = model.to(device)
model.eval()
print("Model loaded successfully!\n")

combinations = list(itertools.combinations(range(len(IMAGE_FILES)), 3))

prompt_text = load_prompt()

print("\n" + "=" * 70)
print(f"Model: {args.model}")
if args.model == DEFAULT_MODEL:
    print(f"Using default small Gemma-family model: {DEFAULT_MODEL}")
print("=" * 70)

for combination in combinations:
    example_indices = combination
    target_index = list(set(range(len(IMAGE_FILES))) - set(example_indices))[0]
    target_image = IMAGE_FILES[target_index]
    example_images = [IMAGE_FILES[idx] for idx in example_indices]

    print(f"\nTarget: {target_image}")
    print(f"Examples: {', '.join(example_images)}")

    run_experiment(
        processor,
        model,
        args.model,
        device,
        prompt_text,
        target_image,
        "fewshot",
        example_images,
        args.max_new_tokens,
        args.temperature,
    )

    run_experiment(
        processor,
        model,
        args.model,
        device,
        prompt_text,
        target_image,
        "zeroshot",
        [],
        args.max_new_tokens,
        args.temperature,
    )

print("\n" + "=" * 70)
print("Experiments complete!")
print(f"Predictions saved to: {PREDICTIONS_DIR}")
print(f"Metadata saved to: {METADATA_DIR}")
print("=" * 70)
