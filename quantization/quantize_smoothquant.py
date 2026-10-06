"""
SmoothQuant W8A8 quantization via llm-compressor.

SmoothQuant applies activation-aware smoothing, then quantizes weights and activations
to INT8 (W8A8). Uses llmcompressor's oneshot API with SmoothQuantModifier.

Usage:
    python quantization/quantize_smoothquant.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --output ./results/qwen0.5b-instruct-smoothquant-w8a8 \
        --alpha 0.5 \
        --num-calibration-samples 512
"""
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def quantize_smoothquant(
    model_name: str,
    output_dir: str,
    alpha: float = 0.5,
    num_calibration_samples: int = 512,
):
    """Apply SmoothQuant W8A8 using llmcompressor oneshot."""
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create recipe: SmoothQuant + W8A8 quantization
    # SmoothQuant migrates difficulty from activations to weights (alpha=0.5 is balanced)
    smoothquant_recipe = SmoothQuantModifier(smoothing_strength=alpha)
    quantization_recipe = QuantizationModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=["lm_head"],
    )

    print(f"Applying SmoothQuant (alpha={alpha}) + W8A8 quantization...")
    print(f"Calibration: wikitext-2, {num_calibration_samples} samples")

    # Apply both modifiers via oneshot
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=[smoothquant_recipe, quantization_recipe],  # Apply in sequence
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        splits="train",
        num_calibration_samples=num_calibration_samples,
        max_seq_length=512,
        output_dir=output_dir,
    )

    tokenizer.save_pretrained(output_dir)
    print(f"✓ SmoothQuant W8A8 model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmoothQuant W8A8 quantization")
    parser.add_argument("--model", required=True, help="Model ID on HF Hub")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="SmoothQuant migration factor (0=activations, 1=weights)",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=512,
        help="Number of wikitext2 calibration samples",
    )
    args = parser.parse_args()

    quantize_smoothquant(
        args.model, args.output, args.alpha, args.num_calibration_samples
    )

    