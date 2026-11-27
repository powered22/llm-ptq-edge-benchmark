"""
RTN (Round-To-Nearest) W4A16 quantization via llm-compressor.
Baseline naïve: bobot langsung dibulatkan ke grid kuantisasi tanpa kalibrasi
maupun koreksi error (tidak seperti AWQ/GPTQ).

Usage:
   python quantization/quantize_rtn.py \
    --model Qwen/Qwen2.5-1.5B \
    --output ./results/qwen2.5-1.5b-rtn-w4a16

   # Custom group size (default: the preset's group size, 128 for W4A16):
   python quantization/quantize_rtn.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --output ./results/qwen2.5_0.5b_instruct_rtn_w4a16_g32 \
    --group-size 32
"""
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_recipe(scheme: str, group_size=None):
    """Preset scheme by default. With group_size, the same weight-only scheme
    (symmetric int, per-group scales, 16-bit activations) with a custom group size."""
    if group_size is None:
        return QuantizationModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])
    num_bits = {"W4A16": 4, "W8A16": 8}[scheme]
    return QuantizationModifier(
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": num_bits,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": group_size,
                },
            }
        },
        ignore=["lm_head"],
    )


def quantize_rtn(model_name: str, output_dir: str, scheme: str = "W4A16", group_size=None):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    recipe = build_recipe(scheme, group_size)

    gs = f", group size {group_size}" if group_size else ""
    print(f"Quantizing with RTN {scheme}{gs} (no calibration data)...")
    oneshot(
        model=model,
        recipe=recipe,
        output_dir=output_dir,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"✓ RTN model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scheme", default="W4A16", choices=["W4A16", "W8A16"])
    parser.add_argument("--group-size", type=int, default=None,
                        help="Weights per scale. Default: the preset's group size (128).")
    args = parser.parse_args()
    quantize_rtn(args.model, args.output, args.scheme, args.group_size)
