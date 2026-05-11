"""
Quantize model using AWQ (W4A16) via llm-compressor (vLLM Project).
Replaces deprecated AutoAWQ.

Usage:
    python quantization/quantize_awq_v2.py \
        --model Qwen/Qwen2.5-1.5B \
        --output ./results/qwen2.5-1.5b-awq-int4
"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def quantize_awq(model_name: str, output_dir: str):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # W4A16 = AWQ-style: weight INT4, activation tetap FP16
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],  # lm_head dibiarkan FP16 untuk menjaga akurasi output
    )

    print("Quantizing with W4A16 (AWQ-style)...")
    oneshot(model=model, recipe=recipe, output_dir=output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    quantize_awq(args.model, args.output)