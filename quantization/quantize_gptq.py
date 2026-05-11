"""Quantize a HuggingFace model using GPTQ.

Usage:
    python quantization/quantize_gptq.py \
        --model Qwen/Qwen2.5-1.5B \
        --output ./results/qwen1.5b-gptq \
        --bits 4
"""
import argparse
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset


def quantize_gptq(model_name: str, output_dir: str, bits: int = 4, group_size: int = 128):
    print(f"Loading {model_name} for GPTQ quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,   # True improves accuracy but slows inference
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        model_name, quantize_config=quantize_config, trust_remote_code=True
    )

    # Calibration: 128 samples from WikiText-2 train split
    print("Preparing calibration data (wikitext2)...")
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:256]")
    examples = [
        tokenizer(t["text"], return_tensors="pt", truncation=True, max_length=512)
        for t in data
        if len(t["text"].strip()) > 50
    ][:128]

    print(f"Quantizing with {bits}-bit, group_size={group_size}...")
    model.quantize(examples)
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ GPTQ model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPTQ quantization")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    quantize_gptq(args.model, args.output, args.bits, args.group_size)
