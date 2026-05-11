"""Quantize a HuggingFace model using AWQ (INT4).

Usage:
    python quantization/quantize_awq.py \
        --model Qwen/Qwen2.5-1.5B \
        --output ./results/qwen1.5b-awq
"""
import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def quantize_awq(model_name: str, output_dir: str, w_bit: int = 4, group_size: int = 128):
    print(f"Loading {model_name} for AWQ quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": w_bit,
        "version": "GEMM",
    }
    print(f"Quantizing with config: {quant_config}")
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ AWQ model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWQ quantization")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--w-bit", type=int, default=4, choices=[3, 4], help="Weight bits")
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    quantize_awq(args.model, args.output, args.w_bit, args.group_size)
