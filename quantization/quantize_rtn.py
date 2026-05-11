"""
RTN (Round-To-Nearest) W4A16 quantization via llm-compressor.
Baseline naïve: bobot langsung dibulatkan ke grid kuantisasi tanpa kalibrasi
maupun koreksi error (tidak seperti AWQ/GPTQ).

Usage:
    python quantization/quantize_rtn.py \
        --model Qwen/Qwen2.5-1.5B \
        --output ./results/qwen2.5-1.5b-rtn-int4
"""
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def quantize_rtn(model_name: str, output_dir: str):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # QuantizationModifier tanpa AWQ/GPTQ = RTN murni (data-free)
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    )

    print("Quantizing with RTN W4A16 (no calibration data)...")
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
    args = parser.parse_args()
    quantize_rtn(args.model, args.output)
