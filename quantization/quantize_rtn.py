"""
RTN (Round-To-Nearest) W4A16 quantization via llm-compressor.
Baseline naïve: bobot langsung dibulatkan ke grid kuantisasi tanpa kalibrasi
maupun koreksi error (tidak seperti AWQ/GPTQ).

Usage:
   python quantization/quantize_rtn.py \
    --model Qwen/Qwen2.5-1.5B \
    --output ./results/qwen2.5-1.5b-rtn-w4a16
"""
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def quantize_rtn(model_name: str, output_dir: str, scheme: str = "W4A16"):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=["lm_head"],
    )

    print(f"Quantizing with RTN {scheme} (no calibration data)...")
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
    args = parser.parse_args()
    quantize_rtn(args.model, args.output, args.scheme)
