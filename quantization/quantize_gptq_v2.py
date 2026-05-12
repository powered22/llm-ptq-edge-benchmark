"""
GPTQ W4A16 quantization via llm-compressor.
Calibration data (wikitext-2) benar-benar digunakan via GPTQModifier.

Usage:
    python quantization/quantize_gptq_v2.py \
        --model Qwen/Qwen2.5-1.5B \
        --output ./results/qwen2.5-1.5b-gptq-int4 \
        --num-samples 512
"""
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def quantize_gptq(model_name: str, output_dir: str, num_samples: int = 512):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # GPTQModifier — calibration data benar-benar digunakan
    # damping_frac: stabilisasi Hessian, 0.01 adalah nilai standar dari paper GPTQ
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
        damping_frac=0.01,
    )

    print(f"Quantizing with GPTQ W4A16 + {num_samples} calibration samples...")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        split="train",
        num_calibration_samples=num_samples,
        max_seq_length=512,
        output_dir=output_dir,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"✓ GPTQ model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=512)
    args = parser.parse_args()
    quantize_gptq(args.model, args.output, args.num_samples)