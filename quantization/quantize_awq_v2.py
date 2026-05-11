"""
quantize_awq_v2.py — AWQ dengan calibration data (API llm-compressor yang benar)
"""
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Suppress tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def quantize_awq(model_name: str, output_dir: str, num_calibration_samples: int = 512):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    )

    print(f"Quantizing with W4A16 + {num_calibration_samples} calibration samples...")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset="wikitext",                      # <-- string nama dataset
        dataset_config_name="wikitext-2-raw-v1", # <-- config dataset
        split="train",
        num_calibration_samples=num_calibration_samples,
        max_seq_length=512,
        output_dir=output_dir,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-calibration-samples", type=int, default=512)
    args = parser.parse_args()
    quantize_awq(args.model, args.output, args.num_calibration_samples)