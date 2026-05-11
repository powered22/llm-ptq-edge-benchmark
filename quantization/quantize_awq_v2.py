"""
quantize_awq_v2.py — AWQ dengan calibration data yang benar
"""
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def quantize_awq(model_name: str, output_dir: str, num_calibration_samples: int = 512):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Siapkan calibration data (wikitext2 — standar untuk PTQ paper)
    print("Preparing calibration data...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    samples = [ 
        tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=512)
        for row in ds
        if len(row["text"].strip()) > 50
    ][:num_calibration_samples]

    # W4A16 dengan calibration = AWQ sejati
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    )

    print(f"Quantizing with W4A16 + {num_calibration_samples} calibration samples...")
    oneshot(
        model=model,
        recipe=recipe,
        dataset=samples,           # <-- ini yang membuat AWQ-style sesungguhnya
        output_dir=output_dir,
        num_calibration_samples=num_calibration_samples,
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