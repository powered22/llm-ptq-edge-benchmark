"""
quantize_awq_v2.py — AWQ (Activation-aware Weight Quantization) dengan calibration data.

Menggunakan AWQModifier dari llm-compressor: bobot di-skala per-channel berdasarkan
statistik aktivasi yang dikumpulkan dari calibration set, lalu dikuantisasi ke W4A16.

python quantization/quantize_awq_v2.py \
  --model Qwen/Qwen2.5-1.5B \
  --output ./out/qwen2.5-1.5b-awq-w4a16 \
  --num-calibration-samples 512 \
  --max-seq-length 512
  
"""
import argparse
import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_calibration_dataset(tokenizer, num_samples: int, max_seq_length: int):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds = ds.filter(lambda x: len(x["text"].strip()) > 0)
    ds = ds.shuffle(seed=42).select(range(num_samples))

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

    return ds.map(tokenize, remove_columns=ds.column_names)


def quantize_awq(
    model_name: str,
    output_dir: str,
    num_calibration_samples: int = 512,
    max_seq_length: int = 512,
):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Preparing {num_calibration_samples} calibration samples from wikitext-2...")
    calib_ds = build_calibration_dataset(tokenizer, num_calibration_samples, max_seq_length)

    recipe = AWQModifier(
        targets=["Linear"],
        scheme="W4A16",
        ignore=["lm_head"],
    )

    print("Running AWQ calibration + quantization (W4A16 symmetric)...")
    oneshot(
        model=model,
        tokenizer=tokenizer,    
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
        output_dir=output_dir,
    )

    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-calibration-samples", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args()
    quantize_awq(
        args.model,
        args.output,
        args.num_calibration_samples,
        args.max_seq_length,
    )