"""Apply SmoothQuant W8A8 quantization using llm-compressor.

SmoothQuant migrates quantization difficulty from activations to weights
by scaling them with a per-channel factor s, controlled by alpha.
Uses llm-compressor's SmoothQuantModifier + QuantizationModifier (W8A8 scheme)
in compressed-tensors format for consistency with Table 1 AWQ/GPTQ/RTN rows.

Reference: https://github.com/mit-han-lab/smoothquant
Paper: Xiao et al. "SmoothQuant: Accurate and Efficient Post-Training
       Quantization for Large Language Models" (ICML 2023)

Usage:
    python quantization/quantize_smoothquant.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --output ./results/qwen0.5b-instruct-smoothquant-w8a8 \
        --alpha 0.5 \
        --num_calib_samples 512
"""
import argparse
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, SparseAutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def get_calibration_data(tokenizer, n_samples: int = 512, seq_len: int = 512):
    """Load wikitext2 calibration data for SmoothQuant."""

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t["text"] for t in data if len(t["text"].strip()) > 50][:n_samples]

    def _loader():
        for text in tqdm(texts, desc="Preparing calibration data"):
            yield tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=seq_len)
    return _loader()

def quantize_smoothquant(model_name: str, output_dir: str, alpha: float = 0.5,
                        num_calib_samples: int = 512):
    """Apply SmoothQuant W8A8 using llm-compressor recipe."""
    print(f"Loading {model_name}...")
    tokenizer = SparseAutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )

    print(f"Preparing calibration data ({num_calib_samples} samples)...")
    calib_data = get_calibration_data(tokenizer, num_calib_samples)

    # llm-compressor recipe: SmoothQuant + W8A8 INT8 quantization
    print(f"Applying SmoothQuant (alpha={alpha}) + W8A8 INT8 quantization...")

    smoothquant_modifier = SmoothQuantModifier(alpha=alpha)
    quantization_modifier = QuantizationModifier(
        scheme="w8a8",
        ignore=["lm_head"],  # Don't quantize output layer
    )

    # Apply modifiers in sequence
    smoothquant_modifier.apply(model)
    quantization_modifier.apply(model, calibration_data=calib_data)

    print(f"Saving quantized model to {output_dir}...")
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ SmoothQuant W8A8 model saved to {output_dir}")
    print(f"  Format: compressed-tensors (compatible with Table 1 evaluation pipeline)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmoothQuant W8A8 quantization")
    parser.add_argument("--model", required=True, help="Model ID on HF Hub")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SmoothQuant migration factor (0=activations, 1=weights)")
    parser.add_argument("--num_calib_samples", type=int, default=512,
                        help="Number of wikitext2 calibration samples")
    args = parser.parse_args()

    quantize_smoothquant(args.model, args.output, args.alpha, args.num_calib_samples)


