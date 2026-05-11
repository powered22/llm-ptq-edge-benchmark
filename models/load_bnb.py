"""Load a model quantized via BitsAndBytes (INT8 or INT4)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_bnb(model_name: str, bits: int = 4, device: str = "cuda"):
    assert bits in (4, 8), "bits must be 4 or 8"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    args = parser.parse_args()
    model, tok = load_bnb(args.model, bits=args.bits)
    print(f"BnB INT{args.bits} model loaded: {args.model}")
