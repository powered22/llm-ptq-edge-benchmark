"""Load a model in full FP16 precision (baseline)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_fp16(model_name: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model, tokenizer = load_fp16(args.model, args.device)
    print(f"Loaded {args.model} in FP16. Params: {sum(p.numel() for p in model.parameters()):,}")
