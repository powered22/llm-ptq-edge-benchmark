"""Load a pre-quantized GPTQ model."""
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer


def load_gptq(model_path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_path, device=device, trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model, tok = load_gptq(args.model_path, args.device)
    print(f"GPTQ model loaded from {args.model_path}")
