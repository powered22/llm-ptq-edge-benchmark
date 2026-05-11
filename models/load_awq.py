"""Load a pre-quantized AWQ model."""
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def load_awq(model_path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_quantized(
        model_path, fuse_layers=True, trust_remote_code=True
    )
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    model, tok = load_awq(args.model_path)
    print(f"AWQ model loaded from {args.model_path}")
