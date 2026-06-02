"""Generate datafiles untuk llama-perplexity tasks (HellaSwag + Winogrande).

Output:
    ./data/hellaswag_val_full.txt       (format llama.cpp --hellaswag)
    ./data/winogrande-debiased-eval.csv (format llama.cpp --winogrande)

Usage: python evaluation/prepare_llama_cpp_eval_data.py
"""
import argparse
from pathlib import Path


def prepare_hellaswag(out_path: Path):
    """Format HellaSwag untuk llama-perplexity --hellaswag flag.

    Setiap line:
        <context>\t<gold_label>\t<ending0>\t<ending1>\t<ending2>\t<ending3>

    Replace tab/newline di field dengan spasi untuk keamanan parsing.
    """
    from datasets import load_dataset
    print("Loading HellaSwag validation set...")
    ds = load_dataset("Rowan/hellaswag", split="validation")
    print(f"  Got {len(ds)} tasks")

    n_skip = 0
    with open(out_path, "w") as f:
        for ex in ds:
            label = ex.get("label", "")
            if label == "" or label is None:
                n_skip += 1
                continue
            ctx = (ex["ctx_a"] + " " + ex["ctx_b"]).replace("\t", " ").replace("\n", " ").strip()
            endings = [e.replace("\t", " ").replace("\n", " ").strip() for e in ex["endings"]]
            f.write(f"{ctx}\t{label}\t" + "\t".join(endings) + "\n")
    print(f"  Saved to: {out_path} (skipped {n_skip} tasks tanpa label)")


def prepare_winogrande(out_path: Path):
    """Format Winogrande untuk llama-perplexity --winogrande flag.

    CSV-style format:
        <sentence>,<option1>,<option2>,<answer_1_or_2>

    Note: HF dataset menggunakan answer = "1" atau "2" (string).
    """
    from datasets import load_dataset
    import csv
    print("Loading Winogrande validation set...")
    ds = load_dataset("allenai/winogrande", "winogrande_xl",
                      split="validation", trust_remote_code=True)
    print(f"  Got {len(ds)} tasks")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        for ex in ds:
            sentence = ex["sentence"]
            opt1 = ex["option1"]
            opt2 = ex["option2"]
            answer = ex["answer"]  # "1" atau "2"
            writer.writerow([sentence, opt1, opt2, answer])
    print(f"  Saved to: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="./data")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    hellaswag_path = data_dir / "hellaswag_val_full.txt"
    winogrande_path = data_dir / "winogrande-debiased-eval.csv"

    if not hellaswag_path.exists():
        prepare_hellaswag(hellaswag_path)
    else:
        print(f"[skip] {hellaswag_path} sudah ada")

    if not winogrande_path.exists():
        prepare_winogrande(winogrande_path)
    else:
        print(f"[skip] {winogrande_path} sudah ada")

    print()
    print("Selesai. Sekarang lanjut jalankan:")
    print("  bash scripts/eval/eval_qwen0.5b_gguf_tasks.sh")


if __name__ == "__main__":
    main()
