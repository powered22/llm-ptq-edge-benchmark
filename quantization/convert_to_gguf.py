"""
convert_to_gguf.py — konversi HuggingFace model ke GGUF (llama.cpp) lalu kuantisasi
ke beberapa skema (Q4_K_M, Q5_K_M, Q8_0, ...).

Alur:
  1. Resolve model ke direktori lokal (download via huggingface_hub kalau perlu).
  2. Konversi ke base GGUF (default outtype=f16) memakai
     external/llama.cpp/convert_hf_to_gguf.py.
  3. Untuk tiap skema, jalankan binary `llama-quantize` untuk hasilkan
     <name>-<SCHEME>.gguf.

Contoh:
  python quantization/convert_to_gguf.py \
    --model Qwen/Qwen2.5-1.5B \
    --output-name qwen2.5_1.5b \
    --schemes Q4_K_M Q5_K_M Q8_0

Prasyarat:
  - external/llama.cpp sudah di-clone (lihat README).
  - `llama-quantize` sudah dibangun. Default lookup:
      external/llama.cpp/build/bin/llama-quantize
    Override via --llama-quantize /path/ke/llama-quantize.
  - Python deps untuk konversi: torch, transformers, sentencepiece, gguf, numpy,
    protobuf. Bisa di-install via:
      pip install -r external/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LLAMA_CPP_DIR = REPO_ROOT / "external" / "llama.cpp"
DEFAULT_OUT_DIR = REPO_ROOT / "results"

# Skema yang umumnya tersedia di llama-quantize. Daftar ini hanya untuk
# validasi argumen lebih awal; nilai aktual diteruskan apa adanya ke
# llama-quantize, jadi skema lain (mis. IQ3_S, Q6_K) tetap bisa dipakai
# dengan --schemes.
KNOWN_SCHEMES = {
    "F32", "F16", "BF16",
    "Q8_0", "Q6_K",
    "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1",
    "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1",
    "Q3_K_L", "Q3_K_M", "Q3_K_S",
    "Q2_K",
    "IQ4_XS", "IQ4_NL", "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S",
}


def resolve_model_dir(model: str) -> Path:
    """Kalau `model` adalah path lokal yang ada, pakai langsung.
    Kalau bentuknya `org/name`, download via huggingface_hub.snapshot_download.
    """
    p = Path(model)
    if p.exists() and p.is_dir():
        return p.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub belum terinstall. Install: pip install huggingface_hub"
        ) from e

    print(f"[snapshot] downloading {model} dari HuggingFace Hub...")
    local = snapshot_download(
        repo_id=model,
        allow_patterns=[
            "*.json", "*.txt", "*.model", "*.tiktoken",
            "*.safetensors", "*.bin",
            "tokenizer.*", "vocab.*", "merges.*",
        ],
    )
    return Path(local).resolve()


def run(cmd, cwd=None):
    print(f"[run] {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def convert_to_base_gguf(
    model_dir: Path,
    out_path: Path,
    outtype: str,
    llama_cpp_dir: Path,
    python_bin: str,
):
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise SystemExit(f"convert_hf_to_gguf.py tidak ditemukan di {convert_script}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin, str(convert_script),
        str(model_dir),
        "--outfile", str(out_path),
        "--outtype", outtype,
    ]
    run(cmd)


def quantize(
    base_gguf: Path,
    out_path: Path,
    scheme: str,
    llama_quantize_bin: Path,
    threads: int,
):
    if not llama_quantize_bin.exists():
        raise SystemExit(
            f"llama-quantize binary tidak ditemukan: {llama_quantize_bin}\n"
            "Build llama.cpp dulu atau lewatkan --llama-quantize /path/ke/llama-quantize."
        )
    cmd = [
        str(llama_quantize_bin),
        str(base_gguf),
        str(out_path),
        scheme,
        str(threads),
    ]
    run(cmd)


def default_llama_quantize(llama_cpp_dir: Path) -> Path:
    candidates = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "build" / "llama-quantize",
        llama_cpp_dir / "llama-quantize",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="HF repo id (mis. Qwen/Qwen2.5-1.5B) atau path direktori lokal.")
    parser.add_argument("--output-name", required=True,
                        help="Base name untuk file output, mis. qwen2.5_1.5b.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR),
                        help="Direktori output (default: ./results). Sub-folder <output-name>_gguf dibuat di dalamnya.")
    parser.add_argument("--schemes", nargs="+", default=["Q4_K_M", "Q5_K_M", "Q8_0"],
                        help="Daftar skema kuantisasi llama.cpp (default: Q4_K_M Q5_K_M Q8_0).")
    parser.add_argument("--base-outtype", default="f16", choices=["f32", "f16", "bf16"],
                        help="Tipe base GGUF sebelum kuantisasi (default: f16).")
    parser.add_argument("--llama-cpp-dir", default=str(DEFAULT_LLAMA_CPP_DIR),
                        help="Path repo llama.cpp.")
    parser.add_argument("--llama-quantize", default=None,
                        help="Path ke binary llama-quantize. Default: auto-detect di build/bin/.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter untuk menjalankan convert_hf_to_gguf.py.")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="Jumlah thread untuk llama-quantize.")
    parser.add_argument("--keep-base", action="store_true",
                        help="Pertahankan file base (F16) setelah semua skema selesai.")
    parser.add_argument("--force", action="store_true",
                        help="Re-quantize meski file output sudah ada.")
    args = parser.parse_args()

    llama_cpp_dir = Path(args.llama_cpp_dir).resolve()
    llama_quantize_bin = Path(args.llama_quantize).resolve() if args.llama_quantize else default_llama_quantize(llama_cpp_dir)

    unknown = [s for s in args.schemes if s not in KNOWN_SCHEMES]
    if unknown:
        print(f"[warn] skema tidak dikenal di daftar internal (tetap diteruskan ke llama-quantize): {unknown}")

    out_dir = Path(args.output_dir).resolve() / f"{args.output_name}_gguf"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] llama.cpp dir        : {llama_cpp_dir}")
    print(f"[info] llama-quantize       : {llama_quantize_bin}")
    print(f"[info] output dir           : {out_dir}")
    print(f"[info] schemes              : {args.schemes}")
    print(f"[info] base outtype         : {args.base_outtype}")

    model_dir = resolve_model_dir(args.model)
    print(f"[info] resolved model dir   : {model_dir}")

    base_path = out_dir / f"{args.output_name}-{args.base_outtype.upper()}.gguf"
    if base_path.exists() and not args.force:
        print(f"[skip] base GGUF sudah ada: {base_path}")
    else:
        convert_to_base_gguf(
            model_dir=model_dir,
            out_path=base_path,
            outtype=args.base_outtype,
            llama_cpp_dir=llama_cpp_dir,
            python_bin=args.python,
        )

    failed = []
    for scheme in args.schemes:
        out_path = out_dir / f"{args.output_name}-{scheme}.gguf"
        if out_path.exists() and not args.force:
            print(f"[skip] {scheme} sudah ada: {out_path}")
            continue
        try:
            quantize(
                base_gguf=base_path,
                out_path=out_path,
                scheme=scheme,
                llama_quantize_bin=llama_quantize_bin,
                threads=args.threads,
            )
        except subprocess.CalledProcessError as e:
            print(f"[error] gagal kuantisasi {scheme}: {e}")
            failed.append(scheme)

    if not args.keep_base and base_path.exists():
        print(f"[cleanup] hapus base file: {base_path}")
        base_path.unlink()

    print("")
    print("=" * 50)
    print(f"  GGUF outputs di: {out_dir}")
    for f in sorted(out_dir.glob("*.gguf")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name:40s} {size_mb:8.1f} MB")
    if failed:
        print(f"  Skema gagal: {failed}")
        sys.exit(1)
    print("=" * 50)


if __name__ == "__main__":
    main()
