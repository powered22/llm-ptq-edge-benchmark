"""Wrapper: register custom llama-cpp-direct backend, lalu jalankan lm-eval CLI.

Cara pakai: sama persis dengan `python -m lm_eval`, tapi support
            `--model llama-cpp-direct` untuk evaluasi GGUF langsung
            via llama-cpp-python (tanpa HTTP).
"""
import sys
from pathlib import Path

# Pastikan repo root ada di sys.path supaya import evaluation.* berhasil
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import backend kustom — eksekusi @register_model decorator
import evaluation.llama_cpp_lm_eval_backend  # noqa: F401

# Sekarang panggil lm-eval CLI seperti biasa
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    sys.exit(cli_evaluate())
