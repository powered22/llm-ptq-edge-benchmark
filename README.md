# LLM PTQ Edge Benchmark

> Comparative Analysis of Post-Training Quantization Methods for LLM Inference on Edge Devices

A reproducible benchmarking framework for evaluating AWQ, GPTQ, SmoothQuant, and BitsAndBytes (INT8/INT4) quantization methods on edge hardware (NVIDIA Jetson) and GPU servers.

## Structure

```
llm-ptq-edge-benchmark/
├── models/           # Model loading utilities
├── quantization/     # Quantization scripts
├── evaluation/       # Accuracy benchmarks
├── benchmark/        # Latency & memory profiling
├── results/          # Output CSV/JSON from experiments
└── notebooks/        # Analysis & visualization
```

## Quickstart

```bash
pip install -r requirements.txt

# Load and quantize a model
python quantization/quantize_awq.py --model Qwen/Qwen2.5-1.5B --output ./results/awq

# Benchmark latency on GPU
python benchmark/benchmark_gpu.py --model-path ./results/awq --method awq

# Run perplexity evaluation
python evaluation/run_perplexity.py --model-path ./results/awq --dataset wikitext2
```

## Supported Methods

| Method | Bits | Backend |
|--------|------|---------|
| AWQ | INT4 | AutoAWQ |
| GPTQ | INT4/INT8 | AutoGPTQ |
| SmoothQuant | INT8 | torch |
| BitsAndBytes | INT8/INT4 | bitsandbytes |

## Target Hardware

- NVIDIA Jetson Orin / AGX Xavier (edge, GPU stack)
- NVIDIA RTX/A-series GPU (server baseline)
- CPU consumer x86 (Windows/Linux) — via GGUF + llama.cpp
- Raspberry Pi 4/5 (ARM64) — via GGUF + llama.cpp

## GGUF / llama.cpp path (CPU & ARM)

Untuk target CPU dan Raspberry Pi, kami konversi model ke **GGUF** dan jalankan
via [llama.cpp](https://github.com/ggml-org/llama.cpp). Format yang sama juga
bisa dipakai di Jetson (build llama.cpp dengan `-DGGML_CUDA=ON`).

### Setup (sekali)

```bash
# 1. Clone llama.cpp ke external/
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git external/llama.cpp

# 2. Install Python deps untuk konversi
pip install -r external/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# 3. Build llama.cpp (pilih salah satu sesuai target)
cd external/llama.cpp
#   CPU (Linux/Windows x86)
cmake -B build -DGGML_NATIVE=ON
#   Raspberry Pi 4/5 (ARM64 + NEON)
# cmake -B build -DGGML_NATIVE=ON -DGGML_CPU_ARM_ARCH=native
#   Jetson Orin (ARM64 + CUDA)
# cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
cd ../..
```

### Konversi + kuantisasi

```bash
python quantization/convert_to_gguf.py \
  --model Qwen/Qwen2.5-1.5B \
  --output-name qwen2.5_1.5b \
  --schemes Q4_K_M Q5_K_M Q8_0
```

Output: `results/qwen2.5_1.5b_gguf/qwen2.5_1.5b-{Q4_K_M,Q5_K_M,Q8_0}.gguf`.



