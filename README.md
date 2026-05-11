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

- NVIDIA Jetson Orin / AGX Xavier (edge)
- NVIDIA RTX/A-series GPU (server baseline)



