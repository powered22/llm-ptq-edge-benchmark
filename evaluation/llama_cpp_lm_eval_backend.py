"""Custom lm-eval-harness backend pakai llama-cpp-python langsung.

Bypass HTTP layer (yang format-nya selalu mismatch antara lm-eval+llama-server).
Implementasi 3 method LM interface: loglikelihood, loglikelihood_rolling,
generate_until. Dengan ini SEMUA lm-eval task bisa dipakai untuk GGUF models.

PENTING — jangan kembalikan create_completion(echo=True, logprobs=1) untuk
loglikelihood. Jalur itu mengonversi matriks logits (n_tokens x 151936 untuk
vocab Qwen2.5) menjadi objek Python float; satu request saja membengkak ke
orde 10 GB dan proses dibunuh OOM killer. Versi ini membaca llm.scores sebagai
array numpy dan menghitung log-softmax sepenuhnya di numpy, per potongan baris,
sehingga puncak memorinya ratusan MB — bukan belasan GB.

Cara pakai (lewat wrapper scripts/run_lm_eval_gguf.py):
    python scripts/run_lm_eval_gguf.py \\
        --model llama-cpp-direct \\
        --model_args pretrained=/path/to/model.gguf,n_ctx=2048,n_gpu_layers=99 \\
        --tasks arc_easy --output_path out.json
"""
from typing import Tuple

import numpy as np
from llama_cpp import Llama
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

# Berapa baris logits diproses sekaligus. 64 x 151936 x 8 byte ~ 78 MB.
_ROW_CHUNK = 64


@register_model("llama-cpp-direct")
class LlamaCppDirectLM(LM):
    def __init__(self, pretrained=None, n_ctx=2048, n_gpu_layers=-1,
                 batch_size=1, n_batch=512, **kwargs):
        super().__init__()
        assert pretrained, "Pass pretrained=<path/to/model.gguf>"
        self.llm = Llama(
            model_path=str(pretrained),
            n_ctx=int(n_ctx),
            n_gpu_layers=int(n_gpu_layers),
            n_batch=int(n_batch),
            logits_all=True,    # WAJIB: butuh logits di SEMUA posisi, bukan hanya terakhir
            verbose=False,
        )
        self._batch_size = int(batch_size) if batch_size else 1
        self._max_length = int(n_ctx)

    # -- properties yang lm-eval expect --
    @property
    def eot_token_id(self):
        return self.llm.token_eos()

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string):
        return self.llm.tokenize(string.encode(), add_bos=False)

    def tok_decode(self, tokens):
        return self.llm.detokenize(tokens).decode("utf-8", errors="ignore")

    # -- helper --
    def _raw_scores(self, n_tokens):
        """Matriks logits sebagai numpy view. Nama atribut beda antar versi."""
        raw = getattr(self.llm, "scores", None)
        if raw is None:
            raw = self.llm._scores
        return np.asarray(raw)[:n_tokens, :]

    def _forward(self, tokens):
        """Satu forward pass. Truncate dari KIRI kalau melebihi n_ctx."""
        if len(tokens) > self._max_length:
            tokens = tokens[-self._max_length:]
        self.llm.reset()
        self.llm.eval(tokens)
        return tokens, self._raw_scores(len(tokens))

    @staticmethod
    def _sum_logprobs(rows, targets) -> Tuple[float, bool]:
        """Sum log-softmax(rows)[targets], diproses per _ROW_CHUNK baris.

        rows[i] adalah logits yang memprediksi targets[i].
        """
        total = 0.0
        greedy = True
        for s in range(0, len(targets), _ROW_CHUNK):
            r = np.asarray(rows[s:s + _ROW_CHUNK], dtype=np.float64)
            t = targets[s:s + _ROW_CHUNK]
            m = r.max(axis=1, keepdims=True)
            log_z = m[:, 0] + np.log(np.exp(r - m).sum(axis=1))
            total += float((r[np.arange(len(t)), t] - log_z).sum())
            if greedy and not bool((r.argmax(axis=1) == t).all()):
                greedy = False
        return total, greedy

    # -- Inti: 3 method yang lm-eval butuh --
    def _loglikelihood_one(self, context: str, continuation: str) -> Tuple[float, bool]:
        """log P(continuation | context) plus is_greedy flag."""
        ctx_tokens = self.llm.tokenize(context.encode(), add_bos=True)
        cont_tokens = self.llm.tokenize(continuation.encode(), add_bos=False)
        if not cont_tokens:
            return 0.0, True

        tokens, scores = self._forward(ctx_tokens + cont_tokens)

        # Dihitung dari BELAKANG supaya tetap benar kalau tadi ter-truncate.
        n_cont = min(len(cont_tokens), len(tokens) - 1)
        if n_cont <= 0:
            return 0.0, True

        # Baris i memprediksi token i+1.
        rows = scores[len(tokens) - n_cont - 1: len(tokens) - 1, :]
        targets = np.asarray(tokens[len(tokens) - n_cont:], dtype=np.int64)
        return self._sum_logprobs(rows, targets)

    def loglikelihood(self, requests):
        results = []
        for req in tqdm(requests, desc="loglikelihood", leave=False):
            context, continuation = req.args
            results.append(self._loglikelihood_one(context, continuation))
        return results

    def loglikelihood_rolling(self, requests):
        """Untuk perplexity tasks (wikitext, dst). Sliding window selebar n_ctx."""
        results = []
        for req in tqdm(requests, desc="loglikelihood_rolling", leave=False):
            (text,) = req.args
            tokens = self.llm.tokenize(text.encode(), add_bos=True)
            total = 0.0
            start = 0
            win = self._max_length
            while start < len(tokens) - 1:
                chunk = tokens[start:start + win]
                if len(chunk) < 2:
                    break
                self.llm.reset()
                self.llm.eval(chunk)
                scores = self._raw_scores(len(chunk))
                targets = np.asarray(chunk[1:], dtype=np.int64)
                part, _ = self._sum_logprobs(scores[:len(chunk) - 1, :], targets)
                total += part
                start += win - 1        # overlap 1 token, tanpa double-count
            results.append((total,))
        return results

    def generate_until(self, requests):
        """Untuk generative tasks (gsm8k, ifeval, dst).

        Aman dari ledakan memori karena TIDAK meminta logprobs.
        """
        results = []
        for req in tqdm(requests, desc="generate_until", leave=False):
            context, gen_kwargs = req.args
            stops = []
            max_gen = self.max_gen_toks
            if isinstance(gen_kwargs, dict):
                stops = gen_kwargs.get("until", []) or []
                max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            result = self.llm.create_completion(
                context,
                max_tokens=int(max_gen),
                stop=stops if stops else None,
                temperature=0.0,
            )
            results.append(result["choices"][0]["text"])
        return results
