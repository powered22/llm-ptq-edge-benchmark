"""Custom lm-eval-harness backend pakai llama-cpp-python langsung.

Bypass HTTP layer (yang format-nya selalu mismatch antara lm-eval+llama-server).
Implementasi 3 method LM interface: loglikelihood, loglikelihood_rolling,
generate_until. Dengan ini SEMUA lm-eval task bisa dipakai untuk GGUF models.

Cara pakai (lewat wrapper scripts/run_lm_eval_gguf.py):
    python scripts/run_lm_eval_gguf.py \\
        --model llama-cpp-direct \\
        --model_args pretrained=/path/to/model.gguf,n_ctx=2048,n_gpu_layers=99 \\
        --tasks arc_easy --output_path out.json
"""
from typing import List, Tuple
from llama_cpp import Llama
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm


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
            logits_all=True,    # WAJIB untuk per-token logprob
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

    # -- Inti: 3 method yang lm-eval butuh --
    def _loglikelihood_one(self, context: str, continuation: str) -> Tuple[float, bool]:
        """log P(continuation | context) plus is_greedy flag."""
        # Tokenize untuk cari boundary antara context dan continuation
        ctx_tokens = self.llm.tokenize(context.encode(), add_bos=True)
        n_ctx_tokens = len(ctx_tokens)

        full_text = context + continuation
        result = self.llm.create_completion(
            full_text,
            max_tokens=0,
            echo=True,
            logprobs=1,
            temperature=0.0,
        )
        all_logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
        # token_logprobs[i] = log P(token_i | token_<i). First adalah None.
        cont_logprobs = all_logprobs[n_ctx_tokens:]
        cont_logprobs = [lp for lp in cont_logprobs if lp is not None]
        log_prob = float(sum(cont_logprobs)) if cont_logprobs else 0.0
        # is_greedy: conservatif, kita selalu return False (lm-eval tidak strict pakai ini)
        return log_prob, False

    def loglikelihood(self, requests):
        results = []
        for req in tqdm(requests, desc="loglikelihood", leave=False):
            context, continuation = req.args
            results.append(self._loglikelihood_one(context, continuation))
        return results

    def loglikelihood_rolling(self, requests):
        """Untuk perplexity tasks (wikitext, dst)."""
        results = []
        for req in tqdm(requests, desc="loglikelihood_rolling", leave=False):
            (text,) = req.args
            result = self.llm.create_completion(
                text, max_tokens=0, echo=True, logprobs=1, temperature=0.0,
            )
            all_logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
            valid = [lp for lp in all_logprobs if lp is not None]
            results.append((float(sum(valid)),))
        return results

    def generate_until(self, requests):
        """Untuk generative tasks (gsm8k, ifeval, dst)."""
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
