"""
Faithful streaming perplexity evaluation with VQBenchCache.

Matches `llama.cpp -ctk turbo*` semantics:
  - Tokens are processed in chunks of size `chunk`.
  - Within a chunk, the current K,V are used EXACTLY in attention.
  - Between chunks, past K,V is served from the compressed VQBenchCache (lossy),
    while the current chunk's K,V remain exact until the next call.

Key correctness property (verified by `test_streaming_ppl.py`):
    baseline PPL with an exact `DynamicCache` is chunk-size invariant
    (up to fp16 accumulation noise). A faithful streaming evaluator MUST have
    this property — if PPL drifts with chunk size, the loss math is wrong.

NOT the same as the monkey-patching path in `monkey_patch.py`:
    monkey-patching quantizes ALL token positions including the current chunk.
    Use THIS module for numbers comparable to llama.cpp PPL results.

Implementation note: we compute per-token CE manually rather than relying on
HF's internal `shift_logits / shift_labels` pass. The HF path silently drops
one token per chunk boundary (the first token of each non-initial chunk is
never scored because the shift skips it), which biases PPL in a chunk-size-
dependent way. The correct behavior is:
    for every token t_i (i > 0):
        nll_i = CE(logit_at_position_{i-1}, t_i)
where logit_at_position_{i-1} may come from the PREVIOUS chunk's forward pass.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from vqbench.torch_wrapper.hook import VQBenchCache, make_vqbench_cache


@torch.no_grad()
def evaluate_streaming_ppl(
    model,
    encodings: torch.Tensor,
    device: str | torch.device = "cpu",
    chunk: int = 512,
    cache: VQBenchCache | None = None,
) -> tuple[float, int]:
    """
    Streaming perplexity over a token sequence with (optional) compressed KV cache.

    Processes tokens in chunks of size `chunk` while maintaining a shared cache
    across chunks. Between chunks, past K,V are served from `cache` (compressed),
    while the current chunk's K,V are used exactly in attention.

    Loss computation is manual (not HF's `labels=` shortcut) to avoid the
    chunk-boundary token drop caused by the internal shift.

    Args:
        model: HuggingFace CausalLM.
        encodings: (1, seq_len) token ids.
        device: compute device.
        chunk: tokens per forward-pass step.
        cache: VQBenchCache instance, or None for fp16 DynamicCache baseline.

    Returns:
        (perplexity, n_tokens_scored)
        where n_tokens_scored = seq_len - 1 (every token except the first has
        context to be predicted from).
    """
    seq_len = encodings.size(1)
    total_nll = 0.0
    total_tokens = 0

    # For baseline (cache=None): let the model create its own cache on the
    # first forward call. This is critical for hybrid-attention models like
    # Qwen3.5 which need a cache with both full_attention and linear_attention
    # layers — a bare DynamicCache() would crash on those models.
    # For quantized (cache=VQBenchCache): use the provided cache directly.
    past = cache  # None on first call → model creates the right cache type

    # The last logit of the previous chunk predicts the first token of the
    # current chunk. We keep it on CPU so it doesn't grow device memory.
    prev_last_logit: torch.Tensor | None = None

    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        input_ids = encodings[:, start:end].to(device)
        chunk_len = end - start

        # NOTE: no labels= — we handle loss manually to avoid HF's shift bug
        out = model(
            input_ids,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        logits = out.logits  # (1, chunk_len, V)

        # --------------------------------------------------------------
        # Boundary prediction: score the FIRST token of this chunk using
        # the LAST logit of the previous chunk. Skipped for the very first
        # chunk (no previous context).
        # --------------------------------------------------------------
        if prev_last_logit is not None:
            # prev_last_logit: (1, V) on `device`
            target = input_ids[:, 0]  # (1,) — the token we're predicting
            nll_boundary = F.cross_entropy(
                prev_last_logit, target, reduction="sum"
            )
            total_nll += float(nll_boundary.item())
            total_tokens += 1

        # --------------------------------------------------------------
        # Within-chunk predictions: for i in [1, chunk_len), predict
        # input_ids[i] from logits[i-1]. This gives (chunk_len - 1)
        # scored positions per chunk.
        # --------------------------------------------------------------
        if chunk_len > 1:
            within_logits = logits[:, :-1, :]   # (1, chunk_len-1, V)
            within_targets = input_ids[:, 1:]    # (1, chunk_len-1)
            nll_within = F.cross_entropy(
                within_logits.reshape(-1, within_logits.shape[-1]),
                within_targets.reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll_within.item())
            total_tokens += chunk_len - 1

        # Save the LAST logit of this chunk for the next chunk's boundary
        prev_last_logit = logits[:, -1, :].detach()  # (1, V)

        if end == seq_len:
            break

    if total_tokens == 0:
        return float("inf"), 0
    ppl = math.exp(total_nll / total_tokens)
    return ppl, total_tokens


def evaluate_model_configs(
    model,
    tokenizer,
    encodings: torch.Tensor,
    device: str | torch.device,
    configs: list[dict],
    chunk: int = 512,
) -> list[dict]:
    """
    Run streaming PPL for fp16 baseline + a list of quantization configs.

    Each config is a dict with keys:
      name, method_key, method_value, num_bits_key, num_bits_value

    Returns a list of result dicts with {name, ppl, delta_ppl, n_tokens}.
    """
    results = []

    # Baseline: exact DynamicCache (no compression)
    ppl0, n_tok = evaluate_streaming_ppl(
        model, encodings, device=device, chunk=chunk, cache=None,
    )
    results.append({
        "name": "fp16 baseline",
        "method_key": "fp16", "method_value": "fp16",
        "num_bits_key": 16, "num_bits_value": 16,
        "ppl": ppl0, "delta_ppl": 0.0, "n_tokens": n_tok,
    })

    for cfg in configs:
        cache = make_vqbench_cache(
            model.config,
            method_key=cfg["method_key"],
            method_value=cfg["method_value"],
            num_bits_key=cfg["num_bits_key"],
            num_bits_value=cfg["num_bits_value"],
            seed=cfg.get("seed", 42),
        )
        ppl, n_tok = evaluate_streaming_ppl(
            model, encodings, device=device, chunk=chunk, cache=cache,
        )
        results.append({
            **cfg,
            "ppl": ppl,
            "delta_ppl": ppl - ppl0,
            "n_tokens": n_tok,
            "compression_ratio": cache.compression_ratio(),
        })

    return results


def print_results_table(results: list[dict]) -> None:
    print()
    print(f"{'Method':<26} {'K':>4} {'V':>4} {'PPL':>10} {'ΔPPL':>10} {'ratio':>8}")
    print("-" * 68)
    for r in results:
        name = r["name"]
        kb = r["num_bits_key"]
        vb = r["num_bits_value"]
        ppl = r["ppl"]
        d = r["delta_ppl"]
        ratio = r.get("compression_ratio", 0.0)
        d_str = "—" if name == "fp16 baseline" else f"{d:+.4f}"
        r_str = "1.00×" if name == "fp16 baseline" else f"{ratio:.2f}×"
        print(f"{name:<26} {kb:>4} {vb:>4} {ppl:>10.4f} {d_str:>10} {r_str:>8}")
