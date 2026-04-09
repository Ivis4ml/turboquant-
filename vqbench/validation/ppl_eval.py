"""
Sliding-window perplexity evaluator.

Modeled on turboquant_plus/benchmarks/benchmark_ppl_tq_vs_rq.py.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def evaluate_ppl(
    model,
    tokenizer,
    encodings: torch.Tensor,
    device: str | torch.device = "cpu",
    max_length: int = 1024,
    stride: int = 512,
) -> tuple[float, int]:
    """
    Sliding-window perplexity on tokenized text.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        encodings: Token IDs, shape (1, seq_len).
        device: Compute device.
        max_length: Context window for each chunk.
        stride: Sliding window stride.

    Returns:
        (perplexity, n_tokens_scored).
    """
    seq_len = encodings.size(1)
    nlls = []
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        input_ids = encodings[:, begin:end].to(device)

        target_len = end - prev_end if begin > 0 else end - begin
        target_ids = input_ids.clone()

        if begin > 0:
            target_ids[:, :-target_len] = -100
        else:
            target_ids[:, 0] = -100
            target_len -= 1

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * target_len
        nlls.append(neg_log_likelihood.item())
        n_tokens += target_len

        prev_end = end
        if end == seq_len:
            break

    ppl = math.exp(sum(nlls) / n_tokens)
    return ppl, n_tokens
