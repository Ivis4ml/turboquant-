"""
WikiText-2 dataset loader for perplexity evaluation.

Supports two modes:
  1. HuggingFace datasets (preferred, auto-downloads)
  2. Local raw file fallback
"""

from __future__ import annotations

import torch


def load_wikitext2_encodings(
    tokenizer,
    split: str = "test",
    max_tokens: int | None = None,
) -> torch.Tensor:
    """
    Load WikiText-2 and tokenize.

    Args:
        tokenizer: HuggingFace tokenizer.
        split: "test", "validation", or "train".
        max_tokens: Truncate to this many tokens (None = full dataset).

    Returns:
        Token IDs tensor, shape (1, seq_len).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(ds["text"])
    except Exception:
        # Fallback: try loading from common local paths
        import os
        local_paths = [
            os.path.expanduser("~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw"),
            "wikitext-2-raw/wiki.test.raw",
        ]
        text = None
        for p in local_paths:
            if os.path.exists(p):
                with open(p) as f:
                    text = f.read()
                break
        if text is None:
            raise RuntimeError(
                "Could not load WikiText-2. Install `datasets` or place "
                "wiki.test.raw in a known location."
            )

    encodings = tokenizer(text, return_tensors="pt")["input_ids"]
    if max_tokens is not None and encodings.size(1) > max_tokens:
        encodings = encodings[:, :max_tokens]
    return encodings
