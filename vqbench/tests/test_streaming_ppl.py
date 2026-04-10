"""
Tests for the faithful streaming PPL evaluator and VQBenchCache semantics.

These tests enforce three correctness properties that are easy to break:

1. **Chunk-invariance**: baseline PPL with an exact `DynamicCache` must be
   the same up to fp16/SDPA accumulation noise regardless of chunk size.
   If this fails, the loss math is wrong (the previous version had an
   HF-shift-induced token drop at every chunk boundary).

2. **Single-chunk bit-identical**: with `chunk = seq_len`, the quantized
   path and the baseline path must produce identical PPL — there's no
   past cache, so quantization has nothing to affect.

3. **Past-lossy / current-exact**: within a forward pass, the current
   chunk's K,V are used exactly in attention, while the past cache is
   read from compressed storage (lossy).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

torch_available = True
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
except Exception:
    torch_available = False


# Use a tiny model so tests run in CI time (~30 seconds total)
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    if not torch_available:
        pytest.skip("transformers not installed")
    try:
        tok = AutoTokenizer.from_pretrained(SMOKE_MODEL, trust_remote_code=True)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        mdl = AutoModelForCausalLM.from_pretrained(
            SMOKE_MODEL, torch_dtype=torch.float16, trust_remote_code=True,
        ).to(device).eval()
    except Exception as e:
        pytest.skip(f"model download failed: {e}")
    return mdl, tok, device


@pytest.fixture(scope="module")
def encodings(model_and_tokenizer):
    """Deterministic 256-token synthetic input (avoids wikitext download)."""
    _, tok, _ = model_and_tokenizer
    # Use a fixed, easy-to-tokenize text so tests are deterministic and quick
    text = "The quick brown fox jumps over the lazy dog. " * 64
    ids = tok(text, return_tensors="pt")["input_ids"]
    return ids[:, :256]


class TestStreamingPPLCorrectness:
    def test_n_scored_equals_seq_len_minus_one(self, model_and_tokenizer, encodings):
        """Every token except the first must be scored exactly once."""
        from vqbench.validation.streaming_ppl import evaluate_streaming_ppl
        model, _, device = model_and_tokenizer
        for chunk in [256, 128, 64, 32]:
            _, n = evaluate_streaming_ppl(model, encodings, device, chunk=chunk, cache=None)
            assert n == encodings.size(1) - 1, (
                f"chunk={chunk}: scored {n} tokens, expected {encodings.size(1) - 1}"
            )

    def test_baseline_chunk_invariance(self, model_and_tokenizer, encodings):
        """
        CRITICAL: exact-cache baseline PPL must be chunk-size invariant.

        If this fails the evaluator has a boundary-token accounting bug
        (this was the High-severity regression flagged in review).
        Tolerance: 1% to allow for fp16 attention accumulation noise.
        """
        from vqbench.validation.streaming_ppl import evaluate_streaming_ppl
        model, _, device = model_and_tokenizer
        ppls = {}
        for chunk in [256, 128, 64, 32]:
            ppl, _ = evaluate_streaming_ppl(model, encodings, device, chunk=chunk, cache=None)
            ppls[chunk] = ppl
        lo, hi = min(ppls.values()), max(ppls.values())
        rel_range = (hi - lo) / lo
        assert rel_range < 0.01, (
            f"Baseline PPL drifts with chunk size: {ppls} (range {rel_range:.2%})"
        )

    def test_single_chunk_quantized_equals_baseline(self, model_and_tokenizer, encodings):
        """
        With chunk = seq_len there is no past, so quantization has nothing
        to affect. Quantized PPL must equal baseline PPL exactly (up to fp16).
        """
        from vqbench.validation.streaming_ppl import evaluate_streaming_ppl
        from vqbench.torch_wrapper.hook import make_vqbench_cache

        model, _, device = model_and_tokenizer
        seq_len = encodings.size(1)

        ppl_base, _ = evaluate_streaming_ppl(
            model, encodings, device, chunk=seq_len, cache=None,
        )

        cache = make_vqbench_cache(
            model.config,
            method_key="TurboQuantMSE",
            method_value="TurboQuantMSE",
            num_bits_key=3, num_bits_value=3,
        )
        ppl_q, _ = evaluate_streaming_ppl(
            model, encodings, device, chunk=seq_len, cache=cache,
        )

        # Expect bit-identical (all past is empty in single-chunk mode)
        assert abs(ppl_q - ppl_base) < 1e-4, (
            f"Single-chunk quantized PPL ({ppl_q:.6f}) differs from baseline "
            f"({ppl_base:.6f}); past-lossy / current-exact invariant broken."
        )


class TestVQBenchCacheFaithful:
    """
    Direct tests of the cache's past-lossy / current-exact contract,
    independent of the PPL evaluator.
    """

    def test_first_update_returns_current_exact(self):
        """Very first update() call: past is empty, so return == input."""
        from vqbench.torch_wrapper.hook import VQBenchCache

        cache = VQBenchCache(
            head_dim=64, num_kv_heads=2, num_layers=1,
            method_key="TurboQuantMSE", method_value="TurboQuantMSE",
            num_bits_key=2, num_bits_value=2,
        )
        torch.manual_seed(0)
        k = torch.randn(1, 2, 16, 64)
        v = torch.randn(1, 2, 16, 64)
        out_k, out_v = cache.layers[0].update(k, v)

        # On the first call past is empty → returned k,v should be bit-identical to input
        assert torch.equal(out_k, k), "First update must return current K exactly"
        assert torch.equal(out_v, v), "First update must return current V exactly"

    def test_second_update_current_is_exact_past_is_lossy(self):
        """
        Two sequential update() calls. After the second:
          - the CURRENT chunk portion of the returned tensor must equal the
            second call's input exactly
          - the PAST portion must NOT equal the first call's input (it went
            through quantize/dequantize)
        """
        from vqbench.torch_wrapper.hook import VQBenchCache

        cache = VQBenchCache(
            head_dim=64, num_kv_heads=2, num_layers=1,
            method_key="TurboQuantMSE", method_value="TurboQuantMSE",
            num_bits_key=2, num_bits_value=2,
        )
        torch.manual_seed(0)
        k1 = torch.randn(1, 2, 16, 64)
        v1 = torch.randn(1, 2, 16, 64)
        k2 = torch.randn(1, 2, 16, 64)
        v2 = torch.randn(1, 2, 16, 64)

        cache.layers[0].update(k1, v1)                       # chunk 1 → stored
        out_k, out_v = cache.layers[0].update(k2, v2)        # past=chunk1 lossy, current=chunk2 exact

        assert out_k.shape[2] == 32, f"expected 32 positions, got {out_k.shape[2]}"

        # Current chunk (last 16 positions) MUST be exact
        assert torch.equal(out_k[:, :, 16:, :], k2), (
            "Current-chunk K must be exact after second update()"
        )
        assert torch.equal(out_v[:, :, 16:, :], v2), (
            "Current-chunk V must be exact after second update()"
        )

        # Past chunk (first 16 positions) MUST be lossy (quantization roundtrip
        # at 2-bit is very lossy — strict inequality should hold)
        past_k_err = (out_k[:, :, :16, :].float() - k1.float()).abs().max().item()
        past_v_err = (out_v[:, :, :16, :].float() - v1.float()).abs().max().item()
        assert past_k_err > 1e-4, (
            f"Past K should be lossy but max diff is only {past_k_err:.2e}"
        )
        assert past_v_err > 1e-4, (
            f"Past V should be lossy but max diff is only {past_v_err:.2e}"
        )
