# VQBench — Project Status and TODO

> **Last updated:** 2026-04-10 · Phase 9.1 complete
> **For:** new collaborators and the main author's future self
>
> This document is the fastest path to being productive on VQBench. Read it before touching the code. For the full technical report with numerical results and discussion, see [`REPORT.md`](REPORT.md). For the implementation plan (historical and forward-looking), see [`PLAN.md`](PLAN.md).

---

## 0. One-minute summary

VQBench is a **unified vector quantization benchmark framework** — eight quantizers (TurboQuant / RaBitQ / PQ families + `BlockTurboQuantMSE`) under a single `VectorQuantizer` ABC, a faithful streaming perplexity evaluator for LLM KV caches, a PyTorch wrapper that plugs into the HuggingFace `transformers` 5.5 Cache API, and an end-to-end validation pipeline on Qwen3 and Qwen3.5.

Current headline result on **Qwen3.5-4B** (`head_dim = 256`, WikiText-2, 512 tokens, monkey-patching full_attention layers):

| Method | bits/val | ΔPPL vs fp16 |
|---|---|---|
| Scalar TQ-MSE 4-bit | 4.06 | **+0.0%** |
| **Block B=64 4-bit** | **4.25** | **+0.0%** |
| Block B=32 3-bit | 3.50 | +0.0% |

These match `turboquant_plus`'s published `turbo4 = 4.25 bits/val, +0.23%` at matching storage. **The algorithm is production-viable on the Qwen3.5 family.**

The **code** is 5,773 lines across 55 Python files. **187 tests**, all passing.

**What we still don't have:** Qwen3.5-27B numbers (needs an AWQ int4 checkpoint), hybrid-attention `VQBenchCache` (monkey-patching is used as a workaround), variance/CI reporting, task-proxy metrics, and any kind of production speed story.

---

## 1. Quick Start (for new collaborators)

### 1.1 Install

```bash
git clone <repo-url> turboquant-
cd turboquant-

# Core (quantizers + tests + synthetic benchmarks)
pip install -e ".[test]"

# Full (with real-model validation — torch, transformers, datasets)
pip install -e ".[test,validation]"
```

Tested on Python 3.13, NumPy 1.26+, SciPy 1.12+, PyTorch 2.10 (`mps` on Apple Silicon), transformers 5.5.0, datasets 4.8.3. M5 Pro 48 GB is the reference hardware.

### 1.2 Run the test suite

```bash
python -m pytest vqbench/tests/ -v
```

Expected: **186 passed, 1 skipped**, ~30 seconds.

### 1.3 Reproduce the headline results

```bash
# Block quantization on Qwen3-4B (head_dim=128)
python /tmp/qwen3_4b_block.py   # see REPORT.md §3.5.6 for the script

# Near-lossless on Qwen3.5-4B (head_dim=256)
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vqbench.datasets.wikitext import load_wikitext2_encodings
from vqbench.validation.streaming_ppl import evaluate_streaming_ppl
from vqbench.validation.monkey_patch import patch_model_kv, unpatch_model
from vqbench.methods.turboquant.mse import TurboQuantMSE

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B', torch_dtype=torch.float16, trust_remote_code=True).to('mps').eval()
enc = load_wikitext2_encodings(tok, max_tokens=512)
ppl0, _ = evaluate_streaming_ppl(m, enc, 'mps', chunk=256, cache=None)
print(f'fp16 baseline: {ppl0:.4f}')

def factory(d, seed=42):
    return TurboQuantMSE(d=d, num_bits=4, seed=seed, norm_correction=True)
hooks = patch_model_kv(m, k_quantizer_factory=factory)
ppl1, _ = evaluate_streaming_ppl(m, enc, 'mps', chunk=256, cache=None)
unpatch_model(hooks)
print(f'TQ-MSE 4-bit K: {ppl1:.4f}  delta={ppl1-ppl0:+.4f}')
"
```

### 1.4 Where to read next

- **[`REPORT.md`](REPORT.md)** — numerical results, metric/task mismatch discussion, Qwen3/Qwen3.5 findings, comparison with `turboquant_plus`
- **[`PLAN.md`](PLAN.md)** — historical implementation plan + Phase 9.1/9.2/9.3 forward plan
- **[`BlockTQ.md`](BlockTQ.md)** — algorithm walkthrough for `BlockTurboQuantMSE` (why it exists, how it works, trade-offs)
- **[`README.md`](README.md)** — install, quick-start snippets, layout

---

## 2. What Works (Phase 1–9.1)

### 2.1 Quantizers (8 methods under `VectorQuantizer` ABC)

| Family | Method | File | Status |
|---|---|---|---|
| TurboQuant | `TurboQuantMSE` | `methods/turboquant/mse.py` | ✅ Matches paper Theorem 1 |
| TurboQuant | `TurboQuantProd` | `methods/turboquant/prod.py` | ✅ Matches paper Theorem 2 |
| TurboQuant | `QJLQuantizer` | `methods/turboquant/qjl.py` | ✅ Unbiased IP verified |
| **TurboQuant** | **`BlockTurboQuantMSE`** | **`methods/turboquant/block_mse.py`** | **✅ NEW (Phase 9.1)** |
| RaBitQ | `RaBitQ1Bit` | `methods/rabitq/rabitq_1bit.py` | ✅ Estimator validated |
| RaBitQ | `ExtRaBitQ` | `methods/rabitq/rabitq_ext.py` | ⚠️ Multi-bit estimator unvalidated (see §3.1) |
| PQ | `ProductQuantizer` | `methods/pq/product_quant.py` | ✅ k-means codebook |
| PQ | `OptimizedPQ` | `methods/pq/opq.py` | ✅ Procrustes rotation |

All numerically cross-validated against `turboquant_plus` (where applicable) to `1e-6` max difference on identical inputs.

### 2.2 Infrastructure

- **Core** — Haar rotation (deterministic, cached), vectorized FWHT, Lloyd-Max codebook, metrics (MSE, IP distortion, bias α, recall@k), bit packing
- **KV cache compressor** — `KVCacheCompressor` supports asymmetric K/V strategy
- **Eval suite** — distortion / bias / recall / speed / compression sweep utilities
- **PyTorch wrapper** — `QuantizedKVCache` + `VQBenchCache` (transformers 5.5 `Cache` subclass); see limitations in §3.2
- **Faithful streaming PPL** — `vqbench/validation/streaming_ppl.py` — manual per-token CE with `prev_last_logit` boundary handling; chunk-invariant baseline verified (§3.5 of `REPORT.md`)
- **Monkey-patching evaluator** — `vqbench/validation/monkey_patch.py` — works across arbitrary attention architectures (used for Qwen3.5 where the hybrid cache isn't supported yet)
- **Datasets** — `random_unit_vectors`, `controlled_ip_pairs`, `wikitext-2` loader

### 2.3 Validated results (from `REPORT.md`)

- **§3.1** — Synthetic unit-vector MSE matches paper predictions within 1%
- **§3.2** — Normalized K-MSE on real Qwen2.5-1.5B activations
- **§3.3** — IP bias α with direct vs estimator path distinction
- **§3.4** — Storage accounting (bits/dim, compression vs fp16)
- **§3.5.1** — Qwen2.5-0.5B (head_dim=64) — the pathological regime, +32% ΔPPL
- **§3.5.6** — Qwen3-4B (head_dim=128) — production-viable, Block B=16 4-bit → +0.7%, Block B=32 4-bit → +1.3%
- **§3.5.7** — **Qwen3.5-4B (head_dim=256)** — near-lossless, 4-bit ΔPPL ≈ 0%
- **§3.5.8** — Dimension scaling story: head_dim 64→128→256, ΔPPL +31.8% → +1.1% → +0.0%

### 2.4 Tests

- **187 tests total** across 16 test files
- **186 pass**, 1 skipped (a deliberately skipped performance baseline)
- Key correctness tests:
  - `test_turboquant_mse.py` — Theorem 1 validation
  - `test_turboquant_prod.py` — Theorem 2 validation (unbiased IP)
  - `test_block_quant.py` — 21 tests, including the hypothesis-lock test "block strictly beats scalar on outlier data"
  - `test_streaming_ppl.py` — 5 regression tests locking down the faithful-cache semantics (the evaluator-bug fix)
  - `test_fairness.py` — same-seed cross-method equivalence
  - `test_torch_wrapper.py` — transformers 5.5 `Cache` integration

Full run: `python -m pytest vqbench/tests/ --timeout=120 -v`

---

## 3. Known Limitations (honest list)

These are the issues a reviewer would flag. None of them are "bugs that make the code wrong"; they are **scope gaps** that limit the strength of our published claims.

### 3.1 Algorithmic

1. **Multi-bit RaBitQ estimator (`ext_rabitq_ip_estimate`) is present but not validated at α ≈ 1.** Only the 1-bit estimator is test-verified. See `REPORT.md` §3.3. This means RaBitQ's published "unbiasedness" guarantee holds in our code **only at 1-bit**; multi-bit uses the direct dequant path which is biased (α ≈ 0.88–0.995).

2. **BlockTurboQuantMSE α-bias depends sensitively on (B, b, norm_correction).** No clean theory for when norm correction helps vs hurts. Empirically: at head_dim=64 (Qwen2.5-0.5B), Block B=16 at b=3 has α=1.025 (over-estimates IP) which causes sharper attention distributions and catastrophic PPL. At head_dim=128, Block B=16 at b=4 has α=0.999 (near unbiased). We shipped with `norm_correction=True` by default but this is not uniformly optimal.

3. **No BlockTurboQuantProd.** The Prod variant (block MSE + per-block QJL residual) might combine the outlier-isolation of block quantization with the unbiased-IP guarantee of QJL. We did not implement it. Worth trying.

4. **Codebook is assumption-based, not data-calibrated.** Our Lloyd-Max codebook is computed for the theoretical N(0, 1/d) distribution. `llama.cpp`'s production path probably uses empirically-calibrated codebooks. Data-aware calibration is an open direction.

### 3.2 Integration / scope

5. **`VQBenchCache` only handles pure full-attention models.** For hybrid architectures like Qwen3.5 (8 full_attention + 24 linear_attention layers), the `Cache` subclass doesn't know how to route updates to the right layer type. We used **monkey-patching** (`vqbench/validation/monkey_patch.py`) as a workaround for the Qwen3.5 numbers in `REPORT.md` §3.5.7. This means for Qwen3.5 the "current chunk exact / past chunk lossy" semantics are **not** enforced — the monkey-patch quantizes all positions. This overestimates the degradation and understates how good Qwen3.5 could actually look. The fact that we still get ΔPPL ≈ 0% in this pessimistic mode is strong evidence, but a proper hybrid cache would be more correct.

6. **`QuantizedKVCache.update()` processes `batch = 0` only.** Batched inference (batch > 1) isn't supported. The comment is on `vqbench/torch_wrapper/module.py` line ~120. This is fine for evaluation but blocks deployment in batched serving.

7. **No production speed path.** Everything goes through NumPy → torch cpu → mps. CPU↔GPU transfer dominates. No MLX port, no llama.cpp integration, no CUDA/Triton kernels. We intentionally do not report tok/s numbers because they would be dominated by overhead, not quantization.

8. **FWHT is vectorized but still loses to NumPy BLAS at d ≤ 512.** The asymptotic O(d log d) advantage only shows at d ≥ 2048 or so. For our target head_dims (128, 256) dense BLAS is still faster. FWHT is there for future use and correctness, not current speedup.

### 3.3 Evaluation methodology

9. **No variance / confidence intervals.** Every number in `REPORT.md` is a single-run point estimate with a fixed seed. Multi-seed or multi-slice reporting would strengthen the claims.

10. **Only WikiText-2 and only 512–1024 tokens.** No C4, no RULER, no NIAH, no MMLU, no long-context (32K+) evaluation. `REPORT.md` is not a paper-quality evaluation; it's a "the algorithm is correct and the numbers are plausible" evaluation.

11. **No task-proxy metrics for K-cache.** We only measure MSE, α, and PPL. Attention-logit correlation, top-k overlap, rank preservation would give a cheaper signal than full PPL and reveal when IP bias matters vs MSE.

12. **No unified storage accounting including amortized shared state.** `REPORT.md` §3.4 reports per-vector storage (indices + metadata) but not the per-instance shared state (rotation matrices, QJL projection, RaBitQ centroid, PQ codebooks). For very small caches these dominate; for typical KV caches they are negligible.

### 3.4 Scale

13. **No Qwen3.5-27B numbers yet.** This is the intended headline target. The algorithm and infrastructure are ready. The gating issue is: fp16 weights (~54 GB) don't fit 48 GB M5 Pro; needs AWQ int4 checkpoint (~18 GB) and `autoawq` package. Estimated 4–8 hours for a full sweep.

14. **No cross-architecture validation.** Only Qwen3 and Qwen3.5. Gemma-4, LLaMA 3, Mistral — not tested. The `monkey_patch.py` is model-family-agnostic (pattern-matches `k_proj` / `v_proj`), so it should work, but we haven't verified.

### 3.5 Documentation

15. **No API reference or tutorial notebook.** REPORT.md is long-form prose; README.md has quick-start snippets; PLAN.md is an implementation plan. We're missing a single "here's how to use VQBench as a library" narrative. The docstrings are good (paper references, formulas), but there's no walkthrough.

---

## 4. TODO List (prioritized)

### P0 — unblock Qwen3.5-27B (the central headline target)

These items finish the Phase 9.3 story. They are the critical path for turning "the algorithm is production-viable" into "the benchmark paper is done."

| # | Task | Size | Blocks | Owner |
|---|------|------|--------|-------|
| P0.1 | **Run Qwen3.5-27B with AWQ int4 weights** | 1–2 days (mostly compute) | S14 | — |
| P0.2 | **Implement `VQBenchCache` for hybrid attention** (Qwen3.5 style) | 2–3 days | proper faithful eval on Qwen3.5 | — |
| P0.3 | **Add variance reporting** — repeat each config 3× with different seeds, report mean ± std | 1 day | strengthens all claims | — |
| P0.4 | **Long-context evaluation** — at least one 8K-32K context run (RULER or NIAH) | 2–3 days | proves KV compression works where it matters | — |

### P1 — strengthen the findings (turn hypotheses into conclusions)

| # | Task | Size | Notes |
|---|------|------|-------|
| P1.1 | **Validate multi-bit `ext_rabitq_ip_estimate`** — prove α ≈ 1 at b=2,3,4, or fix if it doesn't | 1 day | Closes gap in §3.3 |
| P1.2 | **Task-proxy metrics for K-cache** — attention-logit correlation, top-k overlap, rank preservation | 2 days | Cheaper and more diagnostic than PPL |
| P1.3 | **`BlockTurboQuantProd`** — block MSE + per-block QJL residual | 2 days | Might combine best of both |
| P1.4 | **Theoretical analysis of BlockTQ α-bias** — why does α depend on (B, b, NC)? | research | Turn empirical into principled |
| P1.5 | **Unified storage metric with shared state amortization** | 0.5 day | Fixes §3.4 gap |
| P1.6 | **Cross-architecture validation** — run on Gemma-4 or LLaMA 3 (smallest variants) | 1 day | Proves it's not Qwen-specific |

### P2 — production path (speed, not just quality)

| # | Task | Size | Notes |
|---|------|------|-------|
| P2.1 | **Batched inference support** — remove the `batch = 0` constraint in `QuantizedKVCache.update()` | 1 day | Needed for any serving use case |
| P2.2 | **MLX port** — native Apple Silicon path | 1 week | Would give real tok/s numbers on M-series |
| P2.3 | **llama.cpp GGUF integration** — use VQBench quantizers as a `-ctk/-ctv` cache type | 2 weeks | Would be the actual production path |
| P2.4 | **Triton / CUDA kernels** for the quantize/dequantize hot path | 2 weeks | Stretch; only matters if (P2.1) proves the transfer overhead is the bottleneck |

### P3 — documentation and polish

| # | Task | Size | Notes |
|---|------|------|-------|
| P3.1 | **Tutorial notebook** — "how to use VQBench as a library for a new method" | 1 day | Lowers the barrier to contribution |
| P3.2 | **Sphinx or mkdocs API reference** | 2 days | Docstrings are good; surfacing them is the gap |
| P3.3 | **A proper benchmark runner CLI** — one command to reproduce the full `REPORT.md` tables | 1 day | `python -m vqbench.benchmark --all` |
| P3.4 | **CI on GitHub Actions** — lint + test on every push | 0.5 day | Basic hygiene |

---

## 5. Project Structure (where to find things)

```
turboquant-/
├── PLAN.md                       # historical + forward implementation plan
├── REPORT.md                     # numerical results and discussion
├── BlockTQ.md                    # BlockTurboQuantMSE algorithm walkthrough
├── TODO.md                       # (this file)
├── README.md                     # install + quick-start
├── VQ.png                        # architecture diagram (excalidraw)
├── pyproject.toml                # deps + optional groups (test, validation)
│
├── turboquant_plus/              # REFERENCE implementation — do not modify
│                                 # (cloned from github.com/TheTom/turboquant_plus)
│
└── vqbench/                      # the framework itself
    ├── core/                     # base.py ABC, rotation, metrics, packing
    │   ├── base.py               # VectorQuantizer ABC + QuantizedVector
    │   ├── rotation.py           # Haar rotation (cached) + FWHT
    │   ├── metrics.py            # MSE, IP distortion, α, recall@k
    │   └── packing.py            # bit-pack utilities
    │
    ├── methods/                  # the 8 quantizers
    │   ├── __init__.py           # central registry (used by eval/)
    │   ├── turboquant/
    │   │   ├── codebook.py       # Lloyd-Max on Gaussian
    │   │   ├── mse.py            # TurboQuantMSE (Algorithm 1)
    │   │   ├── prod.py           # TurboQuantProd (Algorithm 2 = MSE + QJL)
    │   │   ├── qjl.py            # QJL 1-bit quantizer
    │   │   └── block_mse.py      # BlockTurboQuantMSE (Phase 9.1)
    │   ├── rabitq/
    │   │   ├── rabitq_1bit.py    # RaBitQ 1-bit hypercube
    │   │   ├── rabitq_ext.py     # Extended multi-bit
    │   │   └── estimator.py      # unbiased ip_coeff estimator
    │   └── pq/
    │       ├── product_quant.py
    │       └── opq.py
    │
    ├── eval/                     # evaluation sweeps
    │   ├── distortion.py
    │   ├── bias.py
    │   ├── recall.py
    │   ├── speed.py
    │   └── compression.py        # includes storage accounting
    │
    ├── kv_cache/                 # KV-cache-specific glue
    │   ├── compressor.py         # KVCacheCompressor (asymmetric K/V)
    │   ├── attention.py          # scaled-dot-product attention on compressed cache
    │   └── outlier.py            # outlier channel strategy (per-channel fp16)
    │
    ├── torch_wrapper/            # PyTorch / HuggingFace transformers integration
    │   ├── module.py             # QuantizedKVCache nn.Module
    │   └── hook.py               # VQBenchCache (transformers 5.5 Cache subclass)
    │
    ├── validation/               # real-model evaluation
    │   ├── streaming_ppl.py      # FAITHFUL streaming PPL (current chunk exact, past lossy)
    │   ├── monkey_patch.py       # fallback when Cache doesn't support hybrid attention
    │   ├── k_mse.py              # normalized K-MSE measurement
    │   ├── ppl_eval.py           # (deprecated) naive sliding-window PPL
    │   └── run_quick.py          # one-shot smoke test runner
    │
    ├── datasets/
    │   ├── synthetic.py          # random_unit_vectors, controlled_ip_pairs
    │   └── wikitext.py           # wikitext-2 via HF datasets
    │
    └── tests/                    # 16 test files, 187 tests
        ├── test_block_quant.py   # 21 tests for BlockTurboQuantMSE
        ├── test_streaming_ppl.py # 5 tests locking down faithful cache semantics
        ├── test_turboquant_mse.py
        ├── test_turboquant_prod.py
        ├── test_rabitq.py
        ├── test_pq.py
        ├── test_codebook.py
        ├── test_qjl.py
        ├── test_kv_cache.py
        ├── test_torch_wrapper.py
        ├── test_fairness.py
        ├── test_interface.py
        ├── test_perf.py
        └── __init__.py
```

---

## 6. Key Design Decisions and Their Reasons

**Why pure Python/NumPy for the core?**
Fairness. `turboquant_plus` already exists as a C++/Metal implementation of TurboQuant; reimplementing that wouldn't answer any research questions. VQBench's contribution is putting every method on the same footing — same language, same hardware, same data, same evaluation. Pure NumPy is the only way to guarantee that TurboQuant and RaBitQ get identical treatment.

**Why the same `VectorQuantizer` ABC for everything?**
Adding a new method should touch zero existing code. Subclass `VectorQuantizer`, implement four methods, register in `methods/__init__.py`. That's it. `eval/`, `kv_cache/`, `tests/test_interface.py` pick it up automatically. This has paid off repeatedly — `BlockTurboQuantMSE` (Phase 9.1) took half a day from "start implementing" to "running in the PPL sweep."

**Why `norm_correction=True` by default?**
This matches `turboquant_plus` production behavior. The pure Theorem 1 object is `norm_correction=False`, but nobody actually deploys that. Tests keep both paths working; the default is the practical one.

**Why is the faithful streaming PPL evaluator manual-per-token-CE instead of `labels=input_ids`?**
HF's `labels=input_ids` shortcut internally does `shift_logits / shift_labels` which silently drops one token at every chunk boundary. This was the evaluator bug we found via a chunk-invariance sanity check (baseline PPL drifted 12.365 / 12.407 / 12.391 with different chunk sizes). A correct streaming evaluator must give identical baseline PPL regardless of chunk size up to fp16 noise. See `test_streaming_ppl.py::test_baseline_chunk_invariance` for the regression lock.

**Why do we keep `turboquant_plus/` in the repo?**
It's the reference. Every time we change a TurboQuant file, we can `diff` against their implementation and spot drift. It is `.gitignore`'d from the VQBench package but kept as a sibling directory for quick comparison. **Do not modify `turboquant_plus/`**.

**Why aren't there any speed numbers?**
Because publishing speed numbers from pure-NumPy-on-MPS would be misleading. The quantization operation itself is fast, but the CPU↔GPU transfer dominates. A meaningful speed story requires either MLX (P2.2) or llama.cpp integration (P2.3). Until then, speed is deliberately out of scope. This is stated in `REPORT.md` §5.

---

## 7. Current Status Checklist (by PLAN phase)

| Phase | Task | Status |
|---|---|---|
| 1 | Core framework (base, rotation, metrics) | ✅ |
| 2 | TurboQuant (MSE, Prod, QJL) + tests | ✅ |
| 3 | RaBitQ (1-bit, ExtRaBitQ, estimator) + tests | ✅ (multi-bit estimator unvalidated, see §3.1) |
| 4 | PQ / OPQ + tests | ✅ |
| 5 | Unified evaluation + interface + fairness tests | ✅ |
| 6 | KV-cache compressor + compressed attention | ✅ |
| 7 | Performance: FWHT, bit packing, batch ops, rotation cache | ✅ |
| 8 | PyTorch wrapper + HF transformers Cache integration | ✅ (batch=0 only; hybrid attention unsupported) |
| 9.0 | Norm correction + real-model validation (K-MSE on activations) | ✅ |
| 9.0.5 | Faithful streaming PPL evaluator + bug fix | ✅ |
| 9.1 | `BlockTurboQuantMSE` | ✅ |
| 9.2 | Validation on Qwen3-4B / Qwen3.5-4B | ✅ (Qwen3-4B streaming + Qwen3.5-4B monkey-patch) |
| 9.3 | **Qwen3.5-27B final headline run** | ⏳ **P0, next** |
| 9.4 | Hybrid attention `VQBenchCache` | ⏳ P0 |
| 9.5 | Variance reporting + cross-architecture (Gemma, LLaMA) | ⏳ P1 |

---

## 8. How to Contribute

### 8.1 Workflow

1. Read this file (TODO.md), then `REPORT.md`, then `BlockTQ.md` if you care about the algorithm.
2. Pick a task from §4. If it's P0, coordinate with the main author first (these are on the critical path).
3. Run `python -m pytest vqbench/tests/` — all 186 should pass before you start.
4. Make your change. Follow the existing code style (see `vqbench/methods/turboquant/block_mse.py` as a reference — docstrings with paper references, vectorized batch paths, numeric-parity tests).
5. Add tests. Every new quantizer gets at minimum: shape/dtype roundtrip, storage_bits formula, monotonicity in num_bits, synthetic-data MSE comparison.
6. Run `python -m pytest vqbench/tests/` again — all 186 should still pass, plus your new tests.
7. Update `REPORT.md` only if you're reporting a new measurable finding. Update `TODO.md` (this file) to check off the task.
8. Open a PR against `main`.

### 8.2 Non-negotiables

- **Don't break fairness.** Same `(d, seed)` must give the same Haar rotation across all methods. The test `tests/test_fairness.py::test_rotation_shared_between_methods` enforces this.
- **Don't break `turboquant_plus` parity.** If you touch `methods/turboquant/{codebook,mse,prod,qjl}.py`, `diff` your change against `turboquant_plus/turboquant/` and make sure the numerical outputs still match to 1e-6 on identical inputs.
- **Don't commit `turboquant_plus/` modifications.** It's the reference; it's in `.gitignore` for good reason.
- **Don't add silent failure paths.** If a quantizer can't handle certain input, it should `raise ValueError`, not return garbage. See `BlockTurboQuantMSE.__init__`'s `d % block_size != 0` check for the pattern.
- **Don't remove tests without explaining why in the PR.** The tests encode hard-won correctness invariants.

### 8.3 Things that are easy to get wrong

- **Rotation conventions for batch vs single.** `self._rotation @ x` (single column vector) vs `X @ self._rotation.T` (batch of row vectors) — these compute the same thing but it's easy to mix them up. `test_turboquant_mse.py` has a "batch matches single" check that catches this.
- **fp16 precision in the faithful streaming path.** The past cache goes through NumPy (fp64) → torch (fp16) → attention (fp16 matmul). When debugging why a metric differs across chunk sizes, check whether the difference is below fp16 precision (~0.1%) before assuming it's a bug. See `test_streaming_ppl.py::test_baseline_chunk_invariance` — the allowed tolerance is 1%.
- **`DynamicCache()` vs `None`.** The old faithful PPL evaluator created a `DynamicCache()` as the baseline. This breaks on hybrid-attention models (Qwen3.5) because they expect a cache with both layer types. The fix: pass `past_key_values=None` on the first forward call and let the model create the right cache.
- **HF's `labels=input_ids` shortcut.** Internally does `shift_logits / shift_labels`. If you use it across a streaming forward loop, you silently drop one token per chunk boundary. Don't use it; do manual per-token CE. See `streaming_ppl.py`.

---

## 9. Open Research Questions

These are not on any TODO list because they are open-ended and not on the critical path. They are here so that a collaborator with spare cycles knows what's interesting.

**Q1.** Is there a clean theoretical characterization of when `BlockTurboQuantMSE` α-bias is positive vs negative? Empirically it depends on (B, b, norm_correction) in a non-obvious way. Solving this would give a principled rule for default norm_correction.

**Q2.** `turboquant_plus` reports `turbo4 = 4.25 bits/val` which matches our `Block B=64`. Does llama.cpp's actual implementation use block quantization with `block_size = head_dim / 2`, or is it something else? Reading their code would be the most direct way to answer this.

**Q3.** Qwen3.5's hybrid attention architecture has only 25% full-attention layers. Does this mean aggressive KV compression (2-bit?) is essentially free because the linear-attention layers absorb more of the modeling load? We haven't tested 2-bit block at head_dim=256.

**Q4.** Is there a "block size adaptive" variant where you pick `B` per channel based on outlier concentration? The data we have (§3.5.6 vs §3.5.7) suggests B=16 is best at short context but B=32 wins at long context. A dynamic B would sidestep the trade-off.

**Q5.** The TurboQuant paper's Theorem 2 (unbiased IP via QJL residual) has not been combined with block quantization in VQBench. Would `BlockTurboQuantProd` give `α ≈ 1` at all head_dims AND low MSE? This is P1.3.

**Q6.** `ExtRaBitQ` at multi-bit is biased in the direct dequant path but has a dedicated estimator we haven't validated. If the estimator works, `ExtRaBitQ` might be competitive with `TurboQuantProd` at the storage cost of +64 bits per vector. This is P1.1.

---

## 10. Glossary

- **head_dim** — the dimension of each attention head (128 for Qwen3, 256 for Qwen3.5). Written as `d` in equations.
- **bits/val** — effective bits per dimension = (total storage bits per vector) / d. Includes metadata (norms, scales), not just indices. See `REPORT.md` §3.4 for the formula.
- **ΔPPL** — perplexity difference vs the fp16 baseline, either in absolute terms (e.g., +0.14) or relative (+1.1%).
- **faithful streaming PPL** — the evaluation mode where the current forward pass uses exact K/V, but past chunks are read from a compressed cache. Matches llama.cpp semantics. This is the right way to evaluate KV compression.
- **monkey-patching** — the fallback evaluation mode where we wrap the model's `k_proj` and `v_proj` to inject quantize→dequantize inside the forward pass. Works on any architecture but over-estimates degradation because the current chunk's K/V also get quantized.
- **direct dequant path** — what a standard HuggingFace attention module does: call `cache.update()`, then compute `Q @ K.T` on the returned tensors. Implicit assumption: the returned tensors are exact (they aren't, they're dequantized approximations).
- **estimator path** — the explicit paper-specified path for methods like RaBitQ where you're supposed to call a dedicated estimator function (e.g., `rabitq_ip_estimate`) that uses stored side-information to correct the bias. We only have a 1-bit estimator validated; multi-bit is in `estimator.py` but unverified.
- **norm correction** — for TurboQuantMSE, after dequantizing to centroids, re-normalize ŷ to unit norm before the inverse rotation. Matches turboquant_plus production. Shifts α closer to 1 at the cost of slightly worse MSE.
- **Lloyd-Max** — the optimal (in MSE) scalar quantizer for a given distribution. We use it for Gaussian N(0, 1/d) in `codebook.py`.
- **QJL** — Quantized Johnson-Lindenstrauss: 1-bit quantization via random Gaussian projection + sign. Unbiased IP in expectation. Used as the second stage of `TurboQuantProd`.
- **full attention layer / linear attention layer** — in hybrid architectures like Qwen3.5, only some layers compute attention with a standard KV cache that grows with context. The rest use linear attention with a fixed-size recurrent state. Only the full_attention layers are candidates for KV cache compression.
