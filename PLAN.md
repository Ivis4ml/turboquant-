# VQBench — Design, Roadmap, Phase 10 RaBitQ Redirection

> **Single source of truth for the forward-looking plan.** For current empirical results see [`REPORT.md`](REPORT.md); for the BlockTurboQuantMSE algorithm walkthrough see [`BlockTQ.md`](BlockTQ.md).
>
> **Headline target:** run **Qwen3.5-27B** on a **48 GB M5 Pro**. Algorithm and evaluator are ready; the remaining work is hybrid-cache integration, the Qwen3.5-27B sweep, and the Phase 10 RaBitQ redirection.

---

## 1. Motivation

Existing comparisons between TurboQuant, RaBitQ, and PQ are unreliable because they cross implementation boundaries:

| Problem | Source |
|---------|--------|
| TurboQuant paper uses GPU for TurboQuant but CPU for RaBitQ/PQ | Table 2, arXiv 2504.19874 |
| RaBitQ reference implementation is C++, TurboQuant is Python + Metal | `gaoj0017/RaBitQ` vs `TheTom/turboquant_plus` |
| RaBitQ author (Gao) publicly questioned TurboQuant's RaBitQ reimplementation | Fig. 5 discussion |
| No existing codebase puts all three under the same API | — |

**VQBench's answer:** same language (pure NumPy), same hardware, same evaluation harness, same random seeds. No implementation bias. Production speed is deliberately out of scope for the benchmark itself; reference implementations (`turboquant_plus` for C/Metal, Phase 11 for MLX / llama.cpp) handle that.

---

## 2. Paper References

| ID | Method | Paper | arXiv | Core Idea |
|----|--------|-------|-------|-----------|
| P1 | **TurboQuant** | Zandieh et al., ICLR 2026 | 2504.19874 | Random rotation + Lloyd-Max scalar quantization |
| P2 | **RaBitQ** | Gao & Long, SIGMOD 2024 | 2405.12497 | Random rotation + hypercube + unbiased estimator |
| P3 | **Extended RaBitQ** | Gao et al., SIGMOD 2025 | 2409.09913 | Multi-bit via bit-plane decomposition |
| P4 | **QJL** | Zandieh et al., 2024 | 2406.03482 | 1-bit sign quantization for unbiased IP |
| P5 | **PQ** | Jégou et al., TPAMI 2011 | — | k-means codebook per subspace |
| P6 | **OPQ** | Ge et al., CVPR 2013 | — | PQ with learned rotation |

TurboQuant and RaBitQ both exploit random rotation + high-dimensional concentration. After a Haar rotation, each coordinate's marginal distribution approaches $\mathcal{N}(0, 1/d)$. The split comes after:

- **RaBitQ** rounds to hypercube vertices ($\pm 1/\sqrt{d}$) and achieves unbiased IP via a correction estimator that must be used explicitly at query time.
- **TurboQuant** applies Lloyd-Max scalar quantization for better reconstruction MSE, achieving unbiased IP only through its two-stage MSE+QJL variant (`TurboQuantProd`).

This distinction — *where* each method puts its unbiasedness — is why Phase 10 rebuilds RaBitQ as an attention-native compression representation instead of a reconstruction quantizer (see §5 and [`REPORT.md`](REPORT.md) §8).

---

## 3. Design Principles

### D1 — Adding a new method touches zero existing code

The framework follows the Open-Closed Principle. Subclass `VectorQuantizer`, implement four methods (`quantize`, `dequantize`, `storage_bits`, `name`), register in `methods/__init__.py`. `eval/`, `kv_cache/`, `tests/test_interface.py` pick it up automatically. `BlockTurboQuantMSE` (Phase 9.1) took half a day from "start implementing" to "running in the PPL sweep" because of this. Phase 10 is designed to add two new quantizers without touching any existing code.

### D2 — Every function maps to a paper formula

Every algorithm file has a docstring of the form `Paper: <authors>, <arXiv ID>, <Section/Theorem/Definition>` plus the exact mathematical expression. Example: `methods/turboquant/qjl.py`'s `dequantize` docstring reproduces Zandieh et al. 2406.03482 Definition 1 verbatim. A reviewer can open any function, read the paper side-by-side, and verify correctness.

### D3 — Each algorithm is self-contained

No cross-method imports: `rabitq/` never imports from `turboquant/`. Shared code lives in `core/` (rotation, codebook, metrics, packing). Each method has its own test file; you can run `pytest tests/test_rabitq.py` without any TurboQuant code.

### D4 (new in Phase 9.1) — No silent failures

If a quantizer cannot handle certain input, it raises `ValueError` with an error message pointing to the right alternative. Recent examples: `QuantizedKVCache.update()` rejects `batch > 1` with a clear message (previously silently broadcast batch 0); `RaBitQ1Bit.__init__` rejects `num_bits != 1` and points to `ExtRaBitQ`. The codebase has no "silently return garbage" paths.

---

## 4. Phase Status

### 4.1 Completed (Phases 1–9.1)

| Phase | Deliverable | Status | Key files |
|-------|-------------|--------|-----------|
| 1 | Core framework: ABC, Haar rotation, metrics, packing | ✅ | `core/{base,rotation,metrics,packing}.py` |
| 2 | TurboQuant: Lloyd-Max codebook, MSE, QJL, Prod | ✅ | `methods/turboquant/{codebook,mse,qjl,prod}.py` |
| 3 | RaBitQ: 1-bit hypercube + ExtRaBitQ + unbiased estimator | ✅ (multi-bit estimator unvalidated, see §5.2) | `methods/rabitq/{rabitq_1bit,rabitq_ext,estimator}.py` |
| 4 | PQ / OPQ: k-means + Procrustes rotation | ✅ | `methods/pq/{product_quant,opq}.py` |
| 5 | Unified evaluation + interface / fairness tests | ✅ | `eval/`, `tests/test_{interface,fairness}.py` |
| 6 | KV-cache compressor + compressed attention + outlier strategy | ✅ | `kv_cache/{compressor,attention,outlier}.py` |
| 7 | Performance: vectorized FWHT, bit packing, batch ops, rotation cache | ✅ | `core/{rotation,packing}.py`, `tests/test_perf.py` |
| 8 | PyTorch wrapper + `transformers` 5.5 `Cache` integration | ✅ (batch = 1, full-attention only) | `torch_wrapper/{module,hook}.py` |
| 9.0 | Norm correction (`turboquant_plus` parity) + real-model K-MSE | ✅ | `methods/turboquant/mse.py`, `validation/k_mse.py` |
| 9.0.5 | Faithful streaming PPL evaluator + bug fix | ✅ | `validation/streaming_ppl.py`, `tests/test_streaming_ppl.py` |
| 9.1 | `BlockTurboQuantMSE` + Qwen3-4B / Qwen3.5-4B validation | ✅ | `methods/turboquant/block_mse.py`, [`BlockTQ.md`](BlockTQ.md), [`REPORT.md`](REPORT.md) §3 |

**Codebase snapshot (2026-04-10):** 5,773 lines across 55 Python files, **199 pytest tests** across 13 test files. Excluding `test_perf.py` (23) and `test_streaming_ppl.py` (5) for speed: **170 passed + 1 skipped** in ~25 s on M5 Pro. Four regression tests were added in the 2026-04-10 spring-cleaning pass, locking down the batch-rejection path, the `BlockTurboQuantMSE-B64` registration, and the strict 1-bit enforcement on `QJLQuantizer` / `RaBitQ1Bit`.

### 4.2 In flight — Phase 9.2: hybrid cache + variance + Qwen3.5-27B (P0)

These items finish the Phase 9 story: they turn "the algorithm works on Qwen3.5-4B" into "the algorithm works at the M5 Pro 48 GB headline target."

| # | Task | Size | Files | Blocks |
|---|------|------|-------|--------|
| 9.2.1 | **`VQBenchCache` for hybrid attention.** Route `update()` only to full-attention layers; pass linear-attention state through unchanged. | 2–3 days | `torch_wrapper/hook.py` | Faithful Qwen3.5 eval |
| 9.2.2 | **Qwen3.5-27B AWQ int4 sweep.** 3-bit and 4-bit K, chunk = 256, WikiText-2 1K tokens. | 1–2 days (mostly compute) | `validation/run_quick.py`, new `validation/qwen35_27b.py` | The headline |
| 9.2.3 | **Variance reporting.** ✅ **done 2026-04-11** — §3.1, §4, §5.5, §5.6, §5.7 all extended to 3-seed mean ± std. Pending: 3-seed long-context sweep (§3.4 is still single-seed per length). | ~10 min extra for §3.4 variance | — | — |
| 9.2.4 | **Long-context evaluation.** ✅ **substantially done 2026-04-11** — §3.4 now covers 512 → 8192 tokens (1 → 31 cache boundaries), ΔPPL remains flat. Pending: 16K–32K stress, RULER / NIAH. | partial; 1–2 days for longer context + task metrics | — | Phase 9.2.2 readiness |
| 9.2.5 | **Multi-bit `ext_rabitq_ip_estimate` validation.** ✅ **substantively done 2026-04-11** (see §5.7 of REPORT and §5.2 below). Open: regression test lock. | ~1 h | `tests/test_rabitq.py` | — |

Rationale for 9.2.1 before 9.2.2: the Qwen3.5-27B sweep is expensive (4–8 h per config); running it through the monkey-patch would re-pay the "monkey-patch over-estimates degradation" tax on every run. A native hybrid cache is 2–3 days of work that unlocks clean 27B numbers.

### 4.3 Phase 10 — RaBitQ as Attention-Native Compression (next strategic bet)

See §5 for the full plan.

### 4.4 Phase 11 — Production path (P2, not blocking the benchmark)

| # | Task | Size | Notes |
|---|------|------|-------|
| 11.1 | **Batched inference support.** Remove the `batch == 1` constraint from `QuantizedKVCache.update()`. | 1 day | Needed for any serving use case |
| 11.2 | **MLX port.** Native Apple Silicon path — MLX's unified memory avoids the CPU↔GPU transfer that currently dominates. | 1 week | Unlocks real tok/s numbers on M-series |
| 11.3 | **llama.cpp GGUF integration.** Use VQBench quantizers as a `-ctk / -ctv` cache type. | 2 weeks | The actual production path; complements `turboquant_plus`'s reference integration |
| 11.4 | **Triton / CUDA kernels** for the quantize/dequantize hot path. | 2 weeks | Stretch; only matters after Phase 11.1 proves the transfer overhead is the bottleneck |

---

## 5. Phase 10 — RaBitQ as an Attention-Native Compression Representation

**Framing.** The Phase 9.1 story landed squarely in TurboQuant territory: at head_dim = 256, Scalar TurboQuantMSE 4-bit is already lossless on Qwen3.5-4B, and BlockTurboQuantMSE doesn't help. That leaves a puzzle: we spent a lot of effort making TurboQuant's reconstruction path work, and very little making RaBitQ compete. From a fairness perspective, that's a gap the benchmark should close.

The conclusion reached in discussion with GPT-5.4: **the landing for RaBitQ in VQBench is not "a better reconstruction quantizer."** In the direct-dequant path that HF attention modules actually use, RaBitQ is biased and loses to TurboQuant on reconstruction MSE (see [`REPORT.md`](REPORT.md) §8). Reframing RaBitQ's job:

- **K cache only** — RaBitQ takes over K compression. V stays on BlockTurboQuantMSE / TurboQuantMSE (V compression is free under the MSE metric, §5 of [`REPORT.md`](REPORT.md)).
- **Estimator path, not direct dequant** — attention scores are computed directly from compressed K via the RaBitQ estimator (XNOR+popcount at 1-bit, bit-plane accumulation at multi-bit). The query gets rotated and centroid-corrected on the fly. K is never reconstructed as a dense fp16 tensor.
- **Metadata budget** — RaBitQ's 64-bit (1-bit) / 128-bit (multi-bit) fp32 metadata header is currently ~21% of storage at nominal 4-bit on head_dim = 128. Compressing this is a 1-day change with immediate measurable impact.
- **KV-cache-aware calibration** — per-head / per-layer centroid and rotation, not a single dataset-wide rotation.

**Success criterion for Phase 10 as a whole.** On Qwen3.5-4B (intermediate) or Qwen3.5-27B (final), **4-bit K RaBitQ in the estimator-native attention path delivers ΔPPL within 0.2% of Scalar TQ-MSE at matching or lower bits/val**, with V kept on BlockTQ or TurboQuantMSE. If we can hit that, RaBitQ is a strictly complementary tool (the attention-native path) rather than a dominated reconstruction quantizer.

### 5.1 Phase 10.1 — Metadata compression (FIRST, 1 day)

**Why this goes first.** Every storage comparison in [`REPORT.md`](REPORT.md) §6 counts metadata. At head_dim = 128, ExtRaBitQ 4-bit currently costs 5.00 effective bits/dim (128-bit header / 128 dims = 1 extra bit/dim) while TQ-MSE 4-bit costs 4.12 — a 21% penalty. Without this item, even a perfect estimator-native attention path in 10.3 gets dismissed as "costs too many bits."

**Changes:**

1. `ExtRaBitQ.quantize` stores `norm`, `ip_coeff`, `scale`, `offset` as fp32 today (4 × 32 = 128 bits). Change to fp16 (4 × 16 = 64 bits). This costs marginal precision on IP reconstruction — needs a correctness test to confirm $\alpha$ stays at ≈ 1.
2. `RaBitQ1Bit` stores `‖o‖`, `ip_coeff` as fp32 today (64 bits). Change to fp16 (32 bits).
3. **Per-head shared centroid.** The dataset centroid $c$ is currently computed per-instance via `fit(X)`. For KV-cache use, one `VQBenchCacheLayer` holds all heads of one layer; the centroid should be computed per-head on the first chunk and then held constant (not stored per vector). Add a `VQBenchCacheLayer.fit_centroids(k_first_chunk)` hook.
4. **Calibrated coefficient at multi-bit.** At $b \geq 3$, the per-vector `scale` and `offset` in `ExtRaBitQ` can be replaced by a single per-head-shared scale if the distribution is stationary enough. This needs an ablation: does a per-head fixed scale cost any $\alpha$ or does it match the per-vector version?

**Deliverables:**

- `methods/rabitq/rabitq_ext.py`: add `metadata_precision={"fp16","fp32"}` parameter (default `fp16`), `share_centroid={"per_vector","per_head"}` parameter (default `per_head`)
- `tests/test_rabitq.py`: new tests that fp16 metadata + per-head centroid preserves $\alpha \approx 1$ to within 0.005
- `eval/compression.py`: extended table showing the new metadata counts

**Target storage for head_dim = 128 after Phase 10.1:**

| Config | Before (bits/vec) | After | eff. bits/dim | vs fp16 |
|--------|------------------:|------:|--------------:|--------:|
| ExtRaBitQ 3-bit | 512 | **416** | 3.25 | 4.92× |
| ExtRaBitQ 4-bit | 640 | **544** | 4.25 | 3.76× |
| RaBitQ 1-bit | 192 | **160** | 1.25 | 12.80× |

After 10.1, ExtRaBitQ 4-bit at 4.25 bits/dim matches BlockTQ B=32 4-bit at 4.50 bits/dim and `turboquant_plus` `turbo4 = 4.25` exactly.

### 5.2 Phase 10.2 — Multi-bit estimator validation (**substantively done 2026-04-11**)

**Status update.** Measured on real Qwen3.5-4B K in [`REPORT.md`](REPORT.md) §5.7 via `scripts/ip_bias_alpha_qwen35.py`: `ext_rabitq_ip_estimate` gives $\alpha \in \{0.989, 0.998, 0.998\}$ at $b \in \{2, 3, 4\}$ when measured against its actual target $\langle q - c, k - c \rangle$. The old "not validated" label was a **measurement methodology gap**, not a code bug: the original test compared the estimator's output against $\langle q, k \rangle$ on real data where the centroid $c \neq 0$, producing a spurious bias.

**What made it look broken:** `_fit_alpha_from_estimator` in the initial script version computed `true_ip = dot(q, k)` but the estimator returns `⟨q-c, k-c⟩`. On synthetic unit vectors the centroid is zero so the two coincide; on real K activations the centroid is nontrivial and the comparison gave α ≈ 0.95 instead of 0.99+. After centering the "true" value the estimator matches 1.0 cleanly.

**What remains (1 hour):**

1. Add `tests/test_rabitq.py::test_ext_rabitq_estimator_unbiased_multi_bit`: at $d = 256$, $n = 4000$, for $b \in \{2, 3, 4\}$, quantize with `ExtRaBitQ`, fit $\alpha$ via `_fit_alpha_from_estimator` using centered ground truth, assert $|\alpha - 1| < 0.02$.
2. Update the existing 1-bit estimator test (`test_unbiased_ip_via_estimator`) to use the same centered-target convention so both tests exercise the same API.
3. Once the test lands, mark Phase 10.2 fully done.

**Implication for Phase 10.3.** The prerequisite is cleared. Estimator-native attention can proceed — the multi-bit path is known to be unbiased in expectation, so the remaining work is kernel integration, not algorithm correctness.

### 5.3 Phase 10.3 — Estimator-native attention (research, ~1 week)

**Goal.** Replace the standard `attention_scores = q @ k_dequant.T` path with `attention_scores = rabitq_estimator(q, k_quantized)` for K cache, without reconstructing K as a dense fp16 tensor.

**Architecture:**

```
Standard HF attention:
    k_cache.update() → returns (full_k: fp16 tensor)
    attention_scores = q @ full_k.transpose(-2, -1) / sqrt(d)
    attention_weights = softmax(attention_scores)
    out = attention_weights @ full_v

Phase 10.3 estimator-native attention (K-side only):
    k_cache.update() → returns (full_k_quantized: QuantizedVector list)
    attention_scores = rabitq_attention_scores(q, full_k_quantized)  ← new
    attention_weights = softmax(attention_scores)
    out = attention_weights @ full_v                                  ← unchanged
```

`full_v` still comes from a dequantized BlockTQ / TurboQuantMSE V cache — the V path is unchanged because MSE reconstruction is the right metric for values.

**1-bit fast path.** At $b = 1$, the attention score for one $(q, k_i)$ pair reduces to

$$\langle q, k_i \rangle \;\approx\; \frac{\|o_i\|}{\text{ip\_coeff}_i} \cdot \frac{1}{\sqrt{d}}\,(d - 2 \cdot \text{popcount}(b_q \oplus b_i))$$

where $b_q$ and $b_i$ are the sign bits of the rotated query and rotated stored vector, respectively. This is the RaBitQ paper's Eq. (8) expressed as XNOR + popcount. On modern CPUs / GPUs this is 1–2 orders of magnitude cheaper than fp16 matmul at the byte-per-dim density RaBitQ uses.

**Multi-bit path.** For $b \geq 2$ ExtRaBitQ, decompose the stored indices into $b$ binary planes, compute the 1-bit estimator for each plane, weight by $2^k$, and sum. This is the bit-plane decomposition from arXiv 2409.09913 §3.

**Query-side cost.** The query has to be rotated and centroid-subtracted every decode step, and its sign bits computed for the XNOR fast path. A dense Haar rotation at $d = 256$ is 65k multiplies per query — non-trivial at high decode rates. **Phase 10.3 uses structured rotation (randomized Hadamard + sign flip) instead of dense Haar** to cut this to $O(d \log d) \approx 2k$ ops. The structured rotation code already exists in `core/rotation.py` (Phase 7.1); Phase 10.3 wires it to the RaBitQ path.

**Tasks:**

| # | Task | File |
|---|------|------|
| 10.3.1 | `vqbench/torch_wrapper/attention_rabitq.py`: new `rabitq_attention_scores(q, kv_cache, layer_idx, head_idx)` that computes softmax logits directly from the compressed K store | new |
| 10.3.2 | Asymmetric `VQBenchCache` option: K uses `RaBitQ1Bit` / `ExtRaBitQ` with `estimator_native=True`; V uses `BlockTurboQuantMSE` / `TurboQuantMSE` | `torch_wrapper/hook.py` |
| 10.3.3 | Structured rotation integration: switch RaBitQ path to use `StructuredRotation` from `core/rotation.py` when `rotation_mode="fwht"` | `methods/rabitq/rabitq_*.py` |
| 10.3.4 | Correctness test: estimator-native attention logits match the offline estimator path to fp32 tolerance on a 128-token batch | `tests/test_attention_rabitq.py` (new) |
| 10.3.5 | End-to-end PPL on Qwen3.5-4B: K estimator-native RaBitQ at 2-bit and 4-bit, V BlockTQ at 4-bit | `validation/qwen35_4b_rabitq.py` (new) |

**Success criterion.** On Qwen3.5-4B (the regime where Scalar TQ-MSE is already lossless), estimator-native ExtRaBitQ at 4-bit K / BlockTQ 4-bit V gives ΔPPL within **+0.2%** of Scalar TQ-MSE 4-bit K/V at matching or lower total bits/val. If we get there, §5's framing holds: RaBitQ is not a dominated reconstruction quantizer, it's a complementary attention-native representation.

**What we are explicitly NOT claiming.** Phase 10 is not claiming that RaBitQ wins. TurboQuant's reconstruction path is genuinely strong at head_dim = 256, and it may remain the best choice for most configurations. Phase 10's claim is narrower: *with* its proper estimator-native path and *with* metadata compression, RaBitQ gets back to parity on the relevant metric (attention quality), instead of being dismissed as "TurboQuant does the same thing but better."

### 5.4 Phase 10 dependencies

```
10.1 Metadata compression    (1 day,  no deps)
      ↓
10.2 Multi-bit estimator val (1 day,  independent of 10.1 but gates 10.3)
      ↓
10.3 Estimator-native attn   (~1 week, depends on 10.2)
      ↓
Acceptance: Qwen3.5-4B evaluation, optional Qwen3.5-27B re-run
```

10.1 and 10.2 can run in parallel. 10.3 waits for 10.2 to pass.

---

## 6. Fairness Guarantees

Invariants enforced by tests so no method gets an unfair advantage:

| Guarantee | Mechanism | Test |
|-----------|-----------|------|
| Same rotation | TurboQuant Π and RaBitQ P produced by the same `haar_rotation(d, seed)` | `tests/test_fairness.py::test_rotation_shared_between_methods` |
| Same data | All methods tested on vectors from the same `np.random.default_rng(seed)` | `tests/test_fairness.py::test_shared_data` |
| Same language | Pure NumPy — no SIMD or GPU advantage for any method | N/A (by convention) |
| Same metrics | All measured by `core/metrics.py` | `tests/test_interface.py` |
| Same bit budget | Compared at equal **effective** bits per vector (incl. metadata) | `eval/compression.py` + §6 of [`REPORT.md`](REPORT.md) |
| Reproducible | All randomness from deterministic seed chain | `tests/test_turboquant_mse.py::test_same_seed_same_output` |
| No silent failures | Every method raises `ValueError` on out-of-spec input | `tests/test_interface.py`, Phase 9.1 regression tests |

---

## 7. Success Criteria

| # | Criterion | Metric | Phase | Status |
|---|-----------|--------|-------|--------|
| S1 | All 8 quantizers pass the interface test | `test_interface.py` | 5 | ✅ |
| S2 | TQ-MSE MSE within 10% of Lloyd-Max bound at $d \geq 128$ | `test_turboquant_mse.py` | 2 | ✅ |
| S3 | TQ-Prod unbiased, variance·$d$ within 15% of Theorem 2 | `test_turboquant_prod.py` | 2 | ✅ |
| S4 | TQ-MSE $b = 1$ bias $\alpha \in [0.62, 0.66]$ | `test_turboquant_mse.py` | 2 | ✅ |
| S5 | RaBitQ 1-bit estimator `\|bias\| < 0.01` | `test_rabitq.py` | 3 | ✅ |
| S6 | ExtRaBitQ $B = 1$ matches `RaBitQ1Bit` exactly | `test_rabitq.py` | 3 | ✅ |
| S7 | Fixed seed → reproducible results | `test_fairness.py` | 5 | ✅ |
| S8 | $d = 512$ single-vector quantize < 1 ms | `eval/speed.py` | 7 | ✅ (0.008 ms) |
| S9 | Metadata overhead < 5% for $n \geq 1000$ | `eval/compression.py` | 5 | ✅ |
| S10 | FWHT ≥ 3× faster than dense at $d \geq 2048$ | `test_perf.py` | 7 | ✅ (at $d \geq 2048$) |
| S11 | Bit-packed storage ≥ 2× smaller than int8 | `test_perf.py` | 7 | ✅ |
| S12 | PyTorch wrapper `batch = 1` integration test | `test_torch_wrapper.py` | 8 | ✅ |
| S13 | Streaming PPL evaluator chunk-invariant (< 1%) | `test_streaming_ppl.py` | 9.0.5 | ✅ |
| S14 | `VQBenchCache` past-lossy / current-exact contract | `test_streaming_ppl.py` | 9.0.5 | ✅ |
| S15 | `BlockTurboQuantMSE` 4-bit Qwen3-4B ΔPPL < 2% | Phase 9.1 run | 9.1 | ✅ (+0.7% at B=16) |
| S16 | Qwen3.5-4B 4-bit K ΔPPL < 0.5% (near-lossless) | Phase 9.1 run | 9.1 | ✅ (+0.0%) |
| S17 | **Qwen3.5-27B (AWQ int4) 4-bit K / 4-bit V ΔPPL < 1%** | Phase 9.2.2 run | 9.2 | **⏳ next** |
| S18 | Hybrid-attention `VQBenchCache` routes correctly | new `test_hybrid_cache.py` | 9.2.1 | ⏳ |
| S19 | Multi-seed variance bars on all headline configs | Phase 9.2.3 | 9.2 | ✅ Qwen3.5-4B headline done 2026-04-11 |
| S20 | Multi-bit `ext_rabitq_ip_estimate` $\|\alpha - 1\| < 0.03$ | `test_rabitq.py` | 9.2.5 / 10.2 | ✅ measured on Qwen3.5-4B 2026-04-11 (regression test pending) |
| S21 | ExtRaBitQ metadata → 32 bits/vec at head_dim = 128 | `eval/compression.py` | 10.1 | ⏳ |
| S22 | Estimator-native attention matches offline estimator on 128-token batch | `test_attention_rabitq.py` | 10.3.4 | ⏳ |
| S23 | K RaBitQ estimator-native 4-bit + V BlockTQ 4-bit on Qwen3.5-4B ΔPPL within +0.2% of TQ-MSE | Phase 10.3.5 | 10.3 | ⏳ |

**P0 critical path today:** S17 → S18 → S19 (Phase 9.2). These are the "benchmark paper is done" items.

**P1 strategic bet:** S20 → S21 → S22 → S23 (Phase 10). These are the "the benchmark is fair to RaBitQ" items.

---

## 8. Dependency Graph

```
Phase 1 (Core)           ✅
  ↓
  ├── Phase 2 (TurboQuant)      ✅
  ├── Phase 3 (RaBitQ)          ✅ (multi-bit estimator unvalidated)
  ├── Phase 4 (PQ / OPQ)        ✅
  │       ↓
  │   Phase 5 (Eval)            ✅
  │       ↓
  │   Phase 6 (KV cache)        ✅
  │       ↓
  │   Phase 7 (Perf)            ✅
  │       ↓
  │   Phase 8 (Torch wrapper)   ✅ (batch = 1, full-attention)
  │       ↓
  │   Phase 9.0 (Norm correction + K-MSE)      ✅
  │       ↓
  │   Phase 9.0.5 (Streaming PPL + bug fix)    ✅
  │       ↓
  │   Phase 9.1 (BlockTQ + Qwen3/3.5-4B)       ✅
  │       ↓
  │   ┌───────────────┬──────────────────┐
  │   ↓               ↓                  ↓
  │ Phase 9.2       Phase 10          Phase 11
  │ (P0 scale)      (P1 RaBitQ         (production
  │  9.2.1 Hybrid    redirection)       path, not
  │  9.2.2 27B run   10.1 Metadata      blocking)
  │  9.2.3 Variance  10.2 Estimator     11.1 batched
  │  9.2.4 LongCtx   10.3 Attn-native   11.2 MLX
  │  9.2.5 MultiBit↔─┘                  11.3 llama.cpp
  │                                      11.4 Triton
```

9.2.5 and 10.2 are the same task filed under both phases — doing it once satisfies both. Phase 9.2 and Phase 10 can proceed in parallel after 9.2.5 passes.

---

## 9. Glossary

- **head_dim** — dimension of each attention head ($d$ in equations). Qwen3 = 128, Qwen3.5 / Gemma-4 = 256.
- **bits/val** — effective bits per dimension, including metadata (norms, scales). Not nominal bit-width.
- **ΔPPL** — `PPL_quantized − PPL_fp16`, reported as absolute or relative.
- **Direct dequant path** — what standard HF attention does: `cache.update()` returns a dense tensor, `Q @ K.T` uses it as-is. Implicit assumption: the returned tensor is exact (it isn't; it's a dequantized approximation).
- **Estimator path** — the explicit paper-specified unbiasedness path. For RaBitQ it's `rabitq_ip_estimate` / `ext_rabitq_ip_estimate`. For TurboQuantProd the QJL residual is applied *inside* `dequantize()`, so the direct and estimator paths coincide.
- **Estimator-native attention** (Phase 10.3) — computing attention scores directly from compressed K via the estimator path, without reconstructing K as a dense tensor. The point of Phase 10.
- **Faithful streaming PPL** — current chunk uses exact K/V, past chunks come from compressed cache. Matches `llama.cpp -ctk turbo*` semantics. In `vqbench/validation/streaming_ppl.py`.
- **Monkey-patching** (fallback) — wrap `k_proj` / `v_proj` to inject `quantize → dequantize` inside the forward pass. Works on any architecture including hybrid attention, but over-estimates degradation because the current chunk is quantized too. In `vqbench/validation/monkey_patch.py`.
- **Norm correction** — for TurboQuantMSE, re-normalize $\hat{y}$ to unit norm before the inverse rotation. Matches `turboquant_plus` production. Improves $\alpha$ at the cost of slightly worse reconstruction MSE.
- **Full attention layer / linear attention layer** — in hybrid architectures like Qwen3.5, only some layers have a KV cache that grows with context. The rest use linear attention with a fixed-size recurrent state. Only full-attention layers are candidates for KV cache compression.
