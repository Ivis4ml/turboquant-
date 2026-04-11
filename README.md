# VQBench

> Unified vector quantization benchmark for LLM KV-cache compression: TurboQuant, BlockTurboQuant, RaBitQ, and Product Quantization under one Python API, with faithful streaming-PPL evaluation on real HuggingFace models.
>
> **Headline target:** run **Qwen3.5-27B** on a **48 GB M5 Pro** with quantized KV cache.
> **Current status:** Qwen3.5-4B 4-bit K/V is **near-lossless** (+0.0% ΔPPL). 27B is gated only on inference speed (AWQ int4 on MPS). See [`REPORT.md`](REPORT.md).

---

## Documents in this repo

| File | Purpose |
|------|---------|
| [`README.md`](README.md) | Install, quick-start, layout (you are here) |
| [`REPORT.md`](REPORT.md) | Qwen3.5-4B empirical results, 27B projection, honest limitations |
| [`BlockTQ.md`](BlockTQ.md) | Algorithm walkthrough for `BlockTurboQuantMSE` |
| [`PLAN.md`](PLAN.md) | Design principles, phase status, Phase 10 RaBitQ redirection |
| [`scripts/`](scripts/README.md) | Runnable reproductions — every command in the docs maps to a script here |

Start with `REPORT.md` for what's measured, `PLAN.md` for what's next, `BlockTQ.md` if you want to understand why block quantization exists. This file tells you how to install and run. Any `python scripts/foo.py` command in this doc points to a standalone script — no code blocks embedded in markdown.

---

## What's inside

Eight quantizers implementing a common `VectorQuantizer` ABC:

| Family | Methods | Papers |
|--------|---------|--------|
| **TurboQuant** | `TurboQuantMSE`, `TurboQuantProd`, `QJLQuantizer`, `BlockTurboQuantMSE` (B = 16 / 32 / 64) | Zandieh et al., arXiv 2504.19874 / 2406.03482 |
| **RaBitQ** | `RaBitQ1Bit`, `ExtRaBitQ` | Gao & Long, arXiv 2405.12497 / 2409.09913 |
| **Product Quant.** | `ProductQuantizer`, `OptimizedPQ` | Jégou et al., TPAMI 2011; Ge et al., CVPR 2013 |

Plus infrastructure:

- `vqbench.core` — Haar rotation, vectorized FWHT, Lloyd-Max codebook, metrics, bit packing
- `vqbench.kv_cache` — pluggable K/V compressor, compressed attention, outlier strategy
- `vqbench.torch_wrapper` — `QuantizedKVCache` as a drop-in `transformers` 5.5 `Cache` subclass (single-batch, full-attention)
- `vqbench.validation` — faithful streaming PPL, monkey-patch fallback for hybrid-attention models, real K-MSE measurement, WikiText-2 loader
- **199 tests** across 13 test files (170 passed + 1 skipped excluding the two slow files, in ~25 s on M5 Pro)

---

## Install

### Core (quantizers, synthetic benchmarks, tests)

```bash
pip install -e ".[test]"
```

Installs NumPy, SciPy, pytest — no PyTorch required.

### Full (with real-model validation)

```bash
pip install -e ".[test,validation]"
```

Adds `torch`, `transformers`, `datasets`, `tokenizers`, `safetensors`.

Tested configuration:

| Package | Tested version |
|---------|----------------|
| Python | 3.13 |
| NumPy | 1.26+ |
| SciPy | 1.12+ |
| PyTorch | 2.10 (`mps` on Apple Silicon) |
| transformers | 5.5.0 |
| datasets | 4.8.3 |

Reference hardware: **Apple M5 Pro, 48 GB unified memory**. CUDA untested.

---

## Run the tests

```bash
python -m pytest vqbench/tests/ -v
```

- **199 tests collected.** Full run ~31 s on M5 Pro.
- Quick loop (excludes `test_perf.py` and `test_streaming_ppl.py`): **170 passed + 1 skipped**, ~25 s.

```bash
# Correctness only
python -m pytest vqbench/tests/test_turboquant_mse.py vqbench/tests/test_rabitq.py -v

# Interface + fairness guarantees
python -m pytest vqbench/tests/test_interface.py vqbench/tests/test_fairness.py -v

# Real-model torch wrapper (needs [validation] extras)
python -m pytest vqbench/tests/test_torch_wrapper.py -v
```

---

## Three-minute quick-start

Every demo is a script under `scripts/`. Run with `--help` to see configuration.

```bash
# 1. Quantize a single vector (TurboQuantMSE roundtrip, nMSE vs theory)
python scripts/quickstart_quantize_vector.py

# 2. Asymmetric KV cache compressor (auto-picks BlockTQ at head_dim <= 128)
python scripts/quickstart_kv_cache.py
python scripts/quickstart_kv_cache.py --head-dim 128 --seq-len 256

# 3. BlockTurboQuantMSE vs scalar (pure NumPy demo; add --with-hf-cache for HF integration)
python scripts/quickstart_block_turboquant.py

# 4. Drop VQBenchCache into a HuggingFace model (needs [validation] extras + model download)
python scripts/quickstart_transformers_cache.py
python scripts/quickstart_transformers_cache.py --model Qwen/Qwen2.5-1.5B \
    --method-key BlockTurboQuantMSE-B16 --bits 4
```

Scripts 1–3 and `storage_accounting_table.py` are pure NumPy, no model download. Script 4 needs `pip install -e '.[test,validation]'` and downloads the model on first run.

---

## Reproduce the Qwen3.5-4B headline

```bash
python scripts/reproduce_qwen35_4b_headline.py           # Scalar TQ-MSE 4-bit K
python scripts/reproduce_qwen35_4b_headline.py --bits 3  # 3-bit K sweep
python scripts/reproduce_qwen35_4b_headline.py --method BlockTurboQuantMSE --block-size 64
```

Expected (default run): `fp16 baseline ≈ 10.3866  /  TQ-MSE 4-bit K ≈ 10.3866` → ΔPPL ≈ 0 (near-lossless).

Qwen3.5-4B currently uses the **monkey-patch** fallback because `VQBenchCache` does not yet route hybrid-attention updates; a native hybrid cache is Phase 9.2.1 in [`PLAN.md`](PLAN.md).

For the Qwen3.5-27B headline sweep (Phase 9.2.2, needs AWQ int4 checkpoint and ~22 GB free):

```bash
python scripts/run_qwen35_27b_sweep.py --dry-run          # print plan without loading model
python scripts/run_qwen35_27b_sweep.py --bits 4 --max-tokens 1024
```

---

## Architecture

![VQBench architecture](VQ.png)

Four layers:

- **Core** — rotation, codebook, metrics, packing, `VectorQuantizer` ABC
- **Methods** — 8 quantizers across three families under a common interface
- **Applications** — KV cache compressor, compressed attention, torch wrapper, eval suite
- **Pipeline** — streaming PPL evaluator, monkey-patch fallback, K-MSE measurement

---

## Layout

```
turboquant-/
├── README.md                    # this file
├── REPORT.md                    # Qwen3.5 empirical results + 27B projection
├── BlockTQ.md                   # BlockTurboQuantMSE algorithm walkthrough
├── PLAN.md                      # design + Phase 10 RaBitQ redirection
├── VQ.png                       # architecture diagram
├── pyproject.toml
│
├── scripts/                     # runnable reproductions (no code embedded in md)
│   ├── README.md                # script index
│   ├── quickstart_quantize_vector.py
│   ├── quickstart_kv_cache.py
│   ├── quickstart_block_turboquant.py
│   ├── quickstart_transformers_cache.py
│   ├── storage_accounting_table.py
│   ├── synthetic_mse_table.py          # REPORT §5.5
│   ├── real_k_mse_qwen35.py            # REPORT §5.6
│   ├── ip_bias_alpha_qwen35.py         # REPORT §5.7
│   ├── multi_model_headline_sweep.py   # REPORT §4
│   ├── variance_qwen35_4b.py           # REPORT §3.1 (mean ± std)
│   ├── long_context_qwen35_4b.py       # REPORT §3.4 (length sweep)
│   ├── reproduce_qwen35_4b_headline.py # REPORT §3
│   └── run_qwen35_27b_sweep.py         # Phase 9.2.2 (pending AWQ)
│
├── turboquant_plus/             # REFERENCE implementation — do not modify
│                                #   (production C/Metal path; see its own README)
│
└── vqbench/                     # the framework itself
    ├── core/                    # base ABC, rotation, metrics, packing
    ├── methods/
    │   ├── turboquant/          # codebook, mse, prod, qjl, block_mse
    │   ├── rabitq/              # rabitq_1bit, rabitq_ext, estimator
    │   └── pq/                  # product_quant, opq
    ├── eval/                    # distortion, bias, recall, speed, compression
    ├── kv_cache/                # compressor, attention, outlier
    ├── torch_wrapper/           # module.py (QuantizedKVCache), hook.py (VQBenchCache)
    ├── validation/              # streaming_ppl, monkey_patch, k_mse, run_quick, wikitext, ppl_eval
    ├── datasets/                # synthetic, wikitext
    └── tests/                   # 13 files, 199 tests
```

---

## Contributing

Workflow:

1. Read [`REPORT.md`](REPORT.md) for what's measured, then [`PLAN.md`](PLAN.md) §4.2 (Phase 9.2) and §5 (Phase 10) for what's next.
2. Run `python -m pytest vqbench/tests/` — everything should pass before you start.
3. Pick a task from `PLAN.md` §7 (Success Criteria) with status ⏳. If it's P0 (Phase 9.2), coordinate before starting.
4. Follow the existing code style (see `vqbench/methods/turboquant/block_mse.py` as a reference — docstring with paper reference, vectorized batch path, numeric-parity test).
5. Every new quantizer gets at minimum: shape/dtype roundtrip, `storage_bits` formula, monotonicity in `num_bits`, synthetic-data MSE comparison, `ValueError` on out-of-spec input.
6. Open a PR against `main`.

Non-negotiables:

- **Don't break fairness.** Same `(d, seed)` must give the same Haar rotation across all methods. `tests/test_fairness.py::test_rotation_shared_between_methods` enforces this.
- **Don't break `turboquant_plus` parity.** If you touch `methods/turboquant/{codebook,mse,prod,qjl}.py`, `diff` your change against `turboquant_plus/turboquant/` and make sure outputs still match to `1e-6` on identical inputs.
- **Don't modify `turboquant_plus/`.** It's the reference; `.gitignore`'d from the VQBench package.
- **Don't add silent failure paths.** If a quantizer can't handle certain input, it raises `ValueError`. See the 2026-04-10 spring-cleaning commit for the pattern.

Things that are easy to get wrong:

- **Rotation convention for batch vs single.** `self._rotation @ x` (single) vs `X @ self._rotation.T` (batch) — these compute the same thing but it's easy to mix up. `test_turboquant_mse.py` has a "batch matches single" check that catches this.
- **fp16 precision in the faithful streaming path.** Differences below 0.1% are fp16 noise, not a bug. `test_streaming_ppl.py::test_baseline_chunk_invariance` allows 1% tolerance.
- **HF's `labels=input_ids` shortcut.** Internally does `shift_logits / shift_labels`, silently drops one token per chunk boundary. Don't use it; use manual per-token CE.
- **`DynamicCache()` vs `None`.** The old faithful PPL evaluator created a `DynamicCache()` as the baseline. This breaks hybrid-attention models (Qwen3.5). Pass `past_key_values=None` on the first call instead.

---

## License

MIT
