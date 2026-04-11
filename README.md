# VQBench

> Unified Vector Quantization Benchmark: TurboQuant, RaBitQ, and Product Quantization under a single Python API.
>
> **Status:** Research prototype. Core quantizers are implemented, cross-validated against the reference `turboquant_plus`, and verified on synthetic and real-model data. Faithful streaming PPL is now in place and validated on Qwen3-4B (head_dim = 128) and Qwen3.5-4B (head_dim = 256), with near-lossless 4-bit K/V on the latter. Qwen3.5-27B validation, variance reporting, and task-proxy metrics are the remaining open items — see [`REPORT.md`](REPORT.md).

---

## What's inside

Eight quantizers implementing a common `VectorQuantizer` ABC:

| Family | Methods | Papers |
|---|---|---|
| **TurboQuant** | `TurboQuantMSE`, `QJLQuantizer`, `TurboQuantProd`, `BlockTurboQuantMSE` | Zandieh et al., arXiv 2504.19874 / 2406.03482 |
| **RaBitQ** | `RaBitQ1Bit`, `ExtRaBitQ` | Gao & Long, arXiv 2405.12497 / 2409.09913 |
| **Product Quant** | `ProductQuantizer`, `OptimizedPQ` | Jégou et al., TPAMI 2011; Ge et al., CVPR 2013 |

`BlockTurboQuantMSE` (B = 16 / 32 / 64) extends scalar TurboQuantMSE with per-block fp16 scales — see [`BlockTQ.md`](BlockTQ.md) for the algorithm walkthrough and [`REPORT.md`](REPORT.md) §3.5.5 for the Qwen3-4B quality numbers.

Plus infrastructure:

- `vqbench.core` — Haar rotation, Lloyd-Max codebook, metrics, bit packing, vectorized FWHT
- `vqbench.kv_cache` — pluggable K/V compressor, compressed attention, outlier strategy
- `vqbench.eval` — distortion / bias / recall / speed / compression sweeps
- `vqbench.torch_wrapper` — `QuantizedKVCache` as a drop-in `transformers` 5.5.0 `Cache` subclass
- `vqbench.validation` — faithful streaming PPL (`streaming_ppl.py`), monkey-patch PPL, real K-MSE, WikiText-2 loader
- 186 passing tests cross-validating correctness against paper predictions and `turboquant_plus`

Read [`REPORT.md`](REPORT.md) for the honest benchmark findings, the metric-task mismatch discussion, and the list of what is still missing.

---

## Install

### Core (quantizers, synthetic benchmarks, tests)

```bash
pip install -e ".[test]"
```

This installs NumPy, SciPy, pytest — no PyTorch required.

### Full (with real-model validation)

```bash
pip install -e ".[test,validation]"
```

Adds `torch`, `transformers`, `datasets`, `tokenizers`, `safetensors`.

Tested configuration:

| Package | Tested version |
|---|---|
| Python | 3.13 |
| NumPy | 1.26+ |
| SciPy | 1.12+ |
| PyTorch | 2.10 (`mps` on Apple Silicon) |
| transformers | 5.5.0 |
| datasets | 4.8.3 |

Apple Silicon (M-series) uses `mps`; CUDA is not yet tested but the torch path is device-agnostic.

---

## Running the tests

```bash
python -m pytest vqbench/tests/ -v
```

Expected: **186 passed, 1 skipped**, ≈ 31 seconds on Apple M5 Pro.

Subset runs for faster iteration:

```bash
# Core quantizer correctness only
python -m pytest vqbench/tests/test_turboquant_mse.py vqbench/tests/test_rabitq.py -v

# Interface + fairness guarantees
python -m pytest vqbench/tests/test_interface.py vqbench/tests/test_fairness.py -v

# Real-model torch wrapper (needs [validation] extras)
python -m pytest vqbench/tests/test_torch_wrapper.py -v
```

---

## Reproducing the numbers in REPORT.md

### Table 3.1 — Synthetic normalized MSE at $d = 512$

```bash
python -c "
from vqbench.methods.turboquant.mse import TurboQuantMSE
from vqbench.methods.turboquant.prod import TurboQuantProd
from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
from vqbench.methods.rabitq.rabitq_1bit import RaBitQ1Bit
from vqbench.datasets.synthetic import random_unit_vectors
from vqbench.core.metrics import mse_distortion

D = 512
X = random_unit_vectors(2000, D, seed=0)

for b in [1, 2, 3, 4]:
    q_mse = TurboQuantMSE(d=D, num_bits=b, seed=42, norm_correction=True)
    q_prod = TurboQuantProd(d=D, num_bits=b, seed=42, norm_correction=True)
    q_rabitq = (RaBitQ1Bit(d=D, seed=42) if b == 1
                else ExtRaBitQ(d=D, num_bits=b, seed=42))
    q_rabitq.fit(X)
    print(f'b={b}:',
          f'TQ-MSE={mse_distortion(X, q_mse.dequantize_batch(q_mse.quantize_batch(X))):.4f}',
          f'TQ-Prod={mse_distortion(X, q_prod.dequantize_batch(q_prod.quantize_batch(X))):.4f}',
          f'RaBitQ={mse_distortion(X, q_rabitq.dequantize_batch(q_rabitq.quantize_batch(X))):.4f}')
"
```

Expected output:

```
b=1: TQ-MSE=0.4038 TQ-Prod=1.5696 RaBitQ=0.4037
b=2: TQ-MSE=0.1205 TQ-Prod=0.6342 RaBitQ=0.2636
b=3: TQ-MSE=0.0345 TQ-Prod=0.1892 RaBitQ=0.0589
b=4: TQ-MSE=0.0094 TQ-Prod=0.0541 RaBitQ=0.0135
```

### Table 3.2 — Real-model K-cache MSE on Qwen2.5-1.5B

Requires the `[validation]` extras.

```bash
python -m vqbench.validation.run_quick \
    --model Qwen/Qwen2.5-1.5B \
    --bits 2,3,4 \
    --max-tokens 2048 \
    --k-mse-only
```

The `--k-mse-only` flag skips the pessimistic monkey-patched PPL phase — see §4.7 of REPORT.md for why that path is not a faithful measurement.

First run downloads the model (~3GB) from Hugging Face and caches it locally. Subsequent runs are fast.

### Table 3.3 — Inner-product bias α (direct vs estimator)

```bash
python -c "
import numpy as np
from vqbench.methods.turboquant.mse import TurboQuantMSE
from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
from vqbench.datasets.synthetic import random_unit_vectors
from vqbench.core.metrics import ip_bias

D = 512
X = random_unit_vectors(2000, D, seed=0)
Y = random_unit_vectors(2000, D, seed=1)

for b in [1, 2, 3, 4]:
    q_tqm = TurboQuantMSE(d=D, num_bits=b, seed=42, norm_correction=True)
    alpha_tqm, _ = ip_bias(X, q_tqm.dequantize_batch(q_tqm.quantize_batch(X)), Y)
    if b >= 2:
        q_rq = ExtRaBitQ(d=D, num_bits=b, seed=42); q_rq.fit(X)
        alpha_rq, _ = ip_bias(X, q_rq.dequantize_batch(q_rq.quantize_batch(X)), Y)
        print(f'b={b}: TQ-MSE alpha_direct={alpha_tqm:.3f}  ExtRaBitQ alpha_direct={alpha_rq:.3f}')
    else:
        print(f'b={b}: TQ-MSE alpha_direct={alpha_tqm:.3f}')
"
```

---

## Quick start: use VQBench in your own code

### Quantize a single vector

```python
import numpy as np
from vqbench.methods.turboquant.mse import TurboQuantMSE

q = TurboQuantMSE(d=128, num_bits=3, seed=42)
x = np.random.randn(128)
qv = q.quantize(x)
x_hat = q.dequantize(qv)
print(f'Storage: {q.storage_bits(qv)} bits  (fp32 = {32*128} bits)')
```

### Compress a KV cache

```python
from vqbench.methods.turboquant.mse import TurboQuantMSE
from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE
from vqbench.kv_cache.compressor import KVCacheCompressor

# Recommended K/V strategy (REPORT.md §3.5.2, §3.5.5):
# In the direct-dequant path that a standard attention module uses, both K and
# V are best served by MSE-optimal scalar quantization. TQ-Prod's unbiased-IP
# guarantee only holds in its dedicated estimator path (not exercised by HF
# attention). On Qwen2.5-0.5B, direct-dequant TQ-Prod is 27× worse than TQ-MSE
# at 4-bit K — see REPORT.md §3.5.2c.
#
# K: BlockTurboQuantMSE for outlier-prone head_dim 64-128, scalar TurboQuantMSE
#    for head_dim ≥ 256 where block quantization is unnecessary.
# V: scalar TurboQuantMSE at any head_dim (V compression is essentially free).
key_q = BlockTurboQuantMSE(d=128, num_bits=4, block_size=16, seed=42)
val_q = TurboQuantMSE(d=128, num_bits=4, seed=42)

comp = KVCacheCompressor(key_q, val_q)
comp.compress(keys, values)  # (seq_len, 128) numpy arrays
k_hat = comp.get_keys()
v_hat = comp.get_values()
print(f'Compression ratio: {comp.compression_ratio():.1f}x (vs fp16)')
```

### Drop-in transformers Cache

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from vqbench.torch_wrapper.hook import apply_quantized_cache

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

cache = apply_quantized_cache(
    model,
    method_key="BlockTurboQuantMSE-B16",   # or "TurboQuantMSE" for head_dim ≥ 256
    method_value="TurboQuantMSE",
    num_bits=4,
)

# Note: current integration is batch=1 only; see REPORT.md §2 scope notes.
ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
out = model.generate(ids, max_new_tokens=20, past_key_values=cache)
```

---

## Architecture

![VQBench architecture](VQ.png)

Four layers:

- **Core** — rotation, codebook, metrics, packing, base ABC
- **Methods** — 8 quantizers across three families under a common interface
- **Applications** — KV cache compressor, compressed attention, torch wrapper, eval suite
- **Pipeline** — recommended K/V compression strategy (§3.5.2 and §3.5.5 of REPORT.md)

---

## Known limitations

This is a research prototype. Specifically:

1. **Faithful streaming PPL is in place** (REPORT.md §3.5, `vqbench/validation/streaming_ppl.py`). The older monkey-patched PPL path is still shipped as a pessimistic stress test but is no longer the measurement of record — see REPORT.md §4.7.
2. **HF cache integration processes `batch = 0` only.** Batched generation is not yet supported.
3. **Hybrid-attention models** (Qwen3.5 series — 8 full_attention + 24 linear_attention layers) currently use monkey-patching for the full-attention K projection; a native `VQBenchCache` that passes through linear-attention state is pending.
4. **Multi-bit RaBitQ estimator (`ext_rabitq_ip_estimate`) is present but not validated at $\alpha \approx 1$.** Only the 1-bit estimator is test-verified unbiased.
5. **No variance / confidence intervals on any numbers in REPORT.md.** Single-run point estimates only.
6. **Scale validation now covers Qwen3-4B and Qwen3.5-4B.** Qwen3.5-27B is the final target and is gated only on inference speed (AWQ int4 on MPS, ~4–8 hours for a full sweep). See REPORT.md §3.5.8.
7. **No production quantization path.** No MLX, llama.cpp, CUDA, or Triton kernels. The NumPy → PyTorch bridge has CPU ↔ GPU transfer overhead.
8. **FWHT doesn't beat BLAS at $d \leq 512$** — the vectorized FWHT is correct but NumPy's BLAS matmul is heavily optimized. FWHT's asymptotic $O(d \log d)$ advantage would show at $d \geq 2048$.

All of these are tracked as open items in REPORT.md §6 and [`TODO.md`](TODO.md).

---

## Reference implementation

This framework was developed alongside and cross-validated against [`turboquant_plus`](https://github.com/TheTom/turboquant_plus). The TurboQuant output in VQBench matches `turboquant_plus` to within `1e-6` max absolute difference on identical inputs with identical seeds. See REPORT.md §5 for the full comparison.

VQBench extends `turboquant_plus` with:

- RaBitQ and ExtRaBitQ (turboquant_plus references these but does not implement them)
- PQ and OPQ baselines
- A unified `VectorQuantizer` ABC so all methods can be compared under the same evaluation harness
- transformers 5.5.0 `Cache` protocol integration

`turboquant_plus` remains the reference for the **production** quantization path (llama.cpp cache types, MLX `TurboKVCache`, NIAH benchmarks at scale).

---

## Layout

```
vqbench/
├── core/           base.py, rotation.py, metrics.py, packing.py
├── methods/
│   ├── turboquant/ codebook.py, mse.py, qjl.py, prod.py, block_mse.py
│   ├── rabitq/     rabitq_1bit.py, rabitq_ext.py, estimator.py
│   └── pq/         product_quant.py, opq.py
├── eval/           distortion.py, bias.py, recall.py, speed.py, compression.py
├── kv_cache/       compressor.py, attention.py, outlier.py
├── torch_wrapper/  module.py, hook.py          # transformers 5.5.0 Cache
├── validation/     ppl_eval.py, monkey_patch.py, k_mse.py, run_quick.py, streaming_ppl.py
├── datasets/       synthetic.py, wikitext.py
└── tests/          13 test files, 186 passing tests
```

---

## License

MIT
