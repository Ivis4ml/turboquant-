# scripts/

Runnable reproductions and quickstart demos for VQBench. Every code block that used to live inside `README.md`, `REPORT.md`, or `BlockTQ.md` has a matching script here so you can copy-paste a command, not a snippet.

## Quickstart (no model download, pure NumPy)

| Script | What it shows | Referenced from |
|--------|---------------|-----------------|
| `quickstart_quantize_vector.py` | TurboQuantMSE single-vector roundtrip, nMSE vs theory | `README.md` §Three-minute quick-start #1 |
| `quickstart_kv_cache.py` | Asymmetric K/V compressor with auto-selection of Block K at head_dim ≤ 128 | `README.md` §Three-minute quick-start #2 |
| `quickstart_block_turboquant.py` | BlockTurboQuantMSE vs scalar on one vector; optional HF cache integration | `BlockTQ.md` §9 |
| `storage_accounting_table.py` | REPORT.md §6 per-vector storage table (fp16 baseline) | `REPORT.md` §6 |
| `synthetic_mse_table.py` | REPORT.md §5.5 synthetic MSE table across methods × bits | `REPORT.md` §5.5 |

Run each without arguments for the default demo; pass `--help` for configuration.

## Real-model reproductions (needs `[validation]` extras + HF model download)

Install once: `pip install -e '.[test,validation]'` (+ `pip install autoawq` for the 27B run).

| Script | Compute | What it shows | Referenced from |
|--------|---------|---------------|-----------------|
| `quickstart_transformers_cache.py` | ~1 min + model download | Drop-in `apply_quantized_cache` on any full-attention HF model | `README.md` §Three-minute quick-start #3 |
| `reproduce_qwen35_4b_headline.py` | ~1 min per config | Qwen3.5-4B K-only or K+V headline PPL (REPORT.md §3) | `README.md` §Reproduce, `REPORT.md` §3 |
| `real_k_mse_qwen35.py` | ~60 s | Offline nMSE on real Qwen3.5-4B K activations (REPORT.md §5.6) | `REPORT.md` §5.6 |
| `ip_bias_alpha_qwen35.py` | ~90 s | IP bias α table on real K + synthetic queries (REPORT.md §5.7) | `REPORT.md` §5.7, `PLAN.md` §5.2 |
| `multi_model_headline_sweep.py` | ~8 min | Head_dim scaling sweep across Qwen2.5-0.5B / Qwen3-4B / Qwen3.5-4B | `REPORT.md` §4 |
| `variance_qwen35_4b.py` | ~5 min | 3-seed variance sweep on the §3.1 headline (gives mean ± std bars) | `REPORT.md` §3.1 |
| `long_context_qwen35_4b.py` | ~15 min | Length sweep 512→4096 tokens (1→15 cache boundaries) | `REPORT.md` §3.4 |
| `diagnose_qwen35_lossless.py` | ~60 s | Verifies hook firings + per-layer K perturbation + raw fp32 PPL diff | `REPORT.md` §3.5 |
| `run_qwen35_27b_sweep.py` | 30–60 min per config | Phase 9.2.2 headline sweep with JSONL checkpointing | `REPORT.md` §7, `PLAN.md` §4.2 |

The Qwen3.5-4B headline script uses monkey-patching because `VQBenchCache` does not yet route hybrid-attention updates (see `PLAN.md` §4.2 task 9.2.1 for the native hybrid cache plan).

## Conventions

- **Seeds**: every script separates `--data-seed` (for synthetic test vectors) from `--quant-seed` (Haar rotation). If you set them equal, the data aligns with the rotation matrix and nMSE degrades by ~30× — the scripts warn you loudly when this happens.
- **Exit codes**: `0` on success, non-zero on error.
- **Reproducibility**: default seeds match the numbers in `REPORT.md` wherever applicable.
- **No .md-embedded code**: if a doc refers to a command, the command runs one of these scripts. No doc contains a Python snippet you have to extract.

## Adding a new script

Pattern:

```python
"""
<One-line purpose>.

Reproduces <doc reference>.

Requires <pure NumPy | [validation] extras>.

Usage:
    python scripts/<name>.py [--flags ...]
"""
from __future__ import annotations
import argparse

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--quant-seed", type=int, default=42)
    # ... rest of args
    args = parser.parse_args()
    # ... work
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

Register the new script in this file's tables and add a reference from the doc section the script reproduces.
