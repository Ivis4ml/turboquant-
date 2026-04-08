# VQBench: Unified Vector Quantization Benchmark — Implementation Plan

> **Goal**: A pure Python/NumPy framework that implements TurboQuant, RaBitQ, and PQ
> under a single `VectorQuantizer` interface, benchmarks them on identical data/hardware/metrics,
> and demonstrates KV-cache compression as the primary application.
>
> This is the single source of truth for all implementation work.

---

## 1. Motivation

Existing comparisons between TurboQuant, RaBitQ, and PQ are unreliable:

| Problem | Source |
|---------|--------|
| TurboQuant paper uses GPU for TurboQuant but CPU for RaBitQ/PQ | Table 2, arXiv 2504.19874 |
| RaBitQ reference implementation is C++, TurboQuant is Python+Metal | gaoj0017/RaBitQ vs TheTom/turboquant_plus |
| RaBitQ author (Gao) publicly questioned TurboQuant's RaBitQ implementation quality | Fig. 5 discussion |
| No existing codebase puts all three under the same API | — |

**Our solution**: same language, same hardware, same evaluation, same random seeds. No implementation bias.

---

## 2. Paper References

| ID | Method | Paper | Venue | arXiv | Core Idea |
|----|--------|-------|-------|-------|-----------|
| P1 | **TurboQuant** | Zandieh, Daliri, Hadian, Mirrokni | ICLR 2026 | 2504.19874 | Random rotation + Lloyd-Max scalar quantization |
| P2 | **RaBitQ** | Gao & Long | SIGMOD 2024 | 2405.12497 | Random rotation + hypercube quantization + unbiased estimator |
| P3 | **Extended RaBitQ** | Gao et al. | SIGMOD 2025 | 2409.09913 | Multi-bit via bit-plane decomposition |
| P4 | **QJL** | Zandieh et al. | 2024 | 2406.03482 | 1-bit sign quantization for unbiased inner product |
| P5 | **PQ** | Jegou, Douze, Schmid | TPAMI 2011 | — | k-means codebook per subspace |
| P6 | **OPQ** | Ge et al. | CVPR 2013 | — | PQ + learned rotation |

### Academic Relationship

TurboQuant and RaBitQ both exploit **random rotation + high-dimensional concentration**:
after multiplying by a Haar-random orthogonal matrix, each coordinate follows
`f_X(t) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-t²)^{(d-3)/2}` on [-1,1], approaching N(0,1/d) for large d.

RaBitQ (2024.05) preceded TurboQuant (2025.04). The core difference is **what happens after rotation**:
- RaBitQ rounds to hypercube vertices (±1/√d) and achieves unbiased IP via a correction estimator.
- TurboQuant applies Lloyd-Max optimal scalar quantization for better MSE, achieving unbiased IP
  only through a two-stage MSE+QJL scheme (Algorithm 2).

---

## 3. Requirements

### 3.1 Functional

| Req | Description |
|-----|-------------|
| F1 | Implement all 7 quantizers conforming to `VectorQuantizer` ABC |
| F2 | Shared random rotation module (Haar + optional FWHT), deterministic from seed |
| F3 | Distortion-rate sweep: MSE and IP distortion at d∈{128,256,512,1536,3072}, b∈{1,2,3,4} |
| F4 | IP bias analysis: fit multiplicative factor α for each method/bit-width |
| F5 | ANN recall@k benchmark on synthetic and GloVe data |
| F6 | Pluggable KV-cache compressor: any VQ method for K, any for V |
| F7 | Outlier channel strategy (2.5-bit / 3.5-bit mixed precision), shared across methods |

### 3.2 Correctness

| Req | Description |
|-----|-------------|
| C1 | TurboQuant_mse MSE within 10% of paper values (Theorem 1) at d ≥ 128 |
| C2 | TurboQuant_prod unbiased: \|mean bias\| < 0.01, variance×d within 15% of paper (Theorem 2) |
| C3 | TurboQuant_mse b=1 bias α ≈ 2/π ≈ 0.637 |
| C4 | RaBitQ estimator unbiased: \|bias\| < 0.01 |
| C5 | ExtRaBitQ B=1 matches RaBitQ 1-bit exactly |
| C6 | All methods reproducible: fixed seed → fixed numerical results within fp64 tolerance |

### 3.3 Non-Functional

| Req | Description |
|-----|-------------|
| N1 | Pure Python/NumPy — no compiled extensions in Phase 1-5 |
| N2 | Single vector quantize at d=512 in < 1ms (NumPy) |
| N3 | Batch 1000 vectors at d=128 in < 50ms (NumPy) |
| N4 | Metadata overhead < 5% of total storage for n ≥ 1000 vectors |

### 3.4 Design Principles
![alt text](VQ.png)

#### D1 — Extensibility: Adding a New Method Touches Zero Existing Code

The framework follows the **Open-Closed Principle**. To add a new quantization method
(e.g., ScaNN, FAISS-IVF, or a future paper):

1. Create `methods/newmethod/newmethod.py`
2. Subclass `VectorQuantizer`, implement 4 abstract methods
3. Register in `methods/__init__.py` (a name → class dict)
4. Done. Evaluation, KV-cache, and all benchmarks pick it up automatically.

No changes to `core/`, `eval/`, `kv_cache/`, or any existing method directory.

```python
# Example: adding a hypothetical "FooQuant" method
# File: methods/fooquant/fooquant.py

from vqbench.core.base import VectorQuantizer, QuantizedVector

class FooQuant(VectorQuantizer):
    """Ref: Smith et al., NeurIPS 2027, arXiv 2601.12345, Algorithm 1."""

    @property
    def name(self) -> str:
        return "FooQuant"

    def quantize(self, x):   ...  # Implement
    def dequantize(self, qv): ...  # Implement
    def storage_bits(self, qv): ...  # Implement
```

**What makes this work**:

- `eval/distortion.py` iterates over a `list[VectorQuantizer]` — it never imports a specific method
- `kv_cache/compressor.py` accepts `VectorQuantizer` instances — any method plugs in
- Tests use a `@pytest.fixture(params=ALL_QUANTIZERS)` pattern — new methods auto-included

#### D2 — Traceability: Every Function Maps to a Paper Formula

Each algorithm file follows a strict convention linking code to math:

```python
def dequantize_qjl(signs, S, gamma):
    """
    QJL dequantization.

    Paper: Zandieh et al., arXiv 2406.03482, Definition 1
    Formula: x̃ = √(π/2) / d · γ · Sᵀ · sign(S·r)

    Args:
        signs: sign(S·r) ∈ {-1, +1}^d           ← "Q_qjl(r)" in paper
        S: projection matrix S_{ij} ~ N(0,1)     ← "S" in Definition 1
        gamma: ‖r‖₂                              ← "γ" in Algorithm 2
    Returns:
        x̃_qjl ∈ R^d                              ← "Q⁻¹_qjl" in paper
    """
    d = len(signs)
    # √(π/2) / d · γ · Sᵀ · z          ← Definition 1, dequantization
    return np.sqrt(np.pi / 2) / d * gamma * (S.T @ signs.astype(np.float64))
```

**Convention**:

| Element | Rule |
|---------|------|
| Docstring first line | `Paper: <authors>, <arXiv ID>, <Section/Theorem/Definition>` |
| Formula line | The exact mathematical expression being implemented |
| Inline comments | `← Eq. (7)` or `← Theorem 1` pointing to the specific formula |
| Variable names | Match paper notation where possible (`gamma` not `residual_norm`) |
| Constants | Named constants with paper reference (`BIAS_1BIT = 2/π  # §3.2 Eq.(12)`) |

This means: any reviewer can open a function, see which paper formula it implements,
and verify correctness by reading the two side by side.

#### D3 — Modularity: Each Algorithm is Self-Contained

Each method directory is an independent module:

- **No cross-method imports**: `rabitq/` never imports from `turboquant/`. Shared code lives in `core/`.
- **Internal layering**: complex methods decompose into single-responsibility files
  (e.g., TurboQuant: `codebook.py` → `mse.py` → `qjl.py` → `prod.py`)
- **Testable in isolation**: each method has its own test file; you can run
  `pytest tests/test_rabitq.py` without any TurboQuant code.

---

## 4. Architecture

```
vqbench/
├── core/
│   ├── base.py                    # VectorQuantizer ABC, QuantizedVector
│   ├── rotation.py                # Haar random rotation, FWHT
│   └── metrics.py                 # MSE, IP distortion, bias, recall@k
│
├── methods/
│   ├── __init__.py                # Registry: QUANTIZERS = {"TurboQuantMSE": ..., ...}
│   │
│   ├── turboquant/
│   │   ├── __init__.py
│   │   ├── codebook.py            # Lloyd-Max on Lemma 1 density f_X
│   │   ├── mse.py                 # Algorithm 1: TurboQuantMSE
│   │   ├── prod.py                # Algorithm 2: TurboQuantProd (MSE + QJL)
│   │   └── qjl.py                 # QJL 1-bit quantizer (building block)
│   │
│   ├── rabitq/
│   │   ├── __init__.py
│   │   ├── rabitq_1bit.py         # RaBitQ: hypercube + unbiased estimator
│   │   ├── rabitq_ext.py          # Extended RaBitQ: B-bit bit-plane decomposition
│   │   └── estimator.py           # Unbiased distance/IP estimator (popcount-based)
│   │
│   └── pq/
│       ├── __init__.py
│       ├── product_quant.py       # Standard PQ (k-means codebook)
│       └── opq.py                 # Optimized PQ (learned rotation)
│
├── eval/
│   ├── distortion.py              # MSE & IP distortion sweep
│   ├── bias.py                    # IP bias analysis (α fitting)
│   ├── recall.py                  # ANN recall@k
│   ├── speed.py                   # Quantize/dequantize/IP speed
│   └── compression.py             # True storage cost with metadata
│
├── kv_cache/
│   ├── compressor.py              # Pluggable KVCacheCompressor
│   ├── attention.py               # Attention on compressed KV
│   └── outlier.py                 # Outlier channel strategy
│
├── datasets/
│   ├── synthetic.py               # Random unit vectors, controlled IP
│   ├── glove.py                   # GloVe 200d
│   └── openai.py                  # OpenAI 1536d/3072d (optional)
│
├── tests/
│   ├── test_codebook.py           # Centroids match paper values
│   ├── test_turboquant_mse.py     # Theorem 1 validation
│   ├── test_turboquant_prod.py    # Theorem 2 validation
│   ├── test_qjl.py               # QJL unbiasedness + variance
│   ├── test_rabitq.py             # RaBitQ 1-bit + ExtRaBitQ
│   ├── test_pq.py                 # PQ + OPQ baseline
│   ├── test_interface.py          # All quantizers satisfy VectorQuantizer
│   └── test_fairness.py           # Same seed → same rotation, same data
│
├── notebooks/
│   ├── 01_distortion_comparison.ipynb
│   ├── 02_bias_analysis.ipynb
│   ├── 03_recall_benchmark.ipynb
│   └── 04_kv_cache_demo.ipynb
│
└── pyproject.toml
```

### Unified Interface

```python
class VectorQuantizer(ABC):
    def __init__(self, d: int, num_bits: int, seed: int = 42): ...

    @abstractmethod
    def quantize(self, x: np.ndarray) -> QuantizedVector: ...

    @abstractmethod
    def dequantize(self, qv: QuantizedVector) -> np.ndarray: ...

    @abstractmethod
    def storage_bits(self, qv: QuantizedVector) -> int: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    def quantize_batch(self, X: np.ndarray) -> list[QuantizedVector]:
        """Default: loop. Override for vectorized implementations."""
        return [self.quantize(x) for x in X]


@dataclass
class QuantizedVector:
    indices: np.ndarray        # Quantization indices
    signs: np.ndarray | None   # QJL sign bits (TurboQuant_prod only)
    norms: np.ndarray          # Stored scalars (norms, gamma, etc.)
    metadata: dict             # Method-specific (ip_coeff, scale, etc.)
```

All 7 quantizers — `TurboQuantMSE`, `TurboQuantProd`, `QJLQuantizer`, `RaBitQ1Bit`,
`ExtRaBitQ`, `ProductQuantizer`, `OptimizedPQ` — implement this interface.

### Method Registry (`methods/__init__.py`)

```python
from vqbench.methods.turboquant.mse import TurboQuantMSE
from vqbench.methods.turboquant.prod import TurboQuantProd
from vqbench.methods.turboquant.qjl import QJLQuantizer
from vqbench.methods.rabitq.rabitq_1bit import RaBitQ1Bit
from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
from vqbench.methods.pq.product_quant import ProductQuantizer
from vqbench.methods.pq.opq import OptimizedPQ

# Central registry — eval/ and kv_cache/ import from here, never from individual methods
QUANTIZERS: dict[str, type[VectorQuantizer]] = {
    cls.name.fget(None): cls   # or simply a manual dict
    for cls in [TurboQuantMSE, TurboQuantProd, QJLQuantizer,
                RaBitQ1Bit, ExtRaBitQ, ProductQuantizer, OptimizedPQ]
}

def get_all_quantizers(d, num_bits, seed=42) -> list[VectorQuantizer]:
    """Instantiate all registered quantizers with same params."""
    return [cls(d=d, num_bits=num_bits, seed=seed) for cls in QUANTIZERS.values()]
```

**Adding a new method**: add one import + one entry to the dict. Zero changes elsewhere.

---

## 5. Method-by-Method Specifications

### 5.1 Shared: Random Rotation (`core/rotation.py`)

Both TurboQuant and RaBitQ rotate by a Haar-random orthogonal matrix. They share this module.

```python
def haar_rotation(d: int, seed: int) -> np.ndarray:
    """
    Haar-random orthogonal Π ∈ R^{d×d} via QR of Gaussian matrix.
    Fix sign ambiguity: Q = Q @ diag(sign(diag(R))) for proper Haar distribution.
    """

def fast_walsh_hadamard(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """O(d log d) structured rotation: D1·H·D2·H·D3. Optional optimization."""
```

**Critical invariant**: same `(d, seed)` → identical matrix for TurboQuant and RaBitQ.

### 5.2 Shared: Coordinate Distribution (Lemma 1)

After rotation, each coordinate follows:

```
f_X(t) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-t²)^{(d-3)/2},  t ∈ [-1, 1]
```

Connection to Beta: if U ~ Beta((d-1)/2, (d-1)/2) on [0,1], then X = 2U-1 has this density.
For d ≥ 128: f_X ≈ N(0, 1/d). Coordinates are near-independent.

This distribution is used by TurboQuant for Lloyd-Max codebook design.
RaBitQ doesn't need it explicitly (hypercube quantization doesn't depend on the density shape).

---

### 5.3 TurboQuant_mse — Algorithm 1 (P1)

**Goal**: Minimize MSE ‖x − x̃‖². All b bits for scalar quantization.

**Files**: `methods/turboquant/codebook.py`, `methods/turboquant/mse.py`

#### Pipeline

```
x ∈ R^d → store ‖x‖ → x̂ = x/‖x‖ → y = Π·x̂ → idx[j] = nearest centroid(y[j]) → store idx
DeQuant: ỹ[j] = codebook[idx[j]] → x̃ = ‖x‖ · Πᵀ·ỹ
```

#### Codebook (Lloyd-Max on f_X)

Solve continuous 1D k-means on f_X (or N(0,1/d) for d ≥ 128):
1. Initialize 2^b centroids on [-3/√d, 3/√d]
2. Decision boundaries = midpoints
3. Centroids = conditional means via numerical integration on f_X
4. Iterate until |Δc| < 1e-12

Precomputed centroids (×√d):

| b | Centroids (×√d) |
|---|-----------------|
| 1 | ±0.7979 |
| 2 | ±0.4528, ±1.5104 |
| 3 | 8 values (solve numerically) |
| 4 | 16 values (solve numerically) |

#### Storage

`b·d` bits (indices) + 16 bits (fp16 norm) per vector.

#### MSE Guarantee (Theorem 1)

D_mse ≤ (√(3π)/2) · 4^{-b} ≈ 2.7 · 4^{-b}

| b | Paper MSE | Lower Bound (4^{-b}) | Upper Bound |
|---|-----------|---------------------|-------------|
| 1 | 0.36 | 0.25 | 0.384 |
| 2 | 0.117 | 0.0625 | 0.096 |
| 3 | 0.03 | 0.0156 | 0.024 |
| 4 | 0.009 | 0.0039 | 0.006 |

#### Inner Product Bias

**Biased**: E[⟨y, x̃⟩] = α · ⟨y, x⟩ where α < 1.

| b | α (multiplicative bias) |
|---|------------------------|
| 1 | 2/π ≈ 0.637 (35% bias) |
| 2 | ≈ 0.883 (12% bias) |
| 3 | ≈ 0.974 (2.6% bias) |
| 4 | ≈ 0.991 (0.9% bias) |

This bias is the fundamental reason Algorithm 2 exists.

#### Implementation Notes

- Quantization via `np.searchsorted` on sorted codebook (vectorized, no Python loop over coords)
- Bottleneck: O(d²) matmul for rotation → use `np.dot` (BLAS)
- Orthogonality preserves MSE: ‖x−x̃‖² = ‖y−ỹ‖²

---

### 5.4 QJL — Quantized Johnson-Lindenstrauss (P4)

**Goal**: 1-bit quantization with unbiased inner product recovery. Building block for TurboQuant_prod.

**File**: `methods/turboquant/qjl.py`

#### Pipeline

```
Quantize:   qjl = sign(S · r),  S ∈ R^{d×d}, S_{ij} ~ N(0,1), independent from Π
DeQuant:    r̃ = √(π/2)/d · γ · Sᵀ · qjl,  where γ = ‖r‖
```

#### Mathematical Properties

- **Unbiased**: E[⟨y, r̃⟩] = ⟨y, r⟩ (Lemma 4 of P1)
- **Variance**: Var[⟨y, r̃⟩] ≤ (π/2d) · ‖y‖² · ‖r‖²
- **Storage**: d bits (signs) + 1 scalar (γ)

#### Key Constant

```python
QJL_CONST = np.sqrt(np.pi / 2)  # ≈ 1.2533
```

---

### 5.5 TurboQuant_prod — Algorithm 2 (P1)

**Goal**: Unbiased inner product estimation. (b-1) bits MSE + 1 bit QJL on residual.

**File**: `methods/turboquant/prod.py`

**Depends on**: `mse.py` (for MSE stage), `qjl.py` (for QJL stage)

#### Pipeline

```
x → MSE quantize at (b-1) bits → x̃_mse
  → residual r = x - x̃_mse, store γ = ‖r‖
  → QJL on r/γ: qjl = sign(S · r/γ)
  → store (idx, qjl, γ)

DeQuant: x̃ = x̃_mse + √(π/2)/d · γ · Sᵀ · qjl
```

#### Unbiasedness Proof (Theorem 2)

```
E[⟨y, x̃⟩ | x̃_mse]
  = ⟨y, x̃_mse⟩ + E[⟨y, r̃_qjl⟩ | x̃_mse]
  = ⟨y, x̃_mse⟩ + ⟨y, r⟩           ← QJL unbiasedness
  = ⟨y, x̃_mse + r⟩ = ⟨y, x⟩  ✓
```

#### IP Distortion (Theorem 2)

D_prod ≤ (π/2d) · ‖y‖² · D_mse(b-1) ≤ (√(3π)/2) · (‖y‖²/d) · 4^{-b}

| b | D_prod × d | Paper Value |
|---|-----------|-------------|
| 1 | 1.57 | 1.57 |
| 2 | 0.56 | 0.56 |
| 3 | 0.18 | 0.18 |
| 4 | 0.047 | 0.047 |

#### Special Case b=1

MSE stage = 0 bits (no quantization, x̃_mse = 0). Entire budget → QJL on x itself.
Degenerates to pure QJL. MSE ≈ 1.0 (terrible), but IP is unbiased with Var×d ≈ 1.57.

#### Storage

`(b-1)·d` bits (MSE indices) + `d` bits (QJL signs) + 16 bits (norm) + 16 bits (γ)
= `b·d + 32` bits per vector.

**Important**: S matrix (d²×32 bits) is shared across all vectors, amortized.

---

### 5.6 RaBitQ 1-Bit (P2)

**Goal**: 1-bit quantization with natively unbiased distance estimation via correction estimator.

**Files**: `methods/rabitq/rabitq_1bit.py`, `methods/rabitq/estimator.py`

#### Pipeline

```
x ∈ R^d
  → Centroid subtraction: o = x - c   (c = mean of database, computed once)
  → Normalize: ō = o/‖o‖, store ‖o‖
  → Rotate: y = P·ō
  → Hypercube quantize: signs = sign(y) ∈ {-1,+1}^d
  → Normalize: x̄ = signs/√d           (unit hypercube vertex)
  → Store: ip_coeff = ⟨x̄, y⟩ = Σ(signs·y)/√d
  → Store: sign bits (d bits), ‖o‖ (fp32), ip_coeff (fp32)

DeQuant: ỹ = signs/√d → x̃ = ‖o‖ · Pᵀ·ỹ + c
```

#### Centroid Subtraction (RaBitQ-Specific)

RaBitQ subtracts the dataset centroid before quantization:
- `o_i = x_i - c` where `c = mean(X_database)`
- This centers the distribution and improves quantization quality
- **For VQBench synthetic benchmarks**: random unit vectors have c ≈ 0, effect is negligible
- **For real data**: can be significant. Must include for faithful RaBitQ implementation.

#### Unbiased Distance Estimator (Core Innovation)

Direct ⟨y, x̃⟩ is biased (same problem as TurboQuant_mse). RaBitQ's solution: **correct via ip_coeff**.

```python
def rabitq_ip_estimate(query, qv_x, P, centroid, d):
    """Unbiased estimate of ⟨query-c, x-c⟩."""
    q_rot = P @ (query - centroid)
    x_bar = qv_x.indices / np.sqrt(d)       # ±1/√d
    raw_ip = np.dot(q_rot, x_bar)
    ip_coeff = qv_x.metadata['ip_coeff']    # concentrated around √(2/π)
    return qv_x.norms[0] * raw_ip / ip_coeff

def rabitq_dist_estimate(query, qv_x, P, centroid, d):
    """‖q - x‖² via unbiased IP."""
    q_off = query - centroid
    ip_est = rabitq_ip_estimate(query, qv_x, P, centroid, d)
    return np.linalg.norm(q_off)**2 + qv_x.norms[0]**2 - 2 * ip_est
```

**Why it works**: `ip_coeff = ⟨x̄, Pō⟩`. Dividing by ip_coeff exactly cancels the
quantization-induced multiplicative bias. E[estimate] = ⟨q-c, x-c⟩.

**Computational shortcut**: `⟨x̄, q_rot⟩` where x̄ ∈ {±1/√d}^d reduces to:
`(1/√d) · (d - 2 · popcount(b_x XOR b_q))` — computable via bitwise ops + popcount.

#### Key Difference from TurboQuant at 1-Bit

| Aspect | TurboQuant_mse | RaBitQ |
|--------|---------------|--------|
| Centroid per coord | ±√(2/(πd)) ≈ ±0.7979/√d | ±1/√d |
| Centroid optimality | Lloyd-Max optimal | Hypercube (not optimal) |
| MSE | **0.36** (better) | ~0.41 (worse) |
| IP estimation | Biased (α = 2/π) | **Unbiased** (via estimator) |
| Extra metadata | None | ip_coeff (fp32) |

#### Storage

`d` bits (signs) + 32 bits (fp32 ‖o‖) + 32 bits (fp32 ip_coeff) = `d + 64` bits per vector.
Shared: centroid c (d×32 bits, amortized) + rotation P (d²×32 bits, amortized).

---

### 5.7 Extended RaBitQ — Multi-Bit (P3)

**Goal**: Extend RaBitQ to B-bit with bit-plane decomposition, preserving unbiased estimation.

**File**: `methods/rabitq/rabitq_ext.py`

**Depends on**: `estimator.py`

#### Pipeline

```
x → centroid subtraction → normalize → rotate (same as RaBitQ 1-bit)
  → B-bit integer quantize: idx[j] ∈ {0, ..., 2^B - 1} per coordinate
  → Using shifted integer codebook: c_k = (2k - 2^B + 1)
  → Store: B·d bits (indices), ‖o‖, ip_coeff, scale/offset params
```

#### Codebook: Why NOT Lloyd-Max

RaBitQ's unbiased estimator requires the codebook to have **linear algebraic structure**.
Specifically, a B-bit code decomposes into B binary layers:

```python
def decompose_to_bit_planes(indices, B):
    """B-bit index → B binary planes, each ∈ {-1,+1}^d."""
    planes = []
    for k in range(B):
        plane = 2 * ((indices >> k) & 1).astype(np.int8) - 1
        planes.append(plane)
    return planes
```

The B-bit IP estimate = weighted sum of B binary IP estimates:
```
⟨ō, q⟩ ≈ Σ_{k=0}^{B-1} 2^k · ⟨tilde_o^(k), q_rot⟩
```

Each `⟨tilde_o^(k), q_rot⟩` computable via popcount(XOR) — reuses 1-bit fast path B times.

Lloyd-Max centroids do not decompose into bit planes. Using them would break unbiased estimation.

#### Codebook Construction

```python
def ext_rabitq_codebook(B):
    levels = np.arange(2**B)
    centered = 2 * levels - (2**B - 1)  # B=2: [-3, -1, 1, 3]
    return centered
```

#### Error Bounds

- **Unbiased** IP estimation for all B
- Variance ∝ O(4^{-B}/d) — exponential improvement with B
- MSE slightly worse than TurboQuant (non-Lloyd-Max):

| B | ExtRaBitQ MSE (predicted) | TurboQuant_mse MSE |
|---|--------------------------|-------------------|
| 1 | ~0.41 | 0.36 |
| 2 | ~0.13 | 0.117 |
| 3 | ~0.04 | 0.03 |
| 4 | ~0.012 | 0.009 |

#### Invariant

B=1 must produce identical results to RaBitQ1Bit.

#### Storage

`B·d` bits (indices) + 32+32+32+32 bits (norm + ip_coeff + scale + offset) per vector.

---

### 5.8 Product Quantization — PQ (P5)

**Goal**: Offline baseline. k-means codebook per subspace. Data-dependent.

**File**: `methods/pq/product_quant.py`

#### Pipeline

```
TRAINING (offline):
  X_train ∈ R^{n×d} → split into m subspaces of d/m dims
  → k-means per subspace with k=2^b centroids → codebooks C_i

QUANTIZE:
  x → split [x_1,...,x_m] → idx_i = nearest centroid in C_i → store [idx_1,...,idx_m]

DEQUANT:
  → C_i[idx_i] for each subspace → concatenate → x̃
```

#### Design Choices for Fair Comparison

PQ's bit budget works differently. To compare at "b bits per dimension":
- Total bits = b·d
- Split: m subspaces, each gets `b·d/m` bits → `k = 2^(b·d/m)` centroids
- Default: m = d/4 (4-dim subspaces), then k = 2^(4b)

| d | b | m | k per subspace | Total bits/vector |
|---|---|---|---------------|-------------------|
| 128 | 2 | 32 | 256 | 256 |
| 128 | 3 | 32 | 4096 | 384 |
| 512 | 2 | 128 | 256 | 1024 |

#### IP Estimation via ADC

```python
def ip_adc(query, qv, codebooks, m):
    """Asymmetric Distance Computation — precompute lookup table."""
    d_sub = len(query) // m
    total = 0.0
    for i in range(m):
        q_sub = query[i*d_sub:(i+1)*d_sub]
        lut = codebooks[i] @ q_sub          # precompute all k dot products
        total += lut[qv.indices[i]]          # single lookup
    return total
```

#### Properties

- **Biased** IP (k-means approximation, not corrected)
- **No theoretical error bound** (data-dependent)
- **Requires training** — only offline method in the benchmark
- Competitive on trained distribution, worse on OOD

#### Storage

Per vector: `m · ceil(log2(k))` bits + 16 bits (norm).
Shared: `m · k · (d/m) · 32` bits (codebooks, amortized).

---

### 5.9 Optimized PQ — OPQ (P6)

**Goal**: PQ with learned rotation to reduce quantization error.

**File**: `methods/pq/opq.py`

**Depends on**: `product_quant.py`

#### Training (Alternating Optimization)

1. Initialize R = I
2. Repeat (10-20 iters):
   a. Fix R → train PQ on R·X_train
   b. Fix codebooks → optimize R via Procrustes: R = U·Vᵀ from SVD(X̃ᵀ·X)
3. Store R and codebooks

#### Value

OPQ vs PQ shows the benefit of learned rotation for data-dependent methods.
TurboQuant/RaBitQ use random rotations and are data-oblivious — different tradeoff.

---

## 6. Quantization Strategy Comparison (Summary)

```
                TurboQuant_mse    TurboQuant_prod    RaBitQ 1-bit      ExtRaBitQ          PQ/OPQ
                ──────────────    ───────────────    ────────────      ─────────          ──────
Rotation        Haar Π            Haar Π + QJL S     Haar P            Haar P             None / learned R
Preproc         None              None               Centroid sub      Centroid sub       k-means training
Quantization    Lloyd-Max scalar  (b-1) MSE + QJL    sign → ±1/√d     B-bit integer      k-means/subspace
Codebook        Continuous opt    Continuous opt      Hypercube         Integer grid       Data-dependent
IP Bias         Biased (α<1)     Unbiased            Unbiased*         Unbiased*          Biased
IP Mechanism    Direct ⟨y,x̃⟩     QJL correction     ip_coeff corr     Bit-plane corr     ADC lookup
Online?         Yes               Yes                 Yes               Yes                No (offline)
MSE Quality     Best              Good                Worst (1-bit)     Moderate           Data-dependent
```

\* Via dedicated estimator, not via direct ⟨y, x̃⟩.

---

## 7. Key Mathematical Constants

```python
import numpy as np

# === Shared ===
COORD_STD = lambda d: 1.0 / np.sqrt(d)                    # σ of each rotated coord
MSE_LOWER_BOUND = lambda b: 4 ** (-b)                      # Theorem 3
IP_LOWER_BOUND = lambda b, d: (1.0 / d) * 4 ** (-b)       # Theorem 3

# === TurboQuant ===
UPPER_BOUND_FACTOR = np.sqrt(3 * np.pi) / 2                # ≈ 2.7207
TURBO_MSE_PAPER = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
TURBO_IP_PAPER_D = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047} # multiply by 1/d
BIAS_1BIT = 2 / np.pi                                      # ≈ 0.6366

# TurboQuant Lloyd-Max centroids (×√d)
TURBO_CENTROIDS_SCALED = {
    1: np.array([-0.7979, 0.7979]),
    2: np.array([-1.5104, -0.4528, 0.4528, 1.5104]),
}

# === QJL ===
QJL_CONST = np.sqrt(np.pi / 2)                             # ≈ 1.2533

# === RaBitQ ===
RABITQ_CENTROID_1BIT = lambda d: 1.0 / np.sqrt(d)          # ±1/√d
RABITQ_EXPECTED_IP = np.sqrt(2 / np.pi)                     # ≈ 0.7979 (ip_coeff concentration)
```

---

## 8. Expected Benchmark Results

### 8.1 MSE Distortion

| b | TurboQuant_mse | ExtRaBitQ | RaBitQ (1-bit) | Lower Bound |
|---|---------------|-----------|----------------|-------------|
| 1 | **0.36** | — | ~0.41 | 0.25 |
| 2 | **0.117** | ~0.13 | — | 0.0625 |
| 3 | **0.03** | ~0.04 | — | 0.0156 |
| 4 | **0.009** | ~0.012 | — | 0.0039 |

TurboQuant_mse wins MSE (Lloyd-Max is optimal scalar quantizer).

### 8.2 Inner Product Bias (α)

| b | TQ_mse | TQ_prod | RaBitQ/ExtRaBitQ | PQ |
|---|--------|---------|-----------------|-----|
| 1 | 0.637 | **1.000** | **1.000** | biased |
| 2 | 0.883 | **1.000** | **1.000** | biased |
| 3 | 0.974 | **1.000** | **1.000** | biased |
| 4 | 0.991 | **1.000** | **1.000** | biased |

RaBitQ natively unbiased. TurboQuant needs Prod variant.

### 8.3 Storage (bits per vector, d=128)

| Method | b=2 | b=3 | b=4 | Metadata |
|--------|-----|-----|-----|----------|
| TQ_mse | 272 | 400 | 528 | 16 (norm fp16) |
| TQ_prod | 288 | 416 | 544 | 32 (norm+γ fp16) |
| RaBitQ 1-bit | 192 | — | — | 64 (norm+coeff fp32) |
| ExtRaBitQ | 320 | 448 | 576 | 128 (norm+coeff+scale+offset) |
| PQ (m=32,k=256) | 256 | 384 | 512 | 16 + codebook |

---

## 9. Implementation Roadmap

### Phase 1: Core Framework

| Task | File | Description | Depends |
|------|------|-------------|---------|
| 1.0 | `core/base.py` | VectorQuantizer ABC + QuantizedVector | — |
| 1.1 | `core/rotation.py` | Haar rotation + FWHT | — |
| 1.2 | `core/metrics.py` | MSE, IP distortion, bias, recall@k | — |
| 1.3 | `datasets/synthetic.py` | Random unit vectors, controlled IP | — |
| 1.4 | `pyproject.toml` | Project config, deps (numpy, scipy) | — |

### Phase 2: TurboQuant

| Task | File | Description | Depends |
|------|------|-------------|---------|
| 2.1 | `methods/turboquant/codebook.py` | Lloyd-Max on f_X, precompute b=1..4 | 1.0 |
| 2.2 | `methods/turboquant/mse.py` | TurboQuantMSE class | 1.0, 1.1, 2.1 |
| 2.3 | `methods/turboquant/qjl.py` | QJL quantizer | 1.0 |
| 2.4 | `methods/turboquant/prod.py` | TurboQuantProd class | 2.2, 2.3 |
| 2.5 | `tests/test_codebook.py` | Centroids match paper | 2.1 |
| 2.6 | `tests/test_turboquant_mse.py` | Theorem 1 validation | 2.2 |
| 2.7 | `tests/test_turboquant_prod.py` | Theorem 2 validation | 2.4 |
| 2.8 | `tests/test_qjl.py` | QJL unbiasedness + variance | 2.3 |

### Phase 3: RaBitQ

| Task | File | Description | Depends |
|------|------|-------------|---------|
| 3.1 | `methods/rabitq/estimator.py` | Unbiased IP/distance estimator | 1.0 |
| 3.2 | `methods/rabitq/rabitq_1bit.py` | RaBitQ1Bit class | 1.0, 1.1, 3.1 |
| 3.3 | `methods/rabitq/rabitq_ext.py` | ExtRaBitQ class (bit-plane decomp) | 3.1, 3.2 |
| 3.4 | `tests/test_rabitq.py` | 1-bit + ExtRaBitQ validation | 3.2, 3.3 |

### Phase 4: PQ Baseline

| Task | File | Description | Depends |
|------|------|-------------|---------|
| 4.1 | `methods/pq/product_quant.py` | ProductQuantizer + ADC | 1.0 |
| 4.2 | `methods/pq/opq.py` | OptimizedPQ (Procrustes) | 4.1 |
| 4.3 | `tests/test_pq.py` | PQ + OPQ tests | 4.1, 4.2 |

### Phase 5: Unified Evaluation

| Task | File | Description | Depends |
|------|------|-------------|---------|
| 5.1 | `eval/distortion.py` | MSE & IP sweep: d∈{128..3072}, b∈{1..4} | All methods |
| 5.2 | `eval/bias.py` | Fit α for each method/bit-width | All methods |
| 5.3 | `eval/recall.py` | ANN recall@k on synthetic + GloVe | All methods, datasets |
| 5.4 | `eval/speed.py` | Quantize/dequantize/IP timing | All methods |
| 5.5 | `eval/compression.py` | True storage with metadata | All methods |
| 5.6 | `tests/test_interface.py` | All 7 quantizers satisfy ABC | All methods |
| 5.7 | `tests/test_fairness.py` | Same seed/data/evaluation | All methods |

### Phase 6: KV Cache Application

| Task | File | Description | Depends |
|------|------|-------------|---------|
| 6.1 | `kv_cache/compressor.py` | Pluggable KVCacheCompressor | All methods |
| 6.2 | `kv_cache/attention.py` | Attention on compressed KV | 6.1 |
| 6.3 | `kv_cache/outlier.py` | Outlier channel strategy (2.5/3.5-bit) | 6.1 |
| 6.4 | `tests/test_kv_cache.py` | End-to-end attention test | 6.1, 6.2 |

### Phase 7: Performance Optimization

| Task | File | Description | Depends |
|------|------|-------------|---------|
| 7.1 | `core/rotation.py` | FWHT replaces dense Π: O(d²)→O(d log d) | 1.1 |
| 7.2 | All methods | Vectorized batch quantization | All methods |
| 7.3 | `core/packing.py` | Bit packing: b-bit→uint8, signs→bitfield | All methods |

### Phase 8: Integration Targets (Future)

| Task | Description | Depends |
|------|-------------|---------|
| 8.1 | PyTorch wrapper (torch.nn.Module) | Phase 1-6 |
| 8.2 | llama.cpp C port (AVX2/NEON) | Phase 7 |
| 8.3 | MLX / Metal port | Phase 7 |

### Phase 9: Real Model Validation

End-to-end KV-cache compression on production LLMs — the ultimate acceptance test.

| Task | Description | Depends |
|------|-------------|---------|
| 9.1 | **Qwen3.5-27B** KV-cache compression | 8.1 (PyTorch wrapper) |
| 9.2 | **Gemma-4** KV-cache compression | 8.1 (PyTorch wrapper) |
| 9.3 | Cross-model comparison report | 9.1, 9.2 |

#### 9.1 / 9.2 — Per-Model Evaluation

For each model (Qwen3.5-27B, Gemma-4):

1. **Extract KV cache** from a representative prompt set (long-context tasks)
2. **Compress** with each method at b ∈ {2, 3, 4} using the pluggable KVCacheCompressor
3. **Measure compression metrics**:
   - KV-cache memory reduction (GB)
   - Quantize/dequantize throughput (tokens/s)
4. **Measure quality metrics**:
   - Attention output MSE vs full precision
   - Perplexity on eval set (WikiText-2, C4)
   - Downstream task accuracy (MMLU, HumanEval, or model-appropriate benchmarks)
5. **Recommended config** per model: which method × bit-width gives the best quality/compression tradeoff

#### Model-Specific Considerations

| Model | head_dim | num_kv_heads | Context | Notes |
|-------|----------|-------------|---------|-------|
| Qwen3.5-27B | 128 | GQA | 128K | Large KV cache, high compression value |
| Gemma-4 | 256 | GQA | 128K+ | Larger head_dim, may favor different bit-width |

#### 9.3 — Cross-Model Report

Deliverable: a comparison showing whether the winning method/bit-width is **model-dependent** or **universal**.
Key question: does TurboQuant's Lloyd-Max advantage hold across different architectures and head dimensions?

### Dependency Graph

```
Phase 1 (Core)
  │
  ├──→ Phase 2 (TurboQuant) ──┐
  ├──→ Phase 3 (RaBitQ)    ───┤
  ├──→ Phase 4 (PQ)        ───┤
  │                            │
  │                            ▼
  │                      Phase 5 (Evaluation)
  │                            │
  │                            ▼
  │                      Phase 6 (KV Cache)
  │                            │
  │                            ▼
  └──────────────────→   Phase 7 (Optimization)
                               │
                               ▼
                         Phase 8 (Integration)
                               │
                               ▼
                         Phase 9 (Real Model Validation)
                           Qwen3.5-27B + Gemma-4
```

Phases 2, 3, 4 can proceed **in parallel** after Phase 1.

---

## 10. Fairness Guarantees

These invariants ensure no method gets an unfair advantage:

| Guarantee | Mechanism |
|-----------|-----------|
| Same rotation | TurboQuant Π and RaBitQ P produced by same `haar_rotation(d, seed)` |
| Same data | All methods tested on identical vectors from same `np.random.default_rng(seed)` |
| Same language | Pure NumPy — no SIMD or GPU advantage for any method |
| Same metrics | All measured by `core/metrics.py` |
| Same bit budget | Compare at equal total bits per vector (accounting for metadata) |
| Reproducible | All randomness from deterministic seed chain |

---

## 11. Success Criteria

| # | Criterion | Metric |
|---|-----------|--------|
| S1 | All 7 quantizers pass interface test | `test_interface.py` green |
| S2 | TQ_mse MSE within 10% of paper | Table in §5.3 |
| S3 | TQ_prod unbiased, variance×d within 15% | Table in §5.5 |
| S4 | TQ_mse b=1 bias α ∈ [0.62, 0.66] | §5.3 |
| S5 | RaBitQ estimator \|bias\| < 0.01 | §5.6 |
| S6 | ExtRaBitQ B=1 = RaBitQ 1-bit | §5.7 |
| S7 | Fixed seed → reproducible results | `test_fairness.py` green |
| S8 | d=512 single-vector quantize < 1ms | `eval/speed.py` |
| S9 | Metadata overhead < 5% for n ≥ 1000 | `eval/compression.py` |
| S10 | Complete distortion-rate plot for all methods | `notebooks/01_distortion_comparison.ipynb` |
| S11 | Qwen3.5-27B: perplexity degradation < 0.5 at 3-bit | Phase 9 eval |
| S12 | Gemma-4: perplexity degradation < 0.5 at 3-bit | Phase 9 eval |
| S13 | Cross-model report identifying best method per scenario | Phase 9 deliverable |

---

## 12. Changelog

### v4 (current) — Add Real Model Validation

- Added Phase 9: end-to-end KV-cache compression on Qwen3.5-27B and Gemma-4
- Added success criteria S11-S13 for perplexity and cross-model comparison
- Model-specific considerations (head_dim, GQA, context length)

### v3 — Unified VQBench Plan

- Consolidated from TurboQuant+.md, VQBench.md, VQBench-SubPlans.md into single source of truth
- Complete specifications for all 7 quantizers with pipeline diagrams, math, storage, tests
- RaBitQ: added centroid subtraction preprocessing, ip_coeff correction estimator, popcount formula
- ExtRaBitQ: added bit-plane decomposition detail with power-of-two weights
- PQ: bit-budget normalization for fair cross-method comparison
- Full dependency graph with parallel execution opportunities
- Explicit fairness guarantees and success criteria

### v2

- Fixed TurboQuant architecture (separated Algorithm 1 & 2)
- Corrected Beta distribution, speed targets, bias documentation

### v1

- Initial TurboQuant-only plan (had issues)
