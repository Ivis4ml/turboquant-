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

**Goal**: Make VQBench production-viable. Current pure-NumPy implementation is correct but
leaves 10–100× performance on the table. Phase 7 closes the gap without compiled extensions.

| Task | File | Description | Depends |
|------|------|-------------|---------|
| 7.1 | `core/rotation.py` | Structured FWHT rotation: O(d²)→O(d log d) | 1.1 |
| 7.2 | `core/packing.py` | Bit packing: b-bit indices → uint8/uint64 bitfields | — |
| 7.3 | All methods | Fully vectorized batch: eliminate per-vector Python loops | All methods |
| 7.4 | `core/rotation.py` | Precomputed rotation cache: avoid recomputing Π per instance | 1.1 |
| 7.5 | `tests/test_perf.py` | Performance regression tests | 7.1–7.4 |

#### 7.1 — Structured FWHT Rotation: O(d²) → O(d log d)

**Current**: `haar_rotation(d, seed)` returns a dense d×d matrix; matmul is O(d²) per vector.
At d=3072 (Qwen/Gemma head_dim × num_kv_heads), this is 9.4M FLOPs per vector.

**Target**: Replace with structured rotation D₁·H·D₂·H·D₃ (3-layer randomized Hadamard)
that achieves the same concentration-of-measure guarantee at O(d log d).

```python
class StructuredRotation:
    """
    Pseudo-random rotation via randomized Hadamard: D₁·H·D₂·H·D₃.

    Paper: TurboQuant §4.2; Ailon & Chazelle (2006) "Fast JL Transform"

    Each Dᵢ = diag(sᵢ) where sᵢ ∈ {-1,+1}^d is random sign flip.
    H = normalized Walsh-Hadamard matrix (applied via butterfly in O(d log d)).
    Three rounds suffice for near-Haar distribution at d ≥ 128.

    Requires d = power of 2. Pad with zeros if necessary.
    """
    def __init__(self, d: int, seed: int):
        self.d = d
        self.d_padded = 1 << int(np.ceil(np.log2(d)))  # next power of 2
        rng = np.random.default_rng(seed)
        self.signs = [rng.choice([-1, 1], size=self.d_padded).astype(np.float64)
                      for _ in range(3)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply D₁·H·D₂·H·D₃·x in O(d log d)."""
        y = np.zeros(self.d_padded)
        y[:len(x)] = x
        for s in reversed(self.signs):
            y *= s
            y = _fwht_vectorized(y)
        return y[:self.d]

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """Apply rotation to each row of X ∈ R^{n×d}. Fully vectorized."""
        ...
```

**Implementation plan**:
1. Vectorize the FWHT butterfly with NumPy broadcasting (no Python for-loops)
2. Implement `_fwht_vectorized(Y)` operating on 2D arrays (n, d) — all n vectors at once
3. Handle non-power-of-2 d by zero-padding to `d_padded` and truncating output
4. Add `StructuredRotation` class with `.forward(x)` and `.forward_batch(X)` methods
5. Maintain backward compat: `haar_rotation()` still available, methods accept either
6. Add `rotation_mode` parameter to VectorQuantizer: `"dense"` (default) or `"fwht"`

**Vectorized FWHT butterfly** (critical inner kernel):

```python
def _fwht_vectorized(Y: np.ndarray) -> np.ndarray:
    """In-place FWHT on rows of Y ∈ R^{n×d}, d must be power of 2."""
    d = Y.shape[-1]
    h = 1
    while h < d:
        # Y[..., ::2h, :h] and Y[..., ::2h, h:2h] — butterfly pairs
        # NumPy slice view: reshape to (..., d/2h, 2, h)
        Y_view = Y.reshape(*Y.shape[:-1], -1, 2, h)
        a = Y_view[..., 0, :].copy()
        b = Y_view[..., 1, :]
        Y_view[..., 0, :] = a + b
        Y_view[..., 1, :] = a - b
        h *= 2
    Y /= np.sqrt(d)
    return Y
```

**Speed targets**:

| d | Dense Π (current) | FWHT (target) | Speedup |
|---|-------------------|---------------|---------|
| 128 | 0.008 ms | ~0.002 ms | 4× |
| 512 | 0.04 ms | ~0.004 ms | 10× |
| 3072 | ~2 ms | ~0.03 ms | 60× |

**Correctness test**: verify that structured rotation preserves:
- Coordinate std ≈ 1/√d (concentration of measure)
- MSE distortion within 5% of dense rotation
- IP bias unchanged

#### 7.2 — Bit Packing: `core/packing.py`

**Current**: Indices stored as int8/int16 arrays (8–16 bits per index).
At b=2, each index needs 2 bits but occupies 8 → 4× wasted memory.

**File**: `core/packing.py`

```python
def pack_indices(indices: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Pack b-bit indices into uint8 array.

    Args:
        indices: Integer array with values in [0, 2^b - 1], shape (d,) or (n, d).
        num_bits: Bits per index (1, 2, 3, or 4).

    Returns:
        Packed uint8 array. For b=2, d=512: returns 128 bytes instead of 512.
    """

def unpack_indices(packed: np.ndarray, num_bits: int, d: int) -> np.ndarray:
    """Unpack uint8 array back to integer indices."""

def pack_signs(signs: np.ndarray) -> np.ndarray:
    """
    Pack ±1 sign array into bitfield.

    signs ∈ {-1, +1}^d → uint8 array of ceil(d/8) bytes.
    Convention: +1 → bit=1, -1 → bit=0.
    """

def unpack_signs(packed: np.ndarray, d: int) -> np.ndarray:
    """Unpack bitfield back to ±1 array."""
```

**Packing strategy by bit-width**:

| b | Values | Pack ratio | Method |
|---|--------|-----------|--------|
| 1 | {0, 1} | 8 per byte | bit shifts |
| 2 | {0..3} | 4 per byte | 2-bit fields |
| 3 | {0..7} | 2 per byte + waste | 4-bit nibbles (pad to 4) |
| 4 | {0..15} | 2 per byte | 4-bit nibbles |

**Memory reduction at d=512**:

| b | Current (int8) | Packed | Reduction |
|---|---------------|--------|-----------|
| 1 | 512 B | 64 B | 8× |
| 2 | 512 B | 128 B | 4× |
| 4 | 512 B | 256 B | 2× |

**Integration**: Add `PackedQuantizedVector` dataclass that wraps packed arrays.
Methods gain `quantize_packed()` / `dequantize_packed()` that pack on the fly.
The unpacked path remains the default; packed path is opt-in for memory-critical use.

#### 7.3 — Fully Vectorized Batch Operations

**Current state**: TurboQuantMSE, RaBitQ1Bit, ExtRaBitQ, PQ all have vectorized `quantize_batch`.
Missing vectorized batch: `QJLQuantizer`, `TurboQuantProd`, `OPQ` (delegates to PQ).

**TurboQuantProd batch** (highest priority — most complex method):

```python
def quantize_batch(self, X: np.ndarray) -> list[QuantizedVector]:
    n = len(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)      # (n, 1)
    safe_norms = np.maximum(norms, 1e-30)

    if self._mse_bits > 0:
        X_hat = X / safe_norms
        Y = X_hat @ self._rotation.T                       # (n, d)
        all_indices = np.searchsorted(self._boundaries, Y).astype(np.int8)
        Y_hat = self._centroids[all_indices]                # (n, d) — MSE reconstruction
        X_mse = (norms * (Y_hat @ self._rotation))          # (n, d)
    else:
        all_indices = np.zeros((n, self.d), dtype=np.int8)
        X_mse = np.zeros_like(X)

    residuals = X - X_mse                                   # (n, d)
    gammas = np.linalg.norm(residuals, axis=1)              # (n,)

    # Batch QJL: sign(S @ residual.T).T
    projections = residuals @ self._S.T                     # (n, d)
    all_signs = np.sign(projections).astype(np.int8)
    all_signs[all_signs == 0] = 1
    ...
```

**QJL batch** (simpler):
```python
def quantize_batch(self, X: np.ndarray) -> list[QuantizedVector]:
    gammas = np.linalg.norm(X, axis=1)                     # (n,)
    projections = X @ self._S.T                             # (n, d)
    all_signs = np.sign(projections).astype(np.int8)
    all_signs[all_signs == 0] = 1
    ...
```

**Dequantize batch** — similar vectorization for all methods, operating on stacked arrays.

**Perf targets (post-optimization)**:

| Operation | d=128, n=1000 | d=512, n=1000 | Requirement |
|-----------|--------------|--------------|-------------|
| TQ_mse batch | < 10 ms | < 30 ms | N3: < 50 ms |
| TQ_prod batch | < 20 ms | < 50 ms | — |
| RaBitQ batch | < 10 ms | < 30 ms | — |
| PQ batch | < 30 ms | < 60 ms | (k-means-based) |

#### 7.4 — Rotation Cache

**Problem**: Each VectorQuantizer instance generates its own rotation matrix on `__init__`.
At d=512, that's a 512×512×8 = 2MB allocation per instance. When benchmarking 7 methods
with shared `(d, seed)`, we allocate 14 MB of identical matrices.

**Solution**: Module-level LRU cache keyed on `(d, seed)`.

```python
_rotation_cache: dict[tuple[int, int], np.ndarray] = {}

def haar_rotation(d: int, seed: int) -> np.ndarray:
    key = (d, seed)
    if key not in _rotation_cache:
        _rotation_cache[key] = _haar_rotation_impl(d, seed)
    return _rotation_cache[key]
```

Same for `StructuredRotation` instances. Read-only — callers must never mutate the returned matrix.

#### 7.5 — Performance Regression Tests

```python
# tests/test_perf.py

def test_single_vector_under_1ms():
    """Req N2: single vector d=512 < 1ms."""
    q = TurboQuantMSE(d=512, num_bits=2, seed=42)
    x = random_unit_vectors(1, 512, seed=0)[0]
    t0 = time.perf_counter()
    for _ in range(100):
        q.quantize(x)
    elapsed_ms = (time.perf_counter() - t0) / 100 * 1000
    assert elapsed_ms < 1.0

def test_batch_1000_under_50ms():
    """Req N3: batch 1000 at d=128 < 50ms."""
    q = TurboQuantMSE(d=128, num_bits=2, seed=42)
    X = random_unit_vectors(1000, 128, seed=0)
    t0 = time.perf_counter()
    q.quantize_batch(X)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50.0

def test_fwht_faster_than_dense():
    """FWHT rotation should be ≥3× faster than dense at d=512."""
    ...
```

---

### Phase 8: Integration — PyTorch Wrapper & Native Ports

**Goal**: Bridge VQBench from NumPy benchmark to production-usable KV-cache compression.
Three integration paths, in order of priority.

| Task | File | Description | Depends |
|------|------|-------------|---------|
| 8.1 | `torch_wrapper/module.py` | `QuantizedKVCache` — torch.nn.Module | Phase 6 |
| 8.2 | `torch_wrapper/hook.py` | HuggingFace `transformers` integration hook | 8.1 |
| 8.3 | `torch_wrapper/kernels.py` | Custom CUDA/Triton kernels (optional) | 8.1 |
| 8.4 | `native/mlx_port.py` | Apple MLX port for M-series inference | Phase 7 |
| 8.5 | `benchmarks/torch_bench.py` | PyTorch speed benchmarks vs NumPy | 8.1 |

#### 8.1 — PyTorch Module: `QuantizedKVCache`

```
vqbench/
└── torch_wrapper/
    ├── __init__.py
    ├── module.py          # QuantizedKVCache nn.Module
    ├── hook.py            # HuggingFace model_hook integration
    ├── kernels.py         # Optional Triton/CUDA fused kernels
    └── functional.py      # Functional API: quantize_kv, dequantize_kv
```

**Core class**:

```python
class QuantizedKVCache(torch.nn.Module):
    """
    Drop-in replacement for transformers KV cache with VQ compression.

    Usage:
        cache = QuantizedKVCache(
            head_dim=128,
            num_kv_heads=8,
            method_key="TurboQuantProd",   # unbiased IP for attention
            method_value="TurboQuantMSE",  # low MSE for output
            num_bits_key=3,
            num_bits_value=3,
        )

        # In attention forward:
        compressed_k, compressed_v = cache.update(new_keys, new_values)
        keys, values = cache.get()    # decompressed for attention
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        method_key: str = "TurboQuantProd",
        method_value: str = "TurboQuantMSE",
        num_bits_key: int = 3,
        num_bits_value: int = 3,
        seed: int = 42,
    ): ...

    def update(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Compress and append new KV pairs. keys/values: (batch, heads, seq, d)."""

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress all cached KV. Returns (keys, values)."""

    @property
    def seq_len(self) -> int: ...

    @property
    def memory_bytes(self) -> int:
        """Current compressed cache size in bytes."""
```

**Implementation strategy**:
1. Internal storage: keep compressed representations as Python lists of QuantizedVector
   (NumPy backend). Convert torch→numpy on `update()`, numpy→torch on `get()`.
2. This is simple but has CPU↔GPU transfer overhead — acceptable for Phase 8.1.
3. Phase 8.3 adds fused Triton kernels to avoid the transfer.

**Tensor layout**: Per-head compression. Each KV head's cache is an independent
`KVCacheCompressor` instance. GQA naturally supported: fewer KV heads = fewer compressors.

```python
# Internal layout:
self._compressors: list[KVCacheCompressor]  # one per KV head
# _compressors[h] holds all tokens for head h
```

#### 8.2 — HuggingFace Transformers Hook

Integration with `transformers` model inference via cache replacement:

```python
def apply_quantized_cache(
    model: PreTrainedModel,
    method_key: str = "TurboQuantProd",
    method_value: str = "TurboQuantMSE",
    num_bits: int = 3,
) -> PreTrainedModel:
    """
    Monkey-patch model to use quantized KV cache during generation.

    Replaces the DynamicCache with QuantizedKVCache in the model's
    generate() call. Works with Qwen2, Gemma2, LLaMA, Mistral.

    Usage:
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-27B")
        apply_quantized_cache(model, num_bits=3)
        output = model.generate(input_ids, max_new_tokens=1000)
    """
```

**Hook mechanism**: `transformers` ≥4.38 uses `Cache` objects (DynamicCache, StaticCache).
We subclass `Cache` and override `update()` / `__getitem__()`:

```python
class VQBenchCache(Cache):
    """transformers-compatible Cache backed by VQ compression."""

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Compress incoming KV
        self._layers[layer_idx].compress(
            key_states.cpu().numpy(),    # (batch, heads, seq, d)
            value_states.cpu().numpy(),
        )
        # Return decompressed for current attention step
        return self._get_decompressed(layer_idx)
```

**Supported model families** (all use `Cache` protocol):

| Family | head_dim | GQA ratio | Notes |
|--------|----------|-----------|-------|
| Qwen2/Qwen3 | 128 | 4:1–8:1 | Primary target |
| Gemma2/Gemma4 | 256 | 4:1 | Larger head_dim |
| LLaMA 3.x | 128 | 8:1 | Most common |
| Mistral/Mixtral | 128 | 8:1 | Sliding window variant |

#### 8.3 — Custom Kernels (Optional, Stretch)

For GPU-resident compression (avoid CPU↔GPU transfers):

```python
# Triton kernel for batch TurboQuant quantization on GPU
@triton.jit
def turbo_quant_mse_kernel(
    X_ptr, indices_ptr, norms_ptr,
    rotation_ptr, centroids_ptr, boundaries_ptr,
    d: tl.constexpr, num_bits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused rotate + searchsorted + store on GPU."""
    ...
```

**Priority**: Low. The CPU NumPy path + torch.cuda.synchronize is sufficient
for Phase 9 evaluation. Only build if GPU transfer is the bottleneck.

#### 8.4 — Apple MLX Port

For native M-series inference (M1–M5 Pro):

```python
# native/mlx_port.py
import mlx.core as mx

class MLXTurboQuantMSE:
    """TurboQuantMSE using MLX operations for unified memory."""

    def __init__(self, d, num_bits, seed=42):
        self._rotation = mx.array(haar_rotation(d, seed))
        centroids, boundaries = lloyd_max_codebook(num_bits, d)
        self._centroids = mx.array(centroids)
        self._boundaries = mx.array(boundaries)

    def quantize_batch(self, X: mx.array) -> ...:
        norms = mx.linalg.norm(X, axis=1, keepdims=True)
        X_hat = X / mx.maximum(norms, 1e-30)
        Y = X_hat @ self._rotation.T
        indices = mx.searchsorted(self._boundaries, Y)
        ...
```

**Advantage on M5 Pro**: MLX uses unified memory — no CPU↔GPU transfer.
The rotation matmul and searchsorted run on the Neural Engine / GPU
without data movement, which is the main bottleneck in the NumPy path.

**Test**: Verify MLX output matches NumPy to fp32 tolerance.

#### 8.5 — Integration Benchmarks

```python
# benchmarks/torch_bench.py

def bench_kv_cache_throughput():
    """Measure tokens/s for quantized KV cache at different seq_lens."""
    for seq_len in [1024, 4096, 16384, 65536, 131072]:
        for method in ["TurboQuantMSE", "TurboQuantProd", "RaBitQ1Bit"]:
            for bits in [2, 3, 4]:
                cache = QuantizedKVCache(...)
                # Simulate autoregressive generation
                for t in range(seq_len):
                    cache.update(new_k, new_v)   # compress 1 token
                    k, v = cache.get()            # decompress all
                # Report: tokens/s, peak memory, latency breakdown

def bench_memory_vs_baseline():
    """Compare memory footprint: fp16 cache vs quantized at 128K tokens."""
    # fp16 baseline: 2 * num_layers * num_kv_heads * seq_len * head_dim * 2 bytes
    # Qwen3.5-27B: 2 * 64 * 4 * 131072 * 128 * 2 = ~8.6 GB
    # At 3-bit:    2 * 64 * 4 * 131072 * 128 * 3/8 + metadata ≈ ~1.6 GB
    # Compression ratio: ~5.3×
```

---

### Phase 9: Real Model Validation

End-to-end KV-cache compression on production LLMs — the ultimate acceptance test.
This phase proves that VQBench methods work on real attention patterns, not just
synthetic unit vectors.

| Task | File | Description | Depends |
|------|------|-------------|---------|
| 9.1 | `validation/qwen.py` | Qwen3.5-27B KV-cache compression eval | 8.1, 8.2 |
| 9.2 | `validation/gemma.py` | Gemma-4 KV-cache compression eval | 8.1, 8.2 |
| 9.3 | `validation/report.py` | Cross-model comparison & analysis | 9.1, 9.2 |
| 9.4 | `validation/run_eval.py` | Unified evaluation driver script | 9.1, 9.2 |

#### 9.1 / 9.2 — Per-Model Evaluation Protocol

**Step 1: Setup & Model Loading**

```python
# validation/qwen.py (Gemma equivalent is parallel)

MODEL_ID = "Qwen/Qwen3.5-27B"     # or "google/gemma-4-27b"
DEVICE = "mps"                      # M5 Pro — use MPS backend
DTYPE = torch.bfloat16              # Qwen3.5 native dtype

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

**Step 2: Evaluation Datasets**

| Dataset | Purpose | Metric | Tokens |
|---------|---------|--------|--------|
| WikiText-2 | Language modeling | Perplexity | ~250K |
| C4 (validation, 1K samples) | General LM | Perplexity | ~500K |
| RULER (4K, 8K, 16K, 32K) | Long-context retrieval | Accuracy | Variable |
| MMLU (5-shot) | Knowledge reasoning | Accuracy | ~150K |

Rationale: Perplexity catches subtle quality loss. RULER tests whether
attention still reaches distant tokens after KV compression.
MMLU tests downstream task impact.

**Step 3: Compression Configurations**

Test matrix — exhaustive grid over methods × bit-widths:

| Config | Key Method | Value Method | Key bits | Value bits | Notes |
|--------|-----------|-------------|----------|------------|-------|
| Baseline | fp16 | fp16 | 16 | 16 | No compression |
| TQ-MSE-2 | TurboQuantMSE | TurboQuantMSE | 2 | 2 | Biased keys |
| TQ-MSE-3 | TurboQuantMSE | TurboQuantMSE | 3 | 3 | |
| TQ-MSE-4 | TurboQuantMSE | TurboQuantMSE | 4 | 4 | |
| TQ-Prod-2 | TurboQuantProd | TurboQuantMSE | 2 | 2 | Unbiased keys |
| TQ-Prod-3 | TurboQuantProd | TurboQuantMSE | 3 | 3 | **Expected sweet spot** |
| TQ-Prod-4 | TurboQuantProd | TurboQuantMSE | 4 | 4 | |
| RaBitQ-2 | ExtRaBitQ | ExtRaBitQ | 2 | 2 | Unbiased keys |
| RaBitQ-3 | ExtRaBitQ | ExtRaBitQ | 3 | 3 | |
| RaBitQ-4 | ExtRaBitQ | ExtRaBitQ | 4 | 4 | |
| Mixed-3 | TurboQuantProd | TurboQuantMSE | 3 | 2 | Asym: more bits for keys |

**Key design decision**: Use TurboQuantProd (unbiased) for keys and TurboQuantMSE (low MSE) for values.
Keys participate in softmax(QK^T) — IP bias directly distorts attention weights.
Values are just linearly combined — MSE is what matters.

**Step 4: Measurement Protocol**

```python
def evaluate_config(model, tokenizer, config, datasets):
    """Run full evaluation for one compression config."""
    apply_quantized_cache(model, **config)

    results = {}

    # (a) Perplexity — sliding window, stride = max_length // 2
    for name, dataset in [("wikitext2", wt2), ("c4", c4_val)]:
        ppl = evaluate_perplexity(model, tokenizer, dataset, max_length=2048)
        results[f"ppl_{name}"] = ppl

    # (b) Long-context — RULER benchmark at multiple lengths
    for ctx_len in [4096, 8192, 16384, 32768]:
        acc = evaluate_ruler(model, tokenizer, ctx_len=ctx_len)
        results[f"ruler_{ctx_len}"] = acc

    # (c) MMLU — 5-shot
    results["mmlu"] = evaluate_mmlu(model, tokenizer, n_shot=5)

    # (d) Memory & Speed
    results["peak_memory_gb"] = torch.mps.current_allocated_memory() / 1e9
    results["tokens_per_sec"] = measure_generation_speed(model, tokenizer)

    return results
```

**Step 5: Statistical Rigor**

- Run each perplexity eval 3× with different random seeds for few-shot prompts
- Report mean ± std
- Use paired comparisons: Δperplexity = ppl_compressed − ppl_baseline
- Significance: require |Δppl| confidence interval excludes 0

#### Model-Specific Considerations

| Aspect | Qwen3.5-27B | Gemma-4 |
|--------|------------|---------|
| Architecture | Qwen2-based decoder | Gemma2-based decoder |
| head_dim | 128 | 256 |
| num_kv_heads | 4 (GQA 7:1) | 8 (GQA 4:1) |
| num_layers | 64 | 46 |
| Context window | 128K (YaRN) | 128K+ |
| KV cache size (fp16, 128K) | ~8.6 GB | ~19.2 GB |
| KV cache at 3-bit | ~1.6 GB | ~3.6 GB |
| dtype | bfloat16 | bfloat16 |
| M5 Pro fit? | Yes (36GB unified) | Tight — may need 4-bit or offload |
| Tokenizer | Qwen2Tokenizer | GemmaTokenizer |
| HF model class | Qwen2ForCausalLM | Gemma2ForCausalLM |

**Gemma-4 head_dim=256 hypothesis**: Larger head_dim means each coordinate
has smaller variance (1/256 vs 1/128). Lloyd-Max is more accurate at lower
variance → TurboQuant's advantage over RaBitQ may be larger at head_dim=256.

**GQA interaction**: GQA means fewer KV heads (each shared across multiple Q heads).
Compression applies per KV head, so GQA + VQ is multiplicative compression.
Qwen3.5 GQA 7:1 + 3-bit VQ ≈ 37× compression vs naive fp16.

#### 9.3 — Cross-Model Report

**Deliverable**: A structured comparison answering three questions:

**Q1: Is the winning method model-dependent or universal?**

Expected: TurboQuantProd for keys + TurboQuantMSE for values wins universally,
but the optimal bit-width may differ:
- Qwen3.5 (head_dim=128): 3-bit likely sufficient
- Gemma-4 (head_dim=256): may tolerate 2-bit for values

**Q2: Does TurboQuant's Lloyd-Max advantage hold across architectures?**

Compare TQ-Prod-3 vs RaBitQ-3 on both models:
- If TQ wins on both → Lloyd-Max is universally better
- If RaBitQ wins on Gemma-4 → the advantage is head_dim-dependent

**Q3: What's the Pareto frontier (quality vs compression)?**

```
Perplexity                            Pareto
Degradation  ┤                          Front
             │  ×RaBitQ-2                 │
   1.0 ──────│──×TQ-MSE-2───────────────│──
             │     ×TQ-Prod-2            │
   0.5 ──────│────────×RaBitQ-3──────── │──   ← S11/S12 threshold
             │         ×TQ-Prod-3        │
   0.1 ──────│───────────×TQ-Prod-4─────│──
             │                  ×fp16    │
   0.0 ──────┼───────────────────────────┤──
             2×     5×     10×    15×
                  Compression Ratio
```

**Report format**: Markdown tables + matplotlib plots, generated by `validation/report.py`.

#### 9.4 — Evaluation Driver

```python
# validation/run_eval.py — single command to run everything

"""
Usage:
    python -m vqbench.validation.run_eval --model qwen --bits 2,3,4
    python -m vqbench.validation.run_eval --model gemma --bits 3
    python -m vqbench.validation.run_eval --model all --full   # complete grid
"""

def main():
    args = parse_args()
    configs = build_config_grid(args.model, args.bits)
    results = []
    for config in configs:
        print(f"Evaluating {config['name']}...")
        r = evaluate_config(model, tokenizer, config, datasets)
        results.append(r)
        save_checkpoint(results)   # incremental save
    generate_report(results, output_dir=args.output)
```

**Estimated runtime on M5 Pro (36GB)**:

| Model | Configs | Est. per config | Total |
|-------|---------|----------------|-------|
| Qwen3.5-27B | 11 | ~30 min | ~5.5 hours |
| Gemma-4 | 11 | ~45 min | ~8 hours |
| **Full grid** | **22** | — | **~14 hours** |

Checkpoint after each config so runs can be resumed.

### Dependency Graph

```
Phase 1 (Core)         ✅ DONE
  │
  ├──→ Phase 2 (TurboQuant) ──┐  ✅ DONE
  ├──→ Phase 3 (RaBitQ)    ───┤  ✅ DONE
  ├──→ Phase 4 (PQ)        ───┤  ✅ DONE
  │                            │
  │                            ▼
  │                      Phase 5 (Evaluation)  ✅ DONE
  │                            │
  │                            ▼
  │                      Phase 6 (KV Cache)    ✅ DONE
  │                            │
  │              ┌─────────────┤
  │              ▼             ▼
  │        Phase 7         Phase 8.1
  │     (Optimization)   (PyTorch Wrapper)
  │         │    │             │
  │         │    │             ▼
  │         │    │       Phase 8.2
  │         │    │    (HF Transformers Hook)
  │         │    │             │
  │         ▼    │             ▼
  │      Phase 8.4        Phase 9.1 ──→ Phase 9.3
  │    (MLX Port)         (Qwen3.5)    (Report)
  │                       Phase 9.2 ──↗
  │                       (Gemma-4)
  │
  └──→ Phase 8.3 (Triton Kernels — optional, stretch)
```

**Critical path to Phase 9**: Phase 8.1 → 8.2 → 9.1/9.2 → 9.3
Phase 7 is **not blocking** — NumPy is fast enough for evaluation.
Phase 7 and 8.4 (MLX) can proceed in parallel with Phase 9.

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

| # | Criterion | Metric | Phase | Status |
|---|-----------|--------|-------|--------|
| S1 | All 7 quantizers pass interface test | `test_interface.py` green | 5 | ✅ Done |
| S2 | TQ_mse MSE within 10% of Lloyd-Max | Table in §5.3 | 2 | ✅ Done |
| S3 | TQ_prod unbiased, variance×d within 15% | Table in §5.5 | 2 | ✅ Done |
| S4 | TQ_mse b=1 bias α ∈ [0.62, 0.66] | §5.3 | 2 | ✅ Done |
| S5 | RaBitQ estimator \|bias\| < 0.01 | §5.6 | 3 | ✅ Done |
| S6 | ExtRaBitQ B=1 = RaBitQ 1-bit | §5.7 | 3 | ✅ Done |
| S7 | Fixed seed → reproducible results | `test_fairness.py` green | 5 | ✅ Done |
| S8 | d=512 single-vector quantize < 1ms | `eval/speed.py` | 5 | ✅ 0.008ms |
| S9 | Metadata overhead < 5% for n ≥ 1000 | `eval/compression.py` | 5 | ✅ 1-6% |
| S10 | Complete distortion-rate plot for all methods | `notebooks/01_distortion_comparison.ipynb` | 5 | Pending |
| S11 | FWHT ≥ 3× faster than dense at d ≥ 512 | `test_perf.py` | 7 | Pending |
| S12 | Bit-packed storage ≥ 2× smaller than int8 | `test_perf.py` | 7 | Pending |
| S13 | PyTorch wrapper passes integration test | `test_torch_wrapper.py` | 8 | Pending |
| S14 | Qwen3.5-27B: perplexity degradation < 0.5 at 3-bit | Phase 9 eval | 9 | Pending |
| S15 | Gemma-4: perplexity degradation < 0.5 at 3-bit | Phase 9 eval | 9 | Pending |
| S16 | Cross-model report identifying best method per scenario | Phase 9 deliverable | 9 | Pending |

---

## 12. Changelog

### v5 (current) — Detailed Phase 7/8/9 Execution Plans

- **Phase 7 expanded**: Vectorized FWHT butterfly (O(d²)→O(d log d)), bit packing (2-8× memory savings), rotation cache, perf regression tests
- **Phase 8 expanded**: `QuantizedKVCache` torch.nn.Module, HuggingFace `Cache` subclass integration, MLX port for Apple Silicon, optional Triton kernels
- **Phase 9 expanded**: Full evaluation protocol with 4 datasets (WikiText-2, C4, RULER, MMLU), 11 configs per model, mixed key/value methods, statistical rigor
- Updated dependency graph: Phase 7 no longer blocks Phase 9 (NumPy fast enough); critical path is 8.1→8.2→9.1/9.2→9.3
- Added success criteria S11-S16 with Phase/Status tracking
- Marked Phases 1-6 as ✅ DONE (121 tests passing)

### v4 — Add Real Model Validation

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
