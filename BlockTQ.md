# From TurboQuant to BlockTurboQuant

> 为什么论文算法在真实模型上效果不佳，以及怎么修

---

## 1. TurboQuant 论文算法回顾

TurboQuant (Zandieh et al., ICLR 2026) 的核心思路：

$$x \in \mathbb{R}^d \;\xrightarrow{\text{存 } \|x\|}\; \hat{x} = \frac{x}{\|x\|} \;\xrightarrow{\Pi}\; y = \Pi \hat{x} \;\xrightarrow{\text{标量量化}}\; \text{indices}$$

1. **提取 norm**：存一个标量 $\|x\|$
2. **归一化**：$\hat{x} = x / \|x\|$，变成单位向量
3. **Haar 随机旋转**：$y = \Pi \hat{x}$，旋转后每个坐标 $y_j \sim \mathcal{N}(0, 1/d)$（近似）
4. **逐坐标标量量化**：每个 $y_j$ 用 Lloyd-Max 码本映射到最近的 centroid

反量化就是倒着来：查表 → 逆旋转 → 乘以存储的 norm。

**理论保证（Theorem 1）**：归一化 MSE $\leq \frac{\sqrt{3\pi}}{2} \cdot 4^{-b}$

这个保证是**在合成单位向量上验证过的**，我们的 VQBench 实现与 turboquant_plus 的数值误差 < $10^{-6}$。

---

## 2. 问题：真实 K tensor 上崩了

在 Qwen2.5-0.5B（head_dim=64）上用 4-bit TurboQuantMSE 压缩 K cache：

$$\Delta\text{PPL} = +31.8\%$$

在 Qwen3-4B（head_dim=128）上好一些但依然明显：

$$\Delta\text{PPL} = +3.4\% \quad \text{(1024 tokens, 4 chunks)}$$

而 turboquant_plus 通过 llama.cpp 报告的是 $+0.23\%$。差距在哪里？

---

## 3. Root Cause：一个 norm 管不住 128 个坐标

真实 transformer 的 K tensor 长这样（Qwen2.5-0.5B, layer 0, head 0）：

```
K shape = (512, 64)
K norm 平均 = 250
K std = 30.7
```

关键问题：**不是所有坐标都一样大**。某些 "outlier channel" 的方差是其他 channel 的 10 倍以上。

当 TurboQuantMSE 用一个全局 $\|x\|$ 归一化时：

```
归一化前:  [..., 0.3, 0.2, 50.0, 0.1, 0.4, ...]   ← 第 37 维是 outlier
                                ^^^^
归一化后:  [..., 0.001, 0.0008, 0.2, 0.0004, 0.0016, ...]
```

归一化后，outlier 坐标占据了大部分动态范围。Lloyd-Max 码本的 centroid 分布被 outlier 拽向两端，**其余 127 个"安静"坐标的量化精度被牺牲了**。

旋转 $\Pi$ 能把 outlier 分散到所有坐标上吗？理论上能，但效果取决于 $d$：

| head_dim | 旋转后坐标分布 | 集中度 | 实际效果 |
|---|---|---|---|
| 512+ | 非常接近 $\mathcal{N}(0, 1/d)$ | 强 | 论文理论成立 |
| 128 | 接近但有 heavy tail | 中等 | 勉强可用 |
| 64 | 偏离 Gaussian 明显 | 弱 | 崩溃 |

**结论：旋转分散 outlier 的能力随 $d$ 增大而增强，但在 $d=64$-$128$ 范围内不够强，real K tensor 的 outlier 结构仍然残留在旋转后的坐标里。**

---

## 4. 解法：BlockTurboQuantMSE

### 核心思想

不要用一个 norm 管整个向量。把向量**切成小块**，每个块有自己的 scale。

```
Scalar TQ-MSE:   x ∈ R^128  →  1 个 norm  →  128 个坐标共享一个动态范围
Block B=32:      x ∈ R^128  →  4 个 scale  →  每 32 个坐标有自己的动态范围
Block B=16:      x ∈ R^128  →  8 个 scale  →  每 16 个坐标有自己的动态范围
```

outlier 在 block 2 里？它只影响 block 2 的 scale，其余 3 个 block 完全不受影响。

### 算法步骤

给定 $x \in \mathbb{R}^d$，block size $B$，$n_{\text{blocks}} = d / B$：

**Quantize:**

$$y = \Pi \cdot x \quad \text{(全局 Haar 旋转，和 scalar 完全一样)}$$

$$y_{\text{blocks}} = y.\text{reshape}(n_{\text{blocks}},\; B)$$

$$\text{对每个 block } i:$$

$$s_i = \|y_{\text{blocks}}[i]\|_2 \quad \text{(fp16 存储)}$$

$$\bar{y}_i = y_{\text{blocks}}[i] \;/\; s_i \quad \text{(块内归一化到单位范数)}$$

$$\text{idx}[i, j] = \arg\min_k \;|\bar{y}_{i,j} - c_k| \quad \text{(Lloyd-Max on } \mathcal{N}(0, 1/B) \text{)}$$

**Dequantize:**

$$\hat{y}_{i,j} = c_{\text{idx}[i,j]} \quad \text{(查码本)}$$

$$\text{(可选 norm correction: } \hat{y}_i \leftarrow \hat{y}_i \;/\; \|\hat{y}_i\| \text{)}$$

$$\hat{y}_i \leftarrow \hat{y}_i \cdot s_i \quad \text{(恢复 block scale)}$$

$$\hat{x} = \Pi^\top \cdot \hat{y}.\text{flatten}() \quad \text{(逆旋转)}$$

### 和 Scalar 的关键区别

```
                      Scalar TQ-MSE              Block TQ-MSE (B=32)
                      ─────────────              ───────────────────
旋转矩阵              Π ∈ R^{d×d}                同一个 Π（公平比较）
归一化                 全局: x/‖x‖                每块: block_i / ‖block_i‖
存储的 scale           1 个 fp16 norm             n_blocks 个 fp16 scale
码本                   Lloyd-Max on N(0, 1/d)     Lloyd-Max on N(0, 1/B)
outlier 隔离           ✗ 一个 outlier 毁全局       ✓ 只影响自己的 block
```

### 为什么码本用 $\mathcal{N}(0, 1/B)$ 而不是 $\mathcal{N}(0, 1/d)$

每个 block 归一化后是一个 $B$ 维单位向量。$B$ 维单位向量的每个坐标的边缘分布是：

$$f_X(t) = \frac{\Gamma(B/2)}{\sqrt{\pi}\,\Gamma((B\!-\!1)/2)} \,(1 - t^2)^{(B-3)/2} \quad t \in [-1, 1]$$

对 $B \geq 16$，这近似 $\mathcal{N}(0, 1/B)$。

**核心点**：block 越小（$B$ 越小），$1/B$ 越大，每个坐标的值域越宽，码本的 centroid 间距越大 → 量化精度更高（每个 centroid 负责的概率质量更少）。

这就是为什么 block 在 MSE 上总是优于 scalar：**它用更宽的码本在更短的向量上做量化。**

---

## 5. 代价：多了多少存储

每个 block 多存一个 fp16 scale = 16 bits：

$$\text{总 bits} = d \cdot b + n_{\text{blocks}} \cdot 16 = d \cdot b + \frac{d}{B} \cdot 16$$

$$\text{有效 bits/dim} = b + \frac{16}{B}$$

| 配置 (d=128, b=4) | 总 bits | 有效 bits/dim | 额外开销 | vs fp16 压缩比 |
|---|---|---|---|---|
| Scalar | 528 | 4.12 | +0.12 | 3.88× |
| Block B=64 | 544 | 4.25 | +0.25 | 3.76× |
| Block B=32 | 576 | 4.50 | +0.50 | 3.56× |
| Block B=16 | 640 | 5.00 | +1.00 | 3.20× |

turboquant_plus 的 `turbo4 = 4.25 bits/val` 对应我们的 Block B=64。

**Trade-off**：block 越小 → outlier 隔离越好 → MSE 越低 → 但 fp16 scale 开销越大。

---

## 6. 实测结果

### Synthetic unit vectors (d=128, outlier channels)

有 8 个 outlier channel（方差 10×）的合成数据：

| 方法 | b=3 MSE | b=4 MSE |
|---|---|---|
| Scalar TQ-MSE | 基线 | 基线 |
| **Block B=16** | **严格更低** | **严格更低** |

在没有 outlier 的 isotropic 数据上，block 和 scalar 持平（block 不会更差）。

### Qwen3-4B (head_dim=128), WikiText-2, 512 tokens

| 方法 | K bits | V bits | ΔPPL | 改善 |
|---|---|---|---|---|
| Scalar TQ-MSE | 4 | 4 | +0.14 (+1.1%) | 基线 |
| Block B=32 | 4 | 4 | +0.11 (+0.9%) | 18% better |
| **Block B=16** | **4** | **4** | **+0.09 (+0.7%)** | **37% better** |

### Qwen3-4B, 1024 tokens (4 chunks, 3 cache boundaries, 压力测试)

| 方法 | K bits | V bits | ΔPPL | 改善 |
|---|---|---|---|---|
| Scalar TQ-MSE | 4 | 4 | +0.33 (+3.4%) | 基线 |
| Block B=16 | 4 | 4 | +0.20 (+2.1%) | 39% better |
| **Block B=32** | **4** | **4** | **+0.13 (+1.3%)** | **62% better** |

**长 context 下 B=32 反超 B=16**，因为更多 cache 边界放大了 fp16 scale 的舍入误差累积。B=32 只有 4 个 scale（4 次舍入），B=16 有 8 个（8 次舍入）。

---

## 7. 最优 block size 的选择

没有唯一最优。取决于两个对抗因素：

$$\underbrace{\text{outlier 隔离能力}}_{\text{B 越小越好}} \quad \text{vs} \quad \underbrace{\text{fp16 scale 舍入累积}}_{\text{B 越大越好}}$$

| 场景 | 推荐 B | 理由 |
|---|---|---|
| 短 context（< 4K tokens） | 16 | 少量 cache boundary，fp16 累积可忽略 |
| 中等 context（4K-32K） | 32 | 平衡 outlier 隔离和舍入稳定性 |
| 超长 context（128K+） | 32-64 | 舍入累积是瓶颈，宁可粗粒度 |

一个有趣的研究方向：**自适应 block size**，按 channel variance 动态选择每个 block 的大小。

---

## 8. 和 turboquant_plus / llama.cpp 的关系

| 方面 | VQBench BlockTQ | llama.cpp turbo* |
|---|---|---|
| 分块思路 | 相同：per-block fp16 scale | 相同 |
| 旋转 | Haar 随机矩阵 | 可能有不同的随机化策略 |
| 码本 | Lloyd-Max on $\mathcal{N}(0, 1/B)$ | 可能有 data-aware 调整 |
| block size | 可配置 (16, 32, 64) | 通常 32 |
| 实现 | NumPy (Python, 跨平台) | C + NEON/AVX (Metal 优化) |
| PPL @ 4-bit | +1.3% (Qwen3-4B, 1K tok) | +0.23% (不同模型) |

两者的**算法结构是相同的**。剩余的 PPL 差距（+1.3% vs +0.23%）可能来自：
1. 不同的模型和评估协议（不是 apple-to-apple 比较）
2. llama.cpp 可能有 data-aware codebook tuning
3. SIMD 内核的数值行为（定点 vs 浮点中间计算）

---

## 9. 代码指引

| 文件 | 内容 |
|---|---|
| `vqbench/methods/turboquant/mse.py` | Scalar TurboQuantMSE（论文 Algorithm 1） |
| `vqbench/methods/turboquant/block_mse.py` | **BlockTurboQuantMSE**（本文档描述的改进） |
| `vqbench/methods/turboquant/codebook.py` | Lloyd-Max 码本（scalar 和 block 共用） |
| `vqbench/core/rotation.py` | Haar 旋转 + FWHT（scalar 和 block 共用同一个 $\Pi$） |
| `vqbench/tests/test_block_quant.py` | 21 个测试（含 block 严格优于 scalar 的 hypothesis test） |
| `vqbench/torch_wrapper/module.py` | HF Cache 集成（支持 `BlockTurboQuantMSE-B16/B32`） |

快速使用：

```python
from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE

q = BlockTurboQuantMSE(d=128, num_bits=4, block_size=32, seed=42)
qv = q.quantize(x)        # x ∈ R^128
x_hat = q.dequantize(qv)  # 重建
print(q.storage_bits(qv))  # 576 bits (= 4.50 bits/dim)
```

KV cache 集成：

```python
from vqbench.torch_wrapper.hook import make_vqbench_cache

cache = make_vqbench_cache(
    model.config,
    method_key='BlockTurboQuantMSE-B32',    # K cache: block 4-bit
    method_value='TurboQuantMSE',            # V cache: scalar (V compression is free)
    num_bits_key=4, num_bits_value=4,
)
```
