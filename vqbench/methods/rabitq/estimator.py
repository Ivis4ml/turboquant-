"""
RaBitQ unbiased inner product and distance estimator.

Paper: Gao & Long, arXiv 2405.12497, §3.2 (1-bit)
       Gao et al., arXiv 2409.09913, §3 (multi-bit)

Core idea: direct ⟨y, x̃⟩ is biased (same as TurboQuant_mse). RaBitQ corrects via
ip_coeff = ⟨x̄, Pō⟩ to cancel the multiplicative bias exactly.

  estimate = ‖o‖ · raw_ip / ip_coeff

where raw_ip = ⟨q_rot, x̄⟩ and x̄ = quantized_signs / √d.

Computational shortcut: ⟨x̄, q_rot⟩ where x̄ ∈ {±1/√d}^d reduces to
  (1/√d) · (d − 2·popcount(b_x XOR b_q))
"""

from __future__ import annotations

import numpy as np

from vqbench.core.base import QuantizedVector


def rabitq_ip_estimate(
    query: np.ndarray,
    qv: QuantizedVector,
    rotation: np.ndarray,
    centroid: np.ndarray,
    d: int,
) -> float:
    """
    Unbiased estimate of ⟨query − c, x − c⟩.

    Paper: arXiv 2405.12497, §3.2, Eq. (8)
    Formula: estimate = ‖o‖ · ⟨q_rot, x̄⟩ / ip_coeff
    where:
      q_rot = P · (query − c)
      x̄ = signs / √d   (quantized unit hypercube vertex)
      ip_coeff = ⟨x̄, P · ō⟩  (stored during quantization)
      ‖o‖ = stored norm

    Args:
        query: Query vector q ∈ R^d.
        qv: Quantized representation of database vector x.
        rotation: Orthogonal matrix P ∈ R^{d×d}.
        centroid: Dataset centroid c ∈ R^d.
        d: Dimension.

    Returns:
        Unbiased estimate of ⟨q − c, x − c⟩.
    """
    q_rot = rotation @ (query - centroid)            # ← P · (q − c)
    x_bar = qv.indices.astype(np.float64) / np.sqrt(d)  # ← ±1/√d
    raw_ip = np.dot(q_rot, x_bar)                   # ← ⟨q_rot, x̄⟩
    ip_coeff = qv.metadata["ip_coeff"]               # ← ⟨x̄, Pō⟩
    norm_o = float(qv.norms[0])                      # ← ‖o‖

    if abs(ip_coeff) < 1e-30:
        return 0.0
    return norm_o * raw_ip / ip_coeff                # ← Eq. (8)


def rabitq_dist_estimate(
    query: np.ndarray,
    qv: QuantizedVector,
    rotation: np.ndarray,
    centroid: np.ndarray,
    d: int,
) -> float:
    """
    Unbiased estimate of ‖q − x‖² via unbiased IP.

    Paper: arXiv 2405.12497, §3.2
    Formula: ‖q − x‖² = ‖q−c‖² + ‖x−c‖² − 2⟨q−c, x−c⟩

    Args:
        query: Query vector q.
        qv: Quantized database vector x.
        rotation: Orthogonal matrix P.
        centroid: Dataset centroid c.
        d: Dimension.

    Returns:
        Estimated squared distance ‖q − x‖².
    """
    q_off = query - centroid
    ip_est = rabitq_ip_estimate(query, qv, rotation, centroid, d)
    return float(np.dot(q_off, q_off)) + float(qv.norms[0]) ** 2 - 2 * ip_est


def ext_rabitq_ip_estimate(
    query: np.ndarray,
    qv: QuantizedVector,
    rotation: np.ndarray,
    centroid: np.ndarray,
    d: int,
    num_bits: int,
) -> float:
    """
    Unbiased IP estimate for Extended RaBitQ (B-bit).

    Paper: arXiv 2409.09913, §3
    Formula: B-bit IP = Σ_{k=0}^{B-1} 2^k · ⟨tilde_o^(k), q_rot⟩
    Each bit-plane is a binary vector → popcount-computable.

    The estimate is scaled by norm and corrected by ip_coeff.
    """
    q_rot = rotation @ (query - centroid)
    norm_o = float(qv.norms[0])
    ip_coeff = qv.metadata["ip_coeff"]
    scale = qv.metadata.get("scale", 1.0)
    offset = qv.metadata.get("offset", 0.0)

    if abs(ip_coeff) < 1e-30:
        return 0.0

    # Reconstruct from indices: centered integer codebook
    indices = qv.indices.astype(np.float64)
    # x̄ = (2·indices - (2^B - 1)) · scale + offset, normalized
    x_bar = ((2 * indices - (2**num_bits - 1)) * scale + offset)
    x_bar_norm = np.linalg.norm(x_bar)
    if x_bar_norm < 1e-30:
        return 0.0
    x_bar /= x_bar_norm

    raw_ip = np.dot(q_rot, x_bar)
    return norm_o * raw_ip / ip_coeff
