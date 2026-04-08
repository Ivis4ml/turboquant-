"""
Shared evaluation metrics for all quantization methods.

All methods are measured by these same functions — fairness guarantee (§10 of PLAN).
"""

from __future__ import annotations

import numpy as np


def mse_distortion(X: np.ndarray, X_hat: np.ndarray) -> float:
    """
    Normalized MSE distortion: E[‖x − x̃‖² / ‖x‖²].

    Paper: TurboQuant Theorem 1, RaBitQ §4
    For unit vectors: simplifies to E[‖x − x̃‖²].

    Args:
        X: Original vectors, shape (n, d).
        X_hat: Reconstructed vectors, shape (n, d).

    Returns:
        Average normalized MSE across all vectors.
    """
    diff = X - X_hat
    norms_sq = np.sum(X ** 2, axis=1)
    mse_per_vec = np.sum(diff ** 2, axis=1) / np.maximum(norms_sq, 1e-30)
    return float(np.mean(mse_per_vec))


def ip_distortion(X: np.ndarray, X_hat: np.ndarray, Y: np.ndarray) -> float:
    """
    Inner product distortion: E[(⟨y, x⟩ − ⟨y, x̃⟩)²].

    Paper: TurboQuant Theorem 2
    Measures how well quantized vectors preserve inner products with queries.

    Args:
        X: Original database vectors, shape (n, d).
        X_hat: Reconstructed database vectors, shape (n, d).
        Y: Query vectors, shape (n_q, d).

    Returns:
        Average squared IP error.
    """
    # For each (x, y) pair: (⟨y, x⟩ − ⟨y, x̃⟩)²
    # Use random pairing: pair x_i with y_{i % n_q}
    n = len(X)
    n_q = len(Y)
    errors = []
    for i in range(n):
        y = Y[i % n_q]
        true_ip = np.dot(y, X[i])
        approx_ip = np.dot(y, X_hat[i])
        errors.append((true_ip - approx_ip) ** 2)
    return float(np.mean(errors))


def ip_bias(X: np.ndarray, X_hat: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """
    Fit multiplicative IP bias: E[⟨y, x̃⟩] ≈ α · ⟨y, x⟩.

    Paper: TurboQuant §3.2 — TurboQuant_mse has α < 1 (biased),
           TurboQuant_prod and RaBitQ have α = 1 (unbiased).

    Args:
        X: Original vectors, shape (n, d).
        X_hat: Reconstructed vectors, shape (n, d).
        Y: Query vectors, shape (n_q, d).

    Returns:
        (alpha, mean_residual): best-fit α and mean absolute residual.
    """
    true_ips = []
    approx_ips = []
    n = len(X)
    n_q = len(Y)
    for i in range(n):
        y = Y[i % n_q]
        true_ips.append(np.dot(y, X[i]))
        approx_ips.append(np.dot(y, X_hat[i]))

    true_ips = np.array(true_ips)
    approx_ips = np.array(approx_ips)

    # Least-squares fit: approx ≈ α · true  →  α = Σ(true·approx) / Σ(true²)
    denom = np.sum(true_ips ** 2)
    if denom < 1e-30:
        return 1.0, 0.0
    alpha = float(np.sum(true_ips * approx_ips) / denom)
    residual = float(np.mean(np.abs(approx_ips - alpha * true_ips)))
    return alpha, residual


def recall_at_k(
    X: np.ndarray,
    X_hat: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
) -> float:
    """
    ANN recall@k: fraction of true k-NN found in approximate k-NN.

    Args:
        X: Original database vectors, shape (n, d).
        X_hat: Reconstructed database vectors, shape (n, d).
        queries: Query vectors, shape (n_q, d).
        k: Number of neighbors.

    Returns:
        Average recall@k across all queries.
    """
    # True distances using original vectors
    true_dists = -queries @ X.T  # negative IP → sort ascending = highest IP first
    approx_dists = -queries @ X_hat.T

    n_q = len(queries)
    recalls = []
    for i in range(n_q):
        true_topk = set(np.argsort(true_dists[i])[:k])
        approx_topk = set(np.argsort(approx_dists[i])[:k])
        recalls.append(len(true_topk & approx_topk) / k)
    return float(np.mean(recalls))
