"""
Bit packing utilities for compact storage of quantization indices and sign bits.

Phase 7.2 — memory reduction:
  b=1: 8× (8 indices per byte)
  b=2: 4× (4 indices per byte)
  b=4: 2× (2 indices per byte)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Index packing: b-bit values → uint8 arrays
# ---------------------------------------------------------------------------

def pack_indices(indices: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Pack b-bit indices into uint8 array.

    Args:
        indices: Integer array with values in [0, 2^b - 1], shape (d,) or (n, d).
        num_bits: Bits per index (1, 2, 3, or 4).

    Returns:
        Packed uint8 array. For b=2, d=512: 128 bytes instead of 512.
    """
    if num_bits not in (1, 2, 4):
        # b=3: pad to 4-bit nibbles
        return pack_indices(indices, 4)

    vals_per_byte = 8 // num_bits
    flat = indices.ravel().astype(np.uint8)
    # Pad to multiple of vals_per_byte
    pad_len = (-len(flat)) % vals_per_byte
    if pad_len:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.uint8)])

    groups = flat.reshape(-1, vals_per_byte)
    packed = np.zeros(len(groups), dtype=np.uint8)
    for i in range(vals_per_byte):
        packed |= groups[:, i] << (i * num_bits)
    return packed


def unpack_indices(packed: np.ndarray, num_bits: int, length: int) -> np.ndarray:
    """
    Unpack uint8 array back to integer indices.

    Args:
        packed: Packed uint8 array from pack_indices.
        num_bits: Bits per index (must match packing).
        length: Number of original indices to recover.

    Returns:
        Integer array of shape (length,) with values in [0, 2^b - 1].
    """
    if num_bits == 3:
        num_bits = 4  # was padded to 4-bit

    vals_per_byte = 8 // num_bits
    mask = (1 << num_bits) - 1
    result = np.empty(len(packed) * vals_per_byte, dtype=np.uint8)
    for i in range(vals_per_byte):
        result[i::vals_per_byte] = (packed >> (i * num_bits)) & mask
    return result[:length]


# ---------------------------------------------------------------------------
# Sign packing: ±1 arrays → bitfields
# ---------------------------------------------------------------------------

def pack_signs(signs: np.ndarray) -> np.ndarray:
    """
    Pack ±1 sign array into bitfield.

    Convention: +1 → bit=1, -1 → bit=0.

    Args:
        signs: Array of ±1 values, shape (d,) or (n, d).

    Returns:
        Packed uint8 array of ceil(d/8) bytes (per vector).
    """
    flat = signs.ravel()
    bits = (flat > 0).astype(np.uint8)  # +1 → 1, -1 → 0
    # Pad to multiple of 8
    pad_len = (-len(bits)) % 8
    if pad_len:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
    groups = bits.reshape(-1, 8)
    packed = np.zeros(len(groups), dtype=np.uint8)
    for i in range(8):
        packed |= groups[:, i] << i
    return packed


def unpack_signs(packed: np.ndarray, length: int) -> np.ndarray:
    """
    Unpack bitfield back to ±1 array.

    Args:
        packed: Packed uint8 array from pack_signs.
        length: Number of original signs to recover.

    Returns:
        Array of ±1 int8 values, shape (length,).
    """
    result = np.empty(len(packed) * 8, dtype=np.int8)
    for i in range(8):
        result[i::8] = ((packed >> i) & 1).astype(np.int8)
    # Map 0 → -1, 1 → +1
    result = result * 2 - 1
    return result[:length]


# ---------------------------------------------------------------------------
# Batch packing helpers
# ---------------------------------------------------------------------------

def pack_indices_batch(indices_2d: np.ndarray, num_bits: int) -> np.ndarray:
    """Pack indices for a batch of vectors. Shape (n, d) → (n, packed_d)."""
    return np.array([pack_indices(row, num_bits) for row in indices_2d])


def unpack_indices_batch(packed_2d: np.ndarray, num_bits: int, d: int) -> np.ndarray:
    """Unpack a batch. Shape (n, packed_d) → (n, d)."""
    return np.array([unpack_indices(row, num_bits, d) for row in packed_2d])


def pack_signs_batch(signs_2d: np.ndarray) -> np.ndarray:
    """Pack signs for a batch. Shape (n, d) → (n, packed_d)."""
    return np.array([pack_signs(row) for row in signs_2d])


def unpack_signs_batch(packed_2d: np.ndarray, d: int) -> np.ndarray:
    """Unpack signs batch. Shape (n, packed_d) → (n, d)."""
    return np.array([unpack_signs(row, d) for row in packed_2d])
