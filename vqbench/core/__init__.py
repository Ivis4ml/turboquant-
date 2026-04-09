"""Core abstractions: base classes, rotation, metrics, packing."""

from vqbench.core.base import VectorQuantizer, QuantizedVector
from vqbench.core.rotation import (
    haar_rotation, fast_walsh_hadamard, fwht_batch,
    StructuredRotation, get_structured_rotation,
)
from vqbench.core.metrics import mse_distortion, ip_distortion, ip_bias, recall_at_k
from vqbench.core.packing import (
    pack_indices, unpack_indices, pack_signs, unpack_signs,
)
