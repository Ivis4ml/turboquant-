"""Core abstractions: base classes, rotation, metrics."""

from vqbench.core.base import VectorQuantizer, QuantizedVector
from vqbench.core.rotation import haar_rotation, fast_walsh_hadamard
from vqbench.core.metrics import mse_distortion, ip_distortion, ip_bias, recall_at_k
