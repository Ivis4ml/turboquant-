"""
Method registry — central import point for all quantizers.

eval/ and kv_cache/ import from here, never from individual methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vqbench.core.base import VectorQuantizer


def get_registry() -> dict[str, type[VectorQuantizer]]:
    """Return the quantizer registry, importing lazily to avoid circular deps."""
    from vqbench.methods.turboquant.mse import TurboQuantMSE
    from vqbench.methods.turboquant.prod import TurboQuantProd
    from vqbench.methods.turboquant.qjl import QJLQuantizer
    from vqbench.methods.rabitq.rabitq_1bit import RaBitQ1Bit
    from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
    from vqbench.methods.pq.product_quant import ProductQuantizer
    from vqbench.methods.pq.opq import OptimizedPQ

    return {
        "TurboQuantMSE": TurboQuantMSE,
        "TurboQuantProd": TurboQuantProd,
        "QJL": QJLQuantizer,
        "RaBitQ1Bit": RaBitQ1Bit,
        "ExtRaBitQ": ExtRaBitQ,
        "PQ": ProductQuantizer,
        "OPQ": OptimizedPQ,
    }


def get_all_quantizers(d: int, num_bits: int, seed: int = 42) -> list[VectorQuantizer]:
    """Instantiate all registered quantizers with same params."""
    return [cls(d=d, num_bits=num_bits, seed=seed) for cls in get_registry().values()]
