"""Compatibility imports for the former monolithic V16 adapter.

New code should import from :mod:`bertani.v16`. The implementation now lives
in the independent Rust V16 core plus separate Python trace/opponent modules.
"""

from .v16 import (
    V16CacheStats,
    V16OpponentPolicy,
    V16Trace,
    encode_v16_trace,
    load_v16_actions,
    load_v16_trace,
)

NativeV16Policy = V16OpponentPolicy

__all__ = [
    "NativeV16Policy",
    "V16CacheStats",
    "V16OpponentPolicy",
    "V16Trace",
    "encode_v16_trace",
    "load_v16_actions",
    "load_v16_trace",
]
