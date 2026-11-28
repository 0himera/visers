"""Compatibility import for the Visers Nano Python API."""

from visers import (
    DIMENSIONS as DIMENSIONS,
    DecodeMode as DecodeMode,
    ImageSource as ImageSource,
    dimensions as dimensions,
    distance as distance,
    encode as encode,
    encode_single as encode_single,
    similarity as similarity,
)

__all__ = ["DIMENSIONS", "dimensions", "encode", "encode_single", "similarity", "distance"]
