"""Visers Nano CPU image embeddings."""

from collections.abc import Iterable
from os import PathLike, fspath
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native

__all__ = ["DIMENSIONS", "dimensions", "encode", "encode_single", "similarity", "distance"]

DIMENSIONS: int = _native.DIMENSIONS
ImageSource: TypeAlias = str | PathLike[str] | bytes | bytearray | memoryview
DecodeMode: TypeAlias = Literal["fast", "full"]


def dimensions() -> int:
    """Return the embedding width baked into this build."""
    return DIMENSIONS


def _source(image: ImageSource, index: int) -> str | bytes:
    if isinstance(image, bytes):
        return image
    if isinstance(image, (bytearray, memoryview)):
        return bytes(image)
    if isinstance(image, (str, PathLike)):
        path = fspath(image)
        if isinstance(path, str):
            return path
    raise TypeError(f"image[{index}] must be a text path or encoded image bytes")


def encode(
    images: ImageSource | Iterable[ImageSource],
    *,
    mode: DecodeMode = "fast",
    jobs: int | None = None,
) -> NDArray[np.float32]:
    """Return a contiguous float32 matrix of shape (N, DIMENSIONS), in input order.

    A single input gives one row and an empty iterable gives zero rows. Set mode to
    'full' for Forge preprocessing. jobs limits the number of concurrent images.
    """
    if isinstance(images, (str, PathLike, bytes, bytearray, memoryview)):
        sources = [_source(images, 0)]
    else:
        sources = [_source(image, i) for i, image in enumerate(images)]
    return _native.encode(sources, mode, jobs)


def encode_single(image: ImageSource, *, mode: DecodeMode = "fast") -> NDArray[np.float32]:
    """Return the one-dimensional embedding for a single file or encoded buffer."""
    return _native.encode([_source(image, 0)], mode, 1)[0]


def _vector(values: ArrayLike) -> NDArray[np.float32]:
    array = np.require(values, dtype=np.float32, requirements=["C", "A"])
    if array.ndim != 1:
        raise ValueError("expected a one-dimensional embedding")
    return array


def similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine similarity between two finite nonzero vectors of equal length."""
    return _native.similarity(_vector(a), _vector(b))


def distance(a: ArrayLike, b: ArrayLike) -> float:
    """Euclidean distance between two finite vectors of equal, nonzero length."""
    return _native.distance(_vector(a), _vector(b))
