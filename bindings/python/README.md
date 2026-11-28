# Visers Nano Python API

`visers` computes L2-normalized visual embeddings on the CPU. It calls the same Rust decoder, resizer and Nano model as the `nano_image` CLI. The wheel contains the model weights and returns NumPy arrays directly. Existing applications can also import the API as `visers_nano`.

## Install

Add Visers to your Python 3.10+ project with uv.

```bash
uv add visers
```

To install from source, run these commands from the repository root.

```bash
uv venv
uv pip install .
```

A source build needs Rust 1.88 or newer, a C compiler and CMake. NASM enables SIMD in the bundled JPEG decoder on x86. uv installs NumPy and uses maturin to build the native extension.

## Encode images

Replace `photo.jpg` with a local image. Its path and encoded bytes produce the same embedding.

```python
from pathlib import Path

import numpy as np
import visers as nano

image = Path("photo.jpg")
upload = image.read_bytes()

vector = nano.encode_single(image)
rows = nano.encode([image, upload], jobs=2)

assert vector.shape == (nano.DIMENSIONS,)
assert rows.shape == (2, nano.dimensions())
assert rows.dtype == np.float32
np.testing.assert_array_equal(rows[0], rows[1])
print(nano.similarity(vector, rows[1]))
```

| Call | Result |
|---|---|
| `encode_single(image, *, mode="fast")` | One contiguous `float32` array of shape `(D,)` |
| `encode(images, *, mode="fast", jobs=None)` | A C-contiguous `float32` matrix of shape `(N, D)`, in input order |
| `DIMENSIONS`, `dimensions()` | Embedding width of the installed build, 256 by default |
| `similarity(a, b)` | Cosine similarity of two finite nonzero vectors |
| `distance(a, b)` | Euclidean distance between two finite vectors |

An image can be a `str` or `os.PathLike[str]` path, or the contents of a JPEG, PNG, WebP, GIF, BMP or PNM file as `bytes`, `bytearray` or `memoryview`. The file signature selects the decoder. Immutable `bytes` are borrowed during inference. Mutable buffers and memory views are copied when accepted.

`encode` takes one image or an iterable of images. One image yields shape `(1, D)`, and an empty iterable yields `(0, D)`. The iterable is gathered before processing. For a large collection, submit batches that fit in memory.

The two comparison functions take one-dimensional array-like inputs of equal, nonzero length. They convert values to `float32` and accumulate in `float64`. For example, compare two rows using `nano.distance(rows[0], rows[1])`.

## Preprocessing and workers

- `mode="fast"` uses decoder-side scaling for large JPEGs and opaque still WebPs. This is the default for Python, the Rust convenience functions and the CLI.
- `mode="full"` decodes the full raster, applies EXIF orientation and reproduces Forge's Pillow 13 BOX resize of the whole frame to 128×128. Choose the same mode when building an index and embedding queries.
- `jobs=None` uses available CPU parallelism, limited by the number of inputs. A positive integer caps workers per call. Use `jobs=1` to process a batch sequentially. The extension releases the GIL during decoding and inference.
- The process loads the baked model on first use. A calling thread reuses its decoder and inference workspace across calls. Workers write their results directly into rows of the output matrix.

## Errors

Invalid input types raise `TypeError`. Malformed or unsupported images and invalid modes raise `ValueError`. Filesystem failures raise `OSError` subclasses such as `FileNotFoundError`. Inputs are type-checked before dispatch. For read or decode failures, the exception identifies the lowest failing input index, starting at zero, and includes the path when applicable. A successful call returns all rows.

## Build a different embedding width

The default wheel embeds the 256D head. To build a 64D wheel, run maturin from the repository root with `uvx`. Use `dim-128` for 128D. Each build exports its width through `DIMENSIONS`.

```bash
uvx --from "maturin>=1.12,<2" maturin build --release --locked --features dim-64 --out dist/64
```
