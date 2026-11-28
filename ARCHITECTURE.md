# Visers Nano Architecture

Visers Nano takes 49,152 interleaved RGB bytes for a `128×128` image and returns an L2-normalized `f32` embedding. The model uses a MobileNetV4-Conv-Small-0.50 trunk, global average pooling, and a Matryoshka projection from 480 channels to 256, 128, or 64 dimensions.

## Build and initialization

[Python Forge](forge/README.md) exports a folded W8 checkpoint to `VSN1`. The [build script](build.rs) reads `weights/nano_v1.bin` or the file specified by `VISERS_NANO_WEIGHTS`. It keeps the first `dim` projection rows with their scales and biases and embeds the resulting model in `.rodata`. The runtime normalizes the projected vector after this dimensionality reduction.

The default Cargo features are `dim-256` and `simd`. The build script selects the largest dimension if several dimension features are enabled. It also selects 256 when no dimension feature is enabled.

`NanoEncoder::baked()` loads the embedded model through `NanoEncoder::from_bytes()`. The loader checks the channel chain, convolution geometry, activation and residual sizes, and the connection between the final activation and the projection. Each stored weight expands to `f32` as `weight_i8 * scale`. The model uses one scale per output channel and executes in `f32`. For the bundled 256D model, the decoded weights and biases occupy about 1.75 MiB and the embedded model occupies about 489 KiB.

## Inference path

The [Rust encoder](src/lib.rs) uses a lookup table for `(byte / 255 - mean) / std` for each RGB channel. The bundled model uses `mean = std = 0.5`. The following shapes describe the main stages of the bundled model. Activations use `height × width × channels` order.

| Stage | Output shape |
|---|---|
| `3×3` stride-2 stem and fused first block | `32×32×16` |
| Pointwise and `3×3` stride-2 convolutions | `16×16×48` |
| Inverted residual blocks | `8×8×48`, then `4×4×64` |
| Final `1×1` convolution | `4×4×480` |
| Global average pooling | `480` |
| Linear projection and L2 normalization | `dim` |

The fused stem keeps three rows of its `64×64×32` activation in a 24 KiB ring buffer. Later convolutions alternate between two activation buffers. The workspace also holds the saved residual, normalized RGB rows, and global pooling scratch. The bundled model's 64-byte-aligned workspace occupies about 271 KiB per caller or worker.

With the `simd` feature on x86-64, the encoder selects AVX2/FMA kernels when the CPU supports both features. The main pointwise kernel processes six pixels by 16 output channels, and the depthwise kernels clip padded taps once per output pixel and accumulate up to 64 channels in registers. The fused stem uses vector RGB normalization for the baked mean and standard deviation. Custom normalization uses the model's lookup table. The scalar path runs on other supported targets and provides a reference for [parity tests](tests/nano_parity.rs). The [benchmarks](BENCHMARKS.md) include timings for each stage.

The AVX2 implementation groups the [fused stem](src/avx2/stem.rs), [pointwise](src/avx2/pointwise.rs) and [spatial/depthwise convolutions](src/avx2/spatial.rs) by stage. The [scalar implementation](src/scalar.rs) uses the same model and workspace layout.

## Image file pipeline

The optional [image pipeline](src/image_io/mod.rs) runs one file or encoded byte slice through format detection, decoding, EXIF orientation, BOX resizing and RGB inference. Fast mode applies EXIF transforms to the final thumbnail. Full mode applies them to the source raster before either resize pass. A worker reuses the compressed file buffer, decoded pixel allocation, resize scratch and Nano workspace across calls. At 128×128 the encoder can directly borrow the decoder's RGB buffer. Other pixel layouts can drop alpha or expand gray after the per-channel BOX filter. The low-level RGB path accepts its original packed input directly.

- **Fast** uses the largest JPEG IDCT reduction among 1/8, 1/4 and 1/2 that leaves at least 256 samples on the shorter axis. Opaque still WebP with both sides at least 256 decodes directly into a 128×128 RGB buffer. PNG requires full raster decompression. Remaining resize work uses a SIMD BOX filter for typical inputs.
- **Full** uses full-resolution decoding and the BOX pass order and fixed-point rounding used by Forge. See the [Rust implementation](src/image_io/box_resize.rs) and [Pillow 13 resampler](https://github.com/python-pillow/Pillow/blob/13.0.0/src/libImaging/Resample.c). Both modes fit the whole image to the square, retain the stored RGB under alpha, and take the first frame of animated GIF and WebP.

The [file batch runner](src/image_io/batch.rs) shares immutable model weights across scoped workers. Each worker reads, decodes and encodes one image at a time, and the caller participates. `encode_files` returns a result for each path. The higher-level `encode` and `encode_with_options` fill contiguous embedding rows and report the lowest failing input index. Decoder-side scaling counts toward decode time, while EXIF and the software filter count toward resize time.

The Python extension keeps immutable input bytes alive, releases the GIL and calls the same Rust batch API. When inference completes, NumPy takes ownership of the Rust output buffer and exposes it as a C-contiguous `(N, D)` matrix. The single-image helper returns a one-dimensional view. See the [Python API guide](bindings/python/README.md).

## Allocation and batch execution

`encode_single_rgb()` initializes the embedded encoder once per process and the workspace once per calling thread. `NanoEncoder::encode_rgb()` accepts a reusable workspace and output buffer from the caller. Both paths avoid heap allocations in steady-state inference. `encode_batch_rgb()` collects the outputs into a `Vec` on the caller's thread.

`NanoSpinPool::new(encoder, threads)` counts the calling thread as one of the participants and gives each background worker its own workspace. The caller and workers claim image indexes from an atomic counter. Image `i` writes to `out[i * dim..(i + 1) * dim]`, so completion order has no effect on output order. The pool validates input and output buffers before dispatch. With background workers it serializes batches and waits for every worker before releasing access to the caller's buffers.

## VSN1 binary format

Multi-byte fields are little-endian. The file starts with a 44-byte header followed by convolution records and the linear projection.

| Header offset | Type | Field |
|---:|---|---|
| 0 | `[u8; 4]` | Magic `VSN1` |
| 4 | `u32` | Version `1` |
| 8 | `u32` | Convolution record count |
| 12 | `u32` | Projection input channels |
| 16 | `u32` | Projection output dimensions |
| 20 | `[f32; 3]` | RGB normalization means |
| 32 | `[f32; 3]` | RGB normalization standard deviations |

Each convolution record starts with a 16-byte descriptor. The fields are stored in the following order.

| Type | Fields |
|---|---|
| `u16 × 2` | Input channels, output channels |
| `u8 × 8` | Kernel size, stride, padding, depthwise flag, ReLU flag, save-residual flag, add-residual flag, reserved byte |
| `u32` | Weight payload length in bytes |

The descriptor is followed by one `f32` scale and one `f32` bias per output channel, with the full scale array preceding the bias array. The signed `int8` weights follow in `[kernel_y][kernel_x][input_channel][output_channel]` order for regular convolutions and `[kernel_y][kernel_x][channel]` order for depthwise convolutions.

The projection record contains `head_out` scales, `head_out` biases, and `head_out × head_in` signed `int8` weights. Each weight row stores all input channels for one output dimension. The exporter is in [forge/export_vsn1.py](forge/export_vsn1.py).
