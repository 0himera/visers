# Visers Nano Architecture

Visers Nano takes 49,152 interleaved RGB bytes for a `128×128` image and returns an L2-normalized `f32` embedding. The model uses a MobileNetV4-Conv-Small-0.50 trunk, global average pooling, and a Matryoshka projection from 480 channels to 256, 128, or 64 dimensions.

## Build and initialization

[Python Forge](forge/README.md) exports a folded W8 checkpoint to `VSN1`. The [build script](build.rs) reads `weights/nano_v1.bin` or the file specified by `VISERS_NANO_WEIGHTS`. It keeps the first `dim` projection rows with their scales and biases and embeds the resulting model in `.rodata`. The runtime normalizes the projected vector after this dimensionality reduction.

`NanoEncoder::baked()` loads the embedded model through `NanoEncoder::from_bytes()`. The loader checks tensor shapes, activation and residual sizes, and the connection between the final activation and the projection. It expands signed `int8` weights into aligned `f32` arrays using one scale per output channel. W8 is the storage format and inference arithmetic uses `f32`.

## Inference and workspace

The [Rust encoder](src/lib.rs) normalizes RGB bytes through a lookup table. The fused stem and first block share a rolling three-row buffer. The remaining pointwise, depthwise, and regular convolutions work on channel-last activations in a 64-byte-aligned `NanoWorkspace`. The workspace holds two alternating activation buffers, a residual buffer, and scratch for the stem and global pooling.

With the `simd` feature on x86-64, the encoder selects AVX2/FMA kernels when the CPU supports both features. Those kernels tile pixels and output channels across vector registers. The scalar path runs on other supported targets and provides a reference for parity tests.

Create a workspace and an output buffer once per concurrent caller to avoid allocations in steady-state inference. `NanoSpinPool` gives each background worker its own workspace and uses the calling thread as a participant. Pools with background workers process one batch at a time. Embeddings are written in input order.

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
