# Visers Nano

**Sub-millisecond, zero-allocation CPU visual embedding engine in pure Rust.**  
Weights are quantized (`W8`) and baked directly into the binary's `.rodata` at compile time — zero Python, zero ONNX Runtime, zero runtime model downloads.

Inspired by [`flowercomputers/encoders`](https://github.com/flowercomputers/encoders), **Visers Nano** brings the same compile-time static embedding ergonomics to **128×128 RGB images** with Matryoshka dimensionality slicing (`dim-64`, `dim-128`, `dim-256`).

---

## Highlights

- **0.457 ms** single-thread `P50` (`1,839 img/s`) from raw `128×128×3` `uint8` RGB to L2-normalized `256D` vector on Intel i7-8700K (`0.441 ms` at `64D`, **faster than ONNX Runtime** even before counting ONNX's preprocessing).
- **0.0935 ms/image** (`10,083 img/s`) on 6 cores via the built-in lock-free spin-wait `NanoSpinPool` (bit-exact with single-thread inference).
- **488.8 KiB** (`dim-256`) / **397.3 KiB** (`dim-64`) weights baked into `.rodata` (`build.rs`).
- **0 heap allocations** during inference (`NanoWorkspace` + `NanoSpinPool`).
- **Zero external Rust crate dependencies** (`Cargo.toml` `[dependencies]` is empty).
- **Numerical parity `< 9e-7`** (`cosine > 0.9999994`) against the PyTorch `W8` model.

---

## Quick Start

### Encode a single `128×128×3` RGB buffer

```rust
use visers_nano::{encode_single_rgb, NANO_EMBEDDING_DIM, RGB_BYTES};

fn main() -> Result<(), &'static str> {
    let rgb = vec![128u8; RGB_BYTES]; // 128 * 128 * 3 uint8 RGB
    let emb: [f32; NANO_EMBEDDING_DIM] = encode_single_rgb(&rgb)?;
    assert_eq!(emb.len(), 256); // or 128 / 64 via Cargo features
    Ok(())
}
```

### Zero-allocation explicit workspace & multi-core `NanoSpinPool`

```rust
use std::sync::Arc;
use visers_nano::{NanoEncoder, NanoSpinPool, RGB_BYTES};

fn main() -> Result<(), &'static str> {
    let encoder = Arc::new(NanoEncoder::baked()?);
    let mut ws = encoder.workspace();
    let dim = encoder.embedding_dim();

    // Single image (0 heap allocations)
    let img = vec![0u8; RGB_BYTES];
    let mut out = vec![0.0f32; dim];
    encoder.encode_rgb(&img, &mut ws, &mut out)?;

    // Multi-core batch (lock-free spin-wait, 0 heap allocations, bit-exact)
    let pool = NanoSpinPool::new(Arc::clone(&encoder), 6);
    let batch = vec![img.as_slice(); 128];
    let mut batch_out = vec![0.0f32; 128 * dim];
    pool.encode_batch(&batch, &mut ws, &mut batch_out)?;
    Ok(())
}
```

### Compile-time Matryoshka feature flags

```bash
# Default: 256D + runtime AVX2/FMA SIMD (500,572 B baked)
cargo test --release

# 128D Matryoshka slice
cargo test --release --no-default-features --features dim-128,simd

# 64D Matryoshka slice (406,876 B baked)
cargo test --release --no-default-features --features dim-64,simd
```

To override the baked weights path at build time:

```bash
VISERS_NANO_WEIGHTS=/path/to/nano_v1.bin cargo build --release
```

---

## Comparison Against Baselines

All `abstract-v2` held-out test metrics (`5,940` images: `1,686` art + `4,254` photo) and single-thread `i7-8700K` CPU latencies (`batch = 1`, `P50 ms`, no disk/image-file decode):

| Model | Backend & Input | Weights (INT8 / file) | CPU P50 (ms) | Art Style | Art Genre | Photo (COCO) |
|---|---|---:|---:|---:|---:|---:|
| **Visers Nano (`dim-256`)** | **Rust AVX2/FMA · 128px RGB** | **489 KiB** (`.rodata`) | **0.457** | **0.7686** | **0.5291** | **0.4126** |
| **Visers Nano (`dim-64`)** | **Rust AVX2/FMA · 128px RGB** | **397 KiB** (`.rodata`) | **0.441** | — | — | — |
| Visers Nano (ORT baseline) | ONNX Runtime FP32 · 128px RGB | 1.8 MiB FP32 | 0.748 (0.562 tensor) | 0.7686 | 0.5291 | 0.4126 |
| Visers v0.2 | Rust · 128px RGB | 2.6 MiB FP32 | 1.279 | 0.6451 | 0.3724 | 0.2082 |
| Visers v3 (`v0.3` specialists) | Rust AVX2/FMA · 128px RGB | — | 1.916 | 0.6688 | 0.4107 | 0.2269 |
| CPUBone Nano (zero-shot) | ONNX Runtime FP32 · 128px | 4.7 MiB INT8 | 3.580 (2.634 tensor) | 0.7160 | 0.4675 | 0.3982 |
| CPUBone Nano (zero-shot) | ONNX Runtime FP32 · 224px | 4.7 MiB INT8 | 5.403 | 0.7332 | 0.4883 | 0.4560 |
| CPUBone B0 (zero-shot) | ONNX Runtime FP32 · 224px | 8.4 MiB INT8 | 11.475 | 0.7304 | 0.4843 | 0.4903 |
| TinyCLIP 8M | ONNX Runtime FP32 · 224px | ~8M params | 37.405 | 0.7741 | 0.5363 | 0.5152 |
| MobileCLIP2-S0 | ONNX Runtime FP32 · 256px | — | 65.803 | 0.7718 | 0.5707 | 0.5953 |
| MobileCLIP2-S2 | ONNX Runtime FP32 · 256px | — | 184.226 | 0.8019 | 0.5916 | 0.6274 |
| SigLIP2 Base (teacher) | ViT-B/16 · 224px (GPU) | 92.9M params | — | 0.8380 | 0.5907 | 0.6084 |

See [BENCHMARKS.md](BENCHMARKS.md) for the strict artist-disjoint `visers-v6` test comparison, `Flowers102` out-of-distribution probe, Matryoshka `64/128/256` breakdown, per-stage CPU profile, and PyTorch parity details.

---

## Architecture & Engine Design

- **Backbone**: `mobilenetv4_conv_small_050.e3000_r224_in1k` `features_only` trunk (`45` folded `Conv2d + ReLU` layers, `20.64M` MACs at `128×128`, all intermediate channels multiples of `16`) + `Linear(480 → 256)` Matryoshka head (`458,480` total parameters).
- **Distillation**: `SigLIP2 Base` (`google/siglip2-base-patch16-224`) PCA-256 relational distillation (`cosine + KL rank + Gram`) across `~278k` images (`100k ImageNet-1k + 100k CC3M + 77.8k WikiArt/COCO`), followed by `Conv+BN` folding and `2,000` steps of `W8` quantization-aware training.
- **CPU Runtime (`src/lib.rs`)**:
  - **Fused Stem + Block 0.0**: `conv_stem` (`3→32`, `3.54M` MACs) and `blocks.0.0` (`32→16`, `4.72M` MACs) execute row-by-row through a 64B-aligned `24 KiB` L1d ring buffer + `4.6 KiB` rolling RGB LUT strip, keeping all activations inside L1d/L2.
  - **64-byte aligned arena + anti-4K-aliasing stagger** (`+64B` offset between ping-pong buffers).
  - **Register-tiled AVX2/FMA microkernels** (`8..12` YMM accumulators in flight) + portable scalar fallback.
  - **RAII `FTZ | DAZ`** (`MXCSR |= 0x8040`) during `x86_64` inference.

---

## CLI & Benchmarks

```bash
# Run unit, SIMD/scalar parity, zero-alloc, and spin-pool tests
cargo test --release

# Single-thread CPU latency benchmark on raw 128x128x3 RGB corpus
cargo run --release --example nano_rgb -- /path/to/corpus.rgb --bench

# Per-stage microsecond breakdown
cargo run --release --example nano_rgb -- /path/to/corpus.rgb --profile

# Multi-core lock-free spin-pool benchmark (e.g. 6 threads)
cargo run --release --example nano_rgb -- /path/to/corpus.rgb --bench-batch 6
```
