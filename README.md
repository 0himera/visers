# Visers Nano

**Sub-millisecond CPU visual embeddings in pure Rust.**  
`0.457 ms` per `128×128` RGB image · `10,083 img/s` on 6 cores · `0` heap allocations · `489 KiB` weights embedded in `.rodata`.

![Speed vs quality, per task](assets/pareto-speed-quality.png)

`visers-nano` distills a `SigLIP2 Base` vision teacher into a quantization-aware `MobileNetV4-Conv-Small-0.50` backbone (`W8` symmetric per-channel weights + Matryoshka `256D` / `128D` / `64D` head) and compiles it into a dependency-free Rust crate with hand-tuned **AVX2 + FMA** kernels, fused RGB-to-uint8/float stem, and a **zero-allocation `NanoSpinPool`**.

---

## Quick Start

Add `visers-nano` to your `Cargo.toml` (weights are compiled directly into the binary — no external model files or ONNX Runtime needed):

```toml
[dependencies]
visers-nano = { git = "https://github.com/0himera/visers-nano" }
```

Encode a `128×128×3` interleaved `RGB` (`u8`) image into an L2-normalized embedding:

```rust
use visers_nano::{encode_rgb128, DIM};

fn main() {
    let rgb = vec![128u8; 128 * 128 * 3];
    let embedding: [f32; DIM] = encode_rgb128(&rgb);
    println!("dim={}, norm_sq={:.6}", embedding.len(), embedding.iter().map(|x| x * x).sum::<f32>());
}
```

> **Always run in `--release` mode.** Debug builds disable SIMD inlining and are ~15× slower.

For high-throughput workloads with zero per-call allocations, reuse a `NanoScratch` buffer or `NanoSpinPool`:

```rust
use visers_nano::{encode_rgb128_into, NanoScratch, NanoSpinPool, DIM};

// Single-thread zero-alloc loop
let mut scratch = NanoScratch::new();
let mut out = [0.0f32; DIM];
encode_rgb128_into(&rgb, &mut scratch, &mut out);

// Multi-core batch encoding (10,083 img/s on 6 threads)
let pool = NanoSpinPool::new(6);
pool.encode_batch_rgb128(&batch_rgb, &mut batch_out);
```

---

## Why Visers Nano?

- **Zero runtime dependencies:** No ONNX Runtime, no PyTorch/libtorch, no Python, no filesystem I/O at startup.
- **Sub-millisecond CPU latency:** `0.457 ms` P50 single-thread (`1,839 img/s`), `0.094 ms/img` across 6 cores (`10,083 img/s`).
- **4.2× faster & +0.147 nDCG@10 over `Visers v0.3`:** Replaces legacy custom micro-CNNs with a `SigLIP2`-distilled `MobileNetV4` trunk.
- **11.8× faster than `CPUBone Nano (224px)`** at matching retrieval quality (`0.531` vs `0.533` selection nDCG@10).
- **Matryoshka (MRL) dimension flags:** Select `dim-256` (default, `489 KiB`), `dim-128` (`428 KiB`), or `dim-64` (`397 KiB`) at compile time.
- **Portable x86_64 + scalar fallback:** Uses `is_x86_feature_detected!("avx2")` + `fma` at runtime and automatically falls back to scalar code on non-AVX2 CPUs.

---

## Benchmarks

### Retrieval Quality & Latency (`abstract-v2`, 5,940 images)

Measured single-threaded on **Intel Core i7-8700K @ 3.70 GHz** (`taskset -c 0`, excluding disk I/O and JPEG/PNG decoding):

| Model | Backend | CPU P50 (ms) | Art Style | Art Genre | Cross-Genre | Photo (COCO) |
|---|---|---:|---:|---:|---:|---:|
| **Visers Nano (`W8`, `256D`)** | **Rust AVX2 · 128px** | **0.457** | **0.7686** | **0.5291** | **0.6776** | **0.4126** |
| Visers Nano (`FP32`, `256D`) | ORT · 128px | 0.748 | 0.7686 | 0.5291 | 0.6776 | 0.4126 |
| Visers v0.2 | Rust · 128px | 1.279 | 0.6451 | 0.3724 | — | 0.2082 |
| Visers v3 (`v0.3`) | Rust AVX2 · 128px | 1.916 | 0.6688 | 0.4107 | 0.5888 | 0.2269 |
| CPUBone Nano | ORT FP32 · 128px | 3.580 | 0.7160 | 0.4675 | 0.6081 | 0.3982 |
| CPUBone Nano | ORT FP32 · 224px | 5.403 | 0.7332 | 0.4883 | 0.6203 | 0.4560 |
| CPUBone B0 | ORT FP32 · 224px | 11.475 | 0.7304 | 0.4843 | — | 0.4903 |
| TinyCLIP 8M | ORT FP32 · 224px | 37.405 | 0.7741 | 0.5363 | — | 0.5152 |
| MobileCLIP2-S0 | ORT FP32 · 256px | 65.803 | 0.7718 | 0.5707 | — | 0.5953 |
| MobileCLIP2-S2 | ORT FP32 · 256px | 184.226 | 0.8019 | 0.5916 | — | 0.6274 |
| *SigLIP2 Base (Teacher)* | *ViT-B/16 · 224px* | *—* | *0.8380* | *0.5907* | *—* | *0.6084* |

![Per-task quality and throughput vs larger models](assets/domain-comparison.png)

### Runtime Throughput & Per-Stage Profile

![Throughput vs CPUBone Nano and stage profile](assets/nano-throughput-profile.png)

See [BENCHMARKS.md](BENCHMARKS.md) for full `visers-v6` artist-disjoint metrics, `Flowers102` OOD linear-probe scores, and FP32-vs-Rust parity checks.

---

## Cargo Features

| Feature | Embedding Dim | `.rodata` Weights | Rust vs FP32 Max Diff |
|---|---:|---:|---:|
| `dim-256` *(default)* | `256` | `500,572 B` (`489 KiB`) | `8.34e-7` |
| `dim-128` | `128` | `438,556 B` (`428 KiB`) | `8.94e-7` |
| `dim-64` | `64` | `406,876 B` (`397 KiB`) | `8.94e-7` |

```toml
# Example: 64D embeddings (397 KiB weights, 0.441 ms P50)
visers-nano = { git = "https://github.com/0himera/visers-nano", default-features = false, features = ["dim-64"] }
```

## CLI & Benchmarks

```bash
# Run single-thread + 6-thread NanoSpinPool benchmarks
cargo run --release --bin nano_rgb -- --bench

# Print per-stage microsecond profile
cargo run --release --bin nano_rgb -- --profile

# Encode a raw 128x128 RGB file or directory
cargo run --release --bin nano_rgb -- path/to/image_128x128.rgb
```

## License

Dual-licensed under **MIT** or **Apache-2.0**. See [ARCHITECTURE.md](ARCHITECTURE.md) for details on the `VSN1` weight format and SIMD kernel design.
