# Visers Nano — Benchmarks & Verification

The RGB inference timings were measured on **Intel Core i7-8700K @ 3.70 GHz** (Coffee Lake, AVX2/FMA, 32 KiB L1d / 256 KiB L2 per core, 12 MiB L3), pinned with `taskset`, excluding disk I/O and JPEG/PNG decoding.

![Speed vs quality, per task](assets/pareto-speed-quality.png)

---

## 1. Retrieval Quality

![Per-task quality and throughput vs larger models](assets/domain-comparison.png)

### A. `abstract-v2` Held-Out Test (`5,940` images: `1,686` art + `4,254` photo)

| Model | Backend | CPU P50 (ms) | Art Style | Art Genre | Cross-Genre Style | Photo (COCO) |
|---|---|---:|---:|---:|---:|---:|
| **Visers Nano (`W8`, `256D`)** | **Rust AVX2/FMA · 128px** | **0.431** | **0.7686** | **0.5291** | **0.6776** | **0.4126** |
| Visers v0.2 | Rust · 128px | 1.279 | 0.6451 | 0.3724 | — | 0.2082 |
| Visers v3 (`v0.3`) | Rust AVX2/FMA · 128px | 1.916 | 0.6688 | 0.4107 | 0.5888 | 0.2269 |
| CPUBone Nano (zero-shot) | ORT FP32 · 128px | 2.222 | 0.7160 | 0.4675 | 0.6081 | 0.3982 |
| CPUBone Nano (zero-shot) | ORT FP32 · 224px | 5.727 | 0.7332 | 0.4883 | 0.6203 | 0.4560 |
| CPUBone B0 (zero-shot) | ORT FP32 · 224px | 11.475 | 0.7304 | 0.4843 | — | 0.4903 |
| TinyCLIP 8M | ORT FP32 · 224px | 37.405 | 0.7741 | 0.5363 | — | 0.5152 |
| MobileCLIP2-S0 | ORT FP32 · 256px | 65.803 | 0.7718 | 0.5707 | — | 0.5953 |
| MobileCLIP2-S2 | ORT FP32 · 256px | 184.226 | 0.8019 | 0.5916 | — | 0.6274 |
| SigLIP2 Base (teacher) | ViT-B/16 · 224px | — | 0.8380 | 0.5907 | — | 0.6084 |

### B. Strict Artist-Disjoint `visers-v6` Held-Out Test (`12,075` images: `9,576` art + `2,499` photo)

| Model | Dim | Precision | Art Style | Art Genre | Photo (COCO) | Selection Score |
|---|---:|---|---:|---:|---:|---:|
| Visers v6 (`both-s20261115`) | 256 | FP32 | 0.2575 | 0.4323 | 0.2444 | 0.2946 |
| **Visers Nano** | **256** | **W8** | **0.3587** | **0.6096** | **0.4181** | **0.4511** |
| **Visers Nano** | **128** | **W8** | — | — | — | **0.4501** |
| **Visers Nano** | **64** | **W8** | — | — | — | **0.4435** |

### C. Out-of-Distribution Generalization (`Flowers102`: `1,020` train / `6,149` test)

| Model | Dim | kNN-5 | Ridge Top-1 |
|---|---:|---:|---:|
| Previous `spatial-bottleneck` (trained from scratch) | 128 | 0.1807 | 0.2683 |
| **Visers Nano (`W8`)** | **256** | **0.3706** | **0.5269** |

---

## 2. CPU Latency, Throughput & Memory

The runtime comparison uses the same 128 `128×128` RGB images (`validation.rgb` from the upstream training workspace, SHA-256 `ff37815083df93039770d149fd36e91e2b2f7cb9228393edf6b985cd5e934514`). Single-thread runs use `taskset -c 4`. Six-core runs use `taskset -c 0-5`, which selects six distinct physical cores on this CPU. Timings cover preprocessing and inference from in-memory RGB frames. Throughput uses the number of images divided by elapsed time. Reported figures are medians of three warmed runs. Retrieval scores and timing use separate splits.

### Visers Nano (release, AVX2/FMA)

`--bench` times 640 warmed images per run (`128 × 5`). `--bench-batch 6` times 20 batches of 128 images through `NanoSpinPool`. Its P50 is the median batch time divided by the batch size. The 64D row is from the earlier single-thread run on the same corpus.

| Variant | Threads | Baked `.rodata` | P50 (ms) | P90 (ms) | P99 (ms) | Mean (ms) | Throughput (img/s) | Heap Allocs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dim-256` | 1 | 500,572 B | **0.431** | 0.460 | 0.516 | 0.441 | **2,268.3** | **0** |
| `dim-64` | 1 | 406,876 B | **0.441** | 0.733 | 1.219 | 0.526 | **1,900.5** | **0** |
| `NanoSpinPool` (`dim-256`) | 6 | 500,572 B | **0.0752** | — | — | — | **13,224.2** | **0** |

### Single-core RGB inference

The current 256D encoder measured **0.399 ms P50** on the i7-8700K. The same run set measured **0.438 ms** for the original engine at `3427657` and **0.383 ms** for an experimental build with additional pointwise and 3×3 kernels. The Rust and Python APIs use the 0.399 ms implementation.

Seven interleaved runs per build used `taskset -c 4` and `nano_rgb validation.rgb --bench`. Each run warmed the 128 RGB frames twice and timed five passes from RGB24 in memory through the normalized embedding. Values are medians of the seven per-run P50s. All three builds returned byte-identical embeddings for all 128 frames.

### CPUBone Nano (ONNX Runtime FP32)

The `cpubone-native/encoder-128.onnx` and `fast-baselines/cpubone-nano/encoder.onnx` exports from `visers/artifacts/abstract-v2/` ran with ONNX Runtime 1.31.1 `CPUExecutionProvider`, full graph optimization and one inter-op thread. Each inference used batch size 1. Preprocessing applied ImageNet normalization and, for the 224px model, bicubic resize and center crop.

| Model input | 1 core P50 (ms) | 1 core (img/s) | 6 cores, 6 single-thread sessions (img/s) | 6 cores, 1 six-thread session (img/s) |
|---|---:|---:|---:|---:|
| `128×128` | 2.222 | 445.1 | **1,230.9** | 757.5 |
| `224×224` | 5.727 | 173.6 | **727.0** | 335.2 |

The single-session runs timed 640 images each. The six-session runs timed eight batches of 128 images, distributing each batch across six concurrent workers with one ORT session per worker. The throughput chart uses the six-session results for the 224px model.

### Visers Nano and MobileCLIP2-S0 (ONNX Runtime FP32)

`visers/artifacts/nano-v1/nano-v1-mrl256.onnx` uses the trained W8 weights stored as FP32 and runs in FP32. Its timed preprocessing converts interleaved RGB to a normalized FP32 tensor using a reusable buffer. MobileCLIP2-S0 uses `fast-baselines/mobileclip2-s0/encoder.onnx` from `visers/artifacts/abstract-v2/` with its 256px bilinear preprocessing. It is the closest Art Style baseline in this comparison at `0.7718` nDCG@10 versus `0.7686` for Visers Nano.

| Model | 1 core (img/s) | 6 cores, 6 single-thread sessions (img/s) |
|---|---:|---:|
| Visers Nano 256D · 128px | 1,752.1 | 8,518.2 |
| MobileCLIP2-S0 · 256px | 16.1 | 78.0 |

Both models used the same ONNX Runtime settings as CPUBone Nano. For Visers Nano, each run timed 640 images on one core and 20 batches of 128 images on six cores. For MobileCLIP2-S0, each run timed 384 images on one core and eight batches of 128 images on six cores.

In the domain comparison, TinyCLIP 8M and MobileCLIP2-S2 use the earlier measured single-core throughput from `visers/reports/abstract-v2/fast-baselines.json` on the same RGB corpus.

### Per-Stage Breakdown (`dim-256`, `--profile`, 640 runs)

Mean stage times from the run with the median total profile time of `444.6 µs`. MAC counts describe the nominal convolution shapes, including padding.

| Stage | MACs | Time (µs) | Share |
|---|---:|---:|---:|
| Fused `conv_stem` (`3→32`) + `blocks.0.0` (`32→16`) + `u8→f32` LUT | 8.26M | 147.72 | 33.2% |
| Pointwise `1×1` (27 layers) | 9.44M | 171.11 | 38.5% |
| Depthwise `3×3` / `5×5` (15 layers) | 1.06M | 73.63 | 16.6% |
| `conv3x3` (`blocks.1.0`, `16→48`) | 1.77M | 44.50 | 10.0% |
| Global Avg Pool + `Linear(480→256)` + L2 norm | 0.12M | 7.60 | 1.7% |

---

## 3. PyTorch `W8` ↔ Native Rust Numerical Parity

Measured across all 128 canonical `128×128` RGB images in `validation.rgb` (with `FTZ | DAZ` enabled in Rust):

| Matryoshka Dim | Max Abs Error | Min Cosine Similarity |
|---:|---:|---:|
| `256` | `8.34e-7` | `0.99999946` |
| `128` | `8.94e-7` | `0.99999952` |
| `64` | `8.94e-7` | `0.99999970` |

---

## 4. Image file to 256D embedding

The file benchmark starts with a file path and ends with an embedding in memory. The model is loaded once. File open/read, decode, EXIF orientation, resize and inference are timed. The OS file cache is warm. Process startup and output serialization are outside these timings.

| Input | Pillow + ORT P50, ms | Rust Full P50, ms | Rust Fast P50, ms | Rust Fast decode P50, ms |
|---|---:|---:|---:|---:|
| JPEG, 1920×1080 | 10.82 | 7.66 | 4.41 | 3.82 |
| PNG, 1920×1080 | 48.99 | 20.59 | 19.36 | 18.13 |
| WebP, 1920×1080 | 24.88 | 18.87 | 16.38 | 15.87 |
| 128 original art JPEGs, 512 px on the long side | 2.36 | 1.78 | 1.57 | 0.99 |

The 1080p set contains 12 images from the upstream `abstract-v1/images` directory, resized with Pillow BICUBIC and encoded as JPEG quality 88, RGB PNG compression level 4 and lossy WebP quality 84. The 12 are the first entries from `random.Random(47).sample(sorted(paths), 128)` over the JPEG paths. The original-size row uses all 128 files directly. The Rust and Python paths read the same files. Dimensions, file compressibility and storage latency affect these results.

The single-worker Rust runs used `taskset -c 4` on the i7-8700K with the release profile, AVX2/FMA, system libturbojpeg 3.2.0 and bundled libwebp. After warming the files, each of three measured batches traversed its input three times. Per-file medians use the last two traversals. Stage medians are independent. Pillow 13.0.0 used in-place EXIF transpose, RGB conversion as needed, BOX128 and lookup-table normalization to contiguous FP32 NCHW. ONNX Runtime 1.31.1 used the matching Nano W8-folded FP32 graph, one CPU thread, full graph optimization and prebound input/output. The Python comparison used one warm traversal and three measured traversals.

**Ready RGB and fresh CLI processes.** On the canonical 128-frame RGB corpus, the median of five interleaved single-core P50 measurements was 0.434 ms for the original Rust encoder (commit `3427657`), 0.404 ms for the updated RGB/depthwise kernels, and 0.469 ms for the native ORT C API with SIMD input preparation. All three used warmed models. Launching a fresh `nano_image` process and reading its JSON vector on stdout took 4.59 ms P50 for one original JPEG and 7.14 ms for one 1080p JPEG (30 launches each, warm file cache).

**Python API latency.** On the 128 original JPEGs, `visers_nano.encode_single(path)` returned a NumPy vector in **1.69 ms P50** (2.17 ms P90). Launching `nano_image --jobs 1` for each path and converting its JSON output to a NumPy vector took **4.75 ms P50** (5.48 ms P90). Both used fast mode, 256 dimensions, release builds with the same bundled codecs, and one pinned core on the i7-8700K. After a warm-up pass, three traversals alternated call order for each file, giving 384 measurements per interface. The Python process reused its model and decoder. The CLI timings include process startup, model initialization and the JSON round trip.

**Batch throughput.** The 36 mixed 1080p files in fast mode reached 73.0 images/s on one physical core and 394.2 images/s on six (`taskset -c 0-5`, `--jobs 6`). The original JPEGs reached 617.4 and 3,483.7 images/s respectively. Throughput uses whole batch wall time, including worker startup. Each worker decodes and infers a different image.

**Preprocessing fidelity.** On 200 files, Full produced the same embedding bytes as applying Pillow 13's EXIF transpose, RGB conversion and BOX128 followed by `nano_rgb`. The check includes all eight EXIF orientations, CMYK JPEG, transparency, grayscale/16-bit PNG, animated GIF/WebP and narrow sources. For the 12 large files per format, Fast cosine similarity against that reference was 0.99725 median / 0.98926 minimum for JPEG, 0.99994 / 0.99989 for PNG and 0.99881 / 0.99557 for WebP. These are per-image embedding comparisons.

To profile a local collection with the same stage boundaries, run the CLI on the source files directly.

```bash
TURBOJPEG_SOURCE=pkg-config TURBOJPEG_DYNAMIC=1 cargo build --release --locked --features image-files --bin nano_image
taskset -c 4 target/release/nano_image --jobs 1 --profile -o fast.f32 images/*
taskset -c 4 target/release/nano_image --jobs 1 --full --profile -o full.f32 images/*
```
