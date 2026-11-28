//! Native zero-allocation `nano-v1` (`mobilenetv4_conv_small_050` trunk + MRL head) encoder.
//!
//! Use [`encode_single_rgb`] for a packed 128×128 RGB frame. With the `image-files`
//! feature, `encode_single` accepts a path or an encoded image in memory, and `encode`
//! returns a batch in input order. The baked model produces L2-normalized vectors
//! of [`DIMENSIONS`] elements. Compare them with [`similarity`] or [`distance`].
//! For control over buffers and workers, use [`NanoEncoder`] and [`NanoSpinPool`].
//!
//! ## CPU implementation
//!
//! - Fused `conv_stem` (`3->32`) + `blocks.0.0` (`32->16`) via a 64B-aligned 3-row L1d ring buffer
//!   (`24 KiB`) and rolling 3-row normalized RGB strip (`4.5 KiB`), keeping all activations inside L2.
//! - Single 64-byte-aligned workspace arena with `+64B` anti-4K-aliasing stagger between buffers.
//! - Register-tiled AVX2/FMA microkernels (`8..12` YMM accumulators) saturating 2 FMA/cycle ports.
//! - Stack RAII `FTZ | DAZ` (`MXCSR = 0x8040`) guard during `x86_64` inference.
//! - Persistent `NanoSpinPool` for zero-mutex multi-core batch encoding.

use std::cell::RefCell;
use std::sync::OnceLock;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod avx2;
#[cfg(feature = "image-files")]
pub mod image_io;
mod metrics;
mod pool;
mod scalar;

#[cfg(feature = "image-files")]
pub use image_io::{encode, encode_into, encode_single, encode_with_options, ImageInput};
pub use metrics::{distance, similarity};
pub use pool::NanoSpinPool;

pub const RGB_SIDE: usize = 128;
pub const RGB_BYTES: usize = RGB_SIDE * RGB_SIDE * 3;

const BUF_FLOATS: usize = 24_576; // Max intermediate map: 16*16*96 = 96 KiB
const SKIP_FLOATS: usize = 12_288; // Max skip map: 16*16*48 = 48 KiB
const STEM_RING_FLOATS: usize = 3 * 64 * 32; // 3 stem rows x 64 x 32 = 24 KiB (L1d)
const RGB_STRIP_FLOATS: usize = 3 * RGB_SIDE * 3 + 16; // 3 RGB rows x 384 + pad = 4.6 KiB (L1d)

// Staggered by +16 floats (+64B) to eliminate Coffee Lake 4K-aliasing load/store stalls.
const OFF_BUF_A: usize = 0;
const OFF_BUF_B: usize = OFF_BUF_A + BUF_FLOATS + 16;
const OFF_SKIP: usize = OFF_BUF_B + BUF_FLOATS + 16;
const OFF_STEM_RING: usize = OFF_SKIP + SKIP_FLOATS + 16;
const OFF_RGB_STRIP: usize = OFF_STEM_RING + STEM_RING_FLOATS + 16;
const OFF_POOL: usize = OFF_RGB_STRIP + RGB_STRIP_FLOATS + 16;

mod baked {
    include!(concat!(env!("OUT_DIR"), "/nano_model.rs"));
}

/// Active Matryoshka dimension for `nano-v1` controlled by Cargo features (64, 128, or 256).
pub const NANO_EMBEDDING_DIM: usize = baked::NANO_EMBEDDING_DIM;
/// Width of an embedding from the baked model.
pub use NANO_EMBEDDING_DIM as DIMENSIONS;
/// L2-normalized output from the baked model.
pub type Embedding = [f32; DIMENSIONS];

/// True when `weights/nano_v1.bin` was baked into `.rodata` at build time.
pub const NANO_BAKED_AVAILABLE: bool = baked::NANO_BAKED_AVAILABLE;
/// Byte size of the compile-time sliced `.rodata` model.
pub const NANO_BAKED_BYTES_LEN: usize = baked::NANO_BAKED_BYTES.0.len();

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct CacheLine([f32; 16]);

#[derive(Clone)]
struct AlignedF32 {
    lines: Vec<CacheLine>,
    len: usize,
}

impl AlignedF32 {
    fn zeros(len: usize) -> Self {
        let num_lines = len.div_ceil(16);
        Self {
            lines: vec![CacheLine([0.0; 16]); num_lines],
            len,
        }
    }

    #[inline(always)]
    fn as_slice(&self) -> &[f32] {
        let ptr = self.lines.as_ptr() as *const f32;
        // SAFETY: `lines` has `>= len` contiguous `f32` elements aligned to 64 bytes.
        unsafe { core::slice::from_raw_parts(ptr, self.len) }
    }

    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [f32] {
        let ptr = self.lines.as_mut_ptr() as *mut f32;
        // SAFETY: `lines` has `>= len` contiguous `f32` elements aligned to 64 bytes.
        unsafe { core::slice::from_raw_parts_mut(ptr, self.len) }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline(always)]
    fn as_ptr(&self) -> *const f32 {
        self.lines.as_ptr() as *const f32
    }

    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut f32 {
        self.lines.as_mut_ptr() as *mut f32
    }
}

/// Per-stage microsecond breakdown from `encode_rgb_profiled`.
#[derive(Debug, Default, Clone, Copy)]
pub struct NanoStageTimings {
    pub fused_stem_b0_us: f64,
    pub conv3x3_us: f64,
    pub dw_us: f64,
    pub pw_us: f64,
    pub head_us: f64,
}

#[derive(Clone)]
struct ConvOp {
    ci: usize,
    co: usize,
    k: usize,
    s: usize,
    p: usize,
    dw: bool,
    relu: bool,
    save_skip: bool,
    add_skip: bool,
    biases: AlignedF32,
    weights: AlignedF32,
}

/// Reusable 64-byte-aligned L2-resident arena for zero-allocation `nano-v1` inference.
/// Allocate one workspace per concurrent caller thread.
pub struct NanoWorkspace {
    arena: AlignedF32,
    head_in: usize,
}

impl NanoWorkspace {
    pub fn new(head_in: usize) -> Self {
        let pool_aligned = head_in.div_ceil(16) * 16;
        Self {
            arena: AlignedF32::zeros(OFF_POOL + pool_aligned),
            head_in,
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], &'static str> {
        let end = self.offset.checked_add(n).ok_or("VSN1 offset overflow")?;
        let slice = self
            .data
            .get(self.offset..end)
            .ok_or("truncated VSN1 payload")?;
        self.offset = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<usize, &'static str> {
        Ok(self.take(1)?[0] as usize)
    }

    fn u16(&mut self) -> Result<usize, &'static str> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as usize)
    }

    fn u32(&mut self) -> Result<usize, &'static str> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }

    fn f32(&mut self) -> Result<f32, &'static str> {
        let v = f32::from_le_bytes(self.take(4)?.try_into().unwrap());
        if !v.is_finite() {
            return Err("non-finite VSN1 parameter");
        }
        Ok(v)
    }

    fn f32_aligned(&mut self, n: usize) -> Result<AlignedF32, &'static str> {
        let bytes = n.checked_mul(4).ok_or("VSN1 f32 size overflow")?;
        let raw = self.take(bytes)?;
        let mut out = AlignedF32::zeros(n);
        for (dst, chunk) in out.as_mut_slice().iter_mut().zip(raw.chunks_exact(4)) {
            let v = f32::from_le_bytes(chunk.try_into().unwrap());
            if !v.is_finite() {
                return Err("non-finite VSN1 tensor");
            }
            *dst = v;
        }
        Ok(out)
    }
}

/// Validated `nano-v1` encoder. Weights are unpacked once into 64B-aligned storage on construction.
pub struct NanoEncoder {
    norm_lut: [[f32; 256]; 3],
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    standard_rgb: bool,
    ops: Vec<ConvOp>,
    head_in: usize,
    head_out: usize,
    head_biases: AlignedF32,
    head_weights: AlignedF32,
    use_avx2: bool,
}

impl NanoEncoder {
    /// Parses a `VSN1` model and checks its graph against the inference workspace layout.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        let mut r = Reader {
            data: bytes,
            offset: 0,
        };
        if r.take(4)? != b"VSN1" {
            return Err("invalid VSN1 magic");
        }
        if r.u32()? != 1 {
            return Err("unsupported VSN1 version");
        }
        let num_ops = r.u32()?;
        let head_in = r.u32()?;
        let head_out = r.u32()?;
        if num_ops < 2 || head_in == 0 || head_out == 0 || !head_in.is_multiple_of(16) {
            return Err("invalid VSN1 dimensions");
        }
        let mean = [r.f32()?, r.f32()?, r.f32()?];
        let std = [r.f32()?, r.f32()?, r.f32()?];
        if num_ops > (bytes.len() - r.offset) / 16 {
            return Err("truncated VSN1 op descriptors");
        }
        if std.iter().any(|&s| s <= 0.0) {
            return Err("invalid VSN1 normalization std");
        }
        let mut norm_lut = [[0.0f32; 256]; 3];
        for (c, row) in norm_lut.iter_mut().enumerate() {
            for (v, slot) in row.iter_mut().enumerate() {
                *slot = (v as f32 / 255.0 - mean[c]) / std[c];
            }
        }

        let mut ops = Vec::new();
        let mut side = RGB_SIDE;
        let mut channels = 3;
        let mut skip_shape = None;
        for op_idx in 0..num_ops {
            let ci = r.u16()?;
            let co = r.u16()?;
            let k = r.u8()?;
            let s = r.u8()?;
            let p = r.u8()?;
            let dw = r.u8()? != 0;
            let relu = r.u8()? != 0;
            let save_skip = r.u8()? != 0;
            let add_skip = r.u8()? != 0;
            let _reserved = r.u8()?;
            let weight_len = r.u32()?;

            if ci != channels || co == 0 || !co.is_multiple_of(16) || (dw && ci != co) {
                return Err("invalid VSN1 op shape");
            }
            if op_idx == 0
                && (co != 32 || k != 3 || s != 2 || p != 1 || dw || save_skip || add_skip)
            {
                return Err("unexpected VSN1 stem op");
            }
            if op_idx == 1
                && (co != 16 || k != 3 || s != 2 || p != 1 || dw || save_skip || add_skip)
            {
                return Err("unexpected VSN1 block0.0 op");
            }
            let supported_kernel = if dw {
                matches!(k, 3 | 5)
            } else {
                matches!(k, 1 | 3)
            };
            if !supported_kernel || !matches!(s, 1 | 2) || (k == 1 && s != 1) || p != k / 2 {
                return Err("unsupported VSN1 convolution geometry");
            }
            if !side.is_multiple_of(s) {
                return Err("invalid VSN1 spatial dimensions");
            }
            let out_side = side / s;
            // The first convolution uses the fused stem ring. All later outputs use buf_a/b.
            if op_idx != 0 && out_side * out_side * co > BUF_FLOATS {
                return Err("VSN1 activation exceeds workspace");
            }
            if save_skip {
                if skip_shape.is_some() || side * side * ci > SKIP_FLOATS {
                    return Err("invalid VSN1 residual save");
                }
                skip_shape = Some((side, ci));
            }
            if add_skip && (k != 1 || skip_shape.take() != Some((out_side, co))) {
                return Err("invalid VSN1 residual add");
            }

            let inputs_per_tap = if dw { 1 } else { ci };
            let expected_len = (k * k * inputs_per_tap)
                .checked_mul(co)
                .ok_or("VSN1 op size overflow")?;
            if weight_len != expected_len {
                return Err("invalid VSN1 op weight length");
            }

            let scales = r.f32_aligned(co)?;
            let biases = r.f32_aligned(co)?;
            let q_weights = r.take(weight_len)?;

            let mut weights = AlignedF32::zeros(weight_len);
            let sc_slice = scales.as_slice();
            for (idx, (dst, &qb)) in weights.as_mut_slice().iter_mut().zip(q_weights).enumerate() {
                *dst = (qb as i8) as f32 * sc_slice[idx % co];
            }
            ops.push(ConvOp {
                ci,
                co,
                k,
                s,
                p,
                dw,
                relu,
                save_skip,
                add_skip,
                biases,
                weights,
            });
            side = out_side;
            channels = co;
        }
        if skip_shape.is_some() {
            return Err("unfinished VSN1 residual");
        }
        if head_in != channels {
            return Err("VSN1 head input does not match final activation");
        }

        let head_weight_len = head_out
            .checked_mul(head_in)
            .ok_or("VSN1 head size overflow")?;
        let head_scales = r.f32_aligned(head_out)?;
        let head_biases = r.f32_aligned(head_out)?;
        let q_head = r.take(head_weight_len)?;
        if r.offset != bytes.len() {
            return Err("trailing bytes in VSN1 file");
        }
        let mut head_weights = AlignedF32::zeros(head_weight_len);
        let hs_slice = head_scales.as_slice();
        let hw_slice = head_weights.as_mut_slice();
        for row in 0..head_out {
            let s = hs_slice[row];
            let dst = &mut hw_slice[row * head_in..(row + 1) * head_in];
            let src = &q_head[row * head_in..(row + 1) * head_in];
            for (d, &qb) in dst.iter_mut().zip(src) {
                *d = (qb as i8) as f32 * s;
            }
        }

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        let use_avx2 = false;

        Ok(Self {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            standard_rgb: mean == [0.5; 3] && std == [0.5; 3],
            norm_lut,
            ops,
            head_in,
            head_out,
            head_biases,
            head_weights,
            use_avx2,
        })
    }

    /// Loads the compile-time `.rodata` baked model sliced to `NANO_EMBEDDING_DIM`.
    pub fn baked() -> Result<Self, &'static str> {
        if !NANO_BAKED_AVAILABLE {
            return Err("nano_v1.bin was not baked at build time");
        }
        Self::from_bytes(&baked::NANO_BAKED_BYTES.0)
    }

    pub fn embedding_dim(&self) -> usize {
        self.head_out
    }

    pub fn backend(&self) -> &'static str {
        if self.use_avx2 {
            "avx2+fma(fused-ring-64b-ftz)"
        } else {
            "scalar"
        }
    }

    pub fn workspace(&self) -> NanoWorkspace {
        NanoWorkspace::new(self.head_in)
    }

    /// Encodes a canonical `128x128` RGB24 buffer into `out[..self.embedding_dim()]`.
    /// Performs zero heap allocations.
    pub fn encode_rgb(
        &self,
        rgb: &[u8],
        ws: &mut NanoWorkspace,
        out: &mut [f32],
    ) -> Result<(), &'static str> {
        self.encode_rgb_with_backend(rgb, ws, out, self.use_avx2)
    }

    /// Portable scalar reference path (exposed for parity testing against SIMD).
    pub fn encode_rgb_scalar(
        &self,
        rgb: &[u8],
        ws: &mut NanoWorkspace,
        out: &mut [f32],
    ) -> Result<(), &'static str> {
        self.encode_rgb_with_backend(rgb, ws, out, false)
    }

    /// Encodes a single image while accumulating per-stage microsecond breakdown.
    pub fn encode_rgb_profiled(
        &self,
        rgb: &[u8],
        ws: &mut NanoWorkspace,
        out: &mut [f32],
        prof: &mut NanoStageTimings,
    ) -> Result<(), &'static str> {
        if rgb.len() != RGB_BYTES {
            return Err("nano-v1 expects 128x128x3 RGB bytes");
        }
        if out.len() < self.head_out || ws.head_in != self.head_in {
            return Err("nano-v1 output/workspace mismatch");
        }
        let out_slice = &mut out[..self.head_out];
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if self.use_avx2 {
            unsafe {
                self.run_avx2(rgb, ws, out_slice, Some(prof));
            }
            return Ok(());
        }
        let _ = prof;
        self.run_scalar(rgb, ws, out_slice);
        Ok(())
    }

    fn encode_rgb_with_backend(
        &self,
        rgb: &[u8],
        ws: &mut NanoWorkspace,
        out: &mut [f32],
        use_avx2: bool,
    ) -> Result<(), &'static str> {
        if rgb.len() != RGB_BYTES {
            return Err("nano-v1 expects 128x128x3 RGB bytes");
        }
        if out.len() < self.head_out || ws.head_in != self.head_in {
            return Err("nano-v1 output/workspace mismatch");
        }

        let out_slice = &mut out[..self.head_out];

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if use_avx2 {
            // SAFETY: constructor checked AVX2/FMA support, dimensions, and workspace sizes.
            unsafe {
                self.run_avx2(rgb, ws, out_slice, None);
            }
            return Ok(());
        }
        let _ = use_avx2;
        self.run_scalar(rgb, ws, out_slice);
        Ok(())
    }
}

fn l2_normalize(out: &mut [f32]) {
    let sq: f32 = out.iter().map(|&v| v * v).sum();
    let inv = 1.0 / sq.sqrt().max(1e-12);
    for v in out.iter_mut() {
        *v *= inv;
    }
}

static BAKED_ENCODER: OnceLock<Result<NanoEncoder, &'static str>> = OnceLock::new();

fn baked_encoder() -> Result<&'static NanoEncoder, &'static str> {
    BAKED_ENCODER
        .get_or_init(NanoEncoder::baked)
        .as_ref()
        .map_err(|&err| err)
}

thread_local! {
    static THREAD_WS: RefCell<Option<NanoWorkspace>> = const { RefCell::new(None) };
}

/// Encodes a single `128x128x3` RGB buffer using the compile-time `.rodata` baked `nano-v1` model.
/// Zero heap allocations after the first call on the calling thread.
pub fn encode_single_rgb(rgb: &[u8]) -> Result<[f32; NANO_EMBEDDING_DIM], &'static str> {
    let encoder = baked_encoder()?;
    let mut out = [0.0f32; NANO_EMBEDDING_DIM];
    THREAD_WS.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let ws = borrow.get_or_insert_with(|| encoder.workspace());
        encoder.encode_rgb(rgb, ws, &mut out)
    })?;
    Ok(out)
}

/// Encodes a batch of `128x128x3` RGB buffers on the calling thread.
/// For multi-core batch encoding, use [`NanoSpinPool`].
pub fn encode_batch_rgb(images: &[&[u8]]) -> Result<Vec<[f32; NANO_EMBEDDING_DIM]>, &'static str> {
    images.iter().map(|&rgb| encode_single_rgb(rgb)).collect()
}
