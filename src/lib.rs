//! Native zero-allocation `nano-v1` (`mobilenetv4_conv_small_050` trunk + MRL head) encoder.
//!
//! Low-level CPU microarchitecture optimizations:
//! - Fused `conv_stem` (`3->32`) + `blocks.0.0` (`32->16`) via a 64B-aligned 3-row L1d ring buffer
//!   (`24 KiB`) and rolling 3-row normalized RGB strip (`4.5 KiB`), keeping all activations inside L2.
//! - Single 64-byte-aligned workspace arena with `+64B` anti-4K-aliasing stagger between buffers.
//! - Register-tiled AVX2/FMA microkernels (`8..12` YMM accumulators) saturating 2 FMA/cycle ports.
//! - Stack RAII `FTZ | DAZ` (`MXCSR = 0x8040`) guard during `x86_64` inference.
//! - Persistent `NanoSpinPool` for zero-mutex multi-core batch encoding.

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use core::hint::spin_loop;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::time::Instant;

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

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
struct FtzDazGuard {
    saved_mxcsr: u32,
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl FtzDazGuard {
    #[inline(always)]
    unsafe fn new() -> Self {
        let mut saved_mxcsr = 0u32;
        core::arch::asm!(
            "stmxcsr [{ptr}]",
            ptr = in(reg) &mut saved_mxcsr,
            options(nostack, preserves_flags)
        );
        let next = saved_mxcsr | 0x8040;
        core::arch::asm!(
            "ldmxcsr [{ptr}]",
            ptr = in(reg) &next,
            options(nostack, preserves_flags)
        );
        Self { saved_mxcsr }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl Drop for FtzDazGuard {
    #[inline(always)]
    fn drop(&mut self) {
        // SAFETY: restores original caller MXCSR state.
        unsafe {
            core::arch::asm!(
                "ldmxcsr [{ptr}]",
                ptr = in(reg) &self.saved_mxcsr,
                options(nostack, preserves_flags)
            );
        }
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
        let slice = self.data.get(self.offset..end).ok_or("truncated VSN1 payload")?;
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
        let mut r = Reader { data: bytes, offset: 0 };
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

    fn run_scalar(&self, rgb: &[u8], ws: &mut NanoWorkspace, out: &mut [f32]) {
        let arena_ptr = ws.arena.as_mut_ptr();
        // SAFETY: all non-overlapping sub-slices live within `ws.arena` bounds.
        unsafe {
            let buf_a = core::slice::from_raw_parts_mut(arena_ptr.add(OFF_BUF_A), BUF_FLOATS);
            let buf_b = core::slice::from_raw_parts_mut(arena_ptr.add(OFF_BUF_B), BUF_FLOATS);
            let skip = core::slice::from_raw_parts_mut(arena_ptr.add(OFF_SKIP), SKIP_FLOATS);
            let stem_ring = core::slice::from_raw_parts_mut(arena_ptr.add(OFF_STEM_RING), STEM_RING_FLOATS);
            let rgb_strip = core::slice::from_raw_parts_mut(arena_ptr.add(OFF_RGB_STRIP), RGB_STRIP_FLOATS);
            let pool = core::slice::from_raw_parts_mut(arena_ptr.add(OFF_POOL), self.head_in);

            fused_stem_and_block0_scalar(
                rgb,
                &self.norm_lut,
                &self.ops[0],
                &self.ops[1],
                rgb_strip,
                stem_ring,
                &mut buf_a[..32 * 32 * 16],
            );

            let mut h = 32usize;
            let mut w = 32usize;
            let mut c = self.ops[1].co;
            let mut a_is_src = true;

            for op in &self.ops[2..] {
                let in_len = h * w * c;
                let oh = h / op.s;
                let ow = w / op.s;
                let out_len = oh * ow * op.co;

                let (src, dst): (&[f32], &mut [f32]) = if a_is_src {
                    (&buf_a[..in_len], &mut buf_b[..out_len])
                } else {
                    (&buf_b[..in_len], &mut buf_a[..out_len])
                };

                if op.save_skip {
                    skip[..in_len].copy_from_slice(src);
                }
                let skip_slice = if op.add_skip {
                    Some(&skip[..out_len])
                } else {
                    None
                };

                if op.dw {
                    dw_scalar(src, h, w, op, dst);
                } else if op.k == 1 {
                    pw_scalar(src, h * w, op, skip_slice, dst);
                } else {
                    conv3x3_scalar(src, h, w, op, dst);
                }

                h = oh;
                w = ow;
                c = op.co;
                a_is_src = !a_is_src;
            }

            let feat = if a_is_src {
                &buf_a[..h * w * c]
            } else {
                &buf_b[..h * w * c]
            };
            pool_and_head_scalar(
                feat,
                h * w,
                self.head_in,
                self.head_out,
                self.head_weights.as_slice(),
                self.head_biases.as_slice(),
                pool,
                out,
            );
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_avx2(
        &self,
        rgb: &[u8],
        ws: &mut NanoWorkspace,
        out: &mut [f32],
        mut prof: Option<&mut NanoStageTimings>,
    ) {
        let _ftz = FtzDazGuard::new();
        let arena_ptr = ws.arena.as_mut_ptr();
        let buf_a_ptr = arena_ptr.add(OFF_BUF_A);
        let buf_b_ptr = arena_ptr.add(OFF_BUF_B);
        let skip_ptr = arena_ptr.add(OFF_SKIP);
        let stem_ring_ptr = arena_ptr.add(OFF_STEM_RING);
        let rgb_strip_ptr = arena_ptr.add(OFF_RGB_STRIP);
        let pool_ptr = arena_ptr.add(OFF_POOL);

        let t0 = prof.as_ref().map(|_| Instant::now());
        fused_stem_and_block0_avx2(
            rgb.as_ptr(),
            &self.norm_lut,
            &self.ops[0],
            &self.ops[1],
            rgb_strip_ptr,
            stem_ring_ptr,
            buf_a_ptr,
        );
        if let (Some(ref mut p), Some(start)) = (prof.as_deref_mut(), t0) {
            p.fused_stem_b0_us += start.elapsed().as_secs_f64() * 1e6;
        }

        let mut h = 32usize;
        let mut w = 32usize;
        let mut c = self.ops[1].co;
        let mut a_is_src = true;

        for op in &self.ops[2..] {
            let in_len = h * w * c;
            let oh = h / op.s;
            let ow = w / op.s;

            let (src_ptr, dst_ptr): (*const f32, *mut f32) = if a_is_src {
                (buf_a_ptr, buf_b_ptr)
            } else {
                (buf_b_ptr, buf_a_ptr)
            };

            if op.save_skip {
                core::ptr::copy_nonoverlapping(src_ptr, skip_ptr, in_len);
            }
            let active_skip = if op.add_skip {
                Some(skip_ptr as *const f32)
            } else {
                None
            };

            let ts = prof.as_ref().map(|_| Instant::now());
            if op.dw {
                dw_avx2(src_ptr, h, w, op, dst_ptr);
                if let (Some(ref mut p), Some(s)) = (prof.as_deref_mut(), ts) {
                    p.dw_us += s.elapsed().as_secs_f64() * 1e6;
                }
            } else if op.k == 1 {
                pw_avx2(src_ptr, h * w, op, active_skip, dst_ptr);
                if let (Some(ref mut p), Some(s)) = (prof.as_deref_mut(), ts) {
                    p.pw_us += s.elapsed().as_secs_f64() * 1e6;
                }
            } else {
                conv3x3_avx2(src_ptr, h, w, op, dst_ptr);
                if let (Some(ref mut p), Some(s)) = (prof.as_deref_mut(), ts) {
                    p.conv3x3_us += s.elapsed().as_secs_f64() * 1e6;
                }
            }

            h = oh;
            w = ow;
            c = op.co;
            a_is_src = !a_is_src;
        }

        let feat_ptr: *const f32 = if a_is_src { buf_a_ptr } else { buf_b_ptr };
        let th = prof.as_ref().map(|_| Instant::now());
        pool_and_head_avx2(
            feat_ptr,
            h * w,
            self.head_in,
            self.head_out,
            self.head_weights.as_ptr(),
            self.head_biases.as_ptr(),
            pool_ptr,
            out,
        );
        if let (Some(p), Some(s)) = (prof, th) {
            p.head_us += s.elapsed().as_secs_f64() * 1e6;
        }
    }
}

fn l2_normalize(out: &mut [f32]) {
    let sq: f32 = out.iter().map(|&v| v * v).sum();
    let inv = 1.0 / sq.sqrt().max(1e-12);
    for v in out.iter_mut() {
        *v *= inv;
    }
}

fn convert_rgb_row_scalar(rgb: &[u8], lut: &[[f32; 256]; 3], iy: usize, dst_row: &mut [f32]) {
    let row_bytes = &rgb[iy * RGB_SIDE * 3..(iy + 1) * RGB_SIDE * 3];
    for x in 0..RGB_SIDE {
        let p = x * 3;
        dst_row[p] = lut[0][row_bytes[p] as usize];
        dst_row[p + 1] = lut[1][row_bytes[p + 1] as usize];
        dst_row[p + 2] = lut[2][row_bytes[p + 2] as usize];
    }
}

#[allow(clippy::too_many_arguments)]
fn fused_stem_and_block0_scalar(
    rgb: &[u8],
    lut: &[[f32; 256]; 3],
    stem: &ConvOp,
    b0: &ConvOp,
    rgb_strip: &mut [f32],
    stem_ring: &mut [f32],
    out: &mut [f32],
) {
    let row_rgb_len = RGB_SIDE * 3;
    let row_stem_len = 64 * 32;
    let stem_w = stem.weights.as_slice();
    let stem_b = stem.biases.as_slice();
    let b0_w = b0.weights.as_slice();
    let b0_b = b0.biases.as_slice();

    let mut next_rgb_row = 0usize;
    let mut next_stem_row = 0usize;

    for oy in 0..32usize {
        let needed_stem_hi = oy * 2 + 1; // max 63
        while next_stem_row <= needed_stem_hi {
            let sy = next_stem_row;
            let needed_rgb_hi = (sy * 2 + 1).min(RGB_SIDE - 1);
            while next_rgb_row <= needed_rgb_hi {
                let slot = (next_rgb_row % 3) * row_rgb_len;
                convert_rgb_row_scalar(
                    rgb,
                    lut,
                    next_rgb_row,
                    &mut rgb_strip[slot..slot + row_rgb_len],
                );
                next_rgb_row += 1;
            }

            let dst_stem = &mut stem_ring[(sy % 3) * row_stem_len..(sy % 3 + 1) * row_stem_len];
            for sx in 0..64usize {
                let px_out = &mut dst_stem[sx * 32..(sx + 1) * 32];
                px_out.copy_from_slice(stem_b);
                for ky in 0..3usize {
                    let iy = (sy * 2 + ky).wrapping_sub(1);
                    if iy >= RGB_SIDE {
                        continue;
                    }
                    let r_slot = (iy % 3) * row_rgb_len;
                    for kx in 0..3usize {
                        let ix = (sx * 2 + kx).wrapping_sub(1);
                        if ix >= RGB_SIDE {
                            continue;
                        }
                        let in_px = &rgb_strip[r_slot + ix * 3..r_slot + (ix + 1) * 3];
                        let k_off = (ky * 3 + kx) * 3 * 32;
                        for ic in 0..3usize {
                            let xv = in_px[ic];
                            let w_row = &stem_w[k_off + ic * 32..k_off + (ic + 1) * 32];
                            for o in 0..32usize {
                                px_out[o] += xv * w_row[o];
                            }
                        }
                    }
                }
                if stem.relu {
                    for v in px_out.iter_mut() {
                        if *v < 0.0 {
                            *v = 0.0;
                        }
                    }
                }
            }
            next_stem_row += 1;
        }

        // Now consume stem rows (oy*2 - 1 ..= oy*2 + 1) for b0 row `oy`
        for ox in 0..32usize {
            let dst = &mut out[(oy * 32 + ox) * 16..(oy * 32 + ox + 1) * 16];
            dst.copy_from_slice(b0_b);
            for ky in 0..3usize {
                let sy = (oy * 2 + ky).wrapping_sub(1);
                if sy >= 64 {
                    continue;
                }
                let s_slot = (sy % 3) * row_stem_len;
                for kx in 0..3usize {
                    let sx = (ox * 2 + kx).wrapping_sub(1);
                    if sx >= 64 {
                        continue;
                    }
                    let in_px = &stem_ring[s_slot + sx * 32..s_slot + (sx + 1) * 32];
                    let k_off = (ky * 3 + kx) * 32 * 16;
                    for ic in 0..32usize {
                        let xv = in_px[ic];
                        let w_row = &b0_w[k_off + ic * 16..k_off + (ic + 1) * 16];
                        for o in 0..16usize {
                            dst[o] += xv * w_row[o];
                        }
                    }
                }
            }
            if b0.relu {
                for v in dst.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
        }
    }
}

fn conv3x3_scalar(src: &[f32], ih: usize, iw: usize, op: &ConvOp, out: &mut [f32]) {
    let oh = ih / op.s;
    let ow = iw / op.s;
    let ci = op.ci;
    let co = op.co;
    let biases = op.biases.as_slice();
    let weights = op.weights.as_slice();
    for oy in 0..oh {
        for ox in 0..ow {
            let dst = &mut out[(oy * ow + ox) * co..(oy * ow + ox + 1) * co];
            dst.copy_from_slice(biases);
            for ky in 0..3usize {
                let iy = (oy * op.s + ky).wrapping_sub(op.p);
                if iy >= ih {
                    continue;
                }
                for kx in 0..3usize {
                    let ix = (ox * op.s + kx).wrapping_sub(op.p);
                    if ix >= iw {
                        continue;
                    }
                    let in_px = &src[(iy * iw + ix) * ci..(iy * iw + ix + 1) * ci];
                    let k_off = (ky * 3 + kx) * ci * co;
                    for ic in 0..ci {
                        let xv = in_px[ic];
                        let w_row = &weights[k_off + ic * co..k_off + (ic + 1) * co];
                        for o in 0..co {
                            dst[o] += xv * w_row[o];
                        }
                    }
                }
            }
            if op.relu {
                for v in dst.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
        }
    }
}

fn dw_scalar(src: &[f32], ih: usize, iw: usize, op: &ConvOp, out: &mut [f32]) {
    let oh = ih / op.s;
    let ow = iw / op.s;
    let c = op.co;
    let biases = op.biases.as_slice();
    let weights = op.weights.as_slice();
    for oy in 0..oh {
        for ox in 0..ow {
            let dst = &mut out[(oy * ow + ox) * c..(oy * ow + ox + 1) * c];
            dst.copy_from_slice(biases);
            for ky in 0..op.k {
                let iy = (oy * op.s + ky).wrapping_sub(op.p);
                if iy >= ih {
                    continue;
                }
                for kx in 0..op.k {
                    let ix = (ox * op.s + kx).wrapping_sub(op.p);
                    if ix >= iw {
                        continue;
                    }
                    let in_px = &src[(iy * iw + ix) * c..(iy * iw + ix + 1) * c];
                    let w_row = &weights[(ky * op.k + kx) * c..(ky * op.k + kx + 1) * c];
                    for ch in 0..c {
                        dst[ch] += in_px[ch] * w_row[ch];
                    }
                }
            }
            if op.relu {
                for v in dst.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
        }
    }
}

fn pw_scalar(
    src: &[f32],
    pixels: usize,
    op: &ConvOp,
    skip: Option<&[f32]>,
    out: &mut [f32],
) {
    let ci = op.ci;
    let co = op.co;
    let biases = op.biases.as_slice();
    let weights = op.weights.as_slice();
    for px in 0..pixels {
        let in_px = &src[px * ci..(px + 1) * ci];
        let dst = &mut out[px * co..(px + 1) * co];
        dst.copy_from_slice(biases);
        for ic in 0..ci {
            let xv = in_px[ic];
            let w_row = &weights[ic * co..(ic + 1) * co];
            for o in 0..co {
                dst[o] += xv * w_row[o];
            }
        }
        if op.relu {
            for v in dst.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
        if let Some(sk) = skip {
            let sk_px = &sk[px * co..(px + 1) * co];
            for o in 0..co {
                dst[o] += sk_px[o];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pool_and_head_scalar(
    feat: &[f32],
    pixels: usize,
    head_in: usize,
    head_out: usize,
    head_weights: &[f32],
    head_biases: &[f32],
    pool: &mut [f32],
    out: &mut [f32],
) {
    pool[..head_in].fill(0.0);
    for px in 0..pixels {
        let src = &feat[px * head_in..(px + 1) * head_in];
        for i in 0..head_in {
            pool[i] += src[i];
        }
    }
    let inv_px = 1.0 / (pixels as f32);
    for v in pool[..head_in].iter_mut() {
        *v *= inv_px;
    }
    for row in 0..head_out {
        let w_row = &head_weights[row * head_in..(row + 1) * head_in];
        let mut sum = head_biases[row];
        for i in 0..head_in {
            sum += pool[i] * w_row[i];
        }
        out[row] = sum;
    }
    l2_normalize(out);
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_stem_row_avx2(
    sy: usize,
    rgb_strip_ptr: *const f32,
    stem: &ConvOp,
    dst_row: *mut f32,
) {
    let row_rgb_len = RGB_SIDE * 3;
    let w_ptr = stem.weights.as_ptr();
    let b_ptr = stem.biases.as_ptr();
    let zero = _mm256_setzero_ps();
    let b0 = _mm256_load_ps(b_ptr);
    let b1 = _mm256_load_ps(b_ptr.add(8));
    let b2 = _mm256_load_ps(b_ptr.add(16));
    let b3 = _mm256_load_ps(b_ptr.add(24));

    // sx = 0 (border: kx=0 is -1 out of bounds)
    {
        let mut a0 = b0;
        let mut a1 = b1;
        let mut a2 = b2;
        let mut a3 = b3;
        for ky in 0..3usize {
            let iy = (sy * 2 + ky).wrapping_sub(1);
            if iy >= RGB_SIDE {
                continue;
            }
            let r_ptr = rgb_strip_ptr.add((iy % 3) * row_rgb_len);
            for kx in 1..3usize {
                let ix = kx - 1;
                let px = r_ptr.add(ix * 3);
                let k_off = (ky * 3 + kx) * 96;
                for ic in 0..3usize {
                    let xv = _mm256_broadcast_ss(&*px.add(ic));
                    let wp = w_ptr.add(k_off + ic * 32);
                    a0 = _mm256_fmadd_ps(xv, _mm256_load_ps(wp), a0);
                    a1 = _mm256_fmadd_ps(xv, _mm256_load_ps(wp.add(8)), a1);
                    a2 = _mm256_fmadd_ps(xv, _mm256_load_ps(wp.add(16)), a2);
                    a3 = _mm256_fmadd_ps(xv, _mm256_load_ps(wp.add(24)), a3);
                }
            }
        }
        if stem.relu {
            a0 = _mm256_max_ps(a0, zero);
            a1 = _mm256_max_ps(a1, zero);
            a2 = _mm256_max_ps(a2, zero);
            a3 = _mm256_max_ps(a3, zero);
        }
        _mm256_store_ps(dst_row, a0);
        _mm256_store_ps(dst_row.add(8), a1);
        _mm256_store_ps(dst_row.add(16), a2);
        _mm256_store_ps(dst_row.add(24), a3);
    }

    // sx = 1..64 -> 63 pixels = 21 exact tiles of 3 pixels x 32 channels (12 YMM accumulators)
    let interior_y = sy > 0;
    let r0_ptr = if interior_y {
        rgb_strip_ptr.add(((sy * 2 - 1) % 3) * row_rgb_len)
    } else {
        core::ptr::null()
    };
    let r1_ptr = rgb_strip_ptr.add(((sy * 2) % 3) * row_rgb_len);
    let r2_ptr = rgb_strip_ptr.add(((sy * 2 + 1) % 3) * row_rgb_len);

    let mut sx = 1usize;
    while sx < 64 {
        let mut a00 = b0;
        let mut a01 = b1;
        let mut a02 = b2;
        let mut a03 = b3;
        let mut a10 = b0;
        let mut a11 = b1;
        let mut a12 = b2;
        let mut a13 = b3;
        let mut a20 = b0;
        let mut a21 = b1;
        let mut a22 = b2;
        let mut a23 = b3;

        let base_ix = sx * 2 - 1;
        for ky in 0..3usize {
            let r_ptr = if ky == 0 {
                if !interior_y {
                    continue;
                }
                r0_ptr
            } else if ky == 1 {
                r1_ptr
            } else {
                r2_ptr
            };
            for kx in 0..3usize {
                let p0 = r_ptr.add((base_ix + kx) * 3);
                let p1 = p0.add(6); // +2 RGB pixels
                let p2 = p1.add(6); // +4 RGB pixels
                let k_off = (ky * 3 + kx) * 96;
                for ic in 0..3usize {
                    let wp = w_ptr.add(k_off + ic * 32);
                    let x0 = _mm256_broadcast_ss(&*p0.add(ic));
                    let x1 = _mm256_broadcast_ss(&*p1.add(ic));
                    let x2 = _mm256_broadcast_ss(&*p2.add(ic));

                    let w0 = _mm256_load_ps(wp);
                    a00 = _mm256_fmadd_ps(x0, w0, a00);
                    a10 = _mm256_fmadd_ps(x1, w0, a10);
                    a20 = _mm256_fmadd_ps(x2, w0, a20);

                    let w1 = _mm256_load_ps(wp.add(8));
                    a01 = _mm256_fmadd_ps(x0, w1, a01);
                    a11 = _mm256_fmadd_ps(x1, w1, a11);
                    a21 = _mm256_fmadd_ps(x2, w1, a21);

                    let w2 = _mm256_load_ps(wp.add(16));
                    a02 = _mm256_fmadd_ps(x0, w2, a02);
                    a12 = _mm256_fmadd_ps(x1, w2, a12);
                    a22 = _mm256_fmadd_ps(x2, w2, a22);

                    let w3 = _mm256_load_ps(wp.add(24));
                    a03 = _mm256_fmadd_ps(x0, w3, a03);
                    a13 = _mm256_fmadd_ps(x1, w3, a13);
                    a23 = _mm256_fmadd_ps(x2, w3, a23);
                }
            }
        }

        if stem.relu {
            a00 = _mm256_max_ps(a00, zero);
            a01 = _mm256_max_ps(a01, zero);
            a02 = _mm256_max_ps(a02, zero);
            a03 = _mm256_max_ps(a03, zero);
            a10 = _mm256_max_ps(a10, zero);
            a11 = _mm256_max_ps(a11, zero);
            a12 = _mm256_max_ps(a12, zero);
            a13 = _mm256_max_ps(a13, zero);
            a20 = _mm256_max_ps(a20, zero);
            a21 = _mm256_max_ps(a21, zero);
            a22 = _mm256_max_ps(a22, zero);
            a23 = _mm256_max_ps(a23, zero);
        }

        let d0 = dst_row.add(sx * 32);
        _mm256_store_ps(d0, a00);
        _mm256_store_ps(d0.add(8), a01);
        _mm256_store_ps(d0.add(16), a02);
        _mm256_store_ps(d0.add(24), a03);

        let d1 = d0.add(32);
        _mm256_store_ps(d1, a10);
        _mm256_store_ps(d1.add(8), a11);
        _mm256_store_ps(d1.add(16), a12);
        _mm256_store_ps(d1.add(24), a13);

        let d2 = d1.add(32);
        _mm256_store_ps(d2, a20);
        _mm256_store_ps(d2.add(8), a21);
        _mm256_store_ps(d2.add(16), a22);
        _mm256_store_ps(d2.add(24), a23);

        sx += 3;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_stem_and_block0_avx2(
    rgb_ptr: *const u8,
    lut: &[[f32; 256]; 3],
    stem: &ConvOp,
    b0: &ConvOp,
    rgb_strip_ptr: *mut f32,
    stem_ring_ptr: *mut f32,
    out_ptr: *mut f32,
) {
    let row_rgb_len = RGB_SIDE * 3;
    let row_stem_len = 64 * 32;
    let b0_w = b0.weights.as_ptr();
    let b0_b = b0.biases.as_ptr();
    let zero = _mm256_setzero_ps();
    let bias0 = _mm256_load_ps(b0_b);
    let bias1 = _mm256_load_ps(b0_b.add(8));

    let mut next_rgb_row = 0usize;
    let mut next_stem_row = 0usize;

    for oy in 0..32usize {
        let needed_stem_hi = oy * 2 + 1;
        while next_stem_row <= needed_stem_hi {
            let sy = next_stem_row;
            let needed_rgb_hi = (sy * 2 + 1).min(RGB_SIDE - 1);
            while next_rgb_row <= needed_rgb_hi {
                let src_row = rgb_ptr.add(next_rgb_row * row_rgb_len);
                let dst_row = rgb_strip_ptr.add((next_rgb_row % 3) * row_rgb_len);
                let mut x = 0usize;
                while x < RGB_SIDE {
                    let p = x * 3;
                    *dst_row.add(p) = *lut[0].get_unchecked(*src_row.add(p) as usize);
                    *dst_row.add(p + 1) = *lut[1].get_unchecked(*src_row.add(p + 1) as usize);
                    *dst_row.add(p + 2) = *lut[2].get_unchecked(*src_row.add(p + 2) as usize);
                    x += 1;
                }
                next_rgb_row += 1;
            }
            let dst_stem = stem_ring_ptr.add((sy % 3) * row_stem_len);
            compute_stem_row_avx2(sy, rgb_strip_ptr, stem, dst_stem);
            next_stem_row += 1;
        }

        // Consume stem rows into block0.0 row `oy` (`32 -> 16`, k=3, s=2, p=1)
        let interior_y = oy > 0;
        let s0_ptr = if interior_y {
            stem_ring_ptr.add(((oy * 2 - 1) % 3) * row_stem_len)
        } else {
            core::ptr::null()
        };
        let s1_ptr = stem_ring_ptr.add(((oy * 2) % 3) * row_stem_len);
        let s2_ptr = stem_ring_ptr.add(((oy * 2 + 1) % 3) * row_stem_len);
        let out_row = out_ptr.add(oy * 32 * 16);

        // ox = 0 (border: kx=0 out of bounds)
        {
            let mut a0 = bias0;
            let mut a1 = bias1;
            for ky in 0..3usize {
                let sr = if ky == 0 {
                    if !interior_y {
                        continue;
                    }
                    s0_ptr
                } else if ky == 1 {
                    s1_ptr
                } else {
                    s2_ptr
                };
                for kx in 1..3usize {
                    let px = sr.add((kx - 1) * 32);
                    let kw = b0_w.add((ky * 3 + kx) * 32 * 16);
                    for ic in 0..32usize {
                        let xv = _mm256_broadcast_ss(&*px.add(ic));
                        let wp = kw.add(ic * 16);
                        a0 = _mm256_fmadd_ps(xv, _mm256_load_ps(wp), a0);
                        a1 = _mm256_fmadd_ps(xv, _mm256_load_ps(wp.add(8)), a1);
                    }
                }
            }
            if b0.relu {
                a0 = _mm256_max_ps(a0, zero);
                a1 = _mm256_max_ps(a1, zero);
            }
            _mm256_store_ps(out_row, a0);
            _mm256_store_ps(out_row.add(8), a1);
        }

        // ox = 1..31 -> 30 interior pixels = 5 exact tiles of 6 pixels x 16 channels (12 YMM accumulators)
        let mut ox = 1usize;
        while ox + 6 <= 32 {
            let mut a00 = bias0;
            let mut a01 = bias1;
            let mut a10 = bias0;
            let mut a11 = bias1;
            let mut a20 = bias0;
            let mut a21 = bias1;
            let mut a30 = bias0;
            let mut a31 = bias1;
            let mut a40 = bias0;
            let mut a41 = bias1;
            let mut a50 = bias0;
            let mut a51 = bias1;

            let base_sx = ox * 2 - 1;
            for ky in 0..3usize {
                let sr = if ky == 0 {
                    if !interior_y {
                        continue;
                    }
                    s0_ptr
                } else if ky == 1 {
                    s1_ptr
                } else {
                    s2_ptr
                };
                for kx in 0..3usize {
                    let p0 = sr.add((base_sx + kx) * 32);
                    let p1 = p0.add(64);
                    let p2 = p1.add(64);
                    let p3 = p2.add(64);
                    let p4 = p3.add(64);
                    let p5 = p4.add(64);
                    let kw = b0_w.add((ky * 3 + kx) * 32 * 16);
                    for ic in 0..32usize {
                        let wp = kw.add(ic * 16);
                        let w0 = _mm256_load_ps(wp);
                        let w1 = _mm256_load_ps(wp.add(8));

                        let x0 = _mm256_broadcast_ss(&*p0.add(ic));
                        a00 = _mm256_fmadd_ps(x0, w0, a00);
                        a01 = _mm256_fmadd_ps(x0, w1, a01);

                        let x1 = _mm256_broadcast_ss(&*p1.add(ic));
                        a10 = _mm256_fmadd_ps(x1, w0, a10);
                        a11 = _mm256_fmadd_ps(x1, w1, a11);

                        let x0 = _mm256_broadcast_ss(&*p2.add(ic));
                        a20 = _mm256_fmadd_ps(x0, w0, a20);
                        a21 = _mm256_fmadd_ps(x0, w1, a21);

                        let x1 = _mm256_broadcast_ss(&*p3.add(ic));
                        a30 = _mm256_fmadd_ps(x1, w0, a30);
                        a31 = _mm256_fmadd_ps(x1, w1, a31);

                        let x0 = _mm256_broadcast_ss(&*p4.add(ic));
                        a40 = _mm256_fmadd_ps(x0, w0, a40);
                        a41 = _mm256_fmadd_ps(x0, w1, a41);

                        let x1 = _mm256_broadcast_ss(&*p5.add(ic));
                        a50 = _mm256_fmadd_ps(x1, w0, a50);
                        a51 = _mm256_fmadd_ps(x1, w1, a51);
                    }
                }
            }

            if b0.relu {
                a00 = _mm256_max_ps(a00, zero);
                a01 = _mm256_max_ps(a01, zero);
                a10 = _mm256_max_ps(a10, zero);
                a11 = _mm256_max_ps(a11, zero);
                a20 = _mm256_max_ps(a20, zero);
                a21 = _mm256_max_ps(a21, zero);
                a30 = _mm256_max_ps(a30, zero);
                a31 = _mm256_max_ps(a31, zero);
                a40 = _mm256_max_ps(a40, zero);
                a41 = _mm256_max_ps(a41, zero);
                a50 = _mm256_max_ps(a50, zero);
                a51 = _mm256_max_ps(a51, zero);
            }

            let d = out_row.add(ox * 16);
            _mm256_store_ps(d, a00);
            _mm256_store_ps(d.add(8), a01);
            _mm256_store_ps(d.add(16), a10);
            _mm256_store_ps(d.add(24), a11);
            _mm256_store_ps(d.add(32), a20);
            _mm256_store_ps(d.add(40), a21);
            _mm256_store_ps(d.add(48), a30);
            _mm256_store_ps(d.add(56), a31);
            _mm256_store_ps(d.add(64), a40);
            _mm256_store_ps(d.add(72), a41);
            _mm256_store_ps(d.add(80), a50);
            _mm256_store_ps(d.add(88), a51);

            ox += 6;
        }

        // ox = 31 (final interior pixel: ix = 61, 62, 63 all < 64)
        while ox < 32 {
            let mut a0 = bias0;
            let mut a1 = bias1;
            let base_sx = ox * 2 - 1;
            for ky in 0..3usize {
                let sr = if ky == 0 {
                    if !interior_y {
                        continue;
                    }
                    s0_ptr
                } else if ky == 1 {
                    s1_ptr
                } else {
                    s2_ptr
                };
                for kx in 0..3usize {
                    let px = sr.add((base_sx + kx) * 32);
                    let kw = b0_w.add((ky * 3 + kx) * 32 * 16);
                    for ic in 0..32usize {
                        let xv = _mm256_broadcast_ss(&*px.add(ic));
                        let wp = kw.add(ic * 16);
                        a0 = _mm256_fmadd_ps(xv, _mm256_load_ps(wp), a0);
                        a1 = _mm256_fmadd_ps(xv, _mm256_load_ps(wp.add(8)), a1);
                    }
                }
            }
            if b0.relu {
                a0 = _mm256_max_ps(a0, zero);
                a1 = _mm256_max_ps(a1, zero);
            }
            let d = out_row.add(ox * 16);
            _mm256_store_ps(d, a0);
            _mm256_store_ps(d.add(8), a1);
            ox += 1;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv3x3_avx2(src_ptr: *const f32, ih: usize, iw: usize, op: &ConvOp, out_ptr: *mut f32) {
    let oh = ih / op.s;
    let ow = iw / op.s;
    let ci = op.ci;
    let co = op.co;
    let zero = _mm256_setzero_ps();
    let w_ptr = op.weights.as_ptr();
    let b_ptr = op.biases.as_ptr();

    for oy in 0..oh {
        let int_y = oy * op.s >= op.p && oy * op.s + 2 < ih + op.p;

        // Fast 6-pixel x 16-channel register-tiled interior loop
        let mut o = 0usize;
        while o + 16 <= co {
            let bias0 = _mm256_load_ps(b_ptr.add(o));
            let bias1 = _mm256_load_ps(b_ptr.add(o + 8));

            // Left border pixels while ox * s < p
            let mut ox = 0usize;
            while ox < ow && (!int_y || ox * op.s < op.p) {
                let mut a0 = bias0;
                let mut a1 = bias1;
                for ky in 0..3usize {
                    let iy = (oy * op.s + ky).wrapping_sub(op.p);
                    if iy >= ih {
                        continue;
                    }
                    for kx in 0..3usize {
                        let ix = (ox * op.s + kx).wrapping_sub(op.p);
                        if ix >= iw {
                            continue;
                        }
                        let px = src_ptr.add((iy * iw + ix) * ci);
                        let kw = w_ptr.add((ky * 3 + kx) * ci * co + o);
                        for ic in 0..ci {
                            let xv = _mm256_broadcast_ss(&*px.add(ic));
                            let wp = kw.add(ic * co);
                            a0 = _mm256_fmadd_ps(xv, _mm256_load_ps(wp), a0);
                            a1 = _mm256_fmadd_ps(xv, _mm256_load_ps(wp.add(8)), a1);
                        }
                    }
                }
                if op.relu {
                    a0 = _mm256_max_ps(a0, zero);
                    a1 = _mm256_max_ps(a1, zero);
                }
                let dst = out_ptr.add((oy * ow + ox) * co + o);
                _mm256_store_ps(dst, a0);
                _mm256_store_ps(dst.add(8), a1);
                ox += 1;
            }

            // Interior 6-pixel tiles (12 YMM accumulators)
            let stride_px = op.s * ci;
            while int_y && ox + 6 <= ow && (ox + 5) * op.s + 2 < iw + op.p {
                let mut a00 = bias0;
                let mut a01 = bias1;
                let mut a10 = bias0;
                let mut a11 = bias1;
                let mut a20 = bias0;
                let mut a21 = bias1;
                let mut a30 = bias0;
                let mut a31 = bias1;
                let mut a40 = bias0;
                let mut a41 = bias1;
                let mut a50 = bias0;
                let mut a51 = bias1;

                let base_y = oy * op.s - op.p;
                let base_x = ox * op.s - op.p;
                for ky in 0..3usize {
                    let row_off = (base_y + ky) * iw + base_x;
                    for kx in 0..3usize {
                        let p0 = src_ptr.add((row_off + kx) * ci);
                        let p1 = p0.add(stride_px);
                        let p2 = p1.add(stride_px);
                        let p3 = p2.add(stride_px);
                        let p4 = p3.add(stride_px);
                        let p5 = p4.add(stride_px);
                        let kw = w_ptr.add((ky * 3 + kx) * ci * co + o);
                        for ic in 0..ci {
                            let wp = kw.add(ic * co);
                            let w0 = _mm256_load_ps(wp);
                            let w1 = _mm256_load_ps(wp.add(8));

                            let x0 = _mm256_broadcast_ss(&*p0.add(ic));
                            a00 = _mm256_fmadd_ps(x0, w0, a00);
                            a01 = _mm256_fmadd_ps(x0, w1, a01);

                            let x1 = _mm256_broadcast_ss(&*p1.add(ic));
                            a10 = _mm256_fmadd_ps(x1, w0, a10);
                            a11 = _mm256_fmadd_ps(x1, w1, a11);

                            let x0 = _mm256_broadcast_ss(&*p2.add(ic));
                            a20 = _mm256_fmadd_ps(x0, w0, a20);
                            a21 = _mm256_fmadd_ps(x0, w1, a21);

                            let x1 = _mm256_broadcast_ss(&*p3.add(ic));
                            a30 = _mm256_fmadd_ps(x1, w0, a30);
                            a31 = _mm256_fmadd_ps(x1, w1, a31);

                            let x0 = _mm256_broadcast_ss(&*p4.add(ic));
                            a40 = _mm256_fmadd_ps(x0, w0, a40);
                            a41 = _mm256_fmadd_ps(x0, w1, a41);

                            let x1 = _mm256_broadcast_ss(&*p5.add(ic));
                            a50 = _mm256_fmadd_ps(x1, w0, a50);
                            a51 = _mm256_fmadd_ps(x1, w1, a51);
                        }
                    }
                }

                if op.relu {
                    a00 = _mm256_max_ps(a00, zero);
                    a01 = _mm256_max_ps(a01, zero);
                    a10 = _mm256_max_ps(a10, zero);
                    a11 = _mm256_max_ps(a11, zero);
                    a20 = _mm256_max_ps(a20, zero);
                    a21 = _mm256_max_ps(a21, zero);
                    a30 = _mm256_max_ps(a30, zero);
                    a31 = _mm256_max_ps(a31, zero);
                    a40 = _mm256_max_ps(a40, zero);
                    a41 = _mm256_max_ps(a41, zero);
                    a50 = _mm256_max_ps(a50, zero);
                    a51 = _mm256_max_ps(a51, zero);
                }
                let d0 = out_ptr.add((oy * ow + ox) * co + o);
                _mm256_store_ps(d0, a00);
                _mm256_store_ps(d0.add(8), a01);
                let d1 = d0.add(co);
                _mm256_store_ps(d1, a10);
                _mm256_store_ps(d1.add(8), a11);
                let d2 = d1.add(co);
                _mm256_store_ps(d2, a20);
                _mm256_store_ps(d2.add(8), a21);
                let d3 = d2.add(co);
                _mm256_store_ps(d3, a30);
                _mm256_store_ps(d3.add(8), a31);
                let d4 = d3.add(co);
                _mm256_store_ps(d4, a40);
                _mm256_store_ps(d4.add(8), a41);
                let d5 = d4.add(co);
                _mm256_store_ps(d5, a50);
                _mm256_store_ps(d5.add(8), a51);

                ox += 6;
            }

            // Right tail & border pixels
            while ox < ow {
                let mut a0 = bias0;
                let mut a1 = bias1;
                for ky in 0..3usize {
                    let iy = (oy * op.s + ky).wrapping_sub(op.p);
                    if iy >= ih {
                        continue;
                    }
                    for kx in 0..3usize {
                        let ix = (ox * op.s + kx).wrapping_sub(op.p);
                        if ix >= iw {
                            continue;
                        }
                        let px = src_ptr.add((iy * iw + ix) * ci);
                        let kw = w_ptr.add((ky * 3 + kx) * ci * co + o);
                        for ic in 0..ci {
                            let xv = _mm256_broadcast_ss(&*px.add(ic));
                            let wp = kw.add(ic * co);
                            a0 = _mm256_fmadd_ps(xv, _mm256_load_ps(wp), a0);
                            a1 = _mm256_fmadd_ps(xv, _mm256_load_ps(wp.add(8)), a1);
                        }
                    }
                }
                if op.relu {
                    a0 = _mm256_max_ps(a0, zero);
                    a1 = _mm256_max_ps(a1, zero);
                }
                let dst = out_ptr.add((oy * ow + ox) * co + o);
                _mm256_store_ps(dst, a0);
                _mm256_store_ps(dst.add(8), a1);
                ox += 1;
            }

            o += 16;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dw_avx2(src_ptr: *const f32, ih: usize, iw: usize, op: &ConvOp, out_ptr: *mut f32) {
    let oh = ih / op.s;
    let ow = iw / op.s;
    let c = op.co;
    let k = op.k;
    let zero = _mm256_setzero_ps();
    let w_ptr = op.weights.as_ptr();
    let b_ptr = op.biases.as_ptr();

    for oy in 0..oh {
        let int_y = oy * op.s >= op.p && oy * op.s + k - 1 < ih + op.p;

        let mut ch = 0usize;
        while ch + 16 <= c {
            let bias0 = _mm256_load_ps(b_ptr.add(ch));
            let bias1 = _mm256_load_ps(b_ptr.add(ch + 8));

            let mut ox = 0usize;
            while ox < ow && (!int_y || ox * op.s < op.p) {
                let mut a0 = bias0;
                let mut a1 = bias1;
                for ky in 0..k {
                    let iy = (oy * op.s + ky).wrapping_sub(op.p);
                    if iy >= ih {
                        continue;
                    }
                    for kx in 0..k {
                        let ix = (ox * op.s + kx).wrapping_sub(op.p);
                        if ix >= iw {
                            continue;
                        }
                        let xp = src_ptr.add((iy * iw + ix) * c + ch);
                        let wp = w_ptr.add((ky * k + kx) * c + ch);
                        a0 = _mm256_fmadd_ps(_mm256_load_ps(xp), _mm256_load_ps(wp), a0);
                        a1 = _mm256_fmadd_ps(_mm256_load_ps(xp.add(8)), _mm256_load_ps(wp.add(8)), a1);
                    }
                }
                if op.relu {
                    a0 = _mm256_max_ps(a0, zero);
                    a1 = _mm256_max_ps(a1, zero);
                }
                let dst = out_ptr.add((oy * ow + ox) * c + ch);
                _mm256_store_ps(dst, a0);
                _mm256_store_ps(dst.add(8), a1);
                ox += 1;
            }

            // 4-pixel x 16-channel interior tile (8 YMM accumulators, weights loaded once per tap)
            let stride_c = op.s * c;
            while int_y && ox + 4 <= ow && (ox + 3) * op.s + k - 1 < iw + op.p {
                let mut a00 = bias0;
                let mut a01 = bias1;
                let mut a10 = bias0;
                let mut a11 = bias1;
                let mut a20 = bias0;
                let mut a21 = bias1;
                let mut a30 = bias0;
                let mut a31 = bias1;

                let base_y = oy * op.s - op.p;
                let base_x = ox * op.s - op.p;
                for ky in 0..k {
                    let row_off = (base_y + ky) * iw + base_x;
                    for kx in 0..k {
                        let p0 = src_ptr.add((row_off + kx) * c + ch);
                        let p1 = p0.add(stride_c);
                        let p2 = p1.add(stride_c);
                        let p3 = p2.add(stride_c);
                        let wp = w_ptr.add((ky * k + kx) * c + ch);
                        let w0 = _mm256_load_ps(wp);
                        let w1 = _mm256_load_ps(wp.add(8));

                        a00 = _mm256_fmadd_ps(_mm256_load_ps(p0), w0, a00);
                        a01 = _mm256_fmadd_ps(_mm256_load_ps(p0.add(8)), w1, a01);
                        a10 = _mm256_fmadd_ps(_mm256_load_ps(p1), w0, a10);
                        a11 = _mm256_fmadd_ps(_mm256_load_ps(p1.add(8)), w1, a11);
                        a20 = _mm256_fmadd_ps(_mm256_load_ps(p2), w0, a20);
                        a21 = _mm256_fmadd_ps(_mm256_load_ps(p2.add(8)), w1, a21);
                        a30 = _mm256_fmadd_ps(_mm256_load_ps(p3), w0, a30);
                        a31 = _mm256_fmadd_ps(_mm256_load_ps(p3.add(8)), w1, a31);
                    }
                }
                if op.relu {
                    a00 = _mm256_max_ps(a00, zero);
                    a01 = _mm256_max_ps(a01, zero);
                    a10 = _mm256_max_ps(a10, zero);
                    a11 = _mm256_max_ps(a11, zero);
                    a20 = _mm256_max_ps(a20, zero);
                    a21 = _mm256_max_ps(a21, zero);
                    a30 = _mm256_max_ps(a30, zero);
                    a31 = _mm256_max_ps(a31, zero);
                }
                let d0 = out_ptr.add((oy * ow + ox) * c + ch);
                _mm256_store_ps(d0, a00);
                _mm256_store_ps(d0.add(8), a01);
                let d1 = d0.add(c);
                _mm256_store_ps(d1, a10);
                _mm256_store_ps(d1.add(8), a11);
                let d2 = d1.add(c);
                _mm256_store_ps(d2, a20);
                _mm256_store_ps(d2.add(8), a21);
                let d3 = d2.add(c);
                _mm256_store_ps(d3, a30);
                _mm256_store_ps(d3.add(8), a31);
                ox += 4;
            }

            while ox < ow {
                let mut a0 = bias0;
                let mut a1 = bias1;
                for ky in 0..k {
                    let iy = (oy * op.s + ky).wrapping_sub(op.p);
                    if iy >= ih {
                        continue;
                    }
                    for kx in 0..k {
                        let ix = (ox * op.s + kx).wrapping_sub(op.p);
                        if ix >= iw {
                            continue;
                        }
                        let xp = src_ptr.add((iy * iw + ix) * c + ch);
                        let wp = w_ptr.add((ky * k + kx) * c + ch);
                        a0 = _mm256_fmadd_ps(_mm256_load_ps(xp), _mm256_load_ps(wp), a0);
                        a1 = _mm256_fmadd_ps(_mm256_load_ps(xp.add(8)), _mm256_load_ps(wp.add(8)), a1);
                    }
                }
                if op.relu {
                    a0 = _mm256_max_ps(a0, zero);
                    a1 = _mm256_max_ps(a1, zero);
                }
                let dst = out_ptr.add((oy * ow + ox) * c + ch);
                _mm256_store_ps(dst, a0);
                _mm256_store_ps(dst.add(8), a1);
                ox += 1;
            }

            ch += 16;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn pw_avx2(
    src_ptr: *const f32,
    pixels: usize,
    op: &ConvOp,
    skip_ptr: Option<*const f32>,
    out_ptr: *mut f32,
) {
    let ci = op.ci;
    let co = op.co;
    let zero = _mm256_setzero_ps();
    let w_ptr = op.weights.as_ptr();
    let b_ptr = op.biases.as_ptr();

    let mut px = 0usize;
    // 6-pixel x 16-channel microkernel (12 YMM accumulators)
    while px + 6 <= pixels {
        let p0 = src_ptr.add(px * ci);
        let p1 = p0.add(ci);
        let p2 = p1.add(ci);
        let p3 = p2.add(ci);
        let p4 = p3.add(ci);
        let p5 = p4.add(ci);

        let d0 = out_ptr.add(px * co);
        let d1 = d0.add(co);
        let d2 = d1.add(co);
        let d3 = d2.add(co);
        let d4 = d3.add(co);
        let d5 = d4.add(co);

        let mut o = 0usize;
        while o + 16 <= co {
            let b_lo = _mm256_load_ps(b_ptr.add(o));
            let b_hi = _mm256_load_ps(b_ptr.add(o + 8));
            let mut a00 = b_lo;
            let mut a01 = b_hi;
            let mut a10 = b_lo;
            let mut a11 = b_hi;
            let mut a20 = b_lo;
            let mut a21 = b_hi;
            let mut a30 = b_lo;
            let mut a31 = b_hi;
            let mut a40 = b_lo;
            let mut a41 = b_hi;
            let mut a50 = b_lo;
            let mut a51 = b_hi;

            for ic in 0..ci {
                let wp = w_ptr.add(ic * co + o);
                let w0 = _mm256_load_ps(wp);
                let w1 = _mm256_load_ps(wp.add(8));

                let x0 = _mm256_broadcast_ss(&*p0.add(ic));
                a00 = _mm256_fmadd_ps(x0, w0, a00);
                a01 = _mm256_fmadd_ps(x0, w1, a01);

                let x1 = _mm256_broadcast_ss(&*p1.add(ic));
                a10 = _mm256_fmadd_ps(x1, w0, a10);
                a11 = _mm256_fmadd_ps(x1, w1, a11);

                let x0 = _mm256_broadcast_ss(&*p2.add(ic));
                a20 = _mm256_fmadd_ps(x0, w0, a20);
                a21 = _mm256_fmadd_ps(x0, w1, a21);

                let x1 = _mm256_broadcast_ss(&*p3.add(ic));
                a30 = _mm256_fmadd_ps(x1, w0, a30);
                a31 = _mm256_fmadd_ps(x1, w1, a31);

                let x0 = _mm256_broadcast_ss(&*p4.add(ic));
                a40 = _mm256_fmadd_ps(x0, w0, a40);
                a41 = _mm256_fmadd_ps(x0, w1, a41);

                let x1 = _mm256_broadcast_ss(&*p5.add(ic));
                a50 = _mm256_fmadd_ps(x1, w0, a50);
                a51 = _mm256_fmadd_ps(x1, w1, a51);
            }

            if op.relu {
                a00 = _mm256_max_ps(a00, zero);
                a01 = _mm256_max_ps(a01, zero);
                a10 = _mm256_max_ps(a10, zero);
                a11 = _mm256_max_ps(a11, zero);
                a20 = _mm256_max_ps(a20, zero);
                a21 = _mm256_max_ps(a21, zero);
                a30 = _mm256_max_ps(a30, zero);
                a31 = _mm256_max_ps(a31, zero);
                a40 = _mm256_max_ps(a40, zero);
                a41 = _mm256_max_ps(a41, zero);
                a50 = _mm256_max_ps(a50, zero);
                a51 = _mm256_max_ps(a51, zero);
            }
            if let Some(sk) = skip_ptr {
                let s0 = sk.add(px * co + o);
                let s1 = s0.add(co);
                let s2 = s1.add(co);
                let s3 = s2.add(co);
                let s4 = s3.add(co);
                let s5 = s4.add(co);
                a00 = _mm256_add_ps(a00, _mm256_load_ps(s0));
                a01 = _mm256_add_ps(a01, _mm256_load_ps(s0.add(8)));
                a10 = _mm256_add_ps(a10, _mm256_load_ps(s1));
                a11 = _mm256_add_ps(a11, _mm256_load_ps(s1.add(8)));
                a20 = _mm256_add_ps(a20, _mm256_load_ps(s2));
                a21 = _mm256_add_ps(a21, _mm256_load_ps(s2.add(8)));
                a30 = _mm256_add_ps(a30, _mm256_load_ps(s3));
                a31 = _mm256_add_ps(a31, _mm256_load_ps(s3.add(8)));
                a40 = _mm256_add_ps(a40, _mm256_load_ps(s4));
                a41 = _mm256_add_ps(a41, _mm256_load_ps(s4.add(8)));
                a50 = _mm256_add_ps(a50, _mm256_load_ps(s5));
                a51 = _mm256_add_ps(a51, _mm256_load_ps(s5.add(8)));
            }

            _mm256_store_ps(d0.add(o), a00);
            _mm256_store_ps(d0.add(o + 8), a01);
            _mm256_store_ps(d1.add(o), a10);
            _mm256_store_ps(d1.add(o + 8), a11);
            _mm256_store_ps(d2.add(o), a20);
            _mm256_store_ps(d2.add(o + 8), a21);
            _mm256_store_ps(d3.add(o), a30);
            _mm256_store_ps(d3.add(o + 8), a31);
            _mm256_store_ps(d4.add(o), a40);
            _mm256_store_ps(d4.add(o + 8), a41);
            _mm256_store_ps(d5.add(o), a50);
            _mm256_store_ps(d5.add(o + 8), a51);
            o += 16;
        }
        px += 6;
    }

    // 4-pixel tail
    while px + 4 <= pixels {
        let p0 = src_ptr.add(px * ci);
        let p1 = p0.add(ci);
        let p2 = p1.add(ci);
        let p3 = p2.add(ci);
        let d0 = out_ptr.add(px * co);
        let d1 = d0.add(co);
        let d2 = d1.add(co);
        let d3 = d2.add(co);

        let mut o = 0usize;
        while o + 16 <= co {
            let b_lo = _mm256_load_ps(b_ptr.add(o));
            let b_hi = _mm256_load_ps(b_ptr.add(o + 8));
            let mut a00 = b_lo;
            let mut a01 = b_hi;
            let mut a10 = b_lo;
            let mut a11 = b_hi;
            let mut a20 = b_lo;
            let mut a21 = b_hi;
            let mut a30 = b_lo;
            let mut a31 = b_hi;

            for ic in 0..ci {
                let wp = w_ptr.add(ic * co + o);
                let w0 = _mm256_load_ps(wp);
                let w1 = _mm256_load_ps(wp.add(8));
                let x0 = _mm256_broadcast_ss(&*p0.add(ic));
                let x1 = _mm256_broadcast_ss(&*p1.add(ic));
                let x2 = _mm256_broadcast_ss(&*p2.add(ic));
                let x3 = _mm256_broadcast_ss(&*p3.add(ic));
                a00 = _mm256_fmadd_ps(x0, w0, a00);
                a01 = _mm256_fmadd_ps(x0, w1, a01);
                a10 = _mm256_fmadd_ps(x1, w0, a10);
                a11 = _mm256_fmadd_ps(x1, w1, a11);
                a20 = _mm256_fmadd_ps(x2, w0, a20);
                a21 = _mm256_fmadd_ps(x2, w1, a21);
                a30 = _mm256_fmadd_ps(x3, w0, a30);
                a31 = _mm256_fmadd_ps(x3, w1, a31);
            }
            if op.relu {
                a00 = _mm256_max_ps(a00, zero);
                a01 = _mm256_max_ps(a01, zero);
                a10 = _mm256_max_ps(a10, zero);
                a11 = _mm256_max_ps(a11, zero);
                a20 = _mm256_max_ps(a20, zero);
                a21 = _mm256_max_ps(a21, zero);
                a30 = _mm256_max_ps(a30, zero);
                a31 = _mm256_max_ps(a31, zero);
            }
            if let Some(sk) = skip_ptr {
                let s0 = sk.add(px * co + o);
                let s1 = s0.add(co);
                let s2 = s1.add(co);
                let s3 = s2.add(co);
                a00 = _mm256_add_ps(a00, _mm256_load_ps(s0));
                a01 = _mm256_add_ps(a01, _mm256_load_ps(s0.add(8)));
                a10 = _mm256_add_ps(a10, _mm256_load_ps(s1));
                a11 = _mm256_add_ps(a11, _mm256_load_ps(s1.add(8)));
                a20 = _mm256_add_ps(a20, _mm256_load_ps(s2));
                a21 = _mm256_add_ps(a21, _mm256_load_ps(s2.add(8)));
                a30 = _mm256_add_ps(a30, _mm256_load_ps(s3));
                a31 = _mm256_add_ps(a31, _mm256_load_ps(s3.add(8)));
            }
            _mm256_store_ps(d0.add(o), a00);
            _mm256_store_ps(d0.add(o + 8), a01);
            _mm256_store_ps(d1.add(o), a10);
            _mm256_store_ps(d1.add(o + 8), a11);
            _mm256_store_ps(d2.add(o), a20);
            _mm256_store_ps(d2.add(o + 8), a21);
            _mm256_store_ps(d3.add(o), a30);
            _mm256_store_ps(d3.add(o + 8), a31);
            o += 16;
        }
        px += 4;
    }

    while px < pixels {
        let p0 = src_ptr.add(px * ci);
        let d0 = out_ptr.add(px * co);
        let mut o = 0usize;
        while o + 16 <= co {
            let mut a0 = _mm256_load_ps(b_ptr.add(o));
            let mut a1 = _mm256_load_ps(b_ptr.add(o + 8));
            for ic in 0..ci {
                let x0 = _mm256_broadcast_ss(&*p0.add(ic));
                let wp = w_ptr.add(ic * co + o);
                a0 = _mm256_fmadd_ps(x0, _mm256_load_ps(wp), a0);
                a1 = _mm256_fmadd_ps(x0, _mm256_load_ps(wp.add(8)), a1);
            }
            if op.relu {
                a0 = _mm256_max_ps(a0, zero);
                a1 = _mm256_max_ps(a1, zero);
            }
            if let Some(sk) = skip_ptr {
                let s0 = sk.add(px * co + o);
                a0 = _mm256_add_ps(a0, _mm256_load_ps(s0));
                a1 = _mm256_add_ps(a1, _mm256_load_ps(s0.add(8)));
            }
            _mm256_store_ps(d0.add(o), a0);
            _mm256_store_ps(d0.add(o + 8), a1);
            o += 16;
        }
        px += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hsum_ps(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pool_and_head_avx2(
    feat_ptr: *const f32,
    pixels: usize,
    head_in: usize,
    head_out: usize,
    hw_ptr: *const f32,
    hb_ptr: *const f32,
    pool_ptr: *mut f32,
    out: &mut [f32],
) {
    let inv = _mm256_set1_ps(1.0 / (pixels as f32));
    let mut ch = 0usize;
    while ch + 16 <= head_in {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        for px in 0..pixels {
            let p = feat_ptr.add(px * head_in + ch);
            s0 = _mm256_add_ps(s0, _mm256_load_ps(p));
            s1 = _mm256_add_ps(s1, _mm256_load_ps(p.add(8)));
        }
        _mm256_store_ps(pool_ptr.add(ch), _mm256_mul_ps(s0, inv));
        _mm256_store_ps(pool_ptr.add(ch + 8), _mm256_mul_ps(s1, inv));
        ch += 16;
    }

    // 4-row tiled dot product (8 YMM accumulators, reusing pool loads across 4 rows)
    let mut row = 0usize;
    while row + 4 <= head_out {
        let w0_row = hw_ptr.add(row * head_in);
        let w1_row = w0_row.add(head_in);
        let w2_row = w1_row.add(head_in);
        let w3_row = w2_row.add(head_in);

        let mut a00 = _mm256_setzero_ps();
        let mut a01 = _mm256_setzero_ps();
        let mut a10 = _mm256_setzero_ps();
        let mut a11 = _mm256_setzero_ps();
        let mut a20 = _mm256_setzero_ps();
        let mut a21 = _mm256_setzero_ps();
        let mut a30 = _mm256_setzero_ps();
        let mut a31 = _mm256_setzero_ps();

        let mut i = 0usize;
        while i + 16 <= head_in {
            let p0 = _mm256_load_ps(pool_ptr.add(i));
            let p1 = _mm256_load_ps(pool_ptr.add(i + 8));

            a00 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(w0_row.add(i)), a00);
            a01 = _mm256_fmadd_ps(p1, _mm256_loadu_ps(w0_row.add(i + 8)), a01);
            a10 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(w1_row.add(i)), a10);
            a11 = _mm256_fmadd_ps(p1, _mm256_loadu_ps(w1_row.add(i + 8)), a11);
            a20 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(w2_row.add(i)), a20);
            a21 = _mm256_fmadd_ps(p1, _mm256_loadu_ps(w2_row.add(i + 8)), a21);
            a30 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(w3_row.add(i)), a30);
            a31 = _mm256_fmadd_ps(p1, _mm256_loadu_ps(w3_row.add(i + 8)), a31);
            i += 16;
        }
        out[row] = hsum_ps(_mm256_add_ps(a00, a01)) + *hb_ptr.add(row);
        out[row + 1] = hsum_ps(_mm256_add_ps(a10, a11)) + *hb_ptr.add(row + 1);
        out[row + 2] = hsum_ps(_mm256_add_ps(a20, a21)) + *hb_ptr.add(row + 2);
        out[row + 3] = hsum_ps(_mm256_add_ps(a30, a31)) + *hb_ptr.add(row + 3);
        row += 4;
    }

    while row < head_out {
        let w_row = hw_ptr.add(row * head_in);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= head_in {
            acc0 = _mm256_fmadd_ps(_mm256_load_ps(pool_ptr.add(i)), _mm256_loadu_ps(w_row.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_load_ps(pool_ptr.add(i + 8)), _mm256_loadu_ps(w_row.add(i + 8)), acc1);
            i += 16;
        }
        out[row] = hsum_ps(_mm256_add_ps(acc0, acc1)) + *hb_ptr.add(row);
        row += 1;
    }

    l2_normalize(out);
}

struct SpinPoolShared {
    encoder: Arc<NanoEncoder>,
    worker_count: usize,
    epoch: AtomicU64,
    shutdown: AtomicBool,
    batch_active: AtomicBool,
    next_idx: AtomicUsize,
    done: AtomicUsize,
    images_ptr: AtomicPtr<()>,
    images_len: AtomicUsize,
    out_ptr: AtomicPtr<f32>,
}

struct ActiveBatch<'a>(&'a SpinPoolShared);

impl Drop for ActiveBatch<'_> {
    fn drop(&mut self) {
        // Every worker must stop using the borrowed buffers before this call can return or unwind.
        while self.0.done.load(Ordering::Acquire) < self.0.worker_count {
            spin_loop();
        }
        self.0.batch_active.store(false, Ordering::Release);
    }
}

/// Persistent spin-wait thread pool for multi-core `nano-v1` batch encoding.
pub struct NanoSpinPool {
    shared: Arc<SpinPoolShared>,
    workers: Vec<JoinHandle<()>>,
}

impl NanoSpinPool {
    /// Creates a persistent pool with `threads` total participants (`threads - 1` background
    /// workers + the calling thread). Never allocates or locks OS mutexes during `encode_batch`.
    pub fn new(encoder: Arc<NanoEncoder>, threads: usize) -> Self {
        let total = threads.max(1);
        let worker_count = total - 1;
        let shared = Arc::new(SpinPoolShared {
            encoder,
            worker_count,
            epoch: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            batch_active: AtomicBool::new(false),
            next_idx: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            images_ptr: AtomicPtr::new(core::ptr::null_mut()),
            images_len: AtomicUsize::new(0),
            out_ptr: AtomicPtr::new(core::ptr::null_mut()),
        });

        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let s = Arc::clone(&shared);
            workers.push(thread::spawn(move || {
                let mut ws = s.encoder.workspace();
                let dim = s.encoder.embedding_dim();
                let mut seen_epoch = 0u64;
                loop {
                    let mut spins = 0u32;
                    let curr = loop {
                        let e = s.epoch.load(Ordering::Acquire);
                        if e != seen_epoch {
                            break e;
                        }
                        if s.shutdown.load(Ordering::Acquire) {
                            return;
                        }
                        if spins < 4096 {
                            spin_loop();
                            spins += 1;
                        } else {
                            thread::yield_now();
                        }
                    };
                    if s.shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    seen_epoch = curr;

                    let len = s.images_len.load(Ordering::Relaxed);
                    let img_base = s.images_ptr.load(Ordering::Relaxed).cast::<&[u8]>();
                    let out_base = s.out_ptr.load(Ordering::Relaxed);
                    loop {
                        let idx = s.next_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= len {
                            break;
                        }
                        // SAFETY: the active batch keeps both buffers alive until all workers
                        // finish. Each index has one owner and a disjoint output slice.
                        unsafe {
                            let img = *img_base.add(idx);
                            let dst = core::slice::from_raw_parts_mut(out_base.add(idx * dim), dim);
                            let _ = s.encoder.encode_rgb(img, &mut ws, dst);
                        }
                    }
                    s.done.fetch_add(1, Ordering::Release);
                }
            }));
        }

        Self { shared, workers }
    }

    pub fn threads(&self) -> usize {
        self.shared.worker_count + 1
    }

    /// Encodes `images` into `out` (`images.len() * dim`) with zero heap allocations and zero mutexes.
    /// With background workers, concurrent callers wait for the active batch to finish.
    pub fn encode_batch(
        &self,
        images: &[&[u8]],
        ws_main: &mut NanoWorkspace,
        out: &mut [f32],
    ) -> Result<(), &'static str> {
        let dim = self.shared.encoder.embedding_dim();
        let needed = images.len().checked_mul(dim).ok_or("batch output overflow")?;
        if out.len() < needed {
            return Err("batch output slice too small");
        }
        if images.is_empty() {
            return Ok(());
        }
        for &im in images {
            if im.len() != RGB_BYTES {
                return Err("nano-v1 expects 128x128x3 RGB bytes");
            }
        }
        if ws_main.head_in != self.shared.encoder.head_in {
            return Err("nano-v1 output/workspace mismatch");
        }

        if self.shared.worker_count == 0 {
            for (i, &im) in images.iter().enumerate() {
                self.shared
                    .encoder
                    .encode_rgb(im, ws_main, &mut out[i * dim..(i + 1) * dim])?;
            }
            return Ok(());
        }

        let shared = &*self.shared;
        while shared
            .batch_active
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            thread::yield_now();
        }

        let len = images.len();
        let out_ptr = out.as_mut_ptr();
        shared.next_idx.store(0, Ordering::Relaxed);
        shared.done.store(0, Ordering::Relaxed);
        shared
            .images_ptr
            .store(images.as_ptr().cast_mut().cast(), Ordering::Relaxed);
        shared.images_len.store(len, Ordering::Relaxed);
        shared.out_ptr.store(out_ptr, Ordering::Relaxed);

        // Release all workers simultaneously
        shared.epoch.fetch_add(1, Ordering::Release);
        let _batch = ActiveBatch(shared);

        // Calling thread participates in dynamic work stealing
        loop {
            let idx = shared.next_idx.fetch_add(1, Ordering::Relaxed);
            if idx >= len {
                break;
            }
            // SAFETY: each `idx` is claimed exclusively via atomic `fetch_add`.
            unsafe {
                let dst = core::slice::from_raw_parts_mut(out_ptr.add(idx * dim), dim);
                shared.encoder.encode_rgb(images[idx], ws_main, dst)?;
            }
        }

        Ok(())
    }
}

impl Drop for NanoSpinPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

static BAKED_ENCODER: OnceLock<Result<NanoEncoder, &'static str>> = OnceLock::new();

thread_local! {
    static THREAD_WS: RefCell<Option<NanoWorkspace>> = const { RefCell::new(None) };
}

/// Encodes a single `128x128x3` RGB buffer using the compile-time `.rodata` baked `nano-v1` model.
/// Zero heap allocations after the first call on the calling thread.
pub fn encode_single_rgb(rgb: &[u8]) -> Result<[f32; NANO_EMBEDDING_DIM], &'static str> {
    let encoder = BAKED_ENCODER
        .get_or_init(NanoEncoder::baked)
        .as_ref()
        .map_err(|&err| err)?;
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
