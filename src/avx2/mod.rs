mod pointwise;
mod spatial;
mod stem;

use pointwise::pw_avx2;
use spatial::{conv3x3_avx2, dw_avx2};
use stem::fused_stem_and_block0_avx2;

use core::arch::x86_64::*;
use std::time::Instant;

use crate::{
    l2_normalize, NanoEncoder, NanoStageTimings, NanoWorkspace, OFF_BUF_A, OFF_BUF_B, OFF_POOL,
    OFF_RGB_STRIP, OFF_SKIP, OFF_STEM_RING,
};

struct FtzDazGuard {
    saved_mxcsr: u32,
}

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

impl NanoEncoder {
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn run_avx2(
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
        fused_stem_and_block0_avx2(self, rgb.as_ptr(), rgb_strip_ptr, stem_ring_ptr, buf_a_ptr);
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
            acc0 = _mm256_fmadd_ps(
                _mm256_load_ps(pool_ptr.add(i)),
                _mm256_loadu_ps(w_row.add(i)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_load_ps(pool_ptr.add(i + 8)),
                _mm256_loadu_ps(w_row.add(i + 8)),
                acc1,
            );
            i += 16;
        }
        out[row] = hsum_ps(_mm256_add_ps(acc0, acc1)) + *hb_ptr.add(row);
        row += 1;
    }

    l2_normalize(out);
}
