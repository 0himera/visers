use core::arch::x86_64::*;

use crate::ConvOp;

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn pw_avx2(
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
