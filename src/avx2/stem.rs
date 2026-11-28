use core::arch::x86_64::*;

use crate::{ConvOp, NanoEncoder, RGB_SIDE};

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

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn fused_stem_and_block0_avx2(
    encoder: &NanoEncoder,
    rgb_ptr: *const u8,
    rgb_strip_ptr: *mut f32,
    stem_ring_ptr: *mut f32,
    out_ptr: *mut f32,
) {
    let stem = &encoder.ops[0];
    let b0 = &encoder.ops[1];
    let lut = &encoder.norm_lut;
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
                if encoder.standard_rgb {
                    for offset in (0..row_rgb_len).step_by(8) {
                        let bytes = _mm_loadl_epi64(src_row.add(offset).cast());
                        let value = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
                        let unit = _mm256_div_ps(value, _mm256_set1_ps(255.0));
                        let normalized = _mm256_mul_ps(
                            _mm256_sub_ps(unit, _mm256_set1_ps(0.5)),
                            _mm256_set1_ps(2.0),
                        );
                        _mm256_store_ps(dst_row.add(offset), normalized);
                    }
                } else {
                    let mut x = 0usize;
                    while x < RGB_SIDE {
                        let p = x * 3;
                        *dst_row.add(p) = *lut[0].get_unchecked(*src_row.add(p) as usize);
                        *dst_row.add(p + 1) = *lut[1].get_unchecked(*src_row.add(p + 1) as usize);
                        *dst_row.add(p + 2) = *lut[2].get_unchecked(*src_row.add(p + 2) as usize);
                        x += 1;
                    }
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
