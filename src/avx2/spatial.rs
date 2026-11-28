use core::arch::x86_64::*;

use crate::ConvOp;

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn conv3x3_avx2(
    src_ptr: *const f32,
    ih: usize,
    iw: usize,
    op: &ConvOp,
    out_ptr: *mut f32,
) {
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

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn dw_channels<const VECTORS: usize, const K: usize>(
    input: *const f32,
    weights: *const f32,
    bias: *const f32,
    output: *mut f32,
    input_width: usize,
    channels: usize,
    valid_rows: usize,
    valid_cols: usize,
    relu: bool,
) {
    let mut sums = [_mm256_setzero_ps(); VECTORS];
    for (i, sum) in sums.iter_mut().enumerate() {
        *sum = _mm256_load_ps(bias.add(8 * i));
    }
    for ky in 0..valid_rows {
        let input_row = input.add(ky * input_width * channels);
        let weight_row = weights.add(ky * K * channels);
        for kx in 0..valid_cols {
            let pixel = input_row.add(kx * channels);
            let tap = weight_row.add(kx * channels);
            for (i, sum) in sums.iter_mut().enumerate() {
                *sum = _mm256_fmadd_ps(
                    _mm256_load_ps(pixel.add(8 * i)),
                    _mm256_load_ps(tap.add(8 * i)),
                    *sum,
                );
            }
        }
    }
    for (i, &sum) in sums.iter().enumerate() {
        let value = if relu {
            _mm256_max_ps(sum, _mm256_setzero_ps())
        } else {
            sum
        };
        _mm256_store_ps(output.add(8 * i), value);
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dw_clipped<const K: usize>(
    input: *const f32,
    height: usize,
    width: usize,
    op: &ConvOp,
    output: *mut f32,
) {
    let channels = op.co;
    let stride = op.s;
    let pad = K / 2;
    let output_width = width / stride;
    for y in 0..height / stride {
        let input_y = (y * stride).saturating_sub(pad);
        let kernel_y = pad.saturating_sub(y * stride);
        let valid_rows = (K - kernel_y).min(height - input_y);
        for x in 0..output_width {
            let input_x = (x * stride).saturating_sub(pad);
            let kernel_x = pad.saturating_sub(x * stride);
            let valid_cols = (K - kernel_x).min(width - input_x);
            let first_pixel = input.add((input_y * width + input_x) * channels);
            let first_tap = op
                .weights
                .as_ptr()
                .add((kernel_y * K + kernel_x) * channels);
            let dest = output.add((y * output_width + x) * channels);
            let mut channel = 0;

            // Separate instantiations let LLVM keep each channel block in YMM registers.
            macro_rules! accumulate {
                ($vectors:literal) => {
                    dw_channels::<$vectors, K>(
                        first_pixel.add(channel),
                        first_tap.add(channel),
                        op.biases.as_ptr().add(channel),
                        dest.add(channel),
                        width,
                        channels,
                        valid_rows,
                        valid_cols,
                        op.relu,
                    )
                };
            }
            while channel + 64 <= channels {
                accumulate!(8);
                channel += 64;
            }
            if channel + 32 <= channels {
                accumulate!(4);
                channel += 32;
            }
            // The loader requires channel counts divisible by 16.
            if channel < channels {
                accumulate!(2);
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dw_avx2(
    input: *const f32,
    height: usize,
    width: usize,
    op: &ConvOp,
    output: *mut f32,
) {
    match op.k {
        3 => dw_clipped::<3>(input, height, width, op, output),
        5 => dw_clipped::<5>(input, height, width, op, output),
        _ => unreachable!("unsupported depthwise kernel passed validation"),
    }
}
