use crate::{
    l2_normalize, ConvOp, NanoEncoder, NanoWorkspace, BUF_FLOATS, OFF_BUF_A, OFF_BUF_B, OFF_POOL,
    OFF_RGB_STRIP, OFF_SKIP, OFF_STEM_RING, RGB_SIDE, RGB_STRIP_FLOATS, SKIP_FLOATS,
    STEM_RING_FLOATS,
};

impl NanoEncoder {
    pub(super) fn run_scalar(&self, rgb: &[u8], ws: &mut NanoWorkspace, out: &mut [f32]) {
        let arena_ptr = ws.arena.as_mut_ptr();
        // SAFETY: all non-overlapping sub-slices live within `ws.arena` bounds.
        unsafe {
            let buf_a = core::slice::from_raw_parts_mut(arena_ptr.add(OFF_BUF_A), BUF_FLOATS);
            let buf_b = core::slice::from_raw_parts_mut(arena_ptr.add(OFF_BUF_B), BUF_FLOATS);
            let skip = core::slice::from_raw_parts_mut(arena_ptr.add(OFF_SKIP), SKIP_FLOATS);
            let stem_ring =
                core::slice::from_raw_parts_mut(arena_ptr.add(OFF_STEM_RING), STEM_RING_FLOATS);
            let rgb_strip =
                core::slice::from_raw_parts_mut(arena_ptr.add(OFF_RGB_STRIP), RGB_STRIP_FLOATS);
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

fn pw_scalar(src: &[f32], pixels: usize, op: &ConvOp, skip: Option<&[f32]>, out: &mut [f32]) {
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
