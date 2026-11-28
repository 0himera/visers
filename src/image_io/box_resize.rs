//! Match the training pipeline's Pillow 13 BOX bounds, pass order and 8-bit rounding.
use crate::RGB_SIDE;

const ROW: usize = RGB_SIDE * 3;
const PRECISION: u32 = 22;

#[derive(Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
    weight: u32,
}

impl Span {
    fn average(self, sum: u32) -> u8 {
        let fixed = u64::from(sum) * u64::from(self.weight) + (1 << (PRECISION - 1));
        (fixed >> PRECISION).min(255) as u8
    }
}

fn spans(size: usize) -> [Span; RGB_SIDE] {
    let scale = size as f64 / RGB_SIDE as f64;
    let width = scale.max(1.0);
    let inverse = 1.0 / width;
    std::array::from_fn(|i| {
        let center = (i as f64 + 0.5) * scale;
        let mut start = ((center - 0.5 * width + 0.5) as usize).min(size - 1);
        let mut end = ((center + 0.5 * width + 0.5) as usize).min(size);
        // Use the same half-open support and floating-point boundary decisions as Pillow.
        if (start as f64 + 0.5 - center) * inverse <= -0.5 {
            start += 1;
        }
        if (end as f64 - 0.5 - center) * inverse > 0.5 {
            end -= 1;
        }
        let weight = ((1u32 << PRECISION) as f64 / (end - start) as f64 + 0.5) as u32;
        Span { start, end, weight }
    })
}

fn horizontal<const CHANNELS: usize>(source: &[u8], widths: &[Span; RGB_SIDE], row: &mut [u8]) {
    for (&span, dest) in widths.iter().zip(row.chunks_exact_mut(3)) {
        let pixels = &source[span.start * CHANNELS..span.end * CHANNELS];
        let mut sums = [0u32; 3];
        for pixel in pixels.chunks_exact(CHANNELS) {
            sums[0] += u32::from(pixel[0]);
            if CHANNELS >= 3 {
                sums[1] += u32::from(pixel[1]);
                sums[2] += u32::from(pixel[2]);
            }
        }
        if CHANNELS >= 3 {
            dest.copy_from_slice(&sums.map(|s| span.average(s)));
        } else {
            dest.fill(span.average(sums[0]));
        }
    }
}

#[derive(Default)]
pub(super) struct PillowBox {
    row: Vec<u8>,
    sums: Vec<u32>,
}

impl PillowBox {
    pub(super) fn resize<const CHANNELS: usize>(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
        rgb: &mut [u8],
    ) {
        let widths = spans(width);
        let heights = spans(height);
        let stride = width * CHANNELS;
        // Pillow 13 runs the vertical pass first when it removes over twice as many pixels.
        let dx = width as i64 - RGB_SIDE as i64;
        let dy = height as i64 - RGB_SIDE as i64;
        if dy > 0 && dy > 2 * dx {
            self.row.resize(stride, 0);
            self.sums.resize(stride, 0);
            for (span, dest) in heights.iter().zip(rgb.chunks_exact_mut(ROW)) {
                self.sums.fill(0);
                for row in pixels[span.start * stride..span.end * stride].chunks_exact(stride) {
                    for (sum, &v) in self.sums.iter_mut().zip(row) {
                        *sum += u32::from(v);
                    }
                }
                for (v, &sum) in self.row.iter_mut().zip(&self.sums) {
                    *v = span.average(sum);
                }
                horizontal::<CHANNELS>(&self.row, &widths, dest);
            }
            return;
        }
        // A running sum keeps scratch independent of the original image's height.
        let mut row = [0u8; ROW];
        let mut previous = None;
        for (span, dest) in heights.iter().zip(rgb.chunks_exact_mut(ROW)) {
            let mut sums = [0u32; ROW];
            for y in span.start..span.end {
                if previous != Some(y) {
                    horizontal::<CHANNELS>(
                        &pixels[y * stride..(y + 1) * stride],
                        &widths,
                        &mut row,
                    );
                    previous = Some(y);
                }
                if span.end - span.start > 1 {
                    for (sum, &v) in sums.iter_mut().zip(&row) {
                        *sum += u32::from(v);
                    }
                }
            }
            if span.end - span.start == 1 {
                dest.copy_from_slice(&row);
            } else {
                for (v, sum) in dest.iter_mut().zip(sums) {
                    *v = span.average(sum);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PillowBox;
    use image::Rgb;

    #[test]
    fn full_resizing_matches_forge_rgb_for_both_pass_orders() {
        // FNV-1a hashes of Pillow 13 RGB BOX outputs. A transposed pass order changes the rounding.
        for (w, h, reference) in [
            (287, 133, 0x0e49448feb33b724),
            (133, 287, 0x1a8231a8f572176d),
            (313, 512, 0x1cc4b2bdf4c16d78),
            (640, 383, 0x69e78fa5bc43db23),
            (37, 51, 0x5ea01d95cd157127),
            (1, 17, 0x4f5e849f67d1e725),
            (914, 1, 0x84d56e5cce89eb25),
        ] {
            let img = image::RgbImage::from_fn(w, h, |x, y| {
                Rgb(std::array::from_fn(|c| {
                    (x * 7 + y * 11 + c as u32 * 37 + (x * y) % 91) as u8
                }))
            });
            let mut filter = PillowBox::default();
            let mut out = vec![0; crate::RGB_BYTES];
            filter.resize::<3>(&img, w as usize, h as usize, &mut out);
            let hash = out.iter().fold(0xcbf29ce484222325u64, |h, v| {
                (h ^ u64::from(*v)).wrapping_mul(0x100000001b3)
            });
            assert_eq!(hash, reference, "{w} x {h}");
        }
    }
}
