use fast_image_resize::{
    images::{Image, ImageRef},
    FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
};
use image::DynamicImage;

use super::{box_resize::PillowBox, DecodeMode, ImageError};
use crate::{RGB_BYTES, RGB_SIDE};

#[derive(Default)]
pub(super) struct Thumbnail {
    resizer: Resizer,
    pillow: PillowBox,
    resized: Vec<u8>,
    rgb: Vec<u8>,
}

impl Thumbnail {
    pub(super) fn rgb<'a>(
        &'a mut self,
        image: &'a DynamicImage,
        mode: DecodeMode,
    ) -> Result<&'a [u8], ImageError> {
        let pixel_type = match image {
            DynamicImage::ImageRgb8(_) => PixelType::U8x3,
            DynamicImage::ImageRgba8(_) => PixelType::U8x4,
            DynamicImage::ImageLuma8(_) => PixelType::U8,
            DynamicImage::ImageLumaA8(_) => PixelType::U8x2,
            _ => unreachable!("decoder converts high-bit-depth images to RGB8"),
        };
        let channels = pixel_type.size();
        let side = RGB_SIDE as u32;
        let bytes = if (image.width(), image.height()) == (side, side) {
            image.as_bytes()
        } else if mode == DecodeMode::Full || (image.height() < side && image.width() > side) {
            // A wide, short panorama would expand to 128 full-width rows in the SIMD resizer.
            self.rgb.resize(RGB_BYTES, 0);
            let (w, h) = (image.width() as usize, image.height() as usize);
            match channels {
                1 => self
                    .pillow
                    .resize::<1>(image.as_bytes(), w, h, &mut self.rgb),
                2 => self
                    .pillow
                    .resize::<2>(image.as_bytes(), w, h, &mut self.rgb),
                3 => self
                    .pillow
                    .resize::<3>(image.as_bytes(), w, h, &mut self.rgb),
                4 => self
                    .pillow
                    .resize::<4>(image.as_bytes(), w, h, &mut self.rgb),
                _ => unreachable!(),
            }
            return Ok(&self.rgb);
        } else {
            let source =
                ImageRef::new(image.width(), image.height(), image.as_bytes(), pixel_type)?;
            self.resized.resize(RGB_SIDE * RGB_SIDE * channels, 0);
            let mut dest = Image::from_slice_u8(side, side, &mut self.resized, pixel_type)?;
            // Forge drops alpha before filtering. Independent channel filtering gives the
            // same RGB semantics and avoids converting a full-size RGBA image.
            let options = ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::Box))
                .use_alpha(false);
            self.resizer.resize(&source, &mut dest, &options)?;
            &self.resized
        };
        if channels == 3 {
            return Ok(bytes);
        }
        self.rgb.resize(RGB_BYTES, 0);
        for (src, dst) in bytes
            .chunks_exact(channels)
            .zip(self.rgb.chunks_exact_mut(3))
        {
            if channels == 4 {
                dst.copy_from_slice(&src[..3]);
            } else {
                dst.fill(src[0]);
            }
        }
        Ok(&self.rgb)
    }
}
