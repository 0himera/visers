use std::{io::Cursor, mem};

use image::{
    metadata::Orientation, ColorType, DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits,
};
use turbojpeg::{Colorspace, Decompressor, PixelFormat, ScalingFactor};

use super::{orientation, webp, DecodeMode, ImageError};
use crate::RGB_SIDE;

#[derive(Default)]
pub(super) struct Decoder {
    pixels: Vec<u8>,
    jpeg: Option<Decompressor>,
}

impl Decoder {
    pub(super) fn decode(
        &mut self,
        bytes: &[u8],
        mode: DecodeMode,
    ) -> Result<(DynamicImage, Orientation), ImageError> {
        let format = image::guess_format(bytes)?;
        match format {
            ImageFormat::Jpeg => self.jpeg(bytes, mode),
            ImageFormat::WebP => {
                if let Some(image) = webp::decode_still(bytes, mode, &mut self.pixels)? {
                    return Ok((
                        image,
                        orientation::webp(bytes).unwrap_or(Orientation::NoTransforms),
                    ));
                }
                self.generic(bytes, format)
            }
            _ => self.generic(bytes, format),
        }
    }

    pub(super) fn recycle(&mut self, image: DynamicImage) {
        self.pixels = image.into_bytes();
    }

    fn jpeg(
        &mut self,
        bytes: &[u8],
        mode: DecodeMode,
    ) -> Result<(DynamicImage, Orientation), ImageError> {
        let jpeg = match &mut self.jpeg {
            Some(jpeg) => jpeg,
            slot @ None => slot.insert(Decompressor::new()?),
        };
        let header = jpeg.read_header(bytes)?;
        let cmyk = matches!(header.colorspace, Colorspace::CMYK | Colorspace::YCCK);
        let format = if cmyk {
            PixelFormat::CMYK
        } else {
            PixelFormat::RGB
        };
        let channels = format.size();
        Limits::default().reserve_buffer(
            header.width as u32,
            header.height as u32,
            if cmyk {
                ColorType::Rgba8
            } else {
                ColorType::Rgb8
            },
        )?;

        // Retain at least two samples per output pixel on the shorter axis before the BOX pass.
        let mut factor = ScalingFactor::ONE;
        if mode == DecodeMode::Fast && !header.is_lossless {
            let short = header.width.min(header.height);
            for candidate in [
                ScalingFactor::ONE_EIGHTH,
                ScalingFactor::ONE_QUARTER,
                ScalingFactor::ONE_HALF,
            ] {
                if candidate.scale(short) >= 2 * RGB_SIDE {
                    factor = candidate;
                    break;
                }
            }
        }
        jpeg.set_scaling_factor(factor)?;
        let size = header.scaled(factor);
        self.pixels.resize(size.width * size.height * channels, 0);
        jpeg.decompress(
            bytes,
            turbojpeg::Image {
                pixels: self.pixels.as_mut_slice(),
                width: size.width,
                height: size.height,
                pitch: size.width * channels,
                format,
            },
        )?;
        if cmyk {
            // JPEG/Pillow use inverted Adobe CMYK components. Compact to RGB in place.
            for i in 0..size.width * size.height {
                let [c, m, y, k] = self.pixels[i * 4..i * 4 + 4].try_into().unwrap();
                let rgb = [c, m, y].map(|ink| ((u32::from(ink) * u32::from(k) + 127) / 255) as u8);
                self.pixels[i * 3..i * 3 + 3].copy_from_slice(&rgb);
            }
            self.pixels.truncate(size.width * size.height * 3);
        }
        let image = rgb_image(size.width as u32, size.height as u32, &mut self.pixels);
        Ok((
            image,
            orientation::jpeg(bytes).unwrap_or(Orientation::NoTransforms),
        ))
    }

    fn generic(
        &mut self,
        bytes: &[u8],
        format: ImageFormat,
    ) -> Result<(DynamicImage, Orientation), ImageError> {
        let mut decoder = ImageReader::with_format(Cursor::new(bytes), format).into_decoder()?;
        let (width, height) = decoder.dimensions();
        if width == 0 || height == 0 {
            return Err(ImageError::Decode(image::ImageError::Parameter(
                image::error::ParameterError::from_kind(
                    image::error::ParameterErrorKind::DimensionMismatch,
                ),
            )));
        }
        let color = decoder.color_type();
        let mut limits = Limits::default();
        limits.reserve(decoder.total_bytes())?;
        decoder.set_limits(limits)?;
        let orientation = if format == ImageFormat::WebP {
            orientation::webp(bytes).unwrap_or(Orientation::NoTransforms)
        } else {
            decoder.orientation()?
        };

        // Native-endian 16-bit grayscale is clipped by Pillow's I;16 -> RGB conversion.
        // For multi-channel PNG it retains the high byte in each channel.
        let eight_bit = match color {
            ColorType::L16 => ColorType::L8,
            ColorType::La16 => ColorType::La8,
            ColorType::Rgb16 => ColorType::Rgb8,
            ColorType::Rgba16 => ColorType::Rgba8,
            ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8 => color,
            _ => {
                let rgb = DynamicImage::from_decoder(decoder)?.into_rgb8();
                return Ok((DynamicImage::ImageRgb8(rgb), orientation));
            }
        };
        self.pixels.resize(decoder.total_bytes() as usize, 0);
        decoder.read_image(&mut self.pixels)?;
        if eight_bit != color {
            let samples = self.pixels.len() / 2;
            for i in 0..samples {
                let n = u16::from_ne_bytes([self.pixels[i * 2], self.pixels[i * 2 + 1]]);
                self.pixels[i] = if color == ColorType::L16 {
                    n.min(255) as u8
                } else {
                    (n >> 8) as u8
                };
            }
            self.pixels.truncate(samples);
        }
        let pixels = mem::take(&mut self.pixels);
        macro_rules! image8 {
            ($variant:ident) => {
                DynamicImage::$variant(
                    image::ImageBuffer::from_raw(width, height, pixels)
                        .expect("decoder byte count matches its pixel type"),
                )
            };
        }
        let image = match eight_bit {
            ColorType::L8 => image8!(ImageLuma8),
            ColorType::La8 => image8!(ImageLumaA8),
            ColorType::Rgb8 => image8!(ImageRgb8),
            ColorType::Rgba8 => image8!(ImageRgba8),
            _ => unreachable!(),
        };
        Ok((image, orientation))
    }
}

pub(super) fn rgb_image(width: u32, height: u32, pixels: &mut Vec<u8>) -> DynamicImage {
    DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(width, height, mem::take(pixels))
            .expect("RGB decoder wrote width * height * 3 bytes"),
    )
}
