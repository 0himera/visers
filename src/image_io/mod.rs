//! Decode ordinary image files and run the Nano encoder. Enable the `image-files` Cargo feature.

mod api;
mod batch;
mod box_resize;
mod decode;
mod orientation;
mod resize;
mod webp;

pub use api::{
    encode, encode_into, encode_single, encode_with_options, ImageBatchError, ImageInput,
};
pub use batch::{encode_files, FileEmbedding};

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::{NanoEncoder, NanoWorkspace};

/// Select the image preprocessing used before the Nano model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecodeMode {
    /// Use decoder-side scaling for large JPEGs and opaque still WebPs, and a SIMD BOX resize.
    #[default]
    Fast,
    /// Decode the full raster and use the BOX pass order and rounding from Forge (Pillow 13).
    Full,
}

/// Wall time per stage. Decoder-side scaling is included in `decode`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImageTimings {
    pub read: Duration,
    pub decode: Duration,
    pub resize: Duration,
    pub inference: Duration,
}

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Decode(#[from] image::ImageError),
    #[error(transparent)]
    Jpeg(#[from] turbojpeg::Error),
    #[error("WebP decoding failed ({0})")]
    WebP(String),
    #[error(transparent)]
    Resize(#[from] fast_image_resize::ResizeError),
    #[error(transparent)]
    Buffer(#[from] fast_image_resize::ImageBufferError),
    #[error("{0}")]
    Encode(&'static str),
}

/// A reusable file/bytes-to-embedding pipeline. Give each concurrent worker its own instance.
pub struct ImagePipeline<'a> {
    encoded: Vec<u8>,
    frame: FrameEncoder<'a>,
}

struct FrameEncoder<'a> {
    encoder: &'a NanoEncoder,
    workspace: NanoWorkspace,
    decoder: decode::Decoder,
    thumbnail: resize::Thumbnail,
    oriented: Vec<u8>,
    mode: DecodeMode,
}

impl NanoEncoder {
    pub fn image_pipeline(&self, mode: DecodeMode) -> ImagePipeline<'_> {
        ImagePipeline {
            encoded: Vec::new(),
            frame: FrameEncoder {
                encoder: self,
                workspace: self.workspace(),
                decoder: decode::Decoder::default(),
                thumbnail: resize::Thumbnail::default(),
                oriented: Vec::new(),
                mode,
            },
        }
    }
}

impl ImagePipeline<'_> {
    /// Read an image from disk. Its signature selects the codec.
    pub fn encode_file(
        &mut self,
        path: impl AsRef<Path>,
        out: &mut [f32],
    ) -> Result<ImageTimings, ImageError> {
        let start = Instant::now();
        self.encoded.clear();
        File::open(path)?.read_to_end(&mut self.encoded)?;
        let read = start.elapsed();
        let mut timings = self.frame.encode(&self.encoded, out)?;
        timings.read = read;
        Ok(timings)
    }

    /// Encode an in-memory file, including a file with no name or extension.
    pub fn encode_bytes(
        &mut self,
        bytes: &[u8],
        out: &mut [f32],
    ) -> Result<ImageTimings, ImageError> {
        self.frame.encode(bytes, out)
    }
}

impl FrameEncoder<'_> {
    fn encode(&mut self, bytes: &[u8], out: &mut [f32]) -> Result<ImageTimings, ImageError> {
        if out.len() < self.encoder.embedding_dim() {
            return Err(ImageError::Encode(
                "output slice is too small for the embedding",
            ));
        }
        let start = Instant::now();
        let (mut image, orientation) = self.decoder.decode(bytes, self.mode)?;
        let decoded = Instant::now();
        let orientation = match self.mode {
            DecodeMode::Full => {
                image.apply_orientation(orientation);
                image::metadata::Orientation::NoTransforms
            }
            DecodeMode::Fast => orientation,
        };
        let rgb = self.thumbnail.rgb(&image, self.mode)?;
        let rgb = orientation::apply_to_thumbnail(rgb, orientation, &mut self.oriented)?;
        let resized = Instant::now();
        self.encoder
            .encode_rgb(rgb, &mut self.workspace, out)
            .map_err(ImageError::Encode)?;
        let inference = resized.elapsed();
        self.decoder.recycle(image);
        Ok(ImageTimings {
            read: Duration::ZERO,
            decode: decoded - start,
            resize: resized - decoded,
            inference,
        })
    }
}
