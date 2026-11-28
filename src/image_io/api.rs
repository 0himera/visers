use std::{
    cell::RefCell,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::Mutex,
    thread,
};

use super::{batch::run_workers, DecodeMode, ImageError, ImagePipeline, ImageTimings};
use crate::{baked_encoder, Embedding, NanoEncoder, DIMENSIONS};

/// One image file on disk or an encoded image in memory.
#[derive(Clone, Copy)]
pub enum ImageInput<'a> {
    Path(&'a Path),
    Bytes(&'a [u8]),
}

impl<'a> From<&'a Path> for ImageInput<'a> {
    fn from(path: &'a Path) -> Self {
        Self::Path(path)
    }
}
impl<'a> From<&'a PathBuf> for ImageInput<'a> {
    fn from(path: &'a PathBuf) -> Self {
        Self::Path(path)
    }
}
impl<'a> From<&'a str> for ImageInput<'a> {
    fn from(path: &'a str) -> Self {
        Self::Path(path.as_ref())
    }
}
impl<'a> From<&'a String> for ImageInput<'a> {
    fn from(path: &'a String) -> Self {
        Self::Path(path.as_ref())
    }
}
impl<'a> From<&'a Vec<u8>> for ImageInput<'a> {
    fn from(bytes: &'a Vec<u8>) -> Self {
        Self::Bytes(bytes)
    }
}
impl<'a> From<&'a [u8]> for ImageInput<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        Self::Bytes(bytes)
    }
}
impl<'a, const N: usize> From<&'a [u8; N]> for ImageInput<'a> {
    fn from(bytes: &'a [u8; N]) -> Self {
        Self::Bytes(bytes)
    }
}

impl<'a> From<&ImageInput<'a>> for ImageInput<'a> {
    fn from(input: &ImageInput<'a>) -> Self {
        *input
    }
}

// Iterating a borrowed collection of paths or byte slices yields references to references.
impl<'a> From<&&'a Path> for ImageInput<'a> {
    fn from(path: &&'a Path) -> Self {
        Self::from(*path)
    }
}
impl<'a> From<&&'a str> for ImageInput<'a> {
    fn from(path: &&'a str) -> Self {
        Self::from(*path)
    }
}
impl<'a> From<&&'a [u8]> for ImageInput<'a> {
    fn from(bytes: &&'a [u8]) -> Self {
        Self::from(*bytes)
    }
}

impl ImageInput<'_> {
    fn encode_into(
        self,
        pipeline: &mut ImagePipeline<'_>,
        out: &mut [f32],
    ) -> Result<ImageTimings, ImageError> {
        match self {
            Self::Path(path) => pipeline.encode_file(path, out),
            Self::Bytes(bytes) => pipeline.encode_bytes(bytes, out),
        }
    }
}

/// Failure of the lowest-index invalid image in a batch. The index starts at zero.
#[derive(Debug, thiserror::Error)]
#[error("image[{index}]: {source}")]
pub struct ImageBatchError {
    pub index: usize,
    #[source]
    pub source: ImageError,
}

thread_local! {
    static PIPELINE: RefCell<Option<ImagePipeline<'static>>> = const { RefCell::new(None) };
}

fn with_pipeline<T>(
    encoder: &'static NanoEncoder,
    mode: DecodeMode,
    f: impl FnOnce(&mut ImagePipeline<'_>) -> T,
) -> T {
    PIPELINE.with_borrow_mut(|slot| {
        let pipeline = slot.get_or_insert_with(|| encoder.image_pipeline(mode));
        pipeline.frame.mode = mode;
        f(pipeline)
    })
}

/// Encode one file path or encoded byte slice with the baked model in fast mode.
///
/// The model is shared across calls. Each calling thread reuses its image pipeline.
/// ```no_run
/// let embedding = visers_nano::encode_single("photo.jpg")?;
/// assert_eq!(embedding.len(), visers_nano::DIMENSIONS);
/// # Ok::<(), visers_nano::image_io::ImageError>(())
/// ```
pub fn encode_single<'a>(input: impl Into<ImageInput<'a>>) -> Result<Embedding, ImageError> {
    let mut out = [0.0; DIMENSIONS];
    encode_into(input, &mut out)?;
    Ok(out)
}

/// Write an embedding into the first `DIMENSIONS` elements of the caller's output and return timings.
/// The buffer must hold at least one embedding. Any remaining elements keep their values.
pub fn encode_into<'a>(
    input: impl Into<ImageInput<'a>>,
    out: &mut [f32],
) -> Result<ImageTimings, ImageError> {
    let input = input.into();
    with_pipeline(
        baked_encoder().map_err(ImageError::Encode)?,
        DecodeMode::Fast,
        |pipe| input.encode_into(pipe, out),
    )
}

/// Encode a batch in fast mode using the available CPU parallelism. Rows follow input order.
/// Use [`ImageInput`] to mix paths and encoded bytes, and [`encode_with_options`] to select full
/// preprocessing or set a worker limit.
/// ```no_run
/// let rows = visers_nano::encode(["photo.jpg", "cover.webp"])?;
/// let score = visers_nano::similarity(&rows[0], &rows[1])?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn encode<'a, T: Into<ImageInput<'a>>>(
    inputs: impl IntoIterator<Item = T>,
) -> Result<Vec<Embedding>, ImageBatchError> {
    let inputs: Vec<_> = inputs.into_iter().map(Into::into).collect();
    let jobs = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    encode_with_options(&inputs, DecodeMode::Fast, jobs)
}

/// Encode paths and byte slices into contiguous rows using the requested preprocessing and worker count.
/// Returns an empty vector for an empty batch. If decoding or reading fails, the error
/// identifies the lowest failing input index regardless of the order in which workers finish.
///
/// ```no_run
/// use std::num::NonZeroUsize;
/// use visers_nano::{encode_with_options, image_io::DecodeMode, ImageInput};
///
/// let upload = std::fs::read("cover.webp")?;
/// let inputs = [ImageInput::from("photo.jpg"), ImageInput::from(upload.as_slice())];
/// let rows = encode_with_options(&inputs, DecodeMode::Full, NonZeroUsize::new(2).unwrap())?;
/// assert_eq!(rows.len(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn encode_with_options(
    inputs: &[ImageInput<'_>],
    mode: DecodeMode,
    jobs: NonZeroUsize,
) -> Result<Vec<Embedding>, ImageBatchError> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let encoder = baked_encoder().map_err(|err| ImageBatchError {
        index: 0,
        source: ImageError::Encode(err),
    })?;
    let mut rows = vec![[0.0; DIMENSIONS]; inputs.len()];
    let first_error = Mutex::new(None::<ImageBatchError>);
    run_workers(inputs.iter().zip(&mut rows).enumerate(), jobs, |items| {
        with_pipeline(encoder, mode, |pipeline| {
            for (index, (input, out)) in items {
                if let Err(source) = input.encode_into(pipeline, out) {
                    let mut first = first_error.lock().unwrap();
                    if first.as_ref().is_none_or(|e| index < e.index) {
                        *first = Some(ImageBatchError { index, source });
                    }
                }
            }
        });
    });
    match first_error.into_inner().unwrap() {
        Some(err) => Err(err),
        None => Ok(rows),
    }
}
