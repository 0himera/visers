use std::{num::NonZeroUsize, path::PathBuf, thread};

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::{
    exceptions::{PyOSError, PyRuntimeError, PyValueError},
    prelude::*,
    pybacked::PyBackedBytes,
};
use visers_core::{
    image_io::{DecodeMode, ImageBatchError, ImageError},
    ImageInput, DIMENSIONS,
};

// The Python package normalizes PathLike and mutable buffers before invoking the extension.
// Immutable Python bytes remain owned here while the worker threads borrow them.
#[derive(FromPyObject)]
enum Source {
    Bytes(PyBackedBytes),
    Path(PathBuf),
}

impl Source {
    fn as_input(&self) -> ImageInput<'_> {
        match self {
            Self::Bytes(bytes) => ImageInput::Bytes(bytes),
            Self::Path(path) => ImageInput::Path(path),
        }
    }
}

fn image_error(err: ImageBatchError, sources: &[Source]) -> PyErr {
    let index = err.index;
    let path = match &sources[index] {
        Source::Path(path) => Some(path),
        Source::Bytes(_) => None,
    };
    let location = match path {
        Some(path) => format!("image[{index}] ({})", path.display()),
        None => format!("image[{index}]"),
    };
    let message = format!("{location}: {}", err.source);
    match err.source {
        ImageError::Io(io) => match (io.raw_os_error(), path) {
            (Some(errno), Some(path)) => PyOSError::new_err((errno, message, path.clone())),
            _ => PyErr::from(std::io::Error::new(io.kind(), message)),
        },
        ImageError::Encode(_) => PyRuntimeError::new_err(message),
        _ => PyValueError::new_err(message),
    }
}

#[pyfunction]
#[pyo3(signature = (sources, mode, jobs=None))]
fn encode<'py>(
    py: Python<'py>,
    sources: Vec<Source>,
    mode: &str,
    jobs: Option<isize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mode = match mode {
        "fast" => DecodeMode::Fast,
        "full" => DecodeMode::Full,
        _ => return Err(PyValueError::new_err("mode must be 'fast' or 'full'")),
    };
    let jobs = match jobs {
        Some(n) => usize::try_from(n)
            .ok()
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| PyValueError::new_err("jobs must be a positive integer"))?,
        None if sources.len() <= 1 => NonZeroUsize::MIN,
        None => thread::available_parallelism().unwrap_or(NonZeroUsize::MIN),
    };
    let rows = py
        .detach(|| {
            let inputs: Vec<_> = sources.iter().map(Source::as_input).collect();
            visers_core::encode_with_options(&inputs, mode, jobs)
        })
        .map_err(|err| image_error(err, &sources))?;
    let shape = [rows.len(), DIMENSIONS];
    PyArray1::from_vec(py, rows.into_flattened()).reshape(shape)
}

#[pyfunction]
fn similarity(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f64> {
    visers_core::similarity(a.as_slice()?, b.as_slice()?).map_err(PyValueError::new_err)
}

#[pyfunction]
fn distance(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f64> {
    visers_core::distance(a.as_slice()?, b.as_slice()?).map_err(PyValueError::new_err)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("DIMENSIONS", DIMENSIONS)?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(similarity, m)?)?;
    m.add_function(wrap_pyfunction!(distance, m)?)?;
    Ok(())
}
