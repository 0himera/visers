use std::{num::NonZeroUsize, path::Path, sync::Mutex, thread};

use super::{DecodeMode, ImageError, ImageTimings};
use crate::NanoEncoder;

#[derive(Debug)]
pub struct FileEmbedding {
    pub embedding: Vec<f32>,
    pub timings: ImageTimings,
}

/// Encode files concurrently in input order. Each worker reuses one decoder and inference
/// workspace. An unreadable image yields an error at its own position.
pub fn encode_files<P: AsRef<Path> + Sync>(
    encoder: &NanoEncoder,
    paths: &[P],
    mode: DecodeMode,
    jobs: NonZeroUsize,
) -> Vec<Result<FileEmbedding, ImageError>> {
    let mut results: Vec<_> = paths.iter().map(|_| None).collect();
    run_workers(paths.iter().zip(&mut results), jobs, |items| {
        let mut pipeline = encoder.image_pipeline(mode);
        for (path, result) in items {
            let mut embedding = vec![0.0; encoder.embedding_dim()];
            *result = Some(
                pipeline
                    .encode_file(path, &mut embedding)
                    .map(|timings| FileEmbedding { embedding, timings }),
            );
        }
    });
    results
        .into_iter()
        .map(|r| r.expect("all scoped workers joined"))
        .collect()
}

/// Hand each worker its own iterator over a shared queue. Items can contain disjoint output slots.
pub(super) fn run_workers<I>(
    mut items: I,
    jobs: NonZeroUsize,
    work: impl Fn(&mut dyn Iterator<Item = I::Item>) + Sync,
) where
    I: ExactSizeIterator + Send,
{
    let workers = jobs.get().min(items.len());
    if workers == 0 {
        return;
    }
    if workers == 1 {
        work(&mut items);
        return;
    }
    let queue = Mutex::new(items);
    let worker = || work(&mut std::iter::from_fn(|| queue.lock().unwrap().next()));
    thread::scope(|scope| {
        for _ in 1..workers {
            // Already-started workers and the caller can finish if the OS runs out of threads.
            if thread::Builder::new().spawn_scoped(scope, worker).is_err() {
                break;
            }
        }
        worker();
    });
}
